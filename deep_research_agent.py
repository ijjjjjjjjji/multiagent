import json
import math
import argparse
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import TypedDict, List, Literal
from urllib.parse import urlsplit, urlunsplit
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
import pickle
from pydantic import BaseModel, Field, ValidationError
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from langchain_classic.retrievers import ParentDocumentRetriever, EnsembleRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymilvus import connections
from langgraph.graph import StateGraph, START, END

import dynamic_searcher

def print_step(title):
    print(f"\n{'='*15} {title} {'='*15}")


def _clean_model_output(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        t = "\n".join(lines).strip()

    # 清理模型常见的模板尾注与客套结束语，避免污染最终报告。
    t = re.sub(r"【注】[\s\S]*?【/注】", "", t, flags=re.IGNORECASE).strip()
    t = re.sub(r"希望这些信息对你有所帮助！?", "", t).strip()
    t = re.sub(r"如果有任何其他问题[\s\S]*?随时告知。?", "", t).strip()
    t = re.sub(r"祝你写作顺利！?", "", t).strip()
    return t


def _report_len(text: str) -> int:
    """计算文本实际长度（去除首尾空白）"""
    return len((text or "").strip())


def _parse_fact_check_fallback(raw_text: str) -> "FactCheckResult":
    """解析 Fact-Checker 输出的 JSON，支持双重降级"""
    cleaned = _clean_model_output(raw_text)
    try:
        return FactCheckResult.model_validate_json(cleaned)
    except ValidationError:
        data = json.loads(cleaned)
        return FactCheckResult.model_validate(data)


# ===== 检索与质量控制参数 =====
SEMANTIC_SIM_THRESHOLD = 0.35        # 初始语义相似度阈值
MIN_HIGH_QUALITY_SOURCES = 8         # 目标高质量源数
MIN_HIGH_QUALITY_SOURCES_FALLBACK = 5  # 降级后的最少源数
SEMANTIC_SIM_FALLBACKS = [0.35, 0.25, 0.15]  # 阈值渐进降级策略

# ===== 写作与长度控制参数 =====
MIN_REPORT_CHARS = 4000              # 最低章节长度要求
TARGET_REPORT_CHARS = 6000           # 目标章节长度
MAX_EXPAND_ROUNDS = 2                # 全报告最多扩写轮次
MAX_SECTION_EXPAND_ROUNDS = 1        # 单章最多扩写轮次
MAX_RETRY_COUNT = 3                  # 审查最多重试次数

# ===== 模型超时控制（秒）=====
WRITER_FIRST_DRAFT_TIMEOUT_SEC = 180
WRITER_EXPAND_TIMEOUT_SEC = 150
WRITER_SECTION_TIMEOUT_SEC = 120
WRITER_SECTION_EXPAND_TIMEOUT_SEC = 90
FACT_CHECK_TIMEOUT_SEC = 90

# ===== Ollama 生成参数 =====
OLLAMA_NUM_PREDICT = 2048            # 最多生成 token 数，防止截断
OLLAMA_TOP_P = 0.9                   # 核采样参数，控制多样性


class FactCheckResult(BaseModel):
    is_pass: bool = Field(..., description="是否通过一致性校验")
    error_type: Literal["hallucination", "missing_info", "format_error", "none"] = Field(
        default="none",
        description="错误类型：幻觉、信息不足、格式错误或无错误",
    )
    feedback: str = Field(default="", description="给 Writer 的具体修改指导")
    new_search_query: str = Field(default="", description="若信息不足，给 Retrieval 的新检索词")


class EditorPlanResult(BaseModel):
    sub_queries: List[str] = Field(
        default_factory=list,
        description="围绕研究课题拆解出的子检索词列表，建议 3 条",
    )


class SectionWriteResult(BaseModel):
    section_markdown: str = Field(
        default="",
        description="完整章节 Markdown，必须以对应二级标题开头",
    )


class ReportExpandResult(BaseModel):
    report_markdown: str = Field(
        default="",
        description="完整可替换的 Markdown 报告",
    )

# ===== 核心 1: LangGraph 全局状态定义 =====
class ResearchState(TypedDict):
    """多智能体间的共享状态机。无直接通信，所有数据交换通过 State 字段"""
    topic: str
    sub_queries: List[str]   # Editor 拆解输出
    context: str             # Searcher 检索结果（含来源标注）
    draft: str               # Writer 撰写的报告
    critique: str            # Fact-Checker 的修改意见
    fact_check_is_pass: bool
    fact_check_error_type: str
    fact_check_feedback: str
    new_search_query: str    # 新增补检索词（由 Checker 提出）
    retry_count: int         # 当前循环的重试计数
    trace: List[dict]        # 执行路由追踪
    iterations: int          # 防止死循环的迭代计数

# ===== 核心 2: 全局模型实例（单例模式）===== 
_LLM = None          # ChatOllama 实例
_EMBEDDINGS = None   # HuggingFaceEmbeddings 实例


def _get_llm() -> ChatOllama:
    """获取 LLM 实例（首次调用时初始化"""
    global _LLM
    if _LLM is None:
        print("🧠 正在唤醒本地 4090 算力引擎 (Qwen2.5-7B)...")
        _LLM = ChatOllama(
            model="qwen2.5:7b-instruct",
            temperature=0.2,
            num_predict=OLLAMA_NUM_PREDICT,
            top_p=OLLAMA_TOP_P,
            base_url="http://localhost:11434",
        )
    return _LLM


def _invoke_chain_with_timeout(chain, payload: dict, timeout_sec: int, stage_name: str):
    """执行 LangChain 链调用，带超时保护和降级
    
    Args:
        chain: LangChain Runnable 对象
        payload: 调用输入参数
        timeout_sec: 超时时间（秒）
        stage_name: 阶段名称（用于日志和错误追踪）
    
    Returns:
        模型输出或 None（超时时）
    """
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(chain.invoke, payload)
        try:
            return future.result(timeout=timeout_sec)
        except FuturesTimeoutError:
            print(f"   ⚠️ {stage_name} 超时（>{timeout_sec}s），触发降级策略。")
            future.cancel()
            return None


def _invoke_structured_with_timeout(
    llm,
    prompt,
    output_model,
    timeout_sec: int,
    stage_name: str,
):
    """调用模型并强制返回指定 Pydantic 模式的输出（结构化输出）
    
    这是本系统的核心：确保所有 LLM 调用都返回结构化输出，不产生自由文本。
    模型无工具选择自由，必须输出指定的 Pydantic 字段。
    """
    structured_llm = llm.with_structured_output(output_model)
    chain = prompt | structured_llm
    return _invoke_chain_with_timeout(chain, {}, timeout_sec, stage_name)


def _get_embeddings() -> HuggingFaceEmbeddings:
    """获取嵌入模型实例（首次调用时初始化）"""
    global _EMBEDDINGS
    if _EMBEDDINGS is None:
        _EMBEDDINGS = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={'device': 'cuda'},
            encode_kwargs={'normalize_embeddings': True},
        )
    return _EMBEDDINGS

# ===== 核心 3: Editor 节点（主编智能体）===== 
def editor_node(state: ResearchState) -> dict:
    """将宽泛的研究主题拆解为 3 个具体的搜索关键词，输出结构化 EditorPlanResult"""
    print_step("👔 [Editor Agent] 主编正在拆解研究课题...")
    topic = state["topic"]
    print(f"   ➤ 收到总课题: '{topic}'")
    
    system_prompt = """你是一位资深的行业研究主编。
你的任务是将用户提供的【宏大研究主题】，拆解为 3 个极具针对性的【搜索引擎关键词】。
这些关键词必须涵盖：1. 核心技术突破 2. 市场规模与商业化 3. 行业痛点与竞品。

严格遵守以下纪律：
1. 你必须通过结构化字段 sub_queries 返回结果。
2. sub_queries 目标长度为 3，且去重。
3. 不要在字段内容里包含解释性前后缀。
"""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", f"【宏大研究主题】：{topic}")
    ])

    llm = _get_llm()
    # 调用结构化输出，模型必须返回 EditorPlanResult 格式
    response = _invoke_structured_with_timeout(
        llm=llm,
        prompt=prompt,
        output_model=EditorPlanResult,
        timeout_sec=WRITER_SECTION_TIMEOUT_SEC,
        stage_name="Editor结构化拆题",
    )

    try:
        if response is None:
            raise TimeoutError("Editor 结构化调用超时")

        sub_queries = [q.strip() for q in response.sub_queries if (q or "").strip()]
        # 去重保序以避免重复检索同一关键词
        deduped = []
        seen = set()
        for q in sub_queries:
            if q in seen:
                continue
            seen.add(q)
            deduped.append(q)
        sub_queries = deduped[:3]

        if not sub_queries:
            raise ValueError("解析结果不是列表")
            
    except Exception as e:
        print(f"   ⚠️ Editor 结构化调用失败: {str(e)}。触发 Fallback 机制。")
        # 降级：模型调用失败时，用简单规则生成备选关键词
        sub_queries = [f"{topic} 最新进展", f"{topic} 商业化落地"]
        
    print(f"   ➤ 课题拆解完成！子课题清单: {sub_queries}")
    return {"sub_queries": sub_queries}

# ===== 核心 4: 动态检索器构建 ===== 
def build_dynamic_retriever():
    """重建混合检索器（Milvus 密集向量 + BM25 稀疏）
    
    从 dynamic_searcher 的持久化文件中恢复 DocStore 和 BM25 索引，
    与 Milvus 向量库联合构成 EnsembleRetriever（70% 语义 + 30% 关键词）
    """
    print("⏳ [系统] 正在唤醒动态知识库引擎，准备提炼 Context...")
    
    embeddings = _get_embeddings()
    
    milvus_uri = f"http://{dynamic_searcher.MILVUS_HOST}:{dynamic_searcher.MILVUS_PORT}"
    vectorstore = Milvus(
        embedding_function=embeddings,
        connection_args={"uri": milvus_uri},
        collection_name=dynamic_searcher.COLLECTION_NAME
    )
    connections.connect(alias=vectorstore.alias, uri=milvus_uri)

    # 从文件恢复 DocStore（父文档存储）和 BM25 索引
    with open(dynamic_searcher.STORE_PATH, "rb") as f:
        saved_data = pickle.load(f)
        store = saved_data["docstore"]
        bm25_retriever = saved_data["bm25"]

    # 保持与摄取阶段一致的父子分割参数
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。"],
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=150,
        chunk_overlap=30,
        separators=["。", "！", "；", "，"],
    )

    parent_retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )
    
    # 混合检索：来自向量的语义相似文档（70%）和 BM25 关键词匹配（30%）
    return EnsembleRetriever(retrievers=[parent_retriever, bm25_retriever], weights=[0.7, 0.3])


# ===== 核心 5: Searcher 节点（研究员智能体）===== 
def searcher_node(state: ResearchState) -> dict:
    """执行网络检索、文档摄取、混合检索三步，提炼高价值的 Context
    
    固定流程：search_tool_collect → search_tool_ingest → search_tool_retrieve
    """
    print_step("🕵️ [Searcher Agent] 研究员开始全网扫街与知识淬炼...")
    sub_queries = list(state["sub_queries"])
    topic = state["topic"]

    # 若前一轮检查认为信息不足，优先加入补充检索词
    extra_query = (state.get("new_search_query") or "").strip()
    if extra_query and extra_query not in sub_queries:
        sub_queries.insert(0, extra_query)
        print(f"   ➤ Fact-Checker 触发补检索词: {extra_query}")

    # 三步检索流程：先采集 -> 再入库 -> 最后检索
    
    all_scraped_docs, collect_result = dynamic_searcher.search_tool_collect_documents(
        sub_queries=sub_queries,
        max_results_per_query=8,
    )

    if collect_result.get("status") == "empty":
        print("   ❌ 警告：所有子课题均未抓取到有效数据！")
        return {"context": "检索失败：未抓取到任何有效网络数据。"}

    ingest_result = dynamic_searcher.search_tool_ingest_documents(all_scraped_docs)
    if ingest_result.get("status") == "empty":
        return {"context": "检索失败：抓取结果为空，无法入库。"}

    retrieve_result = dynamic_searcher.search_tool_retrieve_context(
        topic=topic,
        retriever=build_dynamic_retriever(),
        rerank_embeddings=_get_embeddings(),
        semantic_sim_threshold=SEMANTIC_SIM_THRESHOLD,
        semantic_sim_fallbacks=SEMANTIC_SIM_FALLBACKS,
        min_high_quality_sources=MIN_HIGH_QUALITY_SOURCES,
        min_high_quality_sources_fallback=MIN_HIGH_QUALITY_SOURCES_FALLBACK,
        candidate_limit=20,
        max_return_docs=15,
    )

    if retrieve_result.get("status") == "no_candidates":
        return {"context": retrieve_result.get("message", "检索完成，但无候选文档返回。")}
    if retrieve_result.get("status") == "insufficient":
        return {"context": retrieve_result.get("message", "证据不足")}

    print(f"   ➤ {retrieve_result.get('message', '知识淬炼完成。')}")
    return {"context": retrieve_result.get("context", "")}


REPORT_SECTION_PLAN = [
    {
        "title": "引言与方法",
        "focus": "研究范围、关键定义、方法论边界、证据来源可信度说明",
        "min_chars": 320,
    },
    {
        "title": "产业规模与增长测算",
        "focus": "市场规模、出货量、增速、区域差异、驱动因素",
        "min_chars": 450,
    },
    {
        "title": "技术路线与工程可行性",
        "focus": "感知、控制、执行器、具身智能栈、系统集成难点",
        "min_chars": 450,
    },
    {
        "title": "竞品格局与厂商对比",
        "focus": "头部厂商策略、产品定位、能力差异、护城河",
        "min_chars": 450,
    },
    {
        "title": "商业模式与收入结构",
        "focus": "一次性销售、订阅服务、运维、生态合作与渠道",
        "min_chars": 450,
    },
    {
        "title": "成本结构与降本路径",
        "focus": "BOM、研发、制造、部署运维成本及规模化拐点",
        "min_chars": 450,
    },
    {
        "title": "落地场景与实施路径",
        "focus": "制造业、物流、服务业等场景优先级与导入路线图",
        "min_chars": 450,
    },
    {
        "title": "风险与情景预测",
        "focus": "技术、法规、供应链、ROI 风险及基准/乐观/保守情景",
        "min_chars": 450,
    },
    {
        "title": "结论与行动建议",
        "focus": "关键结论、未来 12-24 个月建议、里程碑指标",
        "min_chars": 320,
    },
]


def _ensure_section_header(section_text: str, section_title: str) -> str:
    t = (section_text or "").strip()
    if not t:
        return f"## {section_title}\n\n（该章节生成失败，建议重试。）"
    if not t.startswith("## "):
        t = f"## {section_title}\n\n{t}"
    return t


def _generate_section(
    llm,
    topic: str,
    context: str,
    section_title: str,
    section_focus: str,
    section_min_chars: int,
    critique: str,
    previous_sections: str,
) -> str:
    system_prompt = f"""你是一位顶尖的科技与商业领域首席分析师。
你需要通过结构化字段输出指定章节，不能输出整篇报告。

【最高纪律】
1. 只能依据【参考情报】写作，严禁杜撰。
2. 每个关键事实都必须在句尾标注 [来源 X]。
3. section_markdown 必须是 Markdown，且以二级标题开头：## {section_title}
4. 本章节至少 {section_min_chars} 字符，优先展开论证深度，不要空泛。
5. 严禁输出客套话、注释块、过程说明。
"""

    user_prompt = (
        f"【研究课题】\n{topic}\n\n"
        f"【本章标题】\n{section_title}\n\n"
        f"【本章重点】\n{section_focus}\n\n"
        f"【已写章节摘要（用于衔接风格，禁止重复）】\n{previous_sections[-1200:]}\n\n"
        f"【参考情报】\n{context}\n"
    )
    if critique:
        user_prompt += (
            "\n======================\n"
            f"【核查员打回意见/必须修改点】\n{critique}\n"
            "请在本章中显式修复相关问题。"
        )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", user_prompt),
    ])
    # 固定调用章节生成“函数”，不允许模型自主挑选其他工具。
    resp = _invoke_structured_with_timeout(
        llm=llm,
        prompt=prompt,
        output_model=SectionWriteResult,
        timeout_sec=WRITER_SECTION_TIMEOUT_SEC,
        stage_name=f"Writer章节结构化生成-{section_title}",
    )
    if resp is None:
        return f"## {section_title}\n\n（章节生成超时，建议缩小范围后重试。）"

    return _ensure_section_header(_clean_model_output(resp.section_markdown), section_title)


def _expand_section_if_needed(
    llm,
    topic: str,
    context: str,
    section_title: str,
    section_focus: str,
    section_text: str,
    section_min_chars: int,
) -> str:
    current = section_text
    for round_idx in range(1, MAX_SECTION_EXPAND_ROUNDS + 1):
        if _report_len(current) >= section_min_chars:
            break

        expand_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                f"""你是一位研究报告编辑。你要扩写指定章节，并通过结构化字段输出完整替换版本。
规则：
1. 只能使用【参考情报】和【已有章节】中的事实，禁止杜撰。
2. 保留并补充 [来源 X] 标注。
3. section_markdown 必须以二级标题开头：## {section_title}
4. 扩写后的章节至少 {section_min_chars} 字符。""",
            ),
            (
                "user",
                f"【研究课题】\n{topic}\n\n"
                f"【章节标题】\n{section_title}\n\n"
                f"【章节重点】\n{section_focus}\n\n"
                f"【参考情报】\n{context}\n\n"
                f"【已有章节】\n{current}\n\n"
                "请输出扩写后的完整章节。",
            ),
        ])
        expand_resp = _invoke_structured_with_timeout(
            llm=llm,
            prompt=expand_prompt,
            output_model=SectionWriteResult,
            timeout_sec=WRITER_SECTION_EXPAND_TIMEOUT_SEC,
            stage_name=f"Writer章节结构化补写-{section_title}-轮次{round_idx}",
        )
        if expand_resp is None:
            break

        current = _ensure_section_header(_clean_model_output(expand_resp.section_markdown), section_title)

    return current

# ================= 核心 6: 撰稿人智能体 (Writer Node) =================
def writer_node(state: ResearchState) -> dict:
    """
    Writer Agent: 负责将 Searcher 提炼出的高价值 Context 转化为结构化研报初稿。
    如果收到 Fact-Checker 的打回意见 (critique)，则进行针对性修改。
    """
    print_step("✍️ [Writer Agent] 撰稿人正在起草深度研报...")
    
    topic = state["topic"]
    context = state["context"]
    critique = state.get("critique", "")
    
    # 异常流处理：如果 Searcher 没有找到足够证据，直接终止胡编乱造
    if "证据不足" in context or "检索失败" in context:
        print("   ⚠️ 检测到检索数据不足，Writer 拒绝生成幻觉报告。")
        return {"draft": f"【系统提示】：关于“{topic}”的资料获取失败，原因：{context}\n请尝试更换关键词或扩充信源。"}
    
    llm = _get_llm()
    section_outputs = []

    if critique:
        print("   ➤ 收到核查员打回意见，启用按章节重写模式...")
    else:
        print("   ➤ 启用按章节流水线写作，逐章生成并校验长度...")

    for idx, section in enumerate(REPORT_SECTION_PLAN, start=1):
        section_title = section["title"]
        section_focus = section["focus"]
        section_min_chars = int(section["min_chars"])

        print(f"   ➤ 正在生成章节 [{idx}/{len(REPORT_SECTION_PLAN)}]: {section_title}")
        previous_text = "\n\n".join(section_outputs)
        section_text = _generate_section(
            llm=llm,
            topic=topic,
            context=context,
            section_title=section_title,
            section_focus=section_focus,
            section_min_chars=section_min_chars,
            critique=critique,
            previous_sections=previous_text,
        )

        before_expand_len = _report_len(section_text)
        if before_expand_len < section_min_chars:
            print(
                f"      - 章节长度 {before_expand_len} < {section_min_chars}，触发章节补写..."
            )
            section_text = _expand_section_if_needed(
                llm=llm,
                topic=topic,
                context=context,
                section_title=section_title,
                section_focus=section_focus,
                section_text=section_text,
                section_min_chars=section_min_chars,
            )

        section_outputs.append(section_text)

    draft = f"# {topic}研究报告\n\n" + "\n\n".join(section_outputs)

    # 章节补写后仍偏短，则进行一轮全局定向补写。
    for round_idx in range(1, MAX_EXPAND_ROUNDS + 1):
        current_len = _report_len(draft)
        if current_len >= MIN_REPORT_CHARS:
            break
        print(
            f"   ➤ 全文长度 {current_len}，仍低于 {MIN_REPORT_CHARS}，触发全局补写轮次 {round_idx}/{MAX_EXPAND_ROUNDS}..."
        )

        expand_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                f"""你是一位专业研究报告编辑。请在不引入新事实的前提下做定向扩写，并通过结构化字段输出。
规则：
1. 仅使用【参考情报】与【已有报告】事实，严禁杜撰。
2. 保留并补充 [来源 X] 标注。
3. 保持章节结构不变，只扩展论证与细节。
4. report_markdown 必须是完整可替换 Markdown 报告。
5. 最终报告至少 {MIN_REPORT_CHARS} 字符，目标 {TARGET_REPORT_CHARS} 字符。""",
            ),
            (
                "user",
                f"【研究课题】\n{topic}\n\n【参考情报】\n{context}\n\n【已有报告】\n{draft}\n\n请输出扩写后的完整终稿。",
            ),
        ])
        expand_resp = _invoke_structured_with_timeout(
            llm=llm,
            prompt=expand_prompt,
            output_model=ReportExpandResult,
            timeout_sec=WRITER_EXPAND_TIMEOUT_SEC,
            stage_name=f"Writer全局结构化补写轮次{round_idx}",
        )
        if expand_resp is None:
            print("   ⚠️ 全局补写超时，保留当前内容继续流程。")
            break
        draft = _clean_model_output(expand_resp.report_markdown)
    
    print("   ➤ 报告撰写完成！")
    print(f"   ➤ 最终稿长度: {_report_len(draft)} 字符")
    
    # 将生成的草稿更新到 State 中
    return {"draft": draft}


# ================= 核心 7: 事实核查智能体 (Fact-Checker Node) =================
def fact_checker_node(state: ResearchState) -> dict:
    """
    Fact-Checker Agent: 基于 retrieval grounding 做一致性校验。
    强制返回结构化字段：is_pass / error_type / feedback / new_search_query。
    """
    print_step("🧪 [Fact-Checker Agent] 正在执行 grounding 一致性校验...")

    topic = state["topic"]
    context = state.get("context", "")
    draft = state.get("draft", "")

    if not draft.strip():
        result = FactCheckResult(
            is_pass=False,
            error_type="format_error",
            feedback="草稿为空，请按参考情报重新生成完整报告。",
            new_search_query="",
        )
        return {
            "fact_check_is_pass": result.is_pass,
            "fact_check_error_type": result.error_type,
            "fact_check_feedback": result.feedback,
            "new_search_query": result.new_search_query,
            "critique": result.feedback,
        }

    system_prompt = """你是一位严格的事实核查官。你必须根据【参考情报】对【报告草稿】进行 retrieval grounding 一致性校验。

判定标准：
1. hallucination：草稿包含参考情报中不存在的信息，或与参考情报冲突。
2. missing_info：草稿试图回答的问题在参考情报中证据不足，关键实体/数据缺失，需要补检索。
3. format_error：草稿结构或来源标注严重不符合要求（如大量缺失 [来源 X]）。
4. none：通过校验。

输出要求：
你必须返回一个 JSON 对象，且只包含以下字段：
- is_pass: bool
- error_type: "hallucination" | "missing_info" | "format_error" | "none"
- feedback: string
- new_search_query: string
"""

    user_prompt = (
        f"【研究课题】\n{topic}\n\n"
        f"【参考情报】\n{context}\n\n"
        f"【报告草稿】\n{draft}\n\n"
        "请严格按指定 JSON 字段返回。"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", user_prompt),
    ])

    llm = _get_llm()

    try:
        checker_resp = _invoke_structured_with_timeout(
            llm=llm,
            prompt=prompt,
            output_model=FactCheckResult,
            timeout_sec=FACT_CHECK_TIMEOUT_SEC,
            stage_name="Fact-Checker结构化校验",
        )
        if checker_resp is None:
            raise TimeoutError("Fact-Checker 结构化校验超时")
        result = checker_resp
    except Exception:
        fallback_chain = prompt | llm
        fallback_resp = _invoke_chain_with_timeout(
            fallback_chain,
            {},
            FACT_CHECK_TIMEOUT_SEC,
            "Fact-Checker回退JSON校验",
        )
        raw = fallback_resp.content if fallback_resp is not None else ""
        try:
            result = _parse_fact_check_fallback(raw)
        except Exception:
            result = FactCheckResult(
                is_pass=False,
                error_type="format_error",
                feedback="Fact-Checker 结构化输出解析失败，请按参考情报重写并补全来源标注。",
                new_search_query=topic,
            )

    if result.is_pass:
        result.error_type = "none"

    if (not result.is_pass) and result.error_type == "missing_info" and not result.new_search_query.strip():
        result.new_search_query = topic

    current_retry = state.get("retry_count", 0)
    if not result.is_pass:
        current_retry += 1

    print(
        "   ➤ 校验结果: "
        f"is_pass={result.is_pass}, error_type={result.error_type}, "
        f"new_search_query={result.new_search_query or 'N/A'}, "
        f"retry_count={current_retry}/{MAX_RETRY_COUNT}"
    )

    return {
        "fact_check_is_pass": result.is_pass,
        "fact_check_error_type": result.error_type,
        "fact_check_feedback": result.feedback,
        "new_search_query": result.new_search_query,
        "critique": result.feedback,
        "retry_count": current_retry,
    }


def route_after_verification(state: ResearchState) -> str:
    """Router: 根据 Fact-Checker 结果决定流向。"""
    if state.get("fact_check_is_pass", False):
        return "end"

    if state.get("retry_count", 0) >= MAX_RETRY_COUNT:
        return "end_force"

    err = state.get("fact_check_error_type", "format_error")
    if err == "missing_info":
        return "searcher"
    if err in ("hallucination", "format_error"):
        return "writer"

    return "writer"


def force_end_node(state: ResearchState) -> dict:
    """超过最大重试次数时的降级终止节点。"""
    print("   ⚠️ 达到最大重试次数，触发降级终止。")
    if not state.get("fact_check_is_pass", False):
        draft = state.get("draft", "")
        if not draft.startswith("【系统降级提示】"):
            draft = (
                "【系统降级提示】由于事实冲突/信息不足在最大重试次数内无法完全解决，"
                "以下为部分可用草稿，请人工复核关键结论。\n\n" + draft
            )
        trace = list(state.get("trace", []))
        trace.append(
            {
                "node": "force_end",
                "error_type": state.get("fact_check_error_type", "unknown"),
                "retry_count": state.get("retry_count", 0),
                "next": "END",
            }
        )
        return {"draft": draft, "trace": trace}
    return {}


def trace_router_node(state: ResearchState) -> dict:
    """记录每轮校验后的路由决策，便于实验追踪与稳定性统计。"""
    next_route = route_after_verification(state)
    trace = list(state.get("trace", []))
    trace.append(
        {
            "node": "checker",
            "error_type": state.get("fact_check_error_type", "none"),
            "retry_count": state.get("retry_count", 0),
            "next": next_route,
        }
    )
    return {
        "trace": trace,
        "iterations": len(trace),
    }


def build_research_graph():
    """构建 LangGraph StateGraph 工作流。"""
    graph = StateGraph(ResearchState)

    graph.add_node("editor", editor_node)
    graph.add_node("searcher", searcher_node)
    graph.add_node("writer", writer_node)
    graph.add_node("checker", fact_checker_node)
    graph.add_node("trace_router", trace_router_node)
    graph.add_node("force_end", force_end_node)

    graph.add_edge(START, "editor")
    graph.add_edge("editor", "searcher")
    graph.add_edge("searcher", "writer")
    graph.add_edge("writer", "checker")
    graph.add_edge("checker", "trace_router")

    graph.add_conditional_edges(
        "trace_router",
        route_after_verification,
        {
            "end": END,
            "writer": "writer",
            "searcher": "searcher",
            "end_force": "force_end",
        },
    )
    graph.add_edge("force_end", END)

    return graph.compile()


def run_research(topic: str) -> dict:
    """执行完整研报工作流并返回最终状态。"""
    initial_state = ResearchState(
        topic=topic,
        sub_queries=[],
        context="",
        draft="",
        critique="",
        fact_check_is_pass=False,
        fact_check_error_type="none",
        fact_check_feedback="",
        new_search_query="",
        retry_count=0,
        trace=[],
        iterations=0,
    )

    app = build_research_graph()
    state = app.invoke(initial_state)
    return state


# ================= 阶段性测试：三 Agent 接力流转 =================
if __name__ == "__main__":
    print("\n🚀 启动 [Editor -> Searcher -> Writer -> Checker] LangGraph 深度研报流水线压测...")

    parser = argparse.ArgumentParser(description="Deep Research Agent Runner")
    parser.add_argument("--topic", type=str, default="", help="研究课题（可选）")
    args = parser.parse_args()

    default_topic = "2026年人形机器人商业化落地现状"
    topic = (args.topic or "").strip()
    if not topic:
        try:
            user_input = input(f"请输入研究课题（回车使用默认：{default_topic}）：").strip()
        except EOFError:
            user_input = ""
        topic = user_input or default_topic
    
    state = run_research(topic)
    if state.get("fact_check_is_pass", False):
        print("   ✅ Fact-Checker 校验通过，流程结束。")

    print("\n✨ ================= [最终生成的专业研报] ================= ✨\n")
    print(state['draft'])
    print("\n📊 ================= [路由追踪 Trace] ================= 📊")
    for i, item in enumerate(state.get("trace", []), start=1):
        print(
            f"[{i}] node={item.get('node')} | "
            f"error_type={item.get('error_type')} | "
            f"retry_count={item.get('retry_count')} | "
            f"next={item.get('next')}"
        )
    print(f"总决策轮次: {state.get('iterations', 0)}")
    print("\n=========================================================")
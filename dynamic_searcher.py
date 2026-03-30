import os
import time
import pickle
import atexit
import io
import re
import math
import threading
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from urllib.parse import urlsplit, urlunsplit

try:
    from ddgs import DDGS
    DuckDuckGoSearchException = Exception
except Exception:
    from duckduckgo_search import DDGS
    from duckduckgo_search.exceptions import DuckDuckGoSearchException

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from pymilvus import connections
from milvus_lite.server import Server

# 混合检索（向量 + BM25）与父子文档索引组件
from langchain_classic.retrievers import ParentDocumentRetriever, EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.stores import InMemoryStore

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

try:
    from pdfminer.high_level import extract_text as pdfminer_extract_text
except Exception:
    pdfminer_extract_text = None

try:
    from pdf2image import convert_from_bytes
except Exception:
    convert_from_bytes = None

try:
    import pytesseract
except Exception:
    pytesseract = None

try:
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None

def print_step(title):
    print(f"\n{'-'*15} {title} {'-'*15}")

# ================= 全局配置 =================
DB_PATH = "./milvus_agent_demo.db"
COLLECTION_NAME = "dynamic_research_db"
STORE_PATH = "./dynamic_docstore_and_bm25.pkl"  # 持久化 DocStore 与 BM25 索引
MILVUS_HOST = "127.0.0.1"
MILVUS_PORT = int(os.getenv("MILVUS_LITE_PORT", "19539"))

# 避免本地 Milvus / gRPC 连接被系统代理劫持
os.environ['no_proxy'] = '*'
os.environ['GRPC_PROXY_EXP'] = 'localhost'

SKIP_EXTENSIONS = (
    ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx", ".zip", ".rar", ".7z"
)

QUERY_STOPWORDS = {
    "的", "了", "和", "与", "及", "或", "在", "是", "对", "中", "为", "最新", "进展",
    "the", "a", "an", "and", "or", "of", "in", "on", "for", "to", "with", "latest"
}

BASE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
    "Upgrade-Insecure-Requests": "1",
}

ALT_HEADERS = {
    **BASE_HEADERS,
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
}

SUSPECT_BLOCK_PATTERNS = [
    "captcha", "verify you are human", "access denied", "forbidden",
    "cloudflare", "robot check", "security check", "请完成验证", "访问受限"
]

CROSS_ENCODER_MODEL_NAME = os.getenv("CROSS_ENCODER_MODEL_NAME", "BAAI/bge-reranker-v2-m3")
CROSS_ENCODER_DEVICE = os.getenv("CROSS_ENCODER_DEVICE", "cuda")
CROSS_ENCODER_MAX_CHARS = int(os.getenv("CROSS_ENCODER_MAX_CHARS", "1500"))

_CROSS_ENCODER = None
_CROSS_ENCODER_INIT_FAILED = False
_CROSS_ENCODER_LOCK = threading.Lock()


def _build_http_session() -> requests.Session:
    """创建带重试策略的 HTTP 会话，用于提升抓取稳定性。"""
    session = requests.Session()
    retry = Retry(
        total=2,
        connect=2,
        read=2,
        status=2,
        backoff_factor=0.8,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


HTTP_SESSION = _build_http_session()

def _is_binary_content_type(content_type: str) -> bool:
    """判断响应 Content-Type 是否属于二进制内容。"""
    ct = (content_type or "").lower()
    if not ct:
        return False
    text_like = ("text/html", "application/xhtml+xml", "text/plain")
    if any(t in ct for t in text_like):
        return False
    binary_markers = (
        "application/pdf",
        "application/octet-stream",
        "application/zip",
        "application/msword",
        "application/vnd",
        "application/x-rar-compressed",
        "application/x-7z-compressed",
    )
    return any(m in ct for m in binary_markers)


def _extract_pdf_text_by_pypdf(pdf_bytes: bytes) -> tuple[str, str]:
    if not PdfReader:
        return "", "pypdf_unavailable"
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes), strict=False)

        if reader.is_encrypted:
            try:
                decrypt_state = reader.decrypt("")
                if decrypt_state == 0:
                    return "", "encrypted_pdf"
            except Exception:
                return "", "encrypted_pdf"

        parts = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            if page_text.strip():
                parts.append(page_text)
        text = "\n".join(parts).strip()
        if len(text) < 80:
            return "", "pdf_no_extractable_text"
        return text, "ok"
    except Exception as e:
        return "", f"pdf_parse_error: {e}"


def _extract_pdf_text_by_pymupdf(pdf_bytes: bytes) -> tuple[str, str]:
    if not fitz:
        return "", "pymupdf_unavailable"
    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            parts = []
            for page in doc:
                page_text = page.get_text("text") or ""
                if page_text.strip():
                    parts.append(page_text)
            text = "\n".join(parts).strip()
            if len(text) < 80:
                return "", "pdf_no_extractable_text"
            return text, "ok"
    except Exception as e:
        return "", f"pymupdf_parse_error: {e}"


def _extract_pdf_text_by_pdfminer(pdf_bytes: bytes) -> tuple[str, str]:
    if not pdfminer_extract_text:
        return "", "pdfminer_unavailable"
    try:
        text = (pdfminer_extract_text(io.BytesIO(pdf_bytes)) or "").strip()
        if len(text) < 80:
            return "", "pdf_no_extractable_text"
        return text, "ok"
    except Exception as e:
        return "", f"pdfminer_parse_error: {e}"


def _ocr_pdf_text(pdf_bytes: bytes) -> tuple[str, str]:
    if not convert_from_bytes:
        return "", "ocr_pdf2image_unavailable"
    if not pytesseract:
        return "", "ocr_tesseract_unavailable"

    try:
        images = convert_from_bytes(pdf_bytes, dpi=220, fmt="png")
        parts = []
        for img in images[:20]:
            t = (pytesseract.image_to_string(img, lang="chi_sim+eng") or "").strip()
            if t:
                parts.append(t)
        text = "\n".join(parts).strip()
        if len(text) < 80:
            return "", "ocr_text_too_short"
        return text, "ok"
    except Exception as e:
        return "", f"ocr_error: {e}"


def _extract_pdf_text(pdf_bytes: bytes) -> tuple[str, str]:
    """按优先级提取 PDF 文本：pypdf -> pymupdf -> pdfminer -> OCR。"""
    if not pdf_bytes or not pdf_bytes.startswith(b"%PDF"):
        return "", "not_pdf_signature"

    text, reason = _extract_pdf_text_by_pypdf(pdf_bytes)
    if text:
        return text, "ok"

    if reason.startswith("pdf_parse_error") or reason == "pypdf_unavailable":
        text2, reason2 = _extract_pdf_text_by_pymupdf(pdf_bytes)
        if text2:
            return text2, "ok"
        text3, reason3 = _extract_pdf_text_by_pdfminer(pdf_bytes)
        if text3:
            return text3, "ok"
        return "", f"{reason}; {reason2}; {reason3}"

    if reason == "pdf_no_extractable_text":
        # 仅在“无可提取文本”时触发 OCR 兜底。
        text4, reason4 = _ocr_pdf_text(pdf_bytes)
        if text4:
            return text4, "ok"
        return "", f"{reason}; {reason4}"

    return "", reason


def _tokenize_for_scoring(text: str) -> list[str]:
    if not text:
        return []
    text = text.lower().strip()

    tokens = []

    # 英文/数字词元
    tokens.extend(re.findall(r"[a-z0-9\-]{2,}", text))

    # 中文连续串拆成 2-gram 与 3-gram，避免整句只产生一个 token
    zh_groups = re.findall(r"[\u4e00-\u9fff]+", text)
    for grp in zh_groups:
        if len(grp) < 2:
            continue
        for n in (3, 2):
            if len(grp) >= n:
                tokens.extend(grp[i:i+n] for i in range(len(grp) - n + 1))

    # 去重保序
    seen = set()
    uniq = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq


def _looks_like_block_page(text: str) -> bool:
    low = (text or "").lower()
    return any(p in low for p in SUSPECT_BLOCK_PATTERNS)


def _get_cross_encoder():
    """延迟加载 Cross-Encoder，初始化失败时返回 None 并自动降级。"""
    global _CROSS_ENCODER, _CROSS_ENCODER_INIT_FAILED
    if _CROSS_ENCODER is not None:
        return _CROSS_ENCODER
    if _CROSS_ENCODER_INIT_FAILED:
        return None

    with _CROSS_ENCODER_LOCK:
        if _CROSS_ENCODER is not None:
            return _CROSS_ENCODER
        if _CROSS_ENCODER_INIT_FAILED:
            return None

        if CrossEncoder is None:
            print("   ⚠️ 未安装 sentence-transformers，重排将回退为余弦相似度。")
            _CROSS_ENCODER_INIT_FAILED = True
            return None

        try:
            print(f"   ➤ 正在加载 Cross-Encoder 重排模型: {CROSS_ENCODER_MODEL_NAME}")
            _CROSS_ENCODER = CrossEncoder(
                CROSS_ENCODER_MODEL_NAME,
                device=CROSS_ENCODER_DEVICE,
                trust_remote_code=True,
            )
            return _CROSS_ENCODER
        except Exception as e:
            print(f"   ⚠️ Cross-Encoder 初始化失败，将回退余弦重排: {e}")
            _CROSS_ENCODER_INIT_FAILED = True
            return None


def _cross_encoder_rerank(topic: str, docs: list[Document]) -> list[float]:
    """使用 Cross-Encoder 对候选文档打分；失败时抛出异常供上层降级。"""
    model = _get_cross_encoder()
    if model is None:
        raise RuntimeError("cross_encoder_unavailable")

    pairs = []
    for doc in docs:
        content = (doc.page_content or "").strip()
        pairs.append([topic, content[:CROSS_ENCODER_MAX_CHARS]])

    try:
        scores = model.predict(pairs, batch_size=8, show_progress_bar=False)
        return [float(s) for s in scores]
    except Exception as e:
        raise RuntimeError(f"cross_encoder_predict_error: {e}") from e

# ================= 核心 1: 网页抓取与清洗 =================
def _compute_result_score(query: str, title: str, snippet: str) -> float:
    title = (title or "").lower()
    snippet = (snippet or "").lower()
    merged = f"{title} {snippet}".strip()
    score = 0.0

    query = (query or "").strip().lower()
    if query and query in merged:
        # 完整短语命中加权，能明显区分高度相关结果
        score += 6.0

    # 纯查询相关性重排，不引入固定主题关键词。
    query_tokens = []
    matched_tokens = 0
    for tok in _tokenize_for_scoring(query):
        if tok in QUERY_STOPWORDS:
            continue
        query_tokens.append(tok)
        if tok in title:
            score += 2.0
            matched_tokens += 1
        elif tok in merged:
            score += 0.8
            matched_tokens += 1

    # 覆盖率得分，避免所有结果只有固定常数项。
    if query_tokens:
        coverage = matched_tokens / len(query_tokens)
        score += coverage * 2.5

    # 标题长度适中通常更有效，给轻微稳定项。
    if 8 <= len(title) <= 120:
        score += 0.2

    return score


def _request_with_anti_block(url: str) -> requests.Response:
    # 首次请求使用标准请求头，若疑似被拦截则切换备用请求头重试。
    resp = HTTP_SESSION.get(url, headers=BASE_HEADERS, timeout=10, allow_redirects=True)
    text_preview = (resp.text or "")[:5000] if "text" in (resp.headers.get("Content-Type", "").lower()) else ""
    if resp.status_code in (403, 429, 503) or _looks_like_block_page(text_preview):
        # 疑似拦截时，切换头并补充 Referer 再试一次。
        time.sleep(1.2)
        resp = HTTP_SESSION.get(
            url,
            headers={**ALT_HEADERS, "Referer": "https://www.bing.com/"},
            timeout=12,
            allow_redirects=True,
        )
    resp.raise_for_status()
    return resp


def scrape_webpage(url: str) -> tuple[str, str]:
    """抓取网页并提取纯文本（具备基础防反爬伪装与超时控制）"""
    lower_url = (url or "").lower()
    if lower_url.endswith(SKIP_EXTENSIONS):
        print(f"      [跳过二进制源] {url}")
        return "", "binary"

    try:
        response = _request_with_anti_block(url)

        content_type = response.headers.get("Content-Type", "")
        # 对可访问的 PDF 直接做文本提取，不再一刀切跳过。
        if "application/pdf" in content_type.lower() or lower_url.endswith(".pdf"):
            pdf_text, reason = _extract_pdf_text(response.content)
            if pdf_text:
                print(f"      [PDF提取成功] {url}")
                return pdf_text, "ok"

            # 某些站点后缀是 .pdf，但实际返回 HTML 拦截页，改走网页正文兜底。
            if reason == "not_pdf_signature":
                print(f"      [PDF伪装响应] {url}，改走HTML正文提取")
            else:
                print(f"      [PDF提取失败] {url} - {reason}")
                return "", "binary"

        if _is_binary_content_type(content_type):
            print(f"      [跳过二进制内容] {url} (Content-Type: {content_type})")
            return "", "binary"

        response.encoding = response.apparent_encoding or response.encoding
        soup = BeautifulSoup(response.text, 'lxml')
        
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.extract()
            
        text = soup.get_text(separator='\n')
        
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        return text, "ok"
    except Exception as e:
        print(f"      [抓取失败] {url} - {str(e)}")
        return "", "failed"

# ================= 核心 2: 动态搜索 =================
def run_dynamic_searcher(query: str, max_results: int = 5) -> list[Document]:
    """执行联网搜索并抓取网页（携带高阶降级重试机制）"""
    print_step(f"🕵️ [Searcher Agent] 开始全网检索: '{query}'")
    
    docs = []
    results = []
    filtered_results = []
    print(f"   ➤ 正在查询搜索引擎，获取前 {max_results} 个结果...")

    backends = [None, "html", "lite"]
    for backend in backends:
        try:
            with DDGS() as ddgs:
                kwargs = {"max_results": min(max_results * 4, 20)}
                if backend is not None:
                    kwargs["backend"] = backend
                results = list(ddgs.text(query, **kwargs))
            if results:
                backend_name = backend if backend is not None else "default"
                print(f"   ➤ 搜索后端 {backend_name} 成功返回 {len(results)} 条结果。")
                break
        except DuckDuckGoSearchException as e:
            backend_name = backend if backend is not None else "default"
            print(f"   ➤ 搜索后端 {backend_name} 失败: {e}")
        except Exception as e:
            backend_name = backend if backend is not None else "default"
            print(f"   ➤ 搜索后端 {backend_name} 异常: {e}")

    if not results:
        print("❌ 搜索引擎未返回结果（可能是网络不可达或被限制）。")
        return docs

    # 通用主题相关性重排（标题+摘要），不引入固定主题硬规则。
    for res in results:
        title = res.get("title") or ""
        snippet = res.get("body") or ""
        res["__rank_score"] = _compute_result_score(query, title, snippet)
        filtered_results.append(res)

    filtered_results.sort(
        key=lambda r: -(r.get("__rank_score") or 0.0)
    )

    if filtered_results:
        print("   ➤ 重排后 Top5 相关度预览：")
        for i, r in enumerate(filtered_results[:5], start=1):
            print(f"      [{i}] score={r.get('__rank_score', 0.0):.2f} | {r.get('title')}")

    if not filtered_results:
        print("❌ 搜索结果为空，无法继续抓取。")
        return docs

    accepted = 0
    for idx, res in enumerate(filtered_results):
        url = res.get("href")
        title = res.get("title")
        snippet = res.get("body")
        print(f"   ➤ [{idx+1}/{len(filtered_results)}] 发现目标: {title}")
        print(f"      正在深度抓取: {url}")

        content, fetch_status = scrape_webpage(url)

        if fetch_status == "binary":
            # 明确跳过 PDF/压缩包等二进制来源，避免污染语料库。
            continue

        if len(content) < 100:
            print("      [触发 Fallback] 页面过短或被拦截，使用搜索摘要兜底。")
            content = snippet or ""

        if len(content.strip()) < 30:
            print("      [跳过低质量内容] 抓取正文与摘要均不足，忽略该结果。")
            continue

        # URL 与标题保存在 metadata，后续切分后可用于来源追踪。
        doc = Document(
            page_content=content,
            metadata={"source_url": url, "title": title}
        )
        docs.append(doc)
        accepted += 1
        if accepted >= max_results:
            break
        time.sleep(1)

    return docs

# ================= 核心 3: 混合检索入库 =================
def ingest_to_milvus(docs: list[Document]):
    """使用 [父子文档 + BM25混合检索] 架构将动态数据压入 Milvus"""
    print_step("🧱 开始使用高阶 RAG 架构处理动态数据并入库")
    
    # 1. 启动 Milvus 服务并连通
    milvus_addr = f"{MILVUS_HOST}:{MILVUS_PORT}"
    milvus_uri = f"http://{milvus_addr}"
    milvus_server = Server(DB_PATH, milvus_addr)
    if not milvus_server.init() or not milvus_server.start():
        pass # 已经启动则忽略
    atexit.register(milvus_server.stop)

    print("   ➤ 正在调用 RTX 4090 加载 BGE-m3 稠密向量模型...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True}
    )

    vectorstore = Milvus(
        embedding_function=embeddings,
        connection_args={"uri": milvus_uri},
        collection_name=COLLECTION_NAME,
        drop_old=True # 每次新课题，清空旧知识
    )
    connections.connect(alias=vectorstore.alias, uri=milvus_uri)

    # 2. 实施 Parent-Child 双层切片策略（针对网页做数值微调）
    print("   ➤ 正在构建父子双层切片与映射规则...")
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600, # 网页内容更长，父文档调大一些
        chunk_overlap=50, 
        separators=["\n\n", "\n", "。"]
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=150, # 子文档保持精细
        chunk_overlap=30,
        separators=["。", "！", "；", "，"]
    )

    store = InMemoryStore()
    parent_retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )

    # 3. 核心写入动作：自动切分子文档进 Milvus，父文档进 Store
    print("   ➤ 正在进行向量化运算并写入 Milvus...")
    parent_retriever.add_documents(docs)

    # 4. 构建 BM25 稀疏检索层
    print("   ➤ 正在构建 BM25 关键词倒排索引...")
    parent_docs_for_bm25 = parent_splitter.split_documents(docs)
    bm25_retriever = BM25Retriever.from_documents(parent_docs_for_bm25)

    # 5. 持久化数据供 Agent 工作流读取
    with open(STORE_PATH, "wb") as f:
        pickle.dump({"docstore": store, "bm25": bm25_retriever}, f)
        
    print(f"✅ 高阶动态数据入库大功告成！持久化文件位于: {STORE_PATH}")


def _normalize_source_url(url: str) -> str:
    """标准化来源链接，避免同一页面因 query/fragment 不同而重复计数。"""
    if not url:
        return ""
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc.lower(), parts.path.rstrip('/'), '', ''))


def _is_low_quality_text(text: str) -> bool:
    """过滤过短或疑似二进制残片文本，降低噪声文档进入重排阶段。"""
    t = (text or "").strip()
    if len(t) < 120:
        return True
    binary_markers = ["%PDF-", "endobj", "xref", "stream", "PK\\x03\\x04"]
    return any(m in t for m in binary_markers)


def _cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """计算两向量余弦相似度。"""
    dot = 0.0
    n1 = 0.0
    n2 = 0.0
    for a, b in zip(v1, v2):
        dot += a * b
        n1 += a * a
        n2 += b * b
    if n1 <= 0.0 or n2 <= 0.0:
        return 0.0
    return dot / (math.sqrt(n1) * math.sqrt(n2))


def search_tool_collect_documents(sub_queries: list[str], max_results_per_query: int = 8):
    """显式工具1：执行联网检索与网页抓取。"""
    all_scraped_docs = []
    for i, query in enumerate(sub_queries):
        print(f"\n   🔄 正在执行子课题 [{i+1}/{len(sub_queries)}]: {query}")
        docs = run_dynamic_searcher(query, max_results=max_results_per_query)
        all_scraped_docs.extend(docs)

    result = {
        "collected_docs": len(all_scraped_docs),
        "attempted_queries": len(sub_queries),
        "status": "ok" if all_scraped_docs else "empty",
    }
    return all_scraped_docs, result


def search_tool_ingest_documents(scraped_docs):
    """显式工具2：把抓取内容写入向量库与稀疏索引。"""
    if not scraped_docs:
        return {"ingested_docs": 0, "status": "empty"}
    ingest_to_milvus(scraped_docs)
    return {"ingested_docs": len(scraped_docs), "status": "ok"}


def search_tool_retrieve_context(
    topic: str,
    retriever,
    rerank_embeddings,
    semantic_sim_threshold: float,
    semantic_sim_fallbacks: list[float],
    min_high_quality_sources: int,
    min_high_quality_sources_fallback: int,
    candidate_limit: int = 20,
    max_return_docs: int = 15,
    use_cross_encoder: bool = True,
):
    """显式工具3：基于已入库数据做检索、重排与上下文组装。"""
    candidates = retriever.invoke(topic)[:candidate_limit]
    if not candidates:
        return {
            "status": "no_candidates",
            "message": "检索完成，但无候选文档返回。",
            "context": "",
            "high_quality_sources": 0,
            "used_threshold": semantic_sim_threshold,
        }

    topic_vec = rerank_embeddings.embed_query(topic)

    prepared = []
    seen_prepare_sources = set()
    for doc in candidates:
        source_url = doc.metadata.get('source_url', '') if doc.metadata else ''
        normalized_source = _normalize_source_url(source_url)
        content = (doc.page_content or '').strip()

        if _is_low_quality_text(content):
            continue
        if normalized_source and normalized_source in seen_prepare_sources:
            continue

        doc_vec = rerank_embeddings.embed_query(content[:1200])
        sim = _cosine_similarity(topic_vec, doc_vec)
        prepared.append((sim, doc, normalized_source, content))

        if normalized_source:
            seen_prepare_sources.add(normalized_source)

    rerank_strategy = "cosine"
    if use_cross_encoder and prepared:
        try:
            docs_for_ce = [item[1] for item in prepared]
            ce_scores = _cross_encoder_rerank(topic, docs_for_ce)
            with_ce = []
            for (sim, doc, normalized_source, content), ce_score in zip(prepared, ce_scores):
                with_ce.append((float(ce_score), sim, doc, normalized_source, content))
            with_ce.sort(key=lambda x: (x[0], x[1]), reverse=True)
            prepared = [(sim, doc, normalized_source, content) for _, sim, doc, normalized_source, content in with_ce]
            rerank_strategy = "cross_encoder"
        except Exception as e:
            print(f"   ⚠️ Cross-Encoder 重排失败，自动回退余弦重排: {e}")
            prepared.sort(key=lambda x: x[0], reverse=True)
            rerank_strategy = "cosine_fallback"
    else:
        prepared.sort(key=lambda x: x[0], reverse=True)

    print(f"   ➤ 重排策略: {rerank_strategy}")

    target_min_sources = (
        min_high_quality_sources
        if len(prepared) >= min_high_quality_sources
        else min_high_quality_sources_fallback
    )

    retrieved_docs = []
    used_threshold = semantic_sim_threshold
    for threshold in semantic_sim_fallbacks:
        tmp = []
        for sim, doc, _, _ in prepared:
            if sim < threshold:
                continue
            tmp.append(doc)
            if len(tmp) >= max_return_docs:
                break
        if len(tmp) >= target_min_sources:
            retrieved_docs = tmp
            used_threshold = threshold
            break

    if len(retrieved_docs) < target_min_sources:
        return {
            "status": "insufficient",
            "message": (
                f"证据不足：仅获得 {len(retrieved_docs)} 个高质量来源，"
                f"低于最小门槛 {target_min_sources}。"
                "建议扩充可访问白名单站点、增加搜索结果数或继续放宽语义阈值。"
            ),
            "context": "",
            "high_quality_sources": len(retrieved_docs),
            "used_threshold": used_threshold,
            "rerank_strategy": rerank_strategy,
        }

    if used_threshold < semantic_sim_threshold:
        print(f"   ➤ 语义阈值已自动回退到 {used_threshold:.2f}，以保证证据覆盖。")

    context_parts = []
    for i, doc in enumerate(retrieved_docs):
        source_url = doc.metadata.get('source_url', '未知来源')
        title = doc.metadata.get('title', '无标题')
        content = doc.page_content.strip()
        context_parts.append(
            f"[来源 {i+1}] 网页标题: {title}\n"
            f"原文链接: {source_url}\n"
            f"核心内容:\n{content}"
        )

    context_str = "\n\n" + "=" * 40 + "\n\n".join(context_parts)
    return {
        "status": "ok",
        "message": f"知识淬炼完成，提炼出 {len(retrieved_docs)} 个高质量参考段落。",
        "context": context_str,
        "high_quality_sources": len(retrieved_docs),
        "used_threshold": used_threshold,
        "rerank_strategy": rerank_strategy,
    }


# ================= 单元测试 =================
if __name__ == "__main__":
    sub_topic = "2025年全球制造业自动化与AI应用趋势"
    
    scraped_documents = run_dynamic_searcher(sub_topic, max_results=3)
    
    if scraped_documents:
        ingest_to_milvus(scraped_documents)
    else:
        print("❌ 抓取失败，无法入库。")
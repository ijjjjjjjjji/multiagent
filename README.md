# Multi-Agent Deep Research System

一个基于 LangGraph 的多智能体研报生成项目。
输入研究主题后，系统会自动完成检索、写作、核查与迭代，输出带来源标注的 Markdown 报告。

## 1. 项目简介

本项目聚焦于“可落地的自动化研究流程”，核心目标是：

- 自动拆解研究问题
- 自动抓取并清洗公开网页/PDF信息
- 基于混合检索生成报告草稿
- 通过事实核查节点进行质量回路

## 2. 目录结构

| 文件 | 说明 |
|------|------|
| deep_research_agent.py | 主流程编排与各节点实现（Editor / Searcher / Writer / Checker） |
| dynamic_searcher.py | 搜索抓取、入库、检索重排相关逻辑 |
| streamlit_app.py | Streamlit 可视化界面 |
| requirements.txt | Python 依赖列表 |

## 3. 工作流

```
Editor -> Searcher -> Writer -> Fact Checker -> Router
```

各节点职责：

- Editor: 将主题拆成可检索子问题
- Searcher: 抓取数据、构建检索上下文
- Writer: 按章节生成报告并补写
- Fact Checker: 校验事实一致性并给出反馈
- Router: 根据校验结果决定结束或重试

## 4. 技术栈

- 编排: LangGraph
- 大模型: Ollama (qwen2.5:7b-instruct)
- 向量模型: BAAI/bge-m3
- 重排模型: Cross-Encoder（默认 BAAI/bge-reranker-v2-m3，失败自动回退为余弦重排）
- 向量库: Milvus Lite
- 稀疏检索: BM25
- 前端: Streamlit

## 5. 环境要求

- Python 3.11+
- Ollama 服务可用，且本地已拉取 qwen2.5:7b-instruct
- 可用的 Milvus Lite 运行环境
- 可选: CUDA（用于嵌入计算加速）

## 6. 安装与运行

安装依赖：

```bash
pip install -r requirements.txt
```

命令行运行：

```bash
python deep_research_agent.py --topic "你的研究主题"
```

Web 界面运行：

```bash
streamlit run streamlit_app.py
```

## 7. 输出说明

- 主要输出为 Markdown 研报（`draft`）
- 报告中会包含来源标注（如 `[来源 1]`）
- 同时输出路由追踪信息（`trace`），用于观察每轮校验与回路决策

## 8. 关键配置

可在 deep_research_agent.py 顶部调整：

- 语义阈值与回退策略
- 最低证据源数量
- 章节/全文长度目标
- 各阶段超时配置与重试上限

## 9. 注意事项

- 网络抓取成功率受目标网站反爬策略影响
- PDF 提取质量与文档结构、OCR 环境有关
- 本项目默认面向研究辅助，不替代人工最终审阅

## 10. License

MIT

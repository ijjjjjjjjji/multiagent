import io
import os
import warnings
import threading
import queue
import time
import traceback
from contextlib import redirect_stdout, redirect_stderr

import streamlit as st

# 抑制第三方库的噪声日志（gRPC、TensorFlow、Milvus 等）
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

warnings.filterwarnings(
    "ignore",
    message=r"pkg_resources is deprecated as an API.*",
)
warnings.filterwarnings(
    "ignore",
    message=r"coroutine 'AsyncMilvusClient\._get_connection' was never awaited",
)

from deep_research_agent import run_research

DEFAULT_TOPIC = "2026年人形机器人商业化落地现状"


st.set_page_config(page_title="Deep Research Agent UI", layout="wide")
st.title("Deep Research Agent 研报生成系统")
st.caption("多智能体 LangGraph 工作流：Editor → Searcher → Writer → Fact-Checker")

with st.sidebar:
    st.subheader("运行配置")
    gpu_id = st.text_input("CUDA_VISIBLE_DEVICES", value="6")
    show_logs = st.checkbox("显示后端运行日志", value=False)
    live_logs = st.checkbox("实时刷新日志", value=True)

with st.form("run_form"):
    topic = st.text_area("研究课题", value=DEFAULT_TOPIC, height=120, key="topic_input")
    submitted = st.form_submit_button("开始生成")

if submitted:
    topic = (topic or "").strip()
    if not topic:
        st.error("请输入研究课题")
        st.stop()

    if gpu_id.strip():
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id.strip()

    class _QueueWriter:
        """将标准输出/错误重定向到队列，使 Streamlit 能实时显示后端日志"""
        def __init__(self, q):
            self.q = q

        def write(self, s):
            if s:
                self.q.put(s)

        def flush(self):
            pass

    def _worker(out, log_q):
        """后台工作线程：运行多智能体流程并捕获所有输出"""
        try:
            writer = _QueueWriter(log_q)
            with redirect_stdout(writer), redirect_stderr(writer):
                out["state"] = run_research(topic)
        except Exception:
            out["error"] = traceback.format_exc()

    output = {}
    log_q = queue.Queue()
    logs = []
    log_box = st.empty()

    with st.spinner("正在执行多智能体研报生成，请稍候..."):
        start_ts = time.time()
        t = threading.Thread(target=_worker, args=(output, log_q), daemon=True)
        t.start()

        while t.is_alive():
            while not log_q.empty():
                logs.append(log_q.get_nowait())
            if show_logs and live_logs:
                elapsed = int(time.time() - start_ts)
                header = f"[运行中] 已耗时 {elapsed}s\n"
                log_box.code(header + ("".join(logs)[-20000:] or "等待日志输出..."), language="text")
            time.sleep(0.2)

        while not log_q.empty():
            logs.append(log_q.get_nowait())

    if "error" in output:
        st.error("后端执行失败，请查看错误堆栈。")
        st.text_area("Error", value=output["error"], height=320, key="error_box")
        if show_logs:
            st.subheader("后端运行日志")
            st.text_area("Logs", value="".join(logs), height=260, key="logs_box_error")
        st.stop()

    final_state = output["state"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("是否通过校验", "是" if final_state.get("fact_check_is_pass") else "否")
    col2.metric("错误类型", final_state.get("fact_check_error_type", "none"))
    col3.metric("重试次数", str(final_state.get("retry_count", 0)))
    col4.metric("路由决策轮次", str(final_state.get("iterations", 0)))

    st.subheader("最终研报")
    st.markdown(final_state.get("draft", ""))

    st.subheader("路由追踪 Trace")
    trace = final_state.get("trace", [])
    if trace:
        st.dataframe(trace, width="stretch")
    else:
        st.info("本次无 trace 记录")

    if show_logs:
        st.subheader("后端运行日志")
        st.text_area("Logs", value="".join(logs), height=260, key="logs_box_final")

st.divider()
# st.markdown("运行方式：在当前目录执行 streamlit run streamlit_app.py")

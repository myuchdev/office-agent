"""企业办公智能助手 —— Streamlit 界面。

运行（项目根目录）：
    streamlit run app.py
"""
import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.config import settings
from agent.graph import build_agent

st.set_page_config(page_title="企业办公智能助手", layout="centered")


@st.cache_resource
def get_agent():
    return build_agent()


# 会话标识（LangGraph 按 thread_id 维持记忆）与展示用消息列表
if "thread_id" not in st.session_state:
    st.session_state.thread_id = uuid.uuid4().hex
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("企业办公智能助手")
st.caption("可询问报销 / 请假 / IT 制度，也可查询工单（如 IT-1001）、计算年假")

with st.sidebar:
    st.markdown("### 会话信息")
    st.write(f"模型：`{settings.model}`")
    st.write("编排：LangGraph ReAct")
    st.write("检索：本地向量 RAG")
    if st.button("清空对话（重置记忆）"):
        st.session_state.messages = []
        st.session_state.thread_id = uuid.uuid4().hex
        st.rerun()
    st.markdown("---")
    st.markdown("**可以试试：**")
    st.markdown("- 去上海出差，住宿费最多能报多少？")
    st.markdown("- 我累计工作 8 年，有几天年假？")
    st.markdown("- 帮我查一下工单 IT-1001 的进度")

# 渲染历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 接收并处理用户输入
if prompt := st.chat_input("请输入你的问题…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    final_text = ""

    with st.chat_message("assistant"):
        status = st.status("正在思考…", expanded=True)
        answer_box = st.empty()
        try:
            for event in get_agent().stream(
                {"messages": [HumanMessage(content=prompt)]},
                config=config,
                stream_mode="values",
            ):
                last = event["messages"][-1]
                if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
                    for tc in last.tool_calls:
                        status.write(f"调用工具 `{tc['name']}`，参数：{tc['args']}")
                if isinstance(last, ToolMessage):
                    status.write(f"工具返回：{str(last.content)[:120]}…")
                if isinstance(last, AIMessage) and last.content:
                    final_text = last.content
                    answer_box.markdown(final_text)
            status.update(label="回答完成", state="complete", expanded=False)
        except Exception as exc:  # noqa: BLE001
            final_text = f"运行出错：{exc}"
            status.update(label="出错了", state="error")
            answer_box.error(final_text)

    st.session_state.messages.append({"role": "assistant", "content": final_text})

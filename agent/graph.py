"""
使用 LangGraph 构建 ReAct 智能体：
- agent 节点：LLM 结合上下文判断是否调用工具（Function Calling）；
- tools 节点：执行被调用的工具并把结果回传；
- 条件边：有 tool_calls 则进入 tools，否则结束；tools 执行后回到 agent（ReAct 循环）；
- MemorySaver + thread_id：实现同一会话的多轮记忆。
"""
import logging

from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agent.config import settings
from agent.tools.business import calc_annual_leave, get_ticket_status
from agent.tools.knowledge import search_knowledge

SYSTEM_PROMPT = """你是「星澜科技」的企业办公智能助手，名叫"小澜"，服务对象是公司员工。

你可以做两类事情：
1. 回答公司制度、流程、政策与 IT 使用问题：必须先调用 search_knowledge 检索，严格依据检索到的内容回答，并在回答末尾用一行标注参考的文件来源；检索不到就如实说明，禁止编造公司规定。
2. 办理查询类事务：查询工单用 get_ticket_status，计算年休假天数用 calc_annual_leave。

要求：
- 涉及金额、天数、流程等事实性信息，一律以工具返回为准，不要凭记忆杜撰；
- 若问题超出能力范围或缺少依据，明确告知并建议联系人力资源部（HR）或 IT 服务台；
- 回答简洁、条理清晰，必要时分点说明，使用中文。"""

logger = logging.getLogger(__name__)


def handle_tool_error(exc: Exception) -> str:
    """工具执行异常时：服务端记录完整堆栈，返回给 Agent 的只有通用提示。"""
    logger.error(
        "Agent 工具执行失败",
        exc_info=(type(exc), exc, exc.__traceback__),
    )
    return "工具执行失败，请稍后重试。"


def build_agent():
    """编译并返回带记忆的 ReAct Agent。"""
    if not settings.api_key:
        raise RuntimeError(
            "未配置大模型 API Key。请复制 .env.example 为 .env 并填入 DEEPSEEK_API_KEY。"
        )

    llm = ChatOpenAI(
        model=settings.model,
        api_key=settings.api_key,
        base_url=settings.base_url,
        temperature=settings.temperature,
    )
    tools = [search_knowledge, get_ticket_status, calc_annual_leave]
    llm_with_tools = llm.bind_tools(tools)

    def agent_node(state: MessagesState):
        messages = state["messages"]
        if not any(getattr(m, "type", "") == "system" for m in messages):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    builder = StateGraph(MessagesState)
    builder.add_node("agent", agent_node)
    builder.add_node(
        "tools",
        ToolNode(tools, handle_tool_errors=handle_tool_error),
    )
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")

    return builder.compile(checkpointer=MemorySaver())

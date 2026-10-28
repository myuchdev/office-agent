"""RAG 知识库检索工具。"""
import logging
from functools import lru_cache

from langchain_core.tools import tool

from agent.config import settings
from knowledge.vector_store import VectorStore

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_store() -> VectorStore:
    """加载已构建的向量索引（单例）。"""
    store = VectorStore(
        settings.vector_store_path,
        dim=settings.embedding_dim,
        keyword_weight=settings.retrieval_keyword_weight,
    )
    store.load()
    return store


@tool
def search_knowledge(query: str) -> str:
    """检索公司内部知识库，覆盖差旅报销、请假休假、IT 支持等制度与流程。
    当用户询问公司政策、报销标准、年假/病假、发票、审批流程，或 VPN、密码、打印机等 IT 问题时，必须先调用本工具。
    输入：用户的完整问题或其中的关键词。"""
    store = get_store()
    if len(store) == 0:
        logger.warning("知识库索引为空，无法执行制度检索。")
        return "当前暂时无法查询到相关制度资料，请稍后重试或联系管理员。"

    hits = store.search(query, k=settings.retrieval_top_k)
    if not hits:
        return "知识库中未检索到与该问题直接相关的内容。"

    parts = []
    for _score, chunk in hits:
        meta = chunk["metadata"]
        parts.append(
            f"【来源文件】{meta.get('source')}\n"
            f"【章节】{meta.get('heading')}\n"
            f"{chunk['text']}"
        )
    return "\n\n---\n\n".join(parts)

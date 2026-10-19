"""
全局配置：从项目根目录的 .env 读取大模型与检索参数。
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# 项目根目录（agent/config.py 的上两级）
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")


class Settings:
    """运行配置。"""

    def __init__(self) -> None:
        # DeepSeek 优先，其次通用 OPENAI_API_KEY
        self.api_key: str = (
            os.getenv("DEEPSEEK_API_KEY", "")
            or os.getenv("OPENAI_API_KEY", "")
        ).strip()
        self.base_url: str = os.getenv(
            "OPENAI_BASE_URL", "https://api.deepseek.com"
        ).strip()
        self.model: str = os.getenv("LLM_MODEL", "deepseek-chat").strip()
        self.temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.3"))

        # 本地 embedding / 检索
        self.embedding_dim: int = int(os.getenv("EMBEDDING_DIM", "512"))
        self.retrieval_top_k: int = int(os.getenv("RETRIEVAL_TOP_K", "4"))
        # 混合检索中关键词( IDF )分数的权重，0 表示纯向量检索
        self.retrieval_keyword_weight: float = float(
            os.getenv("RETRIEVAL_KEYWORD_WEIGHT", "0.8")
        )

        # 路径
        self.data_dir: Path = ROOT_DIR / "data"
        self.docs_dir: Path = ROOT_DIR / "knowledge" / "docs"
        self.vector_store_path: Path = self.data_dir / "vector_store.json"
        self.tickets_path: Path = self.data_dir / "tickets.json"


settings = Settings()
settings.data_dir.mkdir(parents=True, exist_ok=True)

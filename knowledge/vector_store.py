"""
轻量级 RAG 向量存储（零第三方依赖、零模型下载、可离线运行）。

包含：
- HashingEmbeddings：基于「特征哈希 (feature hashing / hashing trick)」的本地向量化。
  中文按「单字 + 相邻字 bigram」、英文/数字按词切分，带符号哈希映射到固定维度并做 L2 归一化。
- VectorStore：混合检索（向量余弦相似度 + 关键词 IDF 命中），JSON 持久化。

为什么要混合检索：
  纯词袋向量在「短块 + 通用词」和「长块（含很多数字/列举）」之间容易排错顺序
  ——例如问「上海住宿费多少」，短的「出差申请」块会压过含金额的「住宿费标准」块。
  叠加一个关键词分数，并对稀有词（只在少数块出现，如「住宿费」）给更高 IDF 权重、
  对通用词（在很多块出现，如「出差」「报销」）降权，能显著提升召回准确度。
  这是生产系统「向量 + 关键词（BM25）混合检索」的简化版。

生产环境可把 Embeddings 换成云端语义模型（text-embedding-v4 / bge-m3），
或把存储换成 Chroma / Milvus（检索接口保持一致）。
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import uuid
from pathlib import Path
from typing import Dict, List, Set, Tuple


class HashingEmbeddings:
    """固定维度的带符号特征哈希向量化。"""

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    @staticmethod
    def _tokens(text: str) -> List[str]:
        text = text.lower()
        tokens: List[str] = re.findall(r"[a-z0-9]+", text)
        han = re.findall(r"[\u4e00-\u9fff]", text)
        tokens.extend(han)
        # 中文相邻字 bigram，提升关键词匹配能力
        tokens.extend(han[i] + han[i + 1] for i in range(len(han) - 1))
        return tokens

    @staticmethod
    def _hash(token: str) -> int:
        return int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)

    def embed(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        for tok in self._tokens(text):
            idx = self._hash(tok) % self.dim
            # 带符号哈希可抵消部分哈希碰撞
            sign = 1.0 if self._hash(tok + "#sign") % 2 == 0 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self.embed(text)


def keyword_tokens(text: str) -> Set[str]:
    """用于关键词匹配的 token：英文/数字词 + 中文相邻字 bigram（不保留单字，降低噪声）。"""
    text = text.lower()
    words = re.findall(r"[a-z0-9]+", text)
    han = re.findall(r"[\u4e00-\u9fff]", text)
    bigrams = [han[i] + han[i + 1] for i in range(len(han) - 1)]
    return set(words + bigrams)


class VectorStore:
    """文本块 + 向量的存储，使用「向量余弦 + 关键词 IDF」混合检索，JSON 持久化。"""

    def __init__(
        self,
        persist_path: str | Path,
        dim: int = 512,
        keyword_weight: float = 0.8,
    ) -> None:
        self.persist_path = Path(persist_path)
        self.embeddings = HashingEmbeddings(dim)
        self.keyword_weight = keyword_weight
        self.chunks: List[Dict] = []
        self._idf: Dict[str, float] = {}
        self._chunk_keywords: List[Set[str]] = []

    def add_texts(self, texts: List[str], metadatas: List[Dict]) -> None:
        vectors = self.embeddings.embed_documents(texts)
        for text, meta, vec in zip(texts, metadatas, vectors):
            self.chunks.append(
                {"id": uuid.uuid4().hex, "text": text, "metadata": meta, "vector": vec}
            )
        self._idf = {}  # 内容变化，IDF 需重算

    @staticmethod
    def _match_text(chunk: Dict) -> str:
        """关键词匹配使用的文本：最后一级小标题 + 正文（让高信息量的标题参与匹配）。"""
        sub = chunk["metadata"].get("heading", "").split(" / ")[-1]
        return sub + "。" + chunk["text"]

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        # 向量已 L2 归一化，点积即余弦相似度
        return sum(x * y for x, y in zip(a, b))

    def _ensure_idf(self) -> None:
        """基于当前所有文本块，惰性计算每个关键词的 IDF。"""
        if self._idf and len(self._chunk_keywords) == len(self.chunks):
            return
        self._chunk_keywords = []
        df: Dict[str, int] = {}
        for chunk in self.chunks:
            kws = keyword_tokens(self._match_text(chunk))
            self._chunk_keywords.append(kws)
            for g in kws:
                df[g] = df.get(g, 0) + 1
        n = len(self.chunks)
        # 平滑 IDF：df 越小（越稀有）权重越高
        self._idf = {g: math.log((n + 1) / (d + 1)) + 1.0 for g, d in df.items()}

    def _keyword_score(self, query_keywords: Set[str], chunk_index: int) -> float:
        """查询关键词命中文本块的 IDF 占比，范围 0~1。"""
        if not query_keywords:
            return 0.0
        chunk_keywords = self._chunk_keywords[chunk_index]
        n = len(self.chunks)
        out_of_vocab_idf = math.log(n + 1) + 1.0

        total = 0.0
        hit = 0.0
        for g in query_keywords:
            weight = self._idf.get(g, out_of_vocab_idf)
            total += weight
            if g in chunk_keywords:
                hit += weight
        return hit / total if total > 0 else 0.0

    def search(
        self, query: str, k: int = 4, min_score: float = 0.0
    ) -> List[Tuple[float, Dict]]:
        self._ensure_idf()
        query_vector = self.embeddings.embed_query(query)
        query_keywords = keyword_tokens(query)

        scored: List[Tuple[float, Dict]] = []
        for index, chunk in enumerate(self.chunks):
            vector_score = self._cosine(query_vector, chunk["vector"])
            keyword_score = self._keyword_score(query_keywords, index)
            final_score = vector_score + self.keyword_weight * keyword_score
            scored.append((final_score, chunk))

        scored = [s for s in scored if s[0] >= min_score]
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:k]

    def save(self) -> None:
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.persist_path, "w", encoding="utf-8") as f:
            json.dump(
                {"dim": self.embeddings.dim, "chunks": self.chunks},
                f,
                ensure_ascii=False,
            )

    def load(self) -> "VectorStore":
        if self.persist_path.exists():
            with open(self.persist_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.chunks = data.get("chunks", [])
            self._idf = {}
            self._ensure_idf()
        return self

    def __len__(self) -> int:
        return len(self.chunks)

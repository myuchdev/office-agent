"""
读取 knowledge/docs/*.md，切分为文本块，构建向量索引并持久化。

运行（在项目根目录）：
    python -m knowledge.build_index
"""
from __future__ import annotations

import re
from typing import List, Tuple

from agent.config import settings
from knowledge.vector_store import VectorStore


def split_markdown(
    text: str, source: str, max_chars: int = 350, overlap: int = 60
) -> List[Tuple[str, dict]]:
    """按 markdown 标题切段；超长正文再用滑动窗口切分。返回 (文本, 元数据) 列表。"""
    pattern = re.compile(r"(?m)^(#{1,3})\s+(.*)$")
    matches = list(pattern.finditer(text))

    segments: List[Tuple[str, str]] = []
    if not matches:
        segments.append((source, text))
    else:
        head_paths: List[str] = []
        for i, m in enumerate(matches):
            level = len(m.group(1))
            title = m.group(2).strip()
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[start:end].strip()
            head_paths = head_paths[: level - 1] + [title]
            heading = " / ".join(head_paths)
            if body:
                segments.append((heading, body))

    results: List[Tuple[str, dict]] = []
    for heading, body in segments:
        body = re.sub(r"\n{3,}", "\n\n", body).strip()
        if len(body) <= max_chars:
            results.append((body, {"source": source, "heading": heading}))
        else:
            start = 0
            while start < len(body):
                piece = body[start : start + max_chars].strip()
                if piece:
                    results.append((piece, {"source": source, "heading": heading}))
                if start + max_chars >= len(body):
                    break
                start += max_chars - overlap
    return results


def build() -> None:
    store = VectorStore(settings.vector_store_path, dim=settings.embedding_dim)
    docs = sorted(settings.docs_dir.glob("*.md"))
    if not docs:
        print(f"[!] 未在 {settings.docs_dir} 找到 .md 文档")
        return

    total = 0
    for path in docs:
        text = path.read_text(encoding="utf-8")
        pieces = split_markdown(text, source=path.name)
        store.add_texts(
            [p[0] for p in pieces], [p[1] for p in pieces]
        )
        total += len(pieces)
        print(f"[+] {path.name}: {len(pieces)} 个文本块")

    store.save()
    print(f"[完成] 共 {total} 个文本块，已保存到 {settings.vector_store_path}")


if __name__ == "__main__":
    build()

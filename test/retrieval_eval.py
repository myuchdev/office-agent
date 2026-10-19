# -*- coding: utf-8 -*-
"""
检索质量回归评测（RAG 的「召回命中率」）。

为什么要做这个：
  RAG 系统「答得对不对」，第一步取决于「该召回的文档块有没有进 Top-K」。
  每改一次检索逻辑、每加一份文档，都可以用本脚本一键回归，避免「改好一个问题、
  又改坏另一个问题」。

运行方式（在项目根目录）：
  .\\.venv\\Scripts\\python.exe test\\retrieval_eval.py
"""
import os
import sys

# 允许直接以脚本方式运行（把项目根目录加入模块搜索路径）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.config import settings  # noqa: E402
from knowledge.vector_store import VectorStore  # noqa: E402

# (测试问题, 期望被召回的章节关键字)
CASES = [
    ("去上海出差住宿费最多能报多少", "住宿费标准"),
    ("出差住宿报销标准是多少", "住宿费标准"),
    ("广州住宿一晚能报多少钱", "住宿费标准"),
    ("高铁票报销有什么要求", "交通费标准"),
    ("年假多少天", "年休假"),
    ("VPN怎么开通", "VPN"),
    ("密码忘记了怎么重置", "密码"),
    ("报销多久能到账", "到账时间"),
    ("发票有什么要求", "发票"),
]


def main() -> int:
    store = VectorStore(
        settings.vector_store_path,
        dim=settings.embedding_dim,
        keyword_weight=settings.retrieval_keyword_weight,
    ).load()
    if len(store) == 0:
        print("知识库为空，请先运行：python -m knowledge.build_index")
        return 1

    k = settings.retrieval_top_k
    passed = 0
    print(f"检索方式：向量 + 关键词混合（权重 {settings.retrieval_keyword_weight}），Top-{k}\n")
    for question, expect in CASES:
        hits = store.search(question, k=k)
        headings = [c["metadata"]["heading"].split(" / ")[-1] for _s, c in hits]
        ok = any(expect in h for h in headings)
        passed += ok
        print(f"{'PASS' if ok else 'FAIL'} | {question}")
        print(f"       召回章节：{headings}")

    print(f"\nTop-{k} 召回命中率：{passed}/{len(CASES)}")
    return 0 if passed == len(CASES) else 1


if __name__ == "__main__":
    raise SystemExit(main())

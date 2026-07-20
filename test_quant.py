#!/usr/bin/env python3
"""量化对比测试：float32 vs int8 embedding 的检索质量与存储对比。

对同一代码库分别用 float32 和 int8 索引，对比存储占用与检索结果一致性，
验证 int8 量化在 4x 省向量存储的同时检索质量基本无损。

用法：
  python test_quant.py            # 对比本项目自身
  python test_quant.py <目录>      # 对比指定代码库
"""
from __future__ import annotations

import shutil
import sys
import tempfile
import time
from pathlib import Path

from dotenv import load_dotenv

from embedder import Embedder
from indexer import Indexer
from retriever import Retriever
from store import CodeStore

load_dotenv()

QUERIES = [
    "Tree-sitter AST 把源码按函数和类切分成代码块",
    "向量检索结果和 BM25 全文检索用 RRF 融合排序",
    "基于文件 hash 做增量同步，只重新索引变更的文件",
    "调用 Voyage 接口批量生成 embedding 向量",
]


def build_and_query(target: Path, dtype: str) -> dict:
    """用指定 dtype 索引并查询，返回耗时/存储/检索结果。"""
    db_dir = tempfile.mkdtemp(prefix=f"scm_q_{dtype}_")
    db_path = str(Path(db_dir) / "index.db")
    try:
        embedder = Embedder(output_dtype=dtype)
        store = CodeStore(db_path, embedder.dim, dtype=dtype)
        indexer = Indexer(str(target), store, embedder)
        retriever = Retriever(store, embedder)
        t0 = time.time()
        stats = indexer.sync()
        index_time = time.time() - t0
        db_size = Path(db_path).stat().st_size
        # 每个查询取 top5 主结果（关闭图扩展，便于纯检索对比）
        results = {}
        for q in QUERIES:
            rs = retriever.search(q, top_k=30, top_n=5, expand_graph=False)
            results[q] = [r["symbol"] for r in rs]
        store.close()
        return {
            "stats": stats,
            "index_time": index_time,
            "db_size": db_size,
            "results": results,
        }
    finally:
        shutil.rmtree(db_dir, ignore_errors=True)


def main() -> None:
    if len(sys.argv) > 1:
        target = Path(sys.argv[1]).resolve()
    else:
        target = Path(__file__).parent.resolve()
    print(f"[quant] 目标代码库: {target}\n")

    print("=== float32 索引 ===")
    f = build_and_query(target, "float")
    print(f"  耗时 {f['index_time']:.1f}s | DB {f['db_size'] / 1024:.1f} KB | {f['stats']}")

    print("\n=== int8 索引 ===")
    i = build_and_query(target, "int8")
    print(f"  耗时 {i['index_time']:.1f}s | DB {i['db_size'] / 1024:.1f} KB | {i['stats']}")

    print("\n=== 存储对比 ===")
    saved = (1 - i["db_size"] / f["db_size"]) * 100 if f["db_size"] else 0
    print(f"  float32: {f['db_size'] / 1024:.1f} KB")
    print(f"  int8:    {i['db_size'] / 1024:.1f} KB  (总 DB 节省 {saved:.1f}%)")
    print("  注：DB 含源码文本，向量仅占一部分；大仓库向量占比高时节省接近 75%")

    print("\n=== 检索一致性（top5 主结果，int8 vs float）===")
    total_same = 0
    total = 0
    for q in QUERIES:
        fr = f["results"][q]
        ir = i["results"][q]
        same = sum(1 for s in ir if s in fr)
        total_same += same
        total += len(fr)
        match = "完全一致" if same == len(fr) else f"{same}/{len(fr)} 重合"
        print(f"\n  [{match}] {q[:28]}...")
        print(f"    float: {fr}")
        print(f"    int8:  {ir}")
    rate = total_same / total * 100 if total else 0
    print(f"\n[quant] 总体一致率: {total_same}/{total} = {rate:.0f}%")


if __name__ == "__main__":
    main()

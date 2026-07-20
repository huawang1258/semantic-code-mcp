#!/usr/bin/env python3
"""端到端测试：在真实代码库上跑通 索引 -> embedding -> hybrid 检索 -> (rerank)。

默认索引本项目自身（semantic-code-mcp/），用中文自然语言查询验证
voyage-code-3 embedding 与「向量 + BM25」混合检索是否按预期工作。

用法：
  python test_e2e.py            # 索引本项目自身
  python test_e2e.py <目录>      # 索引指定代码库
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


# 针对本项目自身的验证查询，覆盖各核心模块
QUERIES = [
    "Tree-sitter AST 把源码按函数和类切分成代码块",
    "向量检索结果和 BM25 全文检索用 RRF 融合排序",
    "基于文件 hash 做增量同步，只重新索引变更的文件",
    "调用 Voyage 接口批量生成 embedding 向量",
    "用 Cohere 对候选结果做 rerank 重排",
]


def main() -> None:
    if len(sys.argv) > 1:
        target = Path(sys.argv[1]).resolve()
    else:
        target = Path(__file__).parent.resolve()
    print(f"[e2e] 目标代码库: {target}")

    # 临时索引库，跑完即删，不污染目标目录与默认库
    db_dir = tempfile.mkdtemp(prefix="scm_e2e_")
    db_path = str(Path(db_dir) / "index.db")

    try:
        embedder = Embedder()
        print(f"[e2e] embedding 模型: {embedder.model}, 维度: {embedder.dim}")

        store = CodeStore(db_path, embedder.dim)
        indexer = Indexer(str(target), store, embedder)
        retriever = Retriever(store, embedder)
        mode = "启用 (Cohere)" if retriever.cohere_client else "未配置，使用 RRF 融合结果"
        print(f"[e2e] rerank: {mode}")

        # 1. 全量索引
        t0 = time.time()

        def _progress(done: int, total: int, fp: str) -> None:
            print(f"  索引中 {done}/{total}: {Path(fp).name}")

        stats = indexer.sync(progress=_progress)
        print(f"[e2e] 索引完成，耗时 {time.time() - t0:.1f}s, stats={stats}")

        # 2. 逐个查询，打印主结果 + call graph 关联块
        _REL = {"callee": "→被调用", "caller": "←调用者"}
        for q in QUERIES:
            print(f"\n=== 查询: {q}")
            t0 = time.time()
            results = retriever.search(q, top_k=30, top_n=3)
            n_main = sum(1 for r in results if not r.get("relation"))
            n_rel = len(results) - n_main
            print(f"  ({time.time() - t0:.2f}s, {n_main} 主结果 + {n_rel} call graph 关联)")
            main_n = 0
            for r in results:
                rel = r.get("relation")
                name = Path(r["file_path"]).name
                if rel:
                    print(
                        f"       └─[{_REL.get(rel, rel)}] {name} :: {r['symbol']} "
                        f"(L{r['start_line']}-{r['end_line']})"
                    )
                else:
                    main_n += 1
                    print(
                        f"  {main_n}. {name} :: {r['symbol']} "
                        f"(L{r['start_line']}-{r['end_line']}, score={r.get('score', 0):.3f})"
                    )

        store.close()
    finally:
        shutil.rmtree(db_dir, ignore_errors=True)

    print("\n[e2e] 完成，临时索引已清理。")


if __name__ == "__main__":
    main()

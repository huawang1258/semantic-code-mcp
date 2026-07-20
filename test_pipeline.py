#!/usr/bin/env python3
"""验证通过 WorkspaceManager 跑完整检索管道。"""
from __future__ import annotations

import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from embedder import Embedder
from workspace import WorkspaceManager


def main() -> None:
    mgr = WorkspaceManager(Embedder())
    target = str(Path(__file__).parent.resolve())
    ws = mgr.get(target)

    print("=== Sync ===")
    t0 = time.time()
    stats = mgr.sync_workspace(ws)
    print(f"  {time.time() - t0:.1f}s, {stats}")

    print("\n=== Search: 'hybrid retrieval with RRF fusion' ===")
    t0 = time.time()
    results = ws.retriever.search("hybrid retrieval with RRF fusion", top_k=20, top_n=3)
    elapsed = time.time() - t0
    main_results = [r for r in results if "relation" not in r]
    graph_results = [r for r in results if "relation" in r]
    print(f"  {elapsed:.2f}s, {len(main_results)} main + {len(graph_results)} graph")
    for r in main_results:
        print(f"  - {Path(r['file_path']).name}::{r['symbol']} (score={r.get('score', 0):.3f})")

    print("\n=== Watcher dirty count ===")
    print(f"  dirty_count: {ws.watcher.dirty_count if ws.watcher else 'N/A'}")

    mgr.destroy_all()
    print("\nPIPELINE OK")


if __name__ == "__main__":
    main()

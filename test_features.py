#!/usr/bin/env python3
"""验证新工程特性：LRU 淘汰 + watchdog 文件监控 + sync_dirty。"""
from __future__ import annotations

import os
import shutil
import tempfile
import time
from pathlib import Path

os.environ.setdefault("SCM_MAX_WORKSPACES", "3")

from dotenv import load_dotenv

load_dotenv()

from embedder import Embedder
from workspace import WorkspaceManager


def test_watcher_and_lru() -> None:
    print("=== 1. 基本工作区创建 + Watcher ===")
    embedder = Embedder()
    mgr = WorkspaceManager(embedder)
    test_dir = str(Path(__file__).parent.resolve())
    ws = mgr.get(test_dir)
    print(f"  root: {ws.root}")
    print(f"  watcher: {'ON' if ws.watcher else 'OFF'}")
    print(f"  active: {mgr.active_count}")
    assert ws.watcher is not None, "watcher should be enabled"

    print("\n=== 2. Watcher 文件变更检测 ===")
    tmp = Path(test_dir) / "__test_watch_tmp.py"
    tmp.write_text("# test file for watcher\n", encoding="utf-8")
    time.sleep(1.0)  # 等 watchdog 事件
    print(f"  写入后 dirty_count: {ws.watcher.dirty_count}")
    assert ws.watcher.dirty_count >= 1, "should detect file creation"

    tmp.unlink()
    time.sleep(1.0)
    print(f"  删除后 dirty_count: {ws.watcher.dirty_count}")
    dirty, deleted = ws.watcher.take_dirty()
    print(f"  take_dirty: dirty={len(dirty)}, deleted={len(deleted)}")
    assert len(dirty) + len(deleted) >= 1, "should have changes"

    print("\n=== 3. Sync 测试 ===")
    stats = mgr.sync_workspace(ws)
    print(f"  total_files={stats['total_files']}, changed={stats['changed']}")
    assert stats["total_files"] > 0

    print("\n=== 4. LRU 淘汰 (max=3) ===")
    dirs = []
    for i in range(4):
        d = tempfile.mkdtemp(prefix=f"scm_lru_{i}_")
        Path(d, "test.py").write_text(f"x = {i}\n", encoding="utf-8")
        dirs.append(d)
        mgr.get(d)
        print(f"  +ws#{i + 1}: active={mgr.active_count}")

    print(f"  final active: {mgr.active_count}")
    assert mgr.active_count <= 3, f"LRU should evict to max 3, got {mgr.active_count}"

    print("\n=== 5. sync_dirty 增量验证 ===")
    # 创建一个有初始索引的 workspace，然后 dirty sync
    d = tempfile.mkdtemp(prefix="scm_dirty_")
    Path(d, "hello.py").write_text("def hello(): pass\n", encoding="utf-8")
    ws2 = mgr.get(d)
    stats = mgr.sync_workspace(ws2)
    print(f"  初始 sync: changed={stats['changed']}")
    # 模拟变更
    Path(d, "hello.py").write_text("def hello(): return 42\n", encoding="utf-8")
    time.sleep(1.0)
    stats2 = mgr.sync_workspace(ws2)
    print(f"  dirty sync: changed={stats2['changed']}")
    dirs.append(d)

    # 清理
    mgr.destroy_all()
    for d in dirs:
        shutil.rmtree(d, ignore_errors=True)

    print("\n ALL PASSED")


if __name__ == "__main__":
    test_watcher_and_lru()

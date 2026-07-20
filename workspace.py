"""多工作区 LRU 管理器。

对齐 augment-context-mcp 的工程特性：
- 多工作区并发管理（最多 MAX_ACTIVE 个）
- LRU 淘汰：超限时释放最久未访问的 workspace
- 文件监控集成：变更 debounce 后**后台主动增量同步**，查询时零等待；
  后台同步失败不丢变更（dirty 保留，下次查询前兜底同步）
- 状态追踪：pending / indexing / ready / failed
"""
from __future__ import annotations

import hashlib
import logging
import os
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Callable

from chunker import EXT_TO_LANG
from embedder import Embedder
from indexer import Indexer
from retriever import Retriever
from store import CodeStore
from watcher import FileWatcher

logger = logging.getLogger("semantic-code-mcp")

MAX_ACTIVE = int(os.getenv("SCM_MAX_WORKSPACES", "8"))


class Workspace:
    """单个工作区状态。"""

    def __init__(
        self,
        root: str,
        store: CodeStore,
        retriever: Retriever,
        indexer: Indexer,
        watcher: FileWatcher | None = None,
    ) -> None:
        self.root = root
        self.store = store
        self.retriever = retriever
        self.indexer = indexer
        self.watcher = watcher
        self.last_accessed: float = time.time()
        self.status: str = "pending"  # pending | indexing | ready | failed
        self.initial_indexed: bool = False
        self.last_error: str | None = None
        self._lock = threading.Lock()

    def touch(self) -> None:
        self.last_accessed = time.time()

    def search(self, query: str, **kwargs) -> list:
        """线程安全的检索入口（加锁保护 sqlite 并发访问）。"""
        with self._lock:
            return self.retriever.search(query, **kwargs)

    def destroy(self) -> None:
        """释放资源（先停 watcher 阻断新的后台同步，再等在途操作结束后关库）。"""
        if self.watcher:
            self.watcher.stop()
            self.watcher = None
        with self._lock:
            if self.store:
                self.store.close()
                self.store = None


class WorkspaceManager:
    """管理所有工作区，含 LRU 淘汰和文件监控。"""

    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self._workspaces: OrderedDict[str, Workspace] = OrderedDict()
        self._lock = threading.Lock()
        self._db_dir = os.getenv("SCM_DB_DIR") or str(
            Path.home() / ".semantic-code-mcp"
        )
        Path(self._db_dir).mkdir(parents=True, exist_ok=True)

    @property
    def active_count(self) -> int:
        return len(self._workspaces)

    def get(
        self,
        directory: str,
        on_progress: Callable[[int, int, str], None] | None = None,
    ) -> Workspace:
        """获取或创建 workspace，自动 LRU 淘汰 + 增量同步。"""
        resolved = str(Path(directory).resolve())
        with self._lock:
            if resolved in self._workspaces:
                ws = self._workspaces[resolved]
                # 移到末尾（最近使用）
                self._workspaces.move_to_end(resolved)
                ws.touch()
                return ws
        # 不在锁里创建（避免阻塞）
        ws = self._create_workspace(resolved)
        evicted: list[Workspace] = []
        loser: Workspace | None = None
        with self._lock:
            existing = self._workspaces.get(resolved)
            if existing is not None:
                # 双重检查：另一个线程已抢先创建，丢弃本次创建的实例
                existing.touch()
                self._workspaces.move_to_end(resolved)
                loser = ws
                ws = existing
            else:
                self._workspaces[resolved] = ws
                self._workspaces.move_to_end(resolved)
                evicted = self._evict()
        if loser is not None:
            loser.destroy()
        for w in evicted:
            w.destroy()
        return ws

    def sync_workspace(
        self,
        ws: Workspace,
        on_progress: Callable[[int, int, str], None] | None = None,
        force: bool = False,
    ) -> dict:
        """对 workspace 执行同步（首次全量 / 后续增量 / watcher dirty 增量）。

        Returns:
            indexer.sync() 返回的 stats dict
        """
        ws.touch()
        with ws._lock:
            # 如果 watcher 有 dirty 文件，将其信息传给 indexer（优化：只扫 dirty 文件）
            if ws.watcher and ws.watcher.dirty_count > 0 and ws.initial_indexed and not force:
                dirty, deleted = ws.watcher.take_dirty()
                stats = ws.indexer.sync_dirty(dirty, deleted, progress=on_progress)
            else:
                stats = ws.indexer.sync(progress=on_progress)
            ws.status = "ready"
            ws.initial_indexed = True
            ws.last_error = None
        return stats

    def force_sync(
        self,
        directory: str,
        on_progress: Callable[[int, int, str], None] | None = None,
    ) -> dict:
        """强制全量重同步。"""
        ws = self.get(directory)
        return self.sync_workspace(ws, on_progress=on_progress, force=True)

    def destroy_all(self) -> None:
        """关闭所有工作区。"""
        with self._lock:
            for ws in self._workspaces.values():
                ws.destroy()
            self._workspaces.clear()

    def _create_workspace(self, resolved: str) -> Workspace:
        """创建新的 workspace 实例。"""
        db_path = self._db_path_for(resolved)
        store = CodeStore(db_path, self.embedder.dim, dtype=self.embedder.output_dtype)
        retriever = Retriever(
            store,
            self.embedder,
            rerank_model=os.getenv("SCM_RERANK_MODEL", "rerank-v3.5"),
        )
        indexer = Indexer(resolved, store, self.embedder)
        # 文件监控：debounce 后后台主动增量同步（对齐 augment 的 _doIncrementalSync），
        # 查询时索引已是最新，不再把同步成本摊到首次查询延迟上
        watch_disabled = os.getenv("SCM_WATCH", "1") == "0"
        holder: list[Workspace] = []
        watcher = None
        if not watch_disabled:

            def _bg_sync() -> None:
                if not holder:
                    return
                w = holder[0]
                # 首次全量由首个查询触发，后台不抢跑
                if not w.initial_indexed:
                    return
                try:
                    stats = self.sync_workspace(w)
                    if stats.get("changed"):
                        logger.info("后台增量同步: %s (%d 变更)", resolved, stats["changed"])
                except Exception:
                    logger.exception("后台增量同步失败: %s", resolved)

            watcher = FileWatcher(
                resolved,
                on_dirty=_bg_sync,
                extensions=set(EXT_TO_LANG.keys()),
            )
            watcher.start()
        ws = Workspace(resolved, store, retriever, indexer, watcher)
        holder.append(ws)
        logger.info(f"工作区创建: {resolved} (watcher={'启用' if watcher else '禁用'})")
        return ws

    def _db_path_for(self, resolved_dir: str) -> str:
        h = hashlib.sha256(
            f"{resolved_dir}|{self.embedder.output_dtype}".encode("utf-8")
        ).hexdigest()[:16]
        return str(Path(self._db_dir) / f"{h}.db")

    def _evict(self) -> list:
        """淘汰超限工作区，返回待销毁列表（调用方在锁外 destroy）。"""
        evicted = []
        while len(self._workspaces) > MAX_ACTIVE:
            key, ws = self._workspaces.popitem(last=False)
            logger.info(f"LRU 淘汰工作区: {key}")
            evicted.append(ws)
        return evicted

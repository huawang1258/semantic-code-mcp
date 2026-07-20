"""文件监控：watchdog + debounce，实时追踪工作区变更。

对齐 augment-context-mcp 的 fs.watch 能力：
- 文件创建/修改/删除 → 标记 dirty
- 2s debounce 合并高频变更
- 主动通知 workspace 触发增量同步
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Callable

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from indexer import DEFAULT_IGNORE_DIRS, FORCE_INCLUDE_FILENAMES

# debounce 间隔（秒）
DEBOUNCE_SECONDS = 2.0
# 累积变更超过此数时立即触发（不等 debounce）
FLUSH_THRESHOLD = 50


class _Handler(FileSystemEventHandler):
    """watchdog 事件 → dirty set。"""

    def __init__(self, watcher: FileWatcher) -> None:
        super().__init__()
        self._watcher = watcher

    def on_any_event(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        src = event.src_path
        if src:
            self._watcher._on_file_event(src)
        # 移动事件有 dest_path
        dest = getattr(event, "dest_path", None)
        if dest:
            self._watcher._on_file_event(dest)


class FileWatcher:
    """单个工作区的文件监控器。

    追踪 dirty 文件集合，debounce 后回调通知上层做增量同步。
    """

    def __init__(
        self,
        root: str,
        on_dirty: Callable[[], None] | None = None,
        extensions: set[str] | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self._on_dirty = on_dirty
        self._extensions = extensions  # 为 None 时不过滤扩展名
        self._dirty: set[str] = set()
        self._deleted: set[str] = set()
        self._lock = threading.Lock()
        self._timer: threading.Timer | None = None
        self._observer: Observer | None = None
        self._stopped = False

    @property
    def dirty_count(self) -> int:
        with self._lock:
            return len(self._dirty) + len(self._deleted)

    def take_dirty(self) -> tuple[set[str], set[str]]:
        """取出并清空 dirty/deleted 集合（原子操作）。"""
        with self._lock:
            d, r = self._dirty, self._deleted
            self._dirty = set()
            self._deleted = set()
            return d, r

    def start(self) -> None:
        """启动文件监控。"""
        if self._observer is not None:
            return
        handler = _Handler(self)
        self._observer = Observer()
        self._observer.schedule(handler, str(self.root), recursive=True)
        self._observer.daemon = True
        self._observer.start()

    def stop(self) -> None:
        """停止文件监控并清理。"""
        self._stopped = True
        if self._timer:
            self._timer.cancel()
            self._timer = None
        if self._observer:
            self._observer.stop()
            try:
                self._observer.join(timeout=2)
            except Exception:
                pass
            self._observer = None

    def _on_file_event(self, filepath: str) -> None:
        """处理单个文件事件。"""
        if self._stopped:
            return
        p = Path(filepath)
        # 过滤：忽略目录中的文件
        try:
            rel = p.relative_to(self.root)
        except ValueError:
            return
        # 跳过忽略目录
        parts = rel.parts
        if any(part in DEFAULT_IGNORE_DIRS or part.startswith(".") for part in parts[:-1]):
            return
        # 过滤扩展名（豁免文件如 .windsurfrules 无扩展名，不受此限）
        if (
            self._extensions
            and p.suffix.lower() not in self._extensions
            and p.name not in FORCE_INCLUDE_FILENAMES
        ):
            return
        with self._lock:
            if p.exists():
                self._dirty.add(str(p))
                self._deleted.discard(str(p))
            else:
                self._deleted.add(str(p))
                self._dirty.discard(str(p))
            count = len(self._dirty) + len(self._deleted)
        # debounce 或立即 flush
        if count >= FLUSH_THRESHOLD:
            self._fire_callback()
        else:
            self._schedule_debounce()

    def _schedule_debounce(self) -> None:
        """重置 debounce 计时器。"""
        if self._timer:
            self._timer.cancel()
        self._timer = threading.Timer(DEBOUNCE_SECONDS, self._fire_callback)
        self._timer.daemon = True
        self._timer.start()

    def _fire_callback(self) -> None:
        """触发 dirty 回调。"""
        if self._timer:
            self._timer.cancel()
            self._timer = None
        if self._on_dirty and not self._stopped:
            try:
                self._on_dirty()
            except Exception:
                pass

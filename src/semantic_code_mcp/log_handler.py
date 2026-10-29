"""多进程共享日志文件的滚动 handler（Windows 安全、绝不写 stderr）。

背景（2026-10 事故）：同一台机器上多个 MCP 实例（Windsurf / Devin CLI / uvx 远端版）共写
~/.semantic-code-mcp/server.log，标准库 RotatingFileHandler 在这种场景下三处失效：
  1. 每个实例常驻持有文件句柄：Windows 上 rename 轮转必报 WinError 32，此后每条日志都失败丢弃
  2. py3.12+ 用各进程缓存的 stream.tell() 判断大小：多进程下位置是旧的，文件越过上限
  3. 失败走 Handler.handleError 把 traceback 写 stderr：host 不读 stderr 时 pipe 写满，
     write 同步阻塞在事件循环线程 —— 整个 MCP server 卡死，工具调用永不返回
"""
from __future__ import annotations

import contextlib
import logging
import os
import shutil
import time

if os.name == "nt":
    import msvcrt

    def _try_lock(fd: int) -> None:
        # msvcrt 锁的是"当前文件位置起"的字节区间，固定锁第 0 字节（文件为空也可锁）
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)

    def _unlock(fd: int) -> None:
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
else:
    import fcntl

    def _try_lock(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _unlock(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_UN)

# 抢跨进程锁的轮询间隔（临界区只有 stat + 追加写，正常竞争在毫秒内结束）
_LOCK_POLL_SECONDS = 0.005
# 轮转彻底失败（rename 与截断降级都失败）后的冷却期：避免每条日志都重试拷贝大文件
_ROTATE_RETRY_SECONDS = 60.0

# 本服务不提供 prompts/resources，客户端仍会每 ~15s 轮询；这类请求日志零信息量，
# 曾占 server.log 99.8% 的行数（两周写满 10MB 触发轮转）
_POLLING_REQUESTS = frozenset({
    "ListPromptsRequest",
    "ListResourcesRequest",
    "ListResourceTemplatesRequest",
    "PingRequest",
})


class SharedRotatingFileHandler(logging.Handler):
    """按大小滚动、可被多进程安全共享的文件 handler。

    - 每条记录 open→append→close，不常驻句柄：其它进程随时可以 rename 轮转
    - 查大小 + 轮转 + 写入在跨进程文件锁（<日志>.lock）内完成，按真实文件大小判断
    - 锁有界等待（lock_timeout）：超时就不轮转直接追加，绝不长时间阻塞调用方（可能是事件循环）
    - rename 失败（仍有旧版实例常驻持有句柄）降级为 copy + 原地截断：截断不需要 FILE_SHARE_DELETE，
      CRT 默认共享模式打开的旧句柄挡不住
    - 任何异常静默丢弃（handleError 空实现）：日志永远不能拖垮服务
    """

    terminator = "\n"

    def __init__(
        self,
        filename: str | os.PathLike[str],
        max_bytes: int = 10 * 1024 * 1024,
        backup_count: int = 1,
        encoding: str = "utf-8",
        lock_timeout: float = 1.0,
    ) -> None:
        super().__init__()
        self.baseFilename = os.path.abspath(os.fspath(filename))
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.encoding = encoding
        self.lock_timeout = lock_timeout
        self._lock_fd: int | None = None
        self._rotate_retry_at = 0.0

    def emit(self, record: logging.LogRecord) -> None:
        try:
            data = (self.format(record) + self.terminator).encode(self.encoding, "backslashreplace")
            locked = self._lock_interprocess()
            try:
                if locked:
                    self._rotate_if_needed(len(data))
                with open(self.baseFilename, "ab") as f:
                    f.write(data)
            finally:
                if locked:
                    self._unlock_interprocess()
        except Exception:
            self.handleError(record)

    def handleError(self, record: logging.LogRecord) -> None:
        """静默丢弃。基类实现会把 traceback 写 stderr，host 不读 stderr 时写满 pipe 会卡死进程。"""

    def close(self) -> None:
        self.acquire()
        try:
            if self._lock_fd is not None:
                with contextlib.suppress(OSError):
                    os.close(self._lock_fd)
                self._lock_fd = None
        finally:
            self.release()
            super().close()

    def _lock_interprocess(self) -> bool:
        """有界等待拿跨进程锁；超时或锁文件打不开返回 False（调用方降级为不轮转直接追加）。"""
        try:
            if self._lock_fd is None:
                self._lock_fd = os.open(self.baseFilename + ".lock", os.O_RDWR | os.O_CREAT, 0o644)
            deadline = time.monotonic() + self.lock_timeout
            while True:
                try:
                    _try_lock(self._lock_fd)
                    return True
                except OSError:
                    if time.monotonic() >= deadline:
                        return False
                    time.sleep(_LOCK_POLL_SECONDS)
        except OSError:
            return False

    def _unlock_interprocess(self) -> None:
        try:
            _unlock(self._lock_fd)
        except OSError:
            # 解锁失败就关闭 fd 强制释放（OS 随句柄关闭释放锁），否则其它进程每条日志都要等满超时
            with contextlib.suppress(OSError):
                os.close(self._lock_fd)
            self._lock_fd = None

    def _rotate_if_needed(self, incoming: int) -> None:
        if self.max_bytes <= 0 or time.monotonic() < self._rotate_retry_at:
            return
        try:
            size = os.path.getsize(self.baseFilename)
        except OSError:
            return
        # 空文件永不轮转：单条超长记录也得有地方落盘
        if size == 0 or size + incoming <= self.max_bytes:
            return
        try:
            self._rotate()
        except OSError:
            self._rotate_retry_at = time.monotonic() + _ROTATE_RETRY_SECONDS

    def _rotate(self) -> None:
        base = self.baseFilename
        if self.backup_count <= 0:
            self._truncate(base)
            return
        for i in range(self.backup_count - 1, 0, -1):
            src = f"{base}.{i}"
            if os.path.exists(src):
                os.replace(src, f"{base}.{i + 1}")
        try:
            os.replace(base, base + ".1")
        except PermissionError:
            shutil.copyfile(base, base + ".1")
            self._truncate(base)

    @staticmethod
    def _truncate(path: str) -> None:
        with open(path, "r+b") as f:
            f.truncate(0)


class PollingNoiseFilter(logging.Filter):
    """挂在 mcp.server.lowlevel.server logger 上，丢弃客户端轮询类请求的
    "Processing request of type <X>" 日志。只看 args/msg 不调 getMessage()：
    过滤器抛异常会冒泡到打日志的业务代码。"""

    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args
        if isinstance(args, tuple) and len(args) == 1 and isinstance(args[0], str):
            return args[0] not in _POLLING_REQUESTS
        return not (isinstance(record.msg, str) and record.msg.rpartition(" ")[2] in _POLLING_REQUESTS)

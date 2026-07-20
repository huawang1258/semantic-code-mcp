"""索引编排：扫描目录 → 切分 → embedding → 存储，支持基于文件 hash 的增量同步。

扫描时跳过常见无关目录、gitignore 命中文件、超大文件与不支持的扩展名。
"""
from __future__ import annotations

import hashlib
import logging
import os
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import pathspec

from chunker import EXT_TO_LANG, chunk_file
from embedder import Embedder
from store import CodeStore

logger = logging.getLogger("semantic-code-mcp")


def _bar(pct: int, width: int = 20) -> str:
    """ASCII 进度条，如 █████░░░░░░░░░░░░░░░。"""
    filled = int(width * max(0, min(pct, 100)) / 100)
    return "█" * filled + "░" * (width - filled)


# 默认跳过的目录
DEFAULT_IGNORE_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", "env",
    "dist", "build", "target", "out", ".next", ".nuxt", "vendor",
    ".idea", ".vscode", ".gradle", "bin", "obj", "coverage", ".pytest_cache",
}
# 超过此大小（字节）的文件跳过
MAX_FILE_SIZE = 1_000_000
# 高噪音生成物（锁文件等），即使扩展名在白名单也跳过
SKIP_FILENAMES = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "npm-shrinkwrap.json",
    "poetry.lock", "uv.lock", "Cargo.lock", "composer.lock", "Gemfile.lock",
}
# agent 指导文件：架构/规范信息密度最高，很多仓库把它们 gitignore 了，索引时豁免
FORCE_INCLUDE_FILENAMES = {"AGENTS.md", "CLAUDE.md", "GEMINI.md", "AGENT.md", ".windsurfrules", ".cursorrules"}
# 单批 embedding 的最大累计块数（跨文件合并，减少 API 往返）
BATCH_CHUNKS = 256


def _embed_concurrency(embedder) -> int:
    """embedding 并发数：SCM_EMBED_CONCURRENCY 优先，否则用后端建议值。

    voyage（网络 IO）默认 3；本地推理（算力瓶颈）默认 1；未知后端保守串行。
    """
    env = os.getenv("SCM_EMBED_CONCURRENCY")
    if env:
        try:
            return max(1, int(env))
        except ValueError:
            pass
    return max(1, int(getattr(embedder, "default_concurrency", 1)))


class _EmbedPipeline:
    """有界并发 embedding 流水线。

    worker 线程只跑 embed_documents（纯网络/纯计算）；切分与 sqlite 写回
    全部留在调用线程，保证 store 单线程访问；按提交顺序写回（FIFO），
    在飞批次数有界（workers+1）防内存膨胀。失败时异常在调用线程抛出，
    未写回文件的 hash 不更新，下次同步自然重试。
    """

    def __init__(self, indexer: "Indexer", workers: int,
                 on_write: Callable[[int, int], None] | None = None) -> None:
        self.indexer = indexer
        self.workers = workers
        self.on_write = on_write
        self._executor = ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="scm-embed",
        ) if workers > 1 else None
        self._pending: deque = deque()

    def submit(self, batch: list[tuple[str, str, float, int, list]]) -> None:
        """提交一批 (fp, hash, mtime, size, chunks)；可能阻塞写回最早的在飞批次。"""
        all_chunks = [c for _, _, _, _, cs in batch for c in cs]
        texts = [_embed_text(c, self.indexer.root) for c in all_chunks]
        if self._executor:
            fut = self._executor.submit(
                self.indexer.embedder.embed_documents, texts) if texts else None
            self._pending.append((batch, all_chunks, fut))
            while len(self._pending) > self.workers:
                self._write_oldest()
        else:
            embeddings = self.indexer.embedder.embed_documents(texts) if texts else []
            self._write(batch, all_chunks, embeddings)

    def drain(self) -> None:
        """等待并写回全部在飞批次。"""
        while self._pending:
            self._write_oldest()

    def close(self) -> None:
        """释放线程池（幂等；异常路径下也必须调用，防线程泄漏）。"""
        if self._executor:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None
        self._pending.clear()

    def _write_oldest(self) -> None:
        batch, all_chunks, fut = self._pending.popleft()
        embeddings = fut.result() if fut else []
        self._write(batch, all_chunks, embeddings)

    def _write(self, batch, all_chunks, embeddings) -> None:
        store = self.indexer.store
        offset = 0
        for fp, fh, mt, sz, cs in batch:
            n = len(cs)
            if n:
                store.add_chunks(cs, embeddings[offset:offset + n])
                offset += n
            store.set_file_hash(fp, fh, mtime=mt, size=sz)
        if self.on_write:
            self.on_write(len(batch), len(all_chunks))


def _embed_text(chunk, root: Path) -> str:
    """构造带上下文的 embedding 文本（不改变存储，只影响向量）。

    在裸代码前添加文件路径 + 符号名作为上下文注释，让 embedding 模型
    理解代码的「角色」和「位置」，显著提升语义对齐（Contextual Retrieval）。
    """
    parts: list[str] = []
    fp = chunk.file_path
    try:
        fp = str(Path(fp).relative_to(root))
    except (ValueError, TypeError):
        pass
    if fp:
        parts.append(f"# File: {fp}")
    if chunk.symbol:
        parts.append(f"# Symbol: {chunk.symbol}")
    if chunk.language:
        parts.append(f"# Language: {chunk.language}")
    if parts:
        parts.append("")
    parts.append(chunk.code)
    return "\n".join(parts)


class Indexer:
    """单个工作区的索引器。"""

    def __init__(self, root: str, store: CodeStore, embedder: Embedder) -> None:
        self.root = Path(root).resolve()
        self.store = store
        self.embedder = embedder
        self._gitignore = self._load_gitignore()

    @staticmethod
    def _scope_pattern(line: str, rel_dir: str) -> str:
        """把嵌套 .gitignore 的模式改写为相对于根目录的模式。

        遵循 git 语义：
        - `!` 前缀（否定）必须保留在最前面
        - 不含斜杠的模式匹配该 .gitignore 目录下任意层级 -> 加 `**/`
        - 含斜杠的模式锚定到该 .gitignore 所在目录 -> 直接拼接
        """
        if not rel_dir:
            return line
        neg = line.startswith("!")
        body = line[1:] if neg else line
        anchored = body.startswith("/")
        body = body.lstrip("/")
        if not anchored and "/" not in body.rstrip("/"):
            scoped = f"{rel_dir}/**/{body}"
        else:
            scoped = f"{rel_dir}/{body}"
        return f"!{scoped}" if neg else scoped

    def _load_gitignore(self) -> pathspec.PathSpec | None:
        """收集所有层级 .gitignore 模式，合并为一个 PathSpec。"""
        patterns: list[str] = []
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = [
                d for d in dirnames
                if d not in DEFAULT_IGNORE_DIRS and not d.startswith(".")
            ]
            if ".gitignore" in filenames:
                gi = Path(dirpath) / ".gitignore"
                try:
                    rel_dir = Path(dirpath).relative_to(self.root).as_posix()
                except ValueError:
                    rel_dir = ""
                if rel_dir == ".":
                    rel_dir = ""
                try:
                    for line in gi.read_text(encoding="utf-8", errors="ignore").splitlines():
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        patterns.append(self._scope_pattern(line, rel_dir))
                except Exception:
                    pass
        if not patterns:
            return None
        return pathspec.PathSpec.from_lines("gitwildmatch", patterns)

    def _should_skip(self, p: Path, size: int | None = None) -> bool:
        """统一的文件过滤口径：扩展名 / gitignore / 大小。

        全量扫描与 watcher 增量必须走同一套规则，避免口径不一致。
        """
        if p.name in SKIP_FILENAMES:
            return True
        # 豁免文件跳过扩展名与 gitignore 检查（如无扩展名的 .windsurfrules）
        force = p.name in FORCE_INCLUDE_FILENAMES
        if not force and p.suffix.lower() not in EXT_TO_LANG:
            return True
        try:
            rel = p.relative_to(self.root).as_posix()
        except ValueError:
            return True
        if self._gitignore and not force and self._gitignore.match_file(rel):
            return True
        if size is None:
            try:
                size = p.stat().st_size
            except OSError:
                return True
        return size > MAX_FILE_SIZE

    def _iter_source_files(self):
        for dirpath, dirnames, filenames in os.walk(self.root):
            # 原地过滤忽略目录与隐藏目录
            dirnames[:] = [
                d for d in dirnames
                if d not in DEFAULT_IGNORE_DIRS and not d.startswith(".")
            ]
            for fn in filenames:
                p = Path(dirpath) / fn
                if self._should_skip(p):
                    continue
                yield p

    @staticmethod
    def _file_hash(path: Path) -> str:
        h = hashlib.sha256()
        try:
            with open(path, "rb") as f:
                for block in iter(lambda: f.read(65536), b""):
                    h.update(block)
        except OSError:
            return ""
        return h.hexdigest()

    def sync_dirty(
        self,
        dirty: set[str],
        deleted: set[str],
        progress: Callable[[int, int, str], None] | None = None,
    ) -> dict:
        """Watcher 驱动的增量同步：只处理变更文件，跳过全量扫描。

        比 sync() 快得多（O(变更数) vs O(总文件数)）。
        """
        t0 = time.time()
        # 清除已删除文件
        for fp in deleted:
            self.store.delete_file(fp)
            self.store.remove_file_record(fp)
        if deleted:
            logger.info("[增量] 清除 %d 个已删除文件", len(deleted))
        # 只对 dirty 文件做 hash 对比 + 重索引
        to_index: list[tuple[str, str, float, int]] = []
        for fp in dirty:
            p = Path(fp)
            if not p.exists():
                self.store.delete_file(fp)
                self.store.remove_file_record(fp)
                continue
            try:
                st = p.stat()
            except OSError:
                continue
            # 与全量扫描同口径：gitignore / 扩展名 / 大小过滤
            if self._should_skip(p, size=st.st_size):
                continue
            fh = self._file_hash(p)
            if not fh:
                continue
            if self.store.get_file_hash(fp) != fh:
                to_index.append((fp, fh, st.st_mtime, st.st_size))
        total = len(to_index)
        chunks_total = 0
        pipeline = _EmbedPipeline(self, _embed_concurrency(self.embedder))
        buf: list[tuple[str, str, float, int, list]] = []
        buf_chunks = 0
        try:
            for done, (fp, fh, mt, sz) in enumerate(to_index, 1):
                self.store.delete_file(fp)
                chunks = chunk_file(fp)
                chunks_total += len(chunks)
                buf.append((fp, fh, mt, sz, chunks))
                buf_chunks += len(chunks)
                if progress:
                    progress(done, total, fp)
                if buf_chunks >= BATCH_CHUNKS:
                    pipeline.submit(buf)
                    buf, buf_chunks = [], 0
            if buf:
                pipeline.submit(buf)
            pipeline.drain()
        finally:
            pipeline.close()
        if total:
            logger.info("[增量] 完成: %d 文件 / %d 块 (%.1fs)", total, chunks_total, time.time() - t0)
        stats = self.store.stats()
        return {"total_files": stats.get("files", 0), "changed": total, **stats}

    def sync(self, progress: Callable[[int, int, str], None] | None = None) -> dict:
        """全量/增量同步。

        用 mtime+size 快速预筛，只对变化文件算 hash 对比并重索引。
        """
        # 重载 gitignore（.gitignore 可能已变更）
        self._gitignore = self._load_gitignore()
        t_scan = time.time()
        current_paths: set[str] = set()
        to_index: list[tuple[str, str, float, int]] = []

        for p in self._iter_source_files():
            fp = str(p)
            current_paths.add(fp)
            try:
                st = p.stat()
            except OSError:
                continue
            meta = self.store.get_file_stat(fp)
            if meta and meta["mtime"] == st.st_mtime and meta["size"] == st.st_size:
                continue
            fh = self._file_hash(p)
            if not fh:
                continue
            if meta and meta["hash"] == fh:
                self.store.set_file_hash(fp, fh, mtime=st.st_mtime, size=st.st_size)
                continue
            to_index.append((fp, fh, st.st_mtime, st.st_size))

        indexed = self.store.all_indexed_files()

        # 清除已删除文件
        gone_count = 0
        for gone in indexed - current_paths:
            self.store.delete_file(gone)
            self.store.remove_file_record(gone)
            gone_count += 1

        total = len(to_index)
        logger.info(
            "[1/2 扫描] %s: %d 个源文件, 变更 %d, 删除 %d (%.1fs)",
            self.root.name, len(current_paths), total, gone_count, time.time() - t_scan,
        )

        # 缓冲多个文件的 chunk，攒够一批提交流水线（并发 embedding，顺序写回）
        t_embed = time.time()
        done_files = [0]
        done_chunks = [0]

        def _on_write(nfiles: int, nchunks: int) -> None:
            done_files[0] += nfiles
            done_chunks[0] += nchunks
            # 每批一条进度线：进度条 + 文件/块计数 + 已用/预估剩余时长
            elapsed = time.time() - t_embed
            pct = done_files[0] * 100 // total if total else 100
            eta = elapsed / done_files[0] * (total - done_files[0]) if done_files[0] else 0.0
            logger.info(
                "[2/2 向量化] %s %3d%% | %d/%d 文件 | %d 块 | 已用 %.0fs / 预估剩 %.0fs",
                _bar(pct), pct, done_files[0], total, done_chunks[0], elapsed, eta,
            )

        workers = _embed_concurrency(self.embedder)
        if total and workers > 1:
            logger.info("[2/2 向量化] 并发 %d 路", workers)
        pipeline = _EmbedPipeline(self, workers, on_write=_on_write)
        buf: list[tuple[str, str, float, int, list]] = []
        buf_chunks = 0
        try:
            for done, (fp, fh, mt, sz) in enumerate(to_index, 1):
                self.store.delete_file(fp)
                chunks = chunk_file(fp)
                buf.append((fp, fh, mt, sz, chunks))
                buf_chunks += len(chunks)
                if progress:
                    progress(done, total, fp)
                if buf_chunks >= BATCH_CHUNKS:
                    pipeline.submit(buf)
                    buf, buf_chunks = [], 0
            if buf:
                pipeline.submit(buf)
            pipeline.drain()
        finally:
            pipeline.close()

        stats = self.store.stats()
        return {"total_files": len(current_paths), "changed": total, **stats}

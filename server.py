#!/usr/bin/env python3
"""semantic-code-mcp 入口。

暴露单一 MCP 工具：
  codebase_search  自然语言语义检索代码库（首次自动全量索引，后续 watcher 增量同步）

工程特性（对齐 augment-context-mcp）：
  - 多工作区 LRU 管理（SCM_MAX_WORKSPACES，默认 8）
  - watchdog 文件监控 + 2s debounce（SCM_WATCH=0 禁用）
  - MCP progress notification（索引进度实时回报）
  - Cancel signal 支持（客户端取消即刻中止）
  - 首次全量索引 / 后续 watcher 增量（O(变更数) 极速）
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

from dotenv import load_dotenv
from mcp.server.fastmcp import Context, FastMCP

from embedder import create_embedder
from workspace import WorkspaceManager

# MCP 由 IDE 启动时 CWD 不定，.env 固定从本文件所在目录加载
load_dotenv(Path(__file__).parent / ".env")

# 日志只写本地文件（MCP_HANG_PROOF_DESIGN 原则 2）：
# host 不消费 stderr 时 64KB pipe buffer 写满会同步阻塞进程，
# stderr 仅 SCM_LOG_STDERR=1 显式开启（调试用）。
LOG_FILE = os.getenv("SCM_LOG_FILE") or str(
    Path(os.getenv("SCM_DB_DIR") or Path.home() / ".semantic-code-mcp") / "server.log"
)
_handlers: list[logging.Handler] = []
if os.getenv("SCM_LOG_DISABLE") != "1":
    Path(LOG_FILE).parent.mkdir(parents=True, exist_ok=True)
    _handlers.append(RotatingFileHandler(
        LOG_FILE,
        maxBytes=int(os.getenv("SCM_LOG_MAX_BYTES", str(10 * 1024 * 1024))),
        backupCount=1,
        encoding="utf-8",
    ))
if os.getenv("SCM_LOG_STDERR") == "1":
    _handlers.append(logging.StreamHandler(sys.stderr))
if not _handlers:
    _handlers.append(logging.NullHandler())
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%m-%d %H:%M:%S",
    handlers=_handlers,
)
logger = logging.getLogger("semantic-code-mcp")

mcp = FastMCP("semantic-code-mcp")

# 单文件最多展示行数（控制 token 开销）：前几名全量，后面的递减
_MAX_FILE_LINES = 120
# 排名超过此名次的文件用精简行预算（越靠后相关性越低，不值得等额 token）
_FULL_BUDGET_TOP = 3
_MAX_FILE_LINES_TAIL = 40
# 实体/DTO 块单块上限：字段列表看前几十行足够，140 行全量吐是浪费
_MAX_ENTITY_LINES = 30
# 任意单块上限：类级大块（整个 Service 类头几百行）价值密度低，
# 单块吃满文件预算会挤掉同文件其它命中块
_MAX_CHUNK_LINES = 80
# 单行最大字符数：builder 链/SQL 拼接行能到几百字符，截尾不影响理解
_MAX_LINE_CHARS = 200
# 单文件字符总预算：行数预算管不住长行密集文件（120 行×200 字符最坏 24KB），
# 字符级预算顶住 token 上限
_MAX_FILE_CHARS = 8000
# 关联（call graph）块最多展示行数
_MAX_RELATED_LINES = 20

# 全局单例
_embedder = None
_manager: WorkspaceManager | None = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = create_embedder(model=os.getenv("SCM_EMBED_MODEL", "voyage-code-3"))
    return _embedder


def _get_manager() -> WorkspaceManager:
    global _manager
    if _manager is None:
        _manager = WorkspaceManager(_get_embedder())
    return _manager


_RELATION_LABEL = {"callee": "\u2192 \u88ab\u8c03\u7528", "caller": "\u2190 \u8c03\u7528\u8005"}


def _numbered_lines(code: str, start_line: int, budget: int) -> tuple[list[str], int]:
    """把代码渲染为带行号的行列表，超出预算截断，超长行截尾。返回 (行列表, 消耗行数)。"""
    code_lines = code.split("\n")
    take = code_lines[:budget]
    out = [
        f"{start_line + i:>6}\t"
        + (ln if len(ln) <= _MAX_LINE_CHARS else ln[:_MAX_LINE_CHARS] + " …")
        for i, ln in enumerate(take)
    ]
    omitted = len(code_lines) - len(take)
    if omitted > 0:
        out.append(f"... ({omitted} more lines)")
    return out, len(take)


def _format_file_group(idx: int, file_path: str, chunks: list[dict]) -> str:
    """按文件聚合渲染：块按行号排序，非连续块之间用 ... 分隔。

    分层预算：前 _FULL_BUDGET_TOP 名全量展示，之后的文件用精简预算。"""
    chunks = sorted(chunks, key=lambda c: c.get("start_line", 0))
    best = max(c.get("score", 0.0) for c in chunks)
    symbols = ", ".join(dict.fromkeys(c.get("symbol", "") for c in chunks if c.get("symbol")))
    lang = chunks[0].get("language", "")
    header = f"## {idx}. {file_path} :: {symbols} (score={best:.3f})"
    body: list[str] = []
    budget = _MAX_FILE_LINES if idx <= _FULL_BUDGET_TOP else _MAX_FILE_LINES_TAIL
    prev_end: int | None = None
    for c in chunks:
        if budget <= 0:
            break
        start = c.get("start_line", 1)
        code = c.get("code", "")
        if prev_end is not None:
            if start > prev_end + 1:
                body.append("...")
            elif start <= prev_end:
                # 行窗口切分的重叠块：裁掉已展示过的行，避免重复渲染
                skip = prev_end - start + 1
                code_lines = code.split("\n")[skip:]
                if not code_lines:
                    continue
                code = "\n".join(code_lines)
                start = prev_end + 1
        chunk_budget = min(budget, _MAX_ENTITY_LINES if c.get("entity") else _MAX_CHUNK_LINES)
        lines, used = _numbered_lines(code, start, chunk_budget)
        body.extend(lines)
        budget -= used
        prev_end = max(prev_end or 0, c.get("end_line", start + used - 1))
        if sum(len(ln) + 1 for ln in body) >= _MAX_FILE_CHARS:
            # 超出字符预算：从尾部丢行直到预算内，再标记截断
            total = 0
            keep = 0
            for ln in body:
                total += len(ln) + 1
                if total > _MAX_FILE_CHARS:
                    break
                keep += 1
            body = body[:keep]
            body.append("... (file output budget reached)")
            break
    return f"{header}\n```{lang}\n" + "\n".join(body) + "\n```"


def _format_results(results: list[dict]) -> str:
    main = [r for r in results if not r.get("relation")]
    related = [r for r in results if r.get("relation")]
    blocks: list[str] = []
    # 主结果按文件聚合（文件顺序 = 该文件最佳块的排名顺序）
    order: list[str] = []
    groups: dict[str, list[dict]] = {}
    for r in main:
        fp = r["file_path"]
        if fp not in groups:
            groups[fp] = []
            order.append(fp)
        groups[fp].append(r)
    for i, fp in enumerate(order, 1):
        blocks.append(_format_file_group(i, fp, groups[fp]))
    # 关联块（call graph 扩展）保持紧凑单块展示
    for r in related:
        tag = _RELATION_LABEL.get(r.get("relation"), r.get("relation"))
        header = (
            f"## [\u5173\u8054 {tag}] {r['file_path']} :: {r['symbol']} "
            f"(L{r['start_line']}-{r['end_line']})"
        )
        lines, _ = _numbered_lines(r.get("code", ""), r.get("start_line", 1), _MAX_RELATED_LINES)
        blocks.append(f"{header}\n```{r.get('language', '')}\n" + "\n".join(lines) + "\n```")
    return "\n\n".join(blocks)


@mcp.tool()
async def codebase_search(
    information_request: str,
    directory_path: str,
    ctx: Context,
    top_n: int = 0,
    path_filter: str = "",
    include_related: bool = True,
) -> str:
    """语义检索代码库，返回最相关的代码片段。

    Args:
        information_request: 自然语言查询，例如"用户登录鉴权逻辑在哪里"
        directory_path: 要检索的代码库绝对路径
        top_n: 可选，返回结果数（0 = 用默认值 SCM_TOP_N，默认 10）
        path_filter: 可选，文件路径过滤；含通配符按 glob 匹配（如 "*.java"、"**/service/**"），
            否则按路径子串匹配（如 "controller"）
        include_related: 可选，是否附带 call graph 关联块（调用者/被调用者），默认 True
    """
    directory = Path(directory_path).resolve()
    t0 = time.time()
    logger.info("[Tool] → codebase_search start dir=%s query=%.80s", directory, information_request)
    if not directory.is_dir():
        return f"Error: \u76ee\u5f55\u4e0d\u5b58\u5728\u6216\u4e0d\u662f\u6587\u4ef6\u5939: {directory_path}"
    try:
        manager = _get_manager()
        ws = manager.get(str(directory))
        # Progress: 索引阶段
        await ctx.report_progress(0, 100)
        await ctx.info(f"\u5de5\u4f5c\u533a: {directory} (LRU {manager.active_count}/{os.getenv('SCM_MAX_WORKSPACES', '8')})")

        # 增量同步（watcher dirty 或全量）
        loop = asyncio.get_running_loop()
        _last_pct = [0]

        def _sync_with_progress():
            def _progress(done, total, fp):
                if total <= 0:
                    return
                pct = 10 + int(done / total * 50)
                if pct - _last_pct[0] >= 5:
                    _last_pct[0] = pct
                    asyncio.run_coroutine_threadsafe(ctx.report_progress(pct, 100), loop)
            return manager.sync_workspace(ws, on_progress=_progress)

        await ctx.report_progress(10, 100)
        stats = await loop.run_in_executor(None, _sync_with_progress)
        changed = stats.get("changed", 0)
        if changed > 0:
            await ctx.info(f"\u7d22\u5f15\u540c\u6b65: {changed} \u4e2a\u6587\u4ef6\u53d8\u66f4")
        await ctx.report_progress(60, 100)

        # 检索阶段
        top_k = int(os.getenv("SCM_TOP_K", "50"))
        n = top_n if top_n > 0 else int(os.getenv("SCM_TOP_N", "10"))
        pf = path_filter.strip() or None
        results = await loop.run_in_executor(
            None,
            lambda: ws.search(
                information_request,
                top_k=top_k,
                top_n=n,
                expand_graph=include_related,
                path_filter=pf,
            ),
        )
        await ctx.report_progress(100, 100)
        logger.info("[Tool] ✓ codebase_search done %.1fs results=%d", time.time() - t0, len(results))
        if not results:
            return "\u672a\u627e\u5230\u76f8\u5173\u4ee3\u7801\u3002"
        return _format_results(results)
    except asyncio.CancelledError:
        logger.info("[Tool] ⊗ codebase_search cancelled %.1fs（后台索引线程会继续跑完并持久化）", time.time() - t0)
        return "Error: \u8bf7\u6c42\u5df2\u88ab\u53d6\u6d88"
    except Exception as e:
        logger.exception("[Tool] ✗ codebase_search FAIL %.1fs", time.time() - t0)
        return f"Error: {e}\n\uff08\u8bf7\u786e\u8ba4\u5df2\u5728\u73af\u5883\u53d8\u91cf\u4e2d\u914d\u7f6e VOYAGE_API_KEY\uff09"


def main() -> None:
    logger.info(
        f"========== semantic-code-mcp 启动 pid={os.getpid()} "
        f"(max_workspaces={os.getenv('SCM_MAX_WORKSPACES', '8')}, "
        f"watch={'enabled' if os.getenv('SCM_WATCH', '1') != '0' else 'disabled'}, "
        f"log={LOG_FILE}) =========="
    )
    mcp.run()


if __name__ == "__main__":
    main()

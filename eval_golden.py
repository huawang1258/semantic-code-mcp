#!/usr/bin/env python3
"""50 条分类 golden set 检索质量评测。

目标仓库：CLIProxyAPI（真实中型 Go 仓库）。
10 个类别 × 5 条：
  A 符号定位 / B 调用链 / C 架构职责 / D 配置schema / E 同名干扰
  F SDK边界 / G 协议契约 / H 测试fixtures / I 中文查询 / J 细节实现

附加 K 类（不计入主总分，独立小节）：中文 Java 业务仓库探针回归集，
覆盖主评测盖不到的链路：复合意图拆分/子查询分数混合/保底、trigram 中文召回、
实现意图降权、expand_graph=True 全链路。依赖本机已索引的生产 DB，不存在则跳过。

指标（全部机械化打分，可重复，不依赖 LLM-judge）：
  Top-1 精确命中：Top-1 文件 == 期望文件
  Recall@5：期望文件进入 Top-5（expected_all 时要求全部进入）
  答案可用性：Top-5 命中期望文件，或返回代码中包含 must_contain 关键字

用法：
  python eval_golden.py                    # 评测（索引持久化复用）
  python eval_golden.py --json             # 额外输出 JSON
  python eval_golden.py --backend local    # 本地 embedding 后端（Qwen3-Embedding-0.6B）
  python eval_golden.py --expand           # 启用 HyDE+多查询扩展（需 SCM_LLM_* 环境变量）
  python eval_golden.py --skip-java        # 跳过 K 类 Java 探针
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from embedder import create_embedder
from expander import QueryExpander
from indexer import Indexer
from retriever import Retriever
from store import CodeStore

load_dotenv()

# 评测目标：CLIProxyAPI（github.com/router-for-me/CLIProxyAPI，clone 后用
# SCM_EVAL_TARGET 指向本地路径即可复现）
TARGET = Path(os.getenv("SCM_EVAL_TARGET", r"D:\project\main\CLIProxyAPI"))
TOP_N = 10
BACKEND = "local" if "--backend" in sys.argv and "local" in sys.argv else "voyage"
EXPAND = "--expand" in sys.argv
# 不同后端向量不兼容，DB 分开持久化
_suffix = "" if BACKEND == "voyage" else f"_{BACKEND}"
DB_PATH = str(Path(__file__).parent / f".eval_cliproxy{_suffix}.db")

# 每条：cat / query / expected（相对路径，命中任一即算）/ must_contain（可选，答案可用性关键字，任一命中即可）
GOLDEN: list[dict] = [
    # ---------- A 符号定位 ----------
    {"cat": "A", "query": "where is ParseSuffix defined and what does it return",
     "expected": ["internal/thinking/suffix.go"], "must_contain": ["func ParseSuffix"]},
    {"cat": "A", "query": "ConvertBudgetToLevel threshold based mapping function",
     "expected": ["internal/thinking/convert.go"], "must_contain": ["func ConvertBudgetToLevel"]},
    {"cat": "A", "query": "RoundRobinSelector Pick next available auth",
     "expected": ["sdk/cliproxy/auth/selector.go"], "must_contain": ["func (s *RoundRobinSelector) Pick"]},
    {"cat": "A", "query": "StartModelsUpdater background updater entry function",
     "expected": ["internal/registry/model_updater.go"], "must_contain": ["func StartModelsUpdater"]},
    {"cat": "A", "query": "GetModelGroup classify model name into gpt claude gemini group",
     "expected": ["internal/cache/signature_cache.go"], "must_contain": ["func GetModelGroup"]},
    # ---------- B 调用链（触发 edges 直查） ----------
    {"cat": "B", "query": "ApplyThinking 被哪些文件调用了？",
     "expected": ["internal/runtime/executor/gemini_executor.go", "internal/runtime/executor/claude_executor.go",
                  "internal/runtime/executor/codex_executor.go", "internal/pluginhost/adapters.go"],
     "must_contain": ["ApplyThinking("]},
    {"cat": "B", "query": "who calls GeneratePKCECodes",
     "expected": ["sdk/auth/claude.go", "sdk/auth/codex.go", "sdk/auth/xai.go",
                  "internal/api/handlers/management/auth_files.go"], "must_contain": ["GeneratePKCECodes()"]},
    {"cat": "B", "query": "callers of NotifyUsageRefresh",
     "expected": ["internal/watcher/clients.go"], "must_contain": ["NotifyUsageRefresh()"]},
    {"cat": "B", "query": "哪些文件调用了 NewProxyAwareHTTPClient",
     "expected": ["internal/runtime/executor/gemini_executor.go", "internal/runtime/executor/claude_executor.go",
                  "internal/runtime/executor/codex_executor.go", "internal/runtime/executor/kimi_executor.go",
                  "internal/runtime/executor/antigravity_executor.go"],
     "must_contain": ["NewProxyAwareHTTPClient("]},
    {"cat": "B", "query": "who calls ExchangeCodeForTokens",
     "expected": ["sdk/auth/claude.go", "sdk/auth/antigravity.go",
                  "internal/api/handlers/management/auth_files.go"], "must_contain": ["ExchangeCodeForTokens("]},
    # ---------- C 架构职责 ----------
    {"cat": "C", "query": "what does internal/thinking package do and what is its architecture",
     "expected": ["AGENTS.md", "internal/thinking/apply.go"], "must_contain": ["canonical", "ThinkingConfig"]},
    {"cat": "C", "query": "project architecture overview which directory holds translators executors registry",
     "expected": ["AGENTS.md", "README.md"], "must_contain": ["internal/translator", "internal/registry"]},
    {"cat": "C", "query": "code conventions for error wrapping and logging in this repo",
     "expected": ["AGENTS.md"], "must_contain": ["logrus", "gofmt"]},
    {"cat": "C", "query": "where are provider protocol translators organized",
     "expected": ["internal/translator/init.go", "AGENTS.md"], "must_contain": ["translator"]},
    {"cat": "C", "query": "what is the sdk/cliproxy package for embedding the service",
     "expected": ["sdk/cliproxy/builder.go", "sdk/cliproxy/service.go", "AGENTS.md"], "must_contain": ["Service", "Builder"]},
    # ---------- D 配置 / schema ----------
    {"cat": "D", "query": "config key to disable quota cooldown scheduling",
     "expected": ["internal/config/config.go", "config.example.yaml"], "must_contain": ["disable-cooling"]},
    {"cat": "D", "query": "request-retry and max-retry-credentials config fields meaning",
     "expected": ["internal/config/config.go", "config.example.yaml"], "must_contain": ["request-retry", "MaxRetryCredentials"]},
    {"cat": "D", "query": "usage statistics enabled configuration option",
     "expected": ["internal/config/config.go", "config.example.yaml"], "must_contain": ["usage-statistics-enabled"]},
    {"cat": "D", "query": "how to configure a proxy url per API key",
     "expected": ["internal/config/config.go", "config.example.yaml"], "must_contain": ["proxy-url"]},
    {"cat": "D", "query": "example config file with api-keys list",
     "expected": ["config.example.yaml"], "must_contain": ["api-keys"]},
    # ---------- E 同名/近义干扰 ----------
    {"cat": "E", "query": "xai websocket executor not codex",
     "expected": ["internal/runtime/executor/xai_websockets_executor.go"], "must_contain": ["XAI"]},
    {"cat": "E", "query": "codex OAuth callback server with /auth/callback endpoint",
     "expected": ["internal/auth/codex/oauth_server.go"], "must_contain": ["/auth/callback"]},
    {"cat": "E", "query": "claude provider thinking applier max_tokens constraint",
     "expected": ["internal/thinking/provider/claude/apply.go"], "must_contain": ["normalizeClaudeBudget"]},
    {"cat": "E", "query": "kimi executor for moonshot models",
     "expected": ["internal/runtime/executor/kimi_executor.go"], "must_contain": ["Kimi"]},
    {"cat": "E", "query": "antigravity reasoning replay cache not codex",
     "expected": ["internal/cache/antigravity_reasoning_replay_cache.go"], "must_contain": ["antigravity"]},
    # ---------- F SDK / 模块边界 ----------
    {"cat": "F", "query": "auth scheduler strategy round-robin vs fill-first in sdk",
     "expected": ["sdk/cliproxy/auth/scheduler.go"], "must_contain": ["schedulerStrategyFillFirst"]},
    {"cat": "F", "query": "plugin scheduler capability delegate to builtin scheduler",
     "expected": ["internal/pluginhost/scheduler.go", "sdk/pluginapi/types.go"], "must_contain": ["SchedulerBuiltinRoundRobin", "delegate"]},
    {"cat": "F", "query": "access manager api key authentication registry",
     "expected": ["sdk/access/registry.go"], "must_contain": ["RegisterProvider"]},
    {"cat": "F", "query": "cliproxy service builder pattern entry",
     "expected": ["sdk/cliproxy/builder.go"], "must_contain": ["Builder"]},
    {"cat": "F", "query": "executor request response and options types",
     "expected": ["sdk/cliproxy/executor/executor.go", "sdk/cliproxy/executor/types.go"], "must_contain": ["Request", "Options"]},
    # ---------- G 协议 / 契约 ----------
    {"cat": "G", "query": "wsrelay message types stream_start stream_chunk stream_end",
     "expected": ["internal/wsrelay/message.go"], "must_contain": ["MessageTypeStreamChunk"]},
    {"cat": "G", "query": "OpenAI-Beta responses_websockets header value for codex websocket",
     "expected": ["internal/runtime/executor/codex_websockets_executor.go"], "must_contain": ["responses_websockets"]},
    {"cat": "G", "query": "thinking error codes budget out of range level not supported",
     "expected": ["internal/thinking/errors.go", "internal/thinking/validate.go"], "must_contain": ["ErrBudgetOutOfRange"]},
    {"cat": "G", "query": "gemini thinkingConfig thinkingBudget includeThoughts request format",
     "expected": ["internal/thinking/provider/gemini/apply.go"], "must_contain": ["thinkingBudget", "includeThoughts"]},
    {"cat": "G", "query": "statusErr http status code error wrapper in executors",
     "expected": ["internal/runtime/executor/openai_compat_executor.go"], "must_contain": ["statusErr"]},
    # ---------- H 测试与 fixtures ----------
    {"cat": "H", "query": "signature cache TTL and model group tests",
     "expected": ["internal/cache/signature_cache_test.go"], "must_contain": ["TestCacheSignature"]},
    {"cat": "H", "query": "thinking suffix conversion test cases matrix",
     "expected": ["test/thinking_conversion_test.go"], "must_contain": ["thinkingTestCase"]},
    {"cat": "H", "query": "codex executor retry behavior tests",
     "expected": ["internal/runtime/executor/codex_executor_retry_test.go"], "must_contain": ["retry"]},
    {"cat": "H", "query": "config watcher hot reload tests",
     "expected": ["internal/watcher/watcher_test.go", "internal/watcher/diff/config_diff_test.go"], "must_contain": ["reload"]},
    {"cat": "H", "query": "round robin scheduler benchmark test",
     "expected": ["sdk/cliproxy/auth/scheduler_benchmark_test.go", "sdk/cliproxy/auth/scheduler_test.go"], "must_contain": ["Benchmark"]},
    # ---------- I 中文查询 ----------
    {"cat": "I", "query": "配置文件热重载是怎么做防抖的",
     "expected": ["internal/watcher/config_reload.go"], "must_contain": ["configReloadDebounce", "scheduleConfigReload"]},
    {"cat": "I", "query": "模型目录多久从远端刷新一次",
     "expected": ["internal/registry/model_updater.go"], "must_contain": ["modelsRefreshInterval"]},
    {"cat": "I", "query": "思考预算超出模型范围时是报错还是截断",
     "expected": ["internal/thinking/validate.go"], "must_contain": ["clampBudget"]},
    {"cat": "I", "query": "账号配额耗尽后的冷却调度在哪里实现",
     "expected": ["sdk/cliproxy/auth/scheduler.go", "sdk/cliproxy/auth/conductor.go"], "must_contain": ["Cooldown", "cooldown"]},
    {"cat": "I", "query": "websocket 中继会话的心跳间隔是多少",
     "expected": ["internal/wsrelay/session.go"], "must_contain": ["heartbeatInterval"]},
    # ---------- J 细节实现 ----------
    {"cat": "J", "query": "codex websocket idle timeout duration constant",
     "expected": ["internal/runtime/executor/codex_websockets_executor.go"], "must_contain": ["codexResponsesWebsocketIdleTimeout"]},
    {"cat": "J", "query": "level to budget mapping medium 8192 high 24576",
     "expected": ["internal/thinking/convert.go"], "must_contain": ["24576"]},
    {"cat": "J", "query": "wsrelay max inbound message length limit bytes",
     "expected": ["internal/wsrelay/session.go"], "must_contain": ["maxInboundMessageLen"]},
    {"cat": "J", "query": "models fetch remote urls list for catalog refresh",
     "expected": ["internal/registry/model_updater.go"], "must_contain": ["modelsURLs"]},
    {"cat": "J", "query": "gemini empty thinking sentinel skip_thought_signature_validator",
     "expected": ["internal/cache/signature_cache.go"], "must_contain": ["skip_thought_signature_validator"]},
]

CAT_NAMES = {
    "A": "符号定位", "B": "调用链", "C": "架构职责", "D": "配置schema", "E": "同名干扰",
    "F": "SDK边界", "G": "协议契约", "H": "测试fixtures", "I": "中文查询", "J": "细节实现",
}

# ---------- K 类：本地私有业务仓库探针（回归安全网，不随仓库分发） ----------
# 探针定义在 gitignore 的 eval_local_probes.py（PROBE_TARGET + PROBE_GOLDEN）；
# 文件不存在时 K 类自动跳过，其它机器/CI 零依赖。探针写法见该文件 docstring。
try:
    from eval_local_probes import PROBE_TARGET as JAVA_TARGET, PROBE_GOLDEN as JAVA_GOLDEN
except ImportError:
    JAVA_TARGET = None
    JAVA_GOLDEN = []


def _rel(fp: str) -> str:
    try:
        return Path(fp).resolve().relative_to(TARGET.resolve()).as_posix()
    except ValueError:
        return Path(fp).as_posix()


def _score_row(item: dict, files: list[str], top5_code: str, match) -> dict:
    """机械化打分。match(f, exp) 定义文件匹配口径（主评测精确/K 类子串）。

    expected_all：每个期望文件都要进 Top-5 才算 recall，rank 取最差名次。
    """
    must = item.get("must_contain", [])
    exp_all = item.get("expected_all")
    if exp_all:
        ranks = [next((i for i, f in enumerate(files, 1) if match(f, e)), 0) for e in exp_all]
        rank = max(ranks) if all(ranks) else 0
        top1 = 1 if any(match(files[0], e) for e in exp_all) and files else 0
        recall5 = 1 if rank and rank <= 5 else 0
        expected = sorted(exp_all)
    else:
        expected = sorted(item["expected"])
        rank = next((i for i, f in enumerate(files, 1) if any(match(f, e) for e in expected)), 0)
        top1 = 1 if (files and any(match(files[0], e) for e in expected)) else 0
        recall5 = 1 if 0 < rank <= 5 else 0
    usable = 1 if (recall5 or (must and any(m in top5_code for m in must))) else 0
    return {"expected": expected, "rank": rank, "top1": top1,
            "recall5": recall5, "usable": usable}


def _run_java_probes() -> list[dict]:
    """K 类：连本机生产 DB 只读跑探针（expand_graph=True 验全链路不崩）。

    DB 定位规则与 server.WorkspaceManager 一致；sha256(路径|dtype)[:16]。
    不存在（其它机器/未索引）则跳过，不烧配额不阻断主评测。
    """
    if JAVA_TARGET is None:
        print("\n[K 类] 跳过：本地探针集不存在（eval_local_probes.py，不随仓库分发）")
        return []
    # 复合 query 每条多发 1-2 次子查询 rerank，独立调用本函数时同样要节流
    os.environ.setdefault("SCM_RERANK_MIN_INTERVAL", "6.2")
    embedder = create_embedder()
    target = str(JAVA_TARGET.resolve())
    h = hashlib.sha256(f"{target}|{embedder.output_dtype}".encode("utf-8")).hexdigest()[:16]
    db_path = Path.home() / ".semantic-code-mcp" / f"{h}.db"
    if not db_path.exists():
        print(f"\n[K 类] 跳过：Java 探针仓库未索引（{db_path} 不存在）")
        return []
    store = CodeStore(str(db_path), embedder.dim, dtype=embedder.output_dtype)
    retriever = Retriever(store, embedder)
    print(f"\n[K 类] Java 探针回归（{JAVA_TARGET.name}，{len(JAVA_GOLDEN)} 条，expand_graph=开）")
    rows: list[dict] = []
    match = lambda f, e: e in f  # 文件名子串匹配
    for item in JAVA_GOLDEN:
        t0 = time.time()
        results = retriever.search(item["query"], top_k=100, top_n=5, expand_graph=True)
        elapsed = time.time() - t0
        main_results = [r for r in results if not r.get("relation")]
        files = [Path(r["file_path"]).as_posix() for r in main_results]
        top5_code = "\n".join(r.get("code", "") for r in main_results[:5])
        row = _score_row(item, files, top5_code, match)
        row.update({"cat": "K", "query": item["query"],
                    "returned": [f.rsplit("/", 1)[-1] for f in files[:5]],
                    "latency": round(elapsed, 2)})
        rows.append(row)
        mark = f"rank={row['rank']}" if row["rank"] else ("USABLE" if row["usable"] else "MISS")
        print(f"  [K] [{mark:>7}] {item['query'][:60]}")
    store.close()
    n = len(rows)
    print(f"  [K 小计] Top-1 {sum(r['top1'] for r in rows) / n:.0%} | "
          f"Recall@5 {sum(r['recall5'] for r in rows) / n:.0%} | "
          f"可用性 {sum(r['usable'] for r in rows) / n:.0%}")
    for r in rows:
        if not r["usable"]:
            print(f"      MISS: {r['query'][:50]} 期望={r['expected']} 实际={r['returned']}")
    return rows


def main() -> None:
    os.environ["SCM_EMBED_BACKEND"] = BACKEND
    # Cohere 试用 key 限速 10 次/分钟：50 条 query 连跑必然超限，静默降级 RRF
    # 会让分数随机劣化 ±10%。节流到限额内保证评测可复现（代价是评测变慢）。
    os.environ.setdefault("SCM_RERANK_MIN_INTERVAL", "6.2")
    embedder = create_embedder()
    store = CodeStore(DB_PATH, embedder.dim, dtype=embedder.output_dtype)
    indexer = Indexer(str(TARGET), store, embedder)
    expander = None
    if EXPAND:
        expander = QueryExpander()
        if not expander.enabled:
            print("[eval] --expand 需要 SCM_LLM_BASE_URL / SCM_LLM_MODEL 环境变量")
            sys.exit(1)
    # expander=None 时 Retriever 内部按 SCM_QUERY_EXPANSION 开关创建（未设置则禁用）
    retriever = Retriever(store, embedder, expander=expander)

    print(f"[eval] 目标: {TARGET} | {len(GOLDEN)} 条 query | 后端={BACKEND} | 扩展={'开' if EXPAND else '关'}")
    t0 = time.time()
    stats = indexer.sync()
    print(f"[eval] 索引: {time.time() - t0:.1f}s, {stats}\n")

    rows: list[dict] = []
    for item in GOLDEN:
        q, expected = item["query"], set(item["expected"])
        must = item.get("must_contain", [])
        t0 = time.time()
        results = retriever.search(q, top_k=100, top_n=TOP_N, expand_graph=False)
        elapsed = time.time() - t0
        files = [_rel(r["file_path"]) for r in results]
        top5_code = "\n".join(r.get("code", "") for r in results[:5])
        row = _score_row(item, files, top5_code, lambda f, e: f == e)
        row.update({"cat": item["cat"], "query": q, "returned": files[:5],
                    "latency": round(elapsed, 2)})
        rows.append(row)
        mark = f"rank={row['rank']}" if row["rank"] else ("USABLE" if row["usable"] else "MISS")
        print(f"  [{item['cat']}] [{mark:>7}] {q[:60]}")

    # 汇总
    n = len(rows)
    total = {
        "top1": sum(r["top1"] for r in rows) / n,
        "recall5": sum(r["recall5"] for r in rows) / n,
        "usable": sum(r["usable"] for r in rows) / n,
        "mrr": sum(1 / r["rank"] for r in rows if r["rank"]) / n,
        "latency": sum(r["latency"] for r in rows) / n,
    }
    print("\n" + "=" * 64)
    print(f"[总分] Top-1 {total['top1']:.0%} | Recall@5 {total['recall5']:.0%} | "
          f"答案可用性 {total['usable']:.0%} | MRR {total['mrr']:.3f} | 平均 {total['latency']:.2f}s")
    print("\n[类别热力图]")
    print(f"  {'类别':<8}{'Top-1':>7}{'Recall@5':>10}{'可用性':>8}")
    by_cat: dict[str, dict] = {}
    for cat in CAT_NAMES:
        cr = [r for r in rows if r["cat"] == cat]
        if not cr:
            continue
        m = len(cr)
        c = {"top1": sum(r["top1"] for r in cr) / m, "recall5": sum(r["recall5"] for r in cr) / m,
             "usable": sum(r["usable"] for r in cr) / m}
        by_cat[cat] = c
        print(f"  {cat} {CAT_NAMES[cat]:<7}{c['top1']:>6.0%}{c['recall5']:>9.0%}{c['usable']:>8.0%}")

    misses = [r for r in rows if not r["usable"]]
    if misses:
        print("\n[未达标明细]")
        for r in misses:
            print(f"  [{r['cat']}] {r['query'][:50]}")
            print(f"      期望: {r['expected']}")
            print(f"      实际: {r['returned']}")

    store.close()
    # K 类 Java 探针（独立小节，不计入主总分，保持历史分数可比）
    java_rows = [] if "--skip-java" in sys.argv else _run_java_probes()

    if "--json" in sys.argv:
        out = Path(__file__).parent / "eval_result_scm.json"
        out.write_text(json.dumps({"engine": "semantic-code-mcp", "total": total,
                                   "by_category": by_cat, "rows": rows,
                                   "java_probes": java_rows}, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        print(f"\n[eval] JSON 已写入 {out}")


if __name__ == "__main__":
    main()

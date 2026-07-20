"""针对 2026-07-06 修复的回归测试（不需 API key）。

覆盖：
  1. FTS 旧 schema 迁移后从 chunks 回填（BM25 不归零）
  2. gitignore 嵌套模式改写：否定 / 锚定 / 任意层级语义
  3. _should_skip 统一过滤口径（gitignore 对 dirty 路径生效）
  4. WorkspaceManager.get 并发创建同一目录只产生一个实例
"""
import os
import random
import shutil
import sqlite3
import tempfile
import threading
import time
from pathlib import Path

os.environ.setdefault("SCM_WATCH", "0")
os.environ.setdefault("VOYAGE_API_KEY", os.getenv("VOYAGE_API_KEY", "dummy-key-for-offline-test"))

from chunker import chunk_file
from indexer import Indexer
from retriever import Retriever
from store import CodeStore


def test_fts_migration_backfill() -> None:
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "t.db")
    dim = 8
    store = CodeStore(db_path, dim)
    chunks = chunk_file("chunker.py")
    embs = [[random.random() for _ in range(dim)] for _ in chunks]
    store.add_chunks(chunks, embs)
    store.close()

    # 模拟旧 schema：DROP 新表，建旧版 (code, symbol) 空表
    db = sqlite3.connect(db_path)
    db.execute("DROP TABLE fts_chunks")
    db.execute("CREATE VIRTUAL TABLE fts_chunks USING fts5(code, symbol)")
    db.commit()
    db.close()

    # 重新打开：迁移应 DROP 旧表并从 chunks 回填
    store2 = CodeStore(db_path, dim)
    hits = store2.search_fts("chunk file parser", 5)
    store2.close()
    shutil.rmtree(tmpdir, ignore_errors=True)
    assert hits, "FTS 迁移后未回填，BM25 召回归零"
    print(f"[1] FTS 迁移回填 OK（{len(hits)} 条命中）")


def test_scope_pattern() -> None:
    sp = Indexer._scope_pattern
    assert sp("*.log", "") == "*.log"
    assert sp("*.log", "sub") == "sub/**/*.log"
    assert sp("!keep.py", "sub") == "!sub/**/keep.py"
    assert sp("/build", "sub") == "sub/build"
    assert sp("foo/bar.py", "sub") == "sub/foo/bar.py"
    print("[2] _scope_pattern 语义 OK")


def test_should_skip_gitignore() -> None:
    tmpdir = tempfile.mkdtemp()
    root = Path(tmpdir)
    sub = root / "sub"
    sub.mkdir()
    (sub / ".gitignore").write_text("*.py\n!keep.py\n", encoding="utf-8")
    (sub / "a.py").write_text("x = 1\n", encoding="utf-8")
    (sub / "keep.py").write_text("y = 2\n", encoding="utf-8")
    (root / "b.py").write_text("z = 3\n", encoding="utf-8")

    # agent 指导文件被 gitignore 时仍应索引（豁免白名单）
    (root / ".gitignore").write_text("AGENTS.md\n", encoding="utf-8")
    (root / "AGENTS.md").write_text("# arch\n", encoding="utf-8")
    # 无扩展名的豁免文件（扩展名检查之前就要豁免）
    (root / ".windsurfrules").write_text("rules\n", encoding="utf-8")

    idx = Indexer(str(root), None, None)
    assert idx._should_skip(sub / "a.py") is True, "gitignore 命中文件未被跳过"
    assert idx._should_skip(sub / "keep.py") is False, "否定模式 !keep.py 未生效"
    assert idx._should_skip(root / "b.py") is False, "范围外文件被误杀"
    assert idx._should_skip(root / "AGENTS.md") is False, "AGENTS.md 应豁免 gitignore"
    assert idx._should_skip(root / ".windsurfrules") is False, "无扩展名豁免文件应可索引"
    shutil.rmtree(tmpdir, ignore_errors=True)
    print("[3] _should_skip gitignore 口径 OK（含否定模式 + 豁免白名单）")


def test_name_boost_and_tokens() -> None:
    r = Retriever.__new__(Retriever)  # 不走 __init__，纯函数测试
    qt = Retriever._query_tokens("parse ParseSuffix thinking budget")
    assert "parsesuffix" in qt and "parse" in qt and "suffix" in qt
    # 符号完整命中 -> 1.5
    c1 = {"symbol": "ParseSuffix", "file_path": "a/suffix.go"}
    assert r._name_boost(qt, c1) == 1.5
    # 分词命中（suffix + parse）-> 1 + 0.1*n
    c2 = {"symbol": "parseLevel", "file_path": "a/suffix.go"}
    b2 = r._name_boost(qt, c2)
    assert 1.0 < b2 <= 1.3, b2
    # 无命中 -> 1.0
    c3 = {"symbol": "foo", "file_path": "a/bar.go"}
    assert r._name_boost(qt, c3) == 1.0
    print("[5] _name_boost / _query_tokens OK")


def test_per_file_diversity() -> None:
    """同一文件最多保留 2 块（用无 rerank/无 embedding 的纯截断逻辑验证）。"""
    candidates = [
        {"id": i, "file_path": "same.go" if i < 5 else f"f{i}.go", "symbol": f"s{i}", "score": 10 - i}
        for i in range(8)
    ]
    # 直接复用 search 第 6 步的截断逻辑（内联重写，保持与实现一致）
    results, per_file = [], {}
    for c in sorted(candidates, key=lambda x: x["score"], reverse=True):
        fp = c["file_path"]
        if per_file.get(fp, 0) >= 2:
            continue
        per_file[fp] = per_file.get(fp, 0) + 1
        results.append(c)
        if len(results) >= 5:
            break
    same_count = sum(1 for r in results if r["file_path"] == "same.go")
    assert same_count == 2, f"同文件应限 2 块，实际 {same_count}"
    assert len(results) == 5
    print("[6] 文件多样性截断 OK")


def test_format_results_grouping() -> None:
    from server import _format_results

    results = [
        {"file_path": "a.go", "symbol": "Foo", "start_line": 10, "end_line": 11,
         "code": "func Foo() {\n}", "language": "go", "score": 0.9},
        {"file_path": "b.go", "symbol": "Bar", "start_line": 1, "end_line": 2,
         "code": "func Bar() {\n}", "language": "go", "score": 0.8},
        {"file_path": "a.go", "symbol": "Baz", "start_line": 50, "end_line": 51,
         "code": "func Baz() {\n}", "language": "go", "score": 0.7},
        {"file_path": "c.go", "symbol": "Rel", "start_line": 5, "end_line": 6,
         "code": "func Rel() {\n}", "language": "go", "relation": "caller"},
    ]
    out = _format_results(results)
    # a.go 的两块合并为一个分组，非连续用 ... 分隔
    assert out.count("## 1. a.go") == 1 and "Foo, Baz" in out
    assert "## 2. b.go" in out
    a_block = out.split("## 2.")[0]
    assert "\n...\n" in a_block, "非连续块应用 ... 分隔"
    # 行号前缀
    assert "    10\tfunc Foo() {" in out
    assert "    50\tfunc Baz() {" in out
    # 关联块单独展示
    assert "[关联" in out and "c.go" in out

    # 重叠块裁剪：行窗口 overlap 不应重复渲染
    overlap = [
        {"file_path": "d.md", "symbol": "lines_1", "start_line": 1, "end_line": 3,
         "code": "L1\nL2\nL3", "language": "markdown", "score": 0.9},
        {"file_path": "d.md", "symbol": "lines_3", "start_line": 3, "end_line": 5,
         "code": "L3\nL4\nL5", "language": "markdown", "score": 0.8},
    ]
    out2 = _format_results(overlap)
    assert out2.count("L3") == 1, "重叠行应只渲染一次"
    assert "     4\tL4" in out2 and "     5\tL5" in out2
    print("[7] 按文件聚合格式化 OK（含重叠裁剪）")


def test_caller_intent_and_barrel() -> None:
    import retriever as rmod

    # 意图正则：中英文都能命中
    assert rmod._CALLER_INTENT_RE.search("buildToolPlan 被哪些文件调用了？")
    assert rmod._CALLER_INTENT_RE.search("who calls reload_config")
    assert rmod._CALLER_INTENT_RE.search("callers of ParseSuffix")
    assert not rmod._CALLER_INTENT_RE.search("config hot reload watcher")
    # 测试意图：词边界不误伤 latest/greatest
    assert rmod._TEST_INTENT_RE.search("where are the tests for reload")
    assert rmod._TEST_INTENT_RE.search("配置热重载的测试在哪")
    assert not rmod._TEST_INTENT_RE.search("latest config reload behavior")
    assert not rmod._TEST_INTENT_RE.search("greatest common divisor helper")
    # 加权 RRF：原查询权重加倍后仍能压住单路扩展噪声
    fused = Retriever._rrf([[1, 2], [1, 3], [9, 9, 9]], weights=[2.0, 2.0, 1.0])
    assert fused[0][0] == 1, fused

    # 查询扩展器：JSON 解析容忍 fence，未配置时禁用
    from expander import QueryExpander, create_expander

    parsed = QueryExpander._parse('```json\n{"variants": ["a", "b"], "hypothetical_code": "func X() {}"}\n```')
    assert parsed == {"variants": ["a", "b"], "hyde": "func X() {}"}, parsed
    assert QueryExpander._parse("no json here") is None
    assert QueryExpander._parse('{"variants": []}') is None
    for k in ("SCM_QUERY_EXPANSION", "SCM_LLM_BASE_URL", "SCM_LLM_MODEL"):
        os.environ.pop(k, None)
    assert create_expander() is None, "未开启时应禁用扩展"
    os.environ["SCM_QUERY_EXPANSION"] = "on"
    assert create_expander() is None, "缺端点配置时应禁用扩展"
    os.environ.pop("SCM_QUERY_EXPANSION", None)
    # 标识符提取：只要 camelCase / snake_case
    syms = Retriever._intent_symbols("谁调用了 buildToolPlan 和 reload_config 还有 watcher")
    assert syms == ["buildToolPlan", "reload_config"], syms
    # barrel 判定
    assert Retriever._is_barrel_file("src/tools/index.ts")
    assert Retriever._is_barrel_file("pkg\\__init__.py")
    assert not Retriever._is_barrel_file("src/tools/planner.ts")
    # 非代码乘数：文档意图 > 配置意图 > 默认降权；代码文件恒为 1
    from retriever import _NON_CODE_PENALTY

    assert Retriever._noncode_mult("go", False, True) == 1.0
    assert Retriever._noncode_mult("yaml", False, False) == _NON_CODE_PENALTY
    assert Retriever._noncode_mult("yaml", False, True) == 1.4
    assert Retriever._noncode_mult("yaml", True, True) == 1.5
    assert Retriever._noncode_mult("markdown", False, True) == _NON_CODE_PENALTY  # md 不吃配置意图

    # 端到端：用真实 store 验证 callers_of 路由（chunker.py 里的函数互相调用）
    tmpdir = tempfile.mkdtemp()
    store = CodeStore(os.path.join(tmpdir, "t.db"), 8)
    chunks = chunk_file("chunker.py")
    store.add_chunks(chunks, [[random.random() for _ in range(8)] for _ in chunks])
    callers = store.callers_of("_get_parser")
    assert callers, "callers_of 应命中 chunk_file -> _get_parser 调用边"
    assert any(c["symbol"] == "chunk_file" for c in callers)
    store.close()
    shutil.rmtree(tmpdir, ignore_errors=True)

    # Java 成员调用：接收者字段名进 edges，PascalCase 类名意图查询可命中调用方
    tmpdir = tempfile.mkdtemp()
    java_fp = os.path.join(tmpdir, "Svc.java")
    with open(java_fp, "w", encoding="utf-8") as f:
        f.write(
            "public class Svc {\n"
            "    private final PriceRuleEvaluator priceRuleEvaluator = new PriceRuleEvaluator();\n"
            "    public void run(String v) {\n"
            "        java.util.List<CustomItem> items = new java.util.ArrayList<CustomItem>();\n"
            "        EvaluateResult r = priceRuleEvaluator.evaluate(1, 2, v, null, null);\n"
            "        process(r);\n"
            "    }\n"
            "}\n"
        )
    jchunks = chunk_file(java_fp)
    jcalls = [c for ch in jchunks for c in ch.calls]
    assert "evaluate" in jcalls, jcalls
    assert "priceRuleEvaluator" in jcalls, f"接收者名应进 edges: {jcalls}"
    assert "PriceRuleEvaluator" in jcalls, f"new 类名应进 edges: {jcalls}"
    assert "ArrayList" in jcalls, f"泛型构造应记主类型名: {jcalls}"
    assert "CustomItem" not in jcalls, f"泛型实参不应进 edges: {jcalls}"
    store = CodeStore(os.path.join(tmpdir, "t.db"), 8)
    store.add_chunks(jchunks, [[random.random() for _ in range(8)] for _ in jchunks])
    r = Retriever.__new__(Retriever)
    r.store = store
    hits = r._caller_intent_hits("谁调用了 PriceRuleEvaluator")
    assert hits and any("Svc.java" in h["file_path"] for h in hits), \
        f"PascalCase 意图应通过 lowerCamel 接收者命中调用方: {[h.get('symbol') for h in hits]}"
    store.close()
    shutil.rmtree(tmpdir, ignore_errors=True)
    print("[8] 调用链意图 + barrel 降权 OK（含 Java 接收者边）")


def test_workspace_get_race() -> None:
    from embedder import Embedder
    from workspace import WorkspaceManager

    tmpdir = tempfile.mkdtemp()
    db_dir = tempfile.mkdtemp()
    os.environ["SCM_DB_DIR"] = db_dir
    (Path(tmpdir) / "a.py").write_text("def f():\n    return 1\n", encoding="utf-8")

    manager = WorkspaceManager(Embedder())
    results: list = []
    barrier = threading.Barrier(8)

    def worker() -> None:
        barrier.wait()
        results.append(manager.get(tmpdir))

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    ids = {id(w) for w in results}
    assert len(ids) == 1, f"并发 get 产生了 {len(ids)} 个不同实例"
    assert manager.active_count == 1
    manager.destroy_all()
    shutil.rmtree(tmpdir, ignore_errors=True)
    shutil.rmtree(db_dir, ignore_errors=True)
    print("[4] 并发 get 单实例 OK（8 线程）")


def test_expander_prompt_and_parse() -> None:
    """expander 的 prompt 构建（含字面 JSON 大括号）与响应解析。"""
    import expander as emod
    from expander import QueryExpander

    e = QueryExpander(base_url="http://fake", api_key="k", model="m")
    assert e.enabled
    captured: dict = {}

    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content":
                '前缀噪声 {"variants": ["a", "b"], "hypothetical_code": "code"} 后缀'}}]}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["payload"] = json
        return FakeResp()

    old_post = emod.requests.post
    emod.requests.post = fake_post
    try:
        r = e.expand("配置热重载防抖")
        assert r == {"variants": ["a", "b"], "hyde": "code"}, r
        content = captured["payload"]["messages"][0]["content"]
        assert "配置热重载防抖" in content, "query 应被替换进 prompt"
        assert '{"variants"' in content, "prompt 里的字面 JSON 示例应完整保留"
    finally:
        emod.requests.post = old_post
    print("[10] expander prompt 构建 + 解析 OK（含字面大括号回归）")


def test_concurrent_embed_pipeline() -> None:
    """并发 embedding 流水线：向量-块顺序对齐 + 并发确实快于串行。"""
    import hashlib as _h

    import indexer as imod
    from chunker import chunk_file as _cf
    from indexer import Indexer, _embed_text

    def _vec(t: str) -> list[float]:
        d = _h.sha256(t.encode("utf-8")).digest()
        return [b / 255.0 for b in d[:8]]

    class SlowEmb:
        dim = 8
        output_dtype = "float"
        default_concurrency = 3

        def __init__(self):
            self.calls = 0

        def embed_documents(self, texts):
            self.calls += 1
            time.sleep(0.2)
            return [_vec(t) for t in texts]

        def embed_query(self, text):
            return _vec(text)

    tmpdir = tempfile.mkdtemp()
    for i in range(8):
        (Path(tmpdir) / f"m{i}.py").write_text(
            f"def func_{i}(a, b):\n    x = a + b\n    y = x * {i}\n    return y\n",
            encoding="utf-8",
        )
    old_batch = imod.BATCH_CHUNKS
    imod.BATCH_CHUNKS = 2  # 8 文件 -> 4 批
    emb = SlowEmb()
    store = CodeStore(os.path.join(tmpdir, "t.db"), 8)
    idx = Indexer(tmpdir, store, emb)
    try:
        t0 = time.time()
        stats = idx.sync()
        elapsed = time.time() - t0
        assert stats["changed"] == 8, stats
        assert emb.calls == 4, f"应分 4 批提交: {emb.calls}"
        # 顺序对齐：任取一块，重建其 embed 文本的向量应精确检索回该块
        target = _cf(str(Path(tmpdir) / "m5.py"))[0]
        hits = store.search_vector(_vec(_embed_text(target, idx.root)), 3)
        top_id = hits[0][0]
        ch = store.get_chunks([top_id])[top_id]
        assert ch["file_path"].endswith("m5.py") and ch["symbol"] == "func_5", ch
        # 提速：4 批 × 0.2s 串行 ≥ 0.8s；3 路并发应约 0.4s
        assert elapsed < 0.7, f"并发流水线未生效: {elapsed:.2f}s"
    finally:
        imod.BATCH_CHUNKS = old_batch
        store.close()
        shutil.rmtree(tmpdir, ignore_errors=True)
    print(f"[11] 并发 embedding 流水线 OK（顺序对齐 + 提速 {elapsed:.2f}s < 串行 0.8s）")


def test_background_incremental_sync() -> None:
    """watcher 变更 → debounce → 后台主动增量同步（不依赖查询触发）。"""
    import random as _r

    import watcher as wmod
    from workspace import WorkspaceManager

    class StubEmb:
        dim = 8
        output_dtype = "float"

        def embed_documents(self, texts):
            return [[_r.random() for _ in range(8)] for _ in texts]

        def embed_query(self, text):
            return [_r.random() for _ in range(8)]

    old_debounce = wmod.DEBOUNCE_SECONDS
    wmod.DEBOUNCE_SECONDS = 0.3  # 加速测试
    tmpdir = tempfile.mkdtemp()
    db_dir = tempfile.mkdtemp()
    os.environ["SCM_DB_DIR"] = db_dir
    os.environ["SCM_WATCH"] = "1"  # 本测试需要 watcher（模块默认关）
    try:
        (Path(tmpdir) / "a.py").write_text("def f():\n    return 1\n", encoding="utf-8")
        manager = WorkspaceManager(StubEmb())
        ws = manager.get(tmpdir)
        manager.sync_workspace(ws)  # 首次全量
        assert ws.store.stats().get("files") == 1

        (Path(tmpdir) / "b.py").write_text("def g():\n    return 2\n", encoding="utf-8")
        deadline = time.time() + 10
        files = 0
        while time.time() < deadline:
            try:
                files = ws.store.stats().get("files", 0)
            except Exception:
                files = 0
            if files == 2:
                break
            time.sleep(0.2)
        assert files == 2, f"后台增量同步未生效（files={files}，期望 2）"
        manager.destroy_all()
    finally:
        wmod.DEBOUNCE_SECONDS = old_debounce
        os.environ["SCM_WATCH"] = "0"
        shutil.rmtree(tmpdir, ignore_errors=True)
        shutil.rmtree(db_dir, ignore_errors=True)
    print("[9] 后台主动增量同步 OK（watcher → debounce → sync）")


if __name__ == "__main__":
    test_fts_migration_backfill()
    test_scope_pattern()
    test_should_skip_gitignore()
    test_name_boost_and_tokens()
    test_per_file_diversity()
    test_format_results_grouping()
    test_caller_intent_and_barrel()
    test_workspace_get_race()
    test_expander_prompt_and_parse()
    test_concurrent_embed_pipeline()
    test_background_incremental_sync()
    print("\nOK 回归测试全部通过")

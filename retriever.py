"""检索层：向量 + BM25 hybrid 召回，RRF 融合，Cohere rerank 重排，意图感知打分。

流程：query -> 向量召回 + BM25 召回 -> RRF 融合 -> (可选)rerank
      -> 调用链意图直查 edges 置顶 -> 测试/barrel/非代码降权（测试与文档意图反转为 boost）
      -> 符号/文件名 boost -> 文件多样性截断 -> call graph 图扩展
未配置 COHERE_API_KEY 时自动跳过 rerank，直接使用 RRF 融合分数。
"""
from __future__ import annotations

import fnmatch
import os
import re
import time

from embedder import Embedder
from expander import create_expander
from store import CodeStore, _is_boilerplate_symbol

try:
    import cohere
except Exception:  # 依赖缺失不阻断
    cohere = None


# RRF 融合常数，经验值 60
_RRF_K = 60
# 送入 rerank 的单文档最大字符数
_RERANK_DOC_MAX_CHARS = 4000
# rerank 429 限速重试：次数与退避基数（Cohere 试用 key 10 次/分钟，
# 静默降级会让排序质量随机劣化，退避重试把调用速率压回限额内）
_RERANK_MAX_RETRIES = 2
_RERANK_BACKOFF_S = 6.0


def _rerank_min_interval() -> float:
    """rerank 调用最小间隔（秒），0 = 不节流。懒读 env：评测等密集场景
    用它把速率压到试用 key 限额内，生产零星查询不受影响。"""
    try:
        return float(os.getenv("SCM_RERANK_MIN_INTERVAL", "0") or 0)
    except ValueError:
        return 0.0


# 测试文件 score 惩罚系数（实现优先于测试）
_TEST_FILE_PENALTY = 0.5
# 测试文件名模式
_TEST_PATTERNS = ("_test.", "test_", "Test.", ".test.", ".spec.", "_spec.")
# 查询中完整命中符号名时的 boost（如查询里直接写了函数名）
_SYMBOL_EXACT_BOOST = 1.5
# 符号/文件名分词与查询 token 每命中一个的 boost 增量（最多计 3 个）
_NAME_TOKEN_BOOST = 0.1
# top_n 结果中同一文件最多保留的块数（文件多样性）
_MAX_PER_FILE = 2
# barrel / 纯导出入口文件降权（实现文件优先于 re-export 入口）
_BARREL_PENALTY = 0.85
# 非代码文件（文档/配置）降权：分数接近时实现代码优先，不影响文档类查询的大分差命中
# （doc/config 意图查询走反转 boost 不受此值影响；0.7 治动作型查询里 properties 混尾部）
_NON_CODE_PENALTY = 0.7
_NON_CODE_LANGS = {"markdown", "yaml", "json", "toml", "properties", "text"}
_BARREL_NAMES = {"index.ts", "index.js", "index.tsx", "index.jsx", "index.mjs", "__init__.py", "mod.rs"}
# 调用链意图："谁调用/被哪些...调用/谁写入/在哪被使用/who calls/callers of/writes to"
_CALLER_INTENT_RE = re.compile(
    r"(谁调用|被哪些|哪些[^\s]{0,8}调用|调用了?它|被写入|写入|谁在用|被使用|被引用"
    r"|callers?\s+of|who\s+calls|call\s*sites?|writes?\s+to|who\s+writes|where\s+.{0,20}\bused)", re.I
)
# 测试意图：查询本身在找测试/基准 -> 反转测试降权为 boost
_TEST_INTENT_RE = re.compile(r"(\btests?\b|\bspec\b|benchmark|单测|测试|用例|基准)", re.I)
_TEST_INTENT_BOOST = 1.2
# 文档意图：查询在找架构/规范/文档 -> 反转非代码降权为 boost
_DOC_INTENT_RE = re.compile(
    r"(architect|overview|convention|guideline|readme|documentation|\bdocs?\b|规范|架构|职责|约定|文档|说明)", re.I
)
_DOC_INTENT_BOOST = 1.5
# 配置意图：查询在找配置项/schema -> 配置类文件（yaml/toml/properties/json）反转为 boost
_CONFIG_INTENT_RE = re.compile(r"(config|configuration|settings|\byaml\b|\btoml\b|配置)", re.I)
_CONFIG_LANGS = {"yaml", "toml", "properties", "json"}
_CONFIG_INTENT_BOOST = 1.4
# 纯声明块（Java interface / @FeignClient 契约）降权：实现优先于声明；
# 短而密集命中查询词的接口/DTO 在三路召回里天然压过大方法体，需要纠偏
_DECL_PENALTY = 0.75
# 实体/DTO 块（注解驱动、无控制流）降权
_ENTITY_PENALTY = 0.85
# 查询显式找接口定义/契约时不惩罚声明块
_DECL_INTENT_RE = re.compile(r"(接口定义|契约|declaration|signature|\binterface\b|feign)", re.I)
# 实现/过程意图（"怎么生成"、"how is X done"）：答案在实现代码里，
# DTO/接口即使语义相似也回答不了"怎么做"，加重降权
_IMPL_INTENT_RE = re.compile(r"(怎么|如何|怎样|流程|逻辑|实现|\bhow\b|implement)", re.I)
_IMPL_INTENT_EXTRA = 0.75
# 同名接口/实现配对：结果集同时出现 X 与 XImpl 时，实现继承接口分数排到前面
_IMPL_PAIR_FACTOR = 1.01
# 超短块（≤此行数）降权：单行方法声明这类低信息命中信息密度撑不起
# 独立席位（尾部占位问题）；温和系数，分数接近时才让位给实质内容
_TINY_CHUNK_LINES = 2
_TINY_CHUNK_PENALTY = 0.85
_JAVA_INTERFACE_RE = re.compile(r"^\s*(?:public\s+)?(?:@\w+(?:\([^)]*\))?\s+)*interface\s+\w", re.M)
_CONTROL_FLOW_RE = re.compile(r"\b(if|for|while|switch|return)\b")
# 块内声明方法名提取（caller 方向图扩展用），每块最多取这么多个
_MAX_DECL_NAMES_PER_CHUNK = 12
_DECL_NAME_RES = {
    "java": [
        # 带修饰符的方法声明
        re.compile(r"^\s*(?:public|protected|private)\s+(?:(?:static|final|synchronized|abstract|default|native)\s+)*[\w<>\[\],.\s?]+?\s+(\w+)\s*\(", re.M),
        # 接口方法（无修饰符，; 结尾）
        re.compile(r"^\s*[\w<>\[\],.?]+\s+(\w+)\s*\([^;{}]*\)\s*;", re.M),
    ],
    "python": [re.compile(r"^\s*def\s+(\w+)\s*\(", re.M)],
    "go": [re.compile(r"^func\s+(?:\([^)]*\)\s*)?(\w+)\s*\(", re.M)],
    "typescript": [re.compile(r"^\s*(?:public\s+|private\s+|protected\s+|export\s+|async\s+)*(\w+)\s*\([^)]*\)\s*[:{]", re.M)],
}
_DECL_NAME_STOP = {"if", "for", "while", "switch", "catch", "return", "new", "super", "else", "throw"}
# CJK 字符（汉字 + 假名 + 谚文）：命中时追加 trigram 召回路
_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")
# 复合意图查询拆分：CJK 连接词 / 加号（"生成+清理"、"生成并清理"）。
# 长词优先（以及/并且/同时 先于 和/与/并/及）；"合并""并发"类词素误拆由
# 两段最短长度 + 子查询低权重兼容（主查询权重不变，RRF 容错）。
# 英文 and 不拆：英文复合名词短语（error wrapping and logging）拆开反而稀释信号
_COMPOUND_SPLIT_RE = re.compile(
    r"[+＋]|(?<=[\u4e00-\u9fff])(?:以及|并且|同时|和|与|并|及)(?=[\u4e00-\u9fff])"
)
# 拆分后子查询最短长度（去空白）；2 覆盖中文双字动词（"生成"/"清理"）
_COMPOUND_MIN_LEN = 2
# 子查询召回路权重（低于主查询，防拆分噪声反客为主）
_COMPOUND_SUB_WEIGHT = 0.7
# 子查询保底：rerank 用原始完整 query 重打分会把子意图召回的文档抹平
# （"生成...并清理"的语义重心压向主意图）。保底触发条件：某子意图召回的
# 前 N 名文件与最终结果零交集（完全无代表）；保底块必须来自已过
# RRF 候选门槛的 candidates（全库 top-1 直插会把"清理"这种宽泛词的噪声带进来）
_COMPOUND_PROBE_TOP = 10
# 子查询 rerank 分数混合：探针块用整句 rerank 分天然吃亏（"清理 100 天"
# 在整句里只占小半权重），单独用子查询文本再 rerank 一次，
# 最终分取 max(整句分, 子查询分 × 此系数)
_COMPOUND_SUB_RERANK_MIX = 0.9
# 子查询分参与混合/保底的最低门槛：实测"清理"这种宽泛短词的字面命中
# （clearCache/clear 工具方法）全部 ≤0.14，真业务清理实现 ≥0.22，
# 0.2 是天然分界线；低于门槛的子查询信号不参与任何提分
_COMPOUND_SUB_MIN_SCORE = 0.2


class Retriever:
    """混合检索器。"""

    def __init__(
        self,
        store: CodeStore,
        embedder: Embedder,
        rerank_api_key: str | None = None,
        rerank_model: str = "rerank-v3.5",
        expander=None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.rerank_model = rerank_model
        key = rerank_api_key or os.getenv("COHERE_API_KEY")
        self.cohere_client = cohere.Client(key) if (key and cohere) else None
        # 节流时间戳无锁：SCM_RERANK_MIN_INTERVAL 仅评测等单线程密集场景启用，
        # 生产默认 0（不节流），并发竞态最坏多发一次请求、由 429 重试兜底
        self._last_rerank_ts = 0.0
        # 查询扩展（HyDE + 多查询变体）：默认按 SCM_QUERY_EXPANSION 开关创建
        self.expander = expander if expander is not None else create_expander()

    @staticmethod
    def _is_test_file(file_path: str) -> bool:
        """判断是否是测试文件。"""
        name = file_path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        return any(p in name for p in _TEST_PATTERNS)

    @staticmethod
    def _is_barrel_file(file_path: str) -> bool:
        """判断是否是 barrel / 纯入口文件。"""
        name = file_path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        return name in _BARREL_NAMES

    @staticmethod
    def _noncode_mult(language: str, doc_intent: bool, config_intent: bool) -> float:
        """非代码文件的意图感知乘数：文档意图 > 配置意图 > 默认降权。"""
        if language not in _NON_CODE_LANGS:
            return 1.0
        if doc_intent:
            return _DOC_INTENT_BOOST
        if config_intent and language in _CONFIG_LANGS:
            return _CONFIG_INTENT_BOOST
        return _NON_CODE_PENALTY

    @staticmethod
    def _intent_symbols(query: str) -> list[str]:
        """从调用链意图查询中提取代码标识符（camelCase / snake_case）。"""
        out: list[str] = []
        for ident in re.findall(r"[A-Za-z_][A-Za-z0-9_]{3,}", query):
            if "_" in ident or re.search(r"[a-z][A-Z]", ident):
                out.append(ident)
        return out

    def _caller_intent_hits(self, query: str) -> list[dict]:
        """调用链意图：命中意图模式时直查 edges 表，结构化结果置顶。"""
        if not _CALLER_INTENT_RE.search(query):
            return []
        idents: list[str] = []
        for ident in self._intent_symbols(query):
            idents.append(ident)
            # 表名 → 实体类名（tb_order_item → OrderItem）：
            # edges 记录了构造类型名（new OrderItem()），能直接命中写入方
            low = ident.lower()
            if low.startswith(("tb_", "t_")):
                parts = [p for p in ident.split("_")[1:] if p]
                if parts:
                    idents.append("".join(p[:1].upper() + p[1:] for p in parts))
        hits: list[dict] = []
        seen: set[int] = set()
        for ident in dict.fromkeys(idents):
            # PascalCase 类名同时按实例字段命名约定查一次
            # （Java/Spring: OrderValidator -> orderValidator，edges 记录的是接收者名）
            names = [ident]
            if ident[:1].isupper():
                names.append(ident[:1].lower() + ident[1:])
            for name in names:
                for ch in self.store.callers_of(name):
                    if ch["id"] in seen:
                        continue
                    seen.add(ch["id"])
                    ch = dict(ch)
                    ch["score"] = 9.99  # 结构化命中，排在所有语义结果之前
                    hits.append(ch)
        return hits

    @staticmethod
    def _ident_parts(name: str) -> list[str]:
        """把标识符拆成小写分词（camelCase / snake_case 兼容）。"""
        s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name or "")
        return [p for p in re.findall(r"[a-z0-9]+", s.lower()) if len(p) >= 2]

    @classmethod
    def _query_tokens(cls, query: str) -> set[str]:
        """提取查询中的标识符 token（含整词与分词）。"""
        tokens: set[str] = set()
        for w in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", query):
            tokens.add(w.lower())
            tokens.update(cls._ident_parts(w))
        return tokens

    @staticmethod
    def _is_declaration_chunk(chunk: dict) -> bool:
        """纯声明块：Java interface / @FeignClient 契约（TS interface 同理）。"""
        lang = chunk.get("language") or ""
        code = chunk.get("code") or ""
        if lang == "java":
            return "@FeignClient" in code or bool(_JAVA_INTERFACE_RE.search(code))
        if lang in ("typescript", "tsx"):
            return bool(re.match(r"\s*(?:export\s+)?interface\s", code))
        return False

    @staticmethod
    def _is_entity_chunk(chunk: dict) -> bool:
        """实体/DTO 块：注解驱动的纯数据载体，无控制流。"""
        lang = chunk.get("language") or ""
        code = chunk.get("code") or ""
        if lang != "java":
            return False
        if "@TableName" in code or ("@Data" in code and "class " in code):
            return not _CONTROL_FLOW_RE.search(code)
        return False

    @classmethod
    def _declared_names(cls, code: str, language: str) -> list[str]:
        """提取块内声明的方法名（类/接口级 chunk 的 symbol 只有类名，
        caller 方向图扩展需要方法名才能匹配 edges 的调用边）。"""
        regs = _DECL_NAME_RES.get("typescript" if language == "tsx" else language)
        if not regs:
            return []
        names: list[str] = []
        seen: set[str] = set()
        for rg in regs:
            for m in rg.finditer(code):
                n = m.group(1)
                if len(n) < 4 or n in seen or n in _DECL_NAME_STOP:
                    continue
                # Java/TS 方法名小写开头，滤掉构造器/类名误匹配
                if language in ("java", "typescript", "tsx") and n[:1].isupper():
                    continue
                # accessor/样板方法不参与 caller 反查（"谁调用了 setRemark"是纯噪音）
                if _is_boilerplate_symbol(n):
                    continue
                seen.add(n)
                names.append(n)
                if len(names) >= _MAX_DECL_NAMES_PER_CHUNK:
                    return names
        return names

    def _name_boost(self, qtokens: set[str], chunk: dict) -> float:
        """名称信号 boost：符号完整命中强提升；符号/文件名分词命中按个数提升。"""
        if not qtokens:
            return 1.0
        symbol = chunk.get("symbol") or ""
        sym_parts = self._ident_parts(symbol)
        flat = "".join(sym_parts)
        if flat and flat in qtokens:
            return _SYMBOL_EXACT_BOOST
        fp = chunk.get("file_path") or ""
        stem = fp.rsplit("/", 1)[-1].rsplit("\\", 1)[-1].split(".")[0]
        matched = len((set(sym_parts) | set(self._ident_parts(stem))) & qtokens)
        return 1.0 + _NAME_TOKEN_BOOST * min(matched, 3)

    def search(
        self,
        query: str,
        top_k: int = 100,
        top_n: int = 10,
        expand_graph: bool = True,
        graph_limit: int = 8,
        max_per_file: int = _MAX_PER_FILE,
        path_filter: str | None = None,
    ) -> list[dict]:
        """检索并返回 top_n 个代码块（dict，含 score）。

        排序管线：向量+BM25(+trigram/子查询) 召回 → RRF → rerank + 子查询分数混合
        （可选）→ 调用链意图置顶 → 意图感知降权/boost + 名称 boost → 接口-实现
        配对 → 文件多样性截断（每文件最多 max_per_file 块）→ 子查询保底。

        path_filter：可选文件路径过滤，含通配符按 glob 匹配，否则按子串匹配。

        expand_graph=True 时，在主结果基础上沿 call graph 扩展 1 跳，
        把调用者/被调用者作为关联结果附加（带 relation 字段）。
        """
        # 1. 向量召回 + 2. BM25 召回（原查询）
        q_emb = self.embedder.embed_query(query)
        rank_lists = [
            [cid for cid, _ in self.store.search_vector(q_emb, top_k)],
            [cid for cid, _ in self.store.search_fts(query, top_k)],
        ]
        weights = [1.0, 1.0]
        # 2.5 查询扩展（可选）：变体走 query 模式，HyDE 假想代码走 document 模式；
        #     原查询两路权重加倍，防扩展噪声稀释原始信号
        expansion = self.expander.expand(query) if self.expander else None
        if expansion:
            weights = [2.0, 2.0]
            for v in expansion.get("variants", []):
                v_emb = self.embedder.embed_query(v)
                rank_lists.append([cid for cid, _ in self.store.search_vector(v_emb, top_k)])
                weights.append(1.0)
                rank_lists.append([cid for cid, _ in self.store.search_fts(v, top_k)])
                weights.append(1.0)
            hyde = expansion.get("hyde")
            if hyde:
                h_emb = self.embedder.embed_documents([hyde])[0]
                rank_lists.append([cid for cid, _ in self.store.search_vector(h_emb, top_k)])
                weights.append(1.0)
        # 2.8 CJK 查询追加 trigram 召回路（unicode61 把连续汉字并成单 token，主 FTS 路对中文失效；
        #     纯英文查询不走 trigram，避免子串匹配稀释标识符精确信号）
        if _CJK_RE.search(query):
            tri_list = [cid for cid, _ in self.store.search_fts_trigram(query, top_k)]
            if tri_list:
                rank_lists.append(tri_list)
                weights.append(weights[0])
        # 2.9 复合意图拆分（"生成+清理"）：单次向量召回对双动作查询天然偏科，
        #     各子查询独立召回后低权重并入 RRF，两个意图都能拿到席位
        subs = self._compound_subqueries(query)
        sub_probes: list[list[int]] = []  # 每个子查询自己的前 N 名（保底探针/分数混合用）
        for sub in subs:
            s_emb = self.embedder.embed_query(sub)
            sub_lists = [[cid for cid, _ in self.store.search_vector(s_emb, top_k)]]
            sub_fts = (
                self.store.search_fts_trigram(sub, top_k)
                if _CJK_RE.search(sub) else self.store.search_fts(sub, top_k)
            )
            if sub_fts:
                sub_lists.append([cid for cid, _ in sub_fts])
            for sl in sub_lists:
                rank_lists.append(sl)
                weights.append(_COMPOUND_SUB_WEIGHT)
            mini = self._rrf(sub_lists)
            # 召回为空也 append 空探针，保持与 subs 一一对应（4.2 按下标 zip 配对）
            sub_probes.append([cid for cid, _ in mini[:_COMPOUND_PROBE_TOP]])
        # 3. RRF 融合（仅用排名）
        fused = self._rrf(rank_lists, weights=weights)
        if not fused:
            return []
        cand_ids = [cid for cid, _ in fused[:top_k]]
        chunk_map = self.store.get_chunks(cand_ids)
        candidates: list[dict] = []
        for cid, rrf_score in fused[:top_k]:
            ch = chunk_map.get(cid)
            if ch:
                ch = dict(ch)
                ch["score"] = rrf_score
                candidates.append(ch)
        # 4. rerank（可选，失败降级为 RRF 分数）；多取候选给后续多样性截断留余量
        if self.cohere_client and candidates:
            pre_rerank = {c.get("id"): c for c in candidates}
            try:
                rerank_n = min(len(candidates), max(top_n * 3, 20))
                candidates = self._rerank(query, candidates, rerank_n)
            except Exception:
                pass
            # 4.1 整句 rerank 截断会把整句分低的探针块丢掉，补回（0 分起步，
            #     等 4.2 子查询分数拉起），否则分数混合和保底都看不见它
            if sub_probes:
                out_ids = {c.get("id") for c in candidates}
                for cid in dict.fromkeys(cid for p in sub_probes for cid in p):
                    if cid in pre_rerank and cid not in out_ids:
                        ch = pre_rerank[cid]
                        ch["score"] = 0.0
                        candidates.append(ch)
            # 4.2 子查询分数混合：探针块单独用子查询文本 rerank，取 max 混入。
            #     整句 rerank 把子意图文档压到 0.2x（语义重心在主意图），
            #     子查询文本打分才能反映它对那半意图的真实相关性
            if subs and sub_probes:
                by_id = {c.get("id"): c for c in candidates}
                for sub, probe in zip(subs, sub_probes):
                    sub_cands = [by_id[cid] for cid in probe if cid in by_id]
                    if not sub_cands:
                        continue
                    try:
                        scored = self._rerank(sub, sub_cands, len(sub_cands))
                    except Exception:
                        continue
                    for s in scored:
                        c = by_id.get(s.get("id"))
                        if c is None:
                            continue
                        c["_sub_score"] = max(c.get("_sub_score", 0.0), s["score"])
                        if s["score"] >= _COMPOUND_SUB_MIN_SCORE:
                            c["score"] = max(
                                c["score"], s["score"] * _COMPOUND_SUB_RERANK_MIX
                            )
        # 4.5 调用链意图：结构化命中注入高分（并入统一打分管线，测试调用方会被后续降权拆开）
        intent_hits = self._caller_intent_hits(query)
        if intent_hits:
            by_id = {c.get("id"): c for c in candidates}
            for h in intent_hits:
                if h["id"] in by_id:
                    by_id[h["id"]]["score"] = h["score"]
                else:
                    candidates.append(h)
        # 5. 统一后处理：意图感知的降权/boost + 符号/文件名 boost
        qtokens = self._query_tokens(query)
        test_intent = bool(_TEST_INTENT_RE.search(query))
        doc_intent = bool(_DOC_INTENT_RE.search(query))
        config_intent = bool(_CONFIG_INTENT_RE.search(query))
        decl_intent = bool(_DECL_INTENT_RE.search(query))
        impl_intent = not decl_intent and bool(_IMPL_INTENT_RE.search(query))
        for c in candidates:
            mult = 1.0
            fp = c.get("file_path", "")
            if self._is_test_file(fp):
                mult *= _TEST_INTENT_BOOST if test_intent else _TEST_FILE_PENALTY
            if self._is_barrel_file(fp):
                mult *= _BARREL_PENALTY
            mult *= self._noncode_mult(c.get("language", ""), doc_intent, config_intent)
            is_decl = self._is_declaration_chunk(c)
            is_entity = not is_decl and self._is_entity_chunk(c)
            if is_entity:
                c["entity"] = True  # 展示层据此收紧行预算（DTO 字段列表不值得全量吐）
            if not decl_intent:
                if is_decl:
                    # 记住降权系数：接口-实现配对要用降权前的分数做 base，
                    # 否则声明降权把 base 一起拉低，配对提拔失效
                    c["_decl_mult"] = _DECL_PENALTY * (_IMPL_INTENT_EXTRA if impl_intent else 1.0)
                    mult *= c["_decl_mult"]
                elif is_entity:
                    mult *= _ENTITY_PENALTY * (_IMPL_INTENT_EXTRA if impl_intent else 1.0)
            if c.get("end_line", 0) - c.get("start_line", 0) + 1 <= _TINY_CHUNK_LINES:
                mult *= _TINY_CHUNK_PENALTY
            mult *= self._name_boost(qtokens, c)
            c["score"] *= mult
        # 5.5 接口/实现配对（文件级）：接口块 symbol 是类名、实现块 symbol 是方法名，
        # 按 symbol 配不上；改按文件 stem 配对：XImpl.java 的最高分块继承 X.java
        # （接口文件）的更高分，实现方法排到接口声明前面
        best_by_stem: dict[str, float] = {}
        for c in candidates:
            undecl = c["score"] / c.get("_decl_mult", 1.0)
            best_by_stem[self._file_stem(c)] = max(
                best_by_stem.get(self._file_stem(c), 0.0), undecl
            )
        impl_top: dict[str, dict] = {}  # stem -> 该实现文件最高分块
        for c in candidates:
            stem = self._file_stem(c)
            if stem.endswith("Impl") and len(stem) > 4:
                if stem not in impl_top or c["score"] > impl_top[stem]["score"]:
                    impl_top[stem] = c
        for stem, c in impl_top.items():
            base = best_by_stem.get(stem[:-4])
            if base and base > c["score"]:
                c["score"] = base * _IMPL_PAIR_FACTOR
        candidates.sort(key=lambda x: x["score"], reverse=True)
        # 6. 文件多样性：同一文件最多 max_per_file 块，避免 top_n 被单文件刷屏
        results: list[dict] = []
        per_file: dict[str, int] = {}
        for c in candidates:
            fp = c.get("file_path", "")
            if path_filter and not self._path_match(fp, path_filter):
                continue
            if max_per_file > 0 and per_file.get(fp, 0) >= max_per_file:
                continue
            per_file[fp] = per_file.get(fp, 0) + 1
            results.append(c)
            if len(results) >= top_n:
                break
        # 6.5 子查询保底：某子意图召回的前 N 名文件与结果零交集（完全无代表）时，
        #     从 candidates 里挑子查询 rerank 分最高且达门槛的块替换最低分名额；
        #     无达标块则放弃（"清理"这种宽泛词的 clearCache 字面命中不配强插）；
        #     已有代表则不动（避免顶掉更相关结果）
        if sub_probes and results:
            by_id = {c.get("id"): c for c in candidates}
            probe_ids = [cid for p in sub_probes for cid in p]
            probe_chunks = self.store.get_chunks(probe_ids)
            rescued_ids: set[int] = set()
            for probe in sub_probes:
                result_files = {r.get("file_path", "") for r in results}
                probe_files = {
                    probe_chunks[cid]["file_path"] for cid in probe if cid in probe_chunks
                }
                if result_files & probe_files:
                    continue  # 该子意图已有代表
                best = max(
                    (
                        by_id[cid] for cid in probe
                        if cid in by_id and cid not in rescued_ids
                        and by_id[cid].get("_sub_score", 0.0) >= _COMPOUND_SUB_MIN_SCORE
                        and not (path_filter and not self._path_match(
                            by_id[cid].get("file_path", ""), path_filter))
                    ),
                    key=lambda c: c.get("_sub_score", 0.0),
                    default=None,
                )
                if best is None:
                    continue
                if len(results) >= top_n:
                    low = min(
                        (i for i, r in enumerate(results) if r.get("id") not in rescued_ids),
                        key=lambda i: results[i].get("score", 0.0),
                        default=None,
                    )
                    if low is None:
                        continue
                    results[low] = best
                else:
                    results.append(best)
                rescued_ids.add(best["id"])
            if rescued_ids:
                # 保底替换后按分数重排，避免混合分高于中间名次时展示乱序
                results.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        # 清理打分管线内部字段，不泄漏到返回值（entity 是展示层公开约定，保留）
        for r in results:
            r.pop("_decl_mult", None)
            r.pop("_sub_score", None)
        # 7. call graph 扩展（在主结果基础上连带召回调用关系）
        if expand_graph and results:
            fused_scores = dict(fused)
            results = self._with_graph(results, graph_limit, fused_scores)
        return results

    def _with_graph(
        self, results: list[dict], limit: int, fused_scores: dict[int, float] | None = None
    ) -> list[dict]:
        """在主结果后附加 call graph 关联块（去重，带 relation 标记）。

        关联块按主召回融合分排序（在召回列表里出现过 = 与查询相关），
        同分 caller 优先于 callee（调用点对溯源/影响面分析更有价值）。
        类/接口级块额外提取内部声明的方法名参与 caller 反查。"""
        origin_ids = [r["id"] for r in results if "id" in r]
        symbols = [r.get("symbol", "") for r in results]
        extra_names: list[str] = []
        for r in results:
            extra_names.extend(self._declared_names(r.get("code") or "", r.get("language") or ""))
        related = self.store.expand_graph(
            origin_ids, symbols, limit=limit * 3, extra_callee_names=extra_names
        )
        fused_scores = fused_scores or {}
        related.sort(
            key=lambda r: (
                fused_scores.get(r.get("id"), 0.0),
                1 if r.get("relation") == "caller" else 0,
            ),
            reverse=True,
        )
        existing = set(origin_ids)
        added = 0
        for r in related:
            if r["id"] in existing:
                continue
            r["score"] = 0.0  # 图扩展块无 relevance 分，靠 relation 标记
            if self._is_entity_chunk(r):
                r["entity"] = True
            results.append(r)
            existing.add(r["id"])
            added += 1
            if added >= limit:
                break
        return results

    @staticmethod
    def _compound_subqueries(query: str) -> list[str]:
        """复合意图查询拆分：恰好拆成两段且每段足够长才生效（保守策略，防误拆）。"""
        parts = [p.strip() for p in _COMPOUND_SPLIT_RE.split(query)]
        parts = [p for p in parts if len(p) >= _COMPOUND_MIN_LEN]
        return parts if len(parts) == 2 else []

    @staticmethod
    def _file_stem(chunk: dict) -> str:
        """文件名去扩展名（OrderServiceImpl.java -> OrderServiceImpl）。"""
        fp = chunk.get("file_path") or ""
        return fp.rsplit("/", 1)[-1].rsplit("\\", 1)[-1].split(".")[0]

    @staticmethod
    def _path_match(file_path: str, pattern: str) -> bool:
        """路径过滤：含通配符按 glob（对全路径，自动补 **/ 前缀），否则子串包含。"""
        fp = file_path.replace("\\", "/")
        pat = pattern.replace("\\", "/")
        if any(ch in pat for ch in "*?["):
            return fnmatch.fnmatch(fp, pat) or fnmatch.fnmatch(fp, f"*{pat}" if not pat.startswith("*") else pat)
        return pat.lower() in fp.lower()

    @staticmethod
    def _rrf(
        rank_lists: list[list[int]],
        k: int = _RRF_K,
        weights: list[float] | None = None,
    ) -> list[tuple[int, float]]:
        """Reciprocal Rank Fusion：融合多个召回列表的排名（可选每路权重）。"""
        scores: dict[int, float] = {}
        for i, ranks in enumerate(rank_lists):
            w = weights[i] if weights else 1.0
            for rank, doc_id in enumerate(ranks):
                scores[doc_id] = scores.get(doc_id, 0.0) + w / (k + rank + 1)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def _rerank(self, query: str, candidates: list[dict], top_n: int) -> list[dict]:
        docs = [self._doc_text(c) for c in candidates]
        min_interval = _rerank_min_interval()
        resp = None
        for attempt in range(_RERANK_MAX_RETRIES + 1):
            if min_interval > 0:
                wait = self._last_rerank_ts + min_interval - time.time()
                if wait > 0:
                    time.sleep(wait)
            try:
                self._last_rerank_ts = time.time()
                resp = self.cohere_client.rerank(
                    model=self.rerank_model,
                    query=query,
                    documents=docs,
                    top_n=min(top_n, len(docs)),
                )
                break
            except Exception as e:
                # 仅对限速重试（退避后速率回到限额内）；其它异常继续抛给外层降级 RRF
                if "TooManyRequests" not in type(e).__name__ or attempt >= _RERANK_MAX_RETRIES:
                    raise
                time.sleep(_RERANK_BACKOFF_S * (attempt + 1))
        out: list[dict] = []
        for r in resp.results:
            ch = dict(candidates[r.index])
            ch["score"] = float(r.relevance_score)
            out.append(ch)
        return out

    @staticmethod
    def _doc_text(chunk: dict) -> str:
        """构造送入 rerank 的文档文本（路径 + 符号 + 代码）。"""
        head = f"{chunk.get('file_path', '')} :: {chunk.get('symbol', '')}"
        return f"{head}\n{chunk.get('code', '')}"[:_RERANK_DOC_MAX_CHARS]

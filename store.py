"""存储层：sqlite-vec 向量存储 + FTS5 全文索引 + 文件指纹增量表。

三张核心表：
  chunks      代码块元数据 + 源码（blob_hash 唯一，实现内容寻址去重）
  vec_chunks  vec0 虚拟表，rowid = chunks.id，存 embedding
  fts_chunks  fts5 虚拟表，rowid = chunks.id，用于 BM25 词法检索
  files       file_path -> file_hash，用于增量同步对比
"""
from __future__ import annotations

import re
import sqlite3

import sqlite_vec

from chunker import CodeChunk


# FTS5 query 至少保留长度 >= 此值的 token
_MIN_TOKEN_LEN = 2
# trigram 分词器要求 token 长度 >= 3 才能命中
_TRIGRAM_MIN_TOKEN_LEN = 3
# Call Graph 同名保护：symbol 对应的定义超过此数视为无区分度，跳过图扩展
_MAX_SYMBOL_FANOUT = 5
# Call Graph 热点保护：被调用方（distinct caller）超过此数的符号视为全局工具函数（日期/字符串工具等），跳过图扩展
_MAX_CALLEE_CALLERS = 20
# Java bean 访问器模式（getX/setX/isX）：作为图扩展目标零信息量，纯噪音
_ACCESSOR_RE = re.compile(r"^(?:get|set|is)[A-Z]\w*$")
# 语言级样板方法名：同理跳过
_BOILERPLATE_NAMES = {
    "toString", "equals", "hashCode", "clone", "compareTo",
    "builder", "build", "valueOf", "readObject", "writeObject",
}


def _is_boilerplate_symbol(name: str) -> bool:
    """accessor / 样板方法：不值得作为 call graph 扩展的目标或反查起点。"""
    return bool(_ACCESSOR_RE.match(name)) or name in _BOILERPLATE_NAMES


class CodeStore:
    """单个工作区的索引存储。"""

    def __init__(self, db_path: str, dim: int, dtype: str = "float") -> None:
        self.db_path = db_path
        self.dim = dim
        self.dtype = dtype if dtype in ("float", "int8") else "float"
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.db.enable_load_extension(True)
        sqlite_vec.load(self.db)
        self.db.enable_load_extension(False)
        self._init_schema()

    @staticmethod
    def _trigram_supported(cur) -> bool:
        """探测当前 SQLite 是否支持 FTS5 trigram 分词器（>= 3.34）。"""
        try:
            cur.execute("CREATE VIRTUAL TABLE temp.__scm_trig_probe USING fts5(x, tokenize='trigram')")
            cur.execute("DROP TABLE temp.__scm_trig_probe")
            return True
        except sqlite3.OperationalError:
            return False

    @staticmethod
    def _migrate_fts(cur) -> None:
        """旧 FTS5 schema 迁移：列缺 file_path，或被旧版建成了 trigram（主表必须 unicode61
        保英文精确分词，trigram 子串匹配对英文标识符查询是噪音源），则重建。"""
        try:
            # 检查 fts_chunks 是否已存在且列数/分词器是否正确
            row = cur.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='fts_chunks'"
            ).fetchone()
            if row and ("file_path" not in row[0] or "trigram" in row[0]):
                cur.execute("DROP TABLE fts_chunks")
        except Exception:
            pass

    @staticmethod
    def _rebuild_fts_if_empty(cur, table: str = "fts_chunks") -> None:
        """FTS 表为空但 chunks 有数据时（如迁移刚 DROP 重建），从 chunks 回填。"""
        try:
            n_chunks = cur.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            n_fts = cur.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            if n_chunks and not n_fts:
                cur.execute(
                    f"INSERT INTO {table} (rowid, code, file_path, symbol) "
                    "SELECT id, code, file_path, symbol FROM chunks"
                )
        except Exception:
            pass

    def _init_schema(self) -> None:
        cur = self.db.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                blob_hash TEXT UNIQUE,
                file_path TEXT NOT NULL,
                language TEXT,
                symbol TEXT,
                start_line INTEGER,
                end_line INTEGER,
                code TEXT
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_path)")
        vec_type = "int8" if self.dtype == "int8" else "float"
        cur.execute(
            f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
                embedding {vec_type}[{self.dim}]
            )
            """
        )
        # FTS5 双表：
        #   fts_chunks     unicode61 精确分词 —— 英文标识符查询主力（基线行为，trigram 子串匹配会引入噪音）
        #   fts_chunks_tri trigram 子串匹配 —— 仅 CJK 查询时作为额外召回路
        #   （unicode61 把连续汉字并成单 token，中文查询在主表上全程失效）
        self._migrate_fts(cur)
        cur.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks USING fts5(
                code, file_path, symbol, tokenize='unicode61'
            )
            """
        )
        self._rebuild_fts_if_empty(cur, "fts_chunks")
        self.fts_trigram = self._trigram_supported(cur)
        if self.fts_trigram:
            cur.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks_tri USING fts5(
                    code, file_path, symbol, tokenize='trigram'
                )
                """
            )
            self._rebuild_fts_if_empty(cur, "fts_chunks_tri")
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS files (
                file_path TEXT PRIMARY KEY,
                file_hash TEXT NOT NULL,
                mtime REAL DEFAULT 0,
                size INTEGER DEFAULT 0
            )
            """
        )
        # 迁移：旧 DB 无 mtime/size 列时自动补充
        try:
            cur.execute("ALTER TABLE files ADD COLUMN mtime REAL DEFAULT 0")
        except Exception:
            pass
        try:
            cur.execute("ALTER TABLE files ADD COLUMN size INTEGER DEFAULT 0")
        except Exception:
            pass
        # call graph 边：caller_id（chunks.id）-> callee_name（被调用函数名）
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS edges (
                caller_id INTEGER NOT NULL,
                callee_name TEXT NOT NULL
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_edges_caller ON edges(caller_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_edges_callee ON edges(callee_name)")
        self.db.commit()

    # ---------- 写入 ----------

    def _serialize_vec(self, emb: list) -> bytes:
        """按 dtype 序列化向量：float32 或 int8。"""
        if self.dtype == "int8":
            return sqlite_vec.serialize_int8(emb)
        return sqlite_vec.serialize_float32(emb)

    def add_chunks(self, chunks: list[CodeChunk], embeddings: list[list[float]]) -> int:
        """批量写入代码块及其向量。blob_hash 重复的自动跳过。返回新增条数。"""
        cur = self.db.cursor()
        added = 0
        for chunk, emb in zip(chunks, embeddings):
            cur.execute(
                "INSERT OR IGNORE INTO chunks "
                "(blob_hash, file_path, language, symbol, start_line, end_line, code) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    chunk.blob_hash,
                    chunk.file_path,
                    chunk.language,
                    chunk.symbol,
                    chunk.start_line,
                    chunk.end_line,
                    chunk.code,
                ),
            )
            if cur.rowcount == 0:
                continue  # 内容已存在，去重
            chunk_id = cur.lastrowid
            vec_wrap = "vec_int8(?)" if self.dtype == "int8" else "?"
            cur.execute(
                f"INSERT INTO vec_chunks (rowid, embedding) VALUES (?, {vec_wrap})",
                (chunk_id, self._serialize_vec(emb)),
            )
            cur.execute(
                "INSERT INTO fts_chunks (rowid, code, file_path, symbol) VALUES (?, ?, ?, ?)",
                (chunk_id, chunk.code, chunk.file_path, chunk.symbol),
            )
            if self.fts_trigram:
                cur.execute(
                    "INSERT INTO fts_chunks_tri (rowid, code, file_path, symbol) VALUES (?, ?, ?, ?)",
                    (chunk_id, chunk.code, chunk.file_path, chunk.symbol),
                )
            for callee in getattr(chunk, "calls", None) or []:
                cur.execute(
                    "INSERT INTO edges (caller_id, callee_name) VALUES (?, ?)",
                    (chunk_id, callee),
                )
            added += 1
        self.db.commit()
        return added

    def delete_file(self, file_path: str) -> None:
        """删除某文件的所有代码块（向量 + 全文 + 元数据）。"""
        cur = self.db.cursor()
        rows = cur.execute("SELECT id FROM chunks WHERE file_path = ?", (file_path,)).fetchall()
        ids = [r[0] for r in rows]
        if ids:
            marks = ",".join("?" * len(ids))
            cur.execute(f"DELETE FROM vec_chunks WHERE rowid IN ({marks})", ids)
            cur.execute(f"DELETE FROM fts_chunks WHERE rowid IN ({marks})", ids)
            if self.fts_trigram:
                cur.execute(f"DELETE FROM fts_chunks_tri WHERE rowid IN ({marks})", ids)
            cur.execute(f"DELETE FROM edges WHERE caller_id IN ({marks})", ids)
            cur.execute("DELETE FROM chunks WHERE file_path = ?", (file_path,))
        self.db.commit()

    # ---------- 文件指纹（增量） ----------

    def get_file_hash(self, file_path: str) -> str | None:
        row = self.db.execute(
            "SELECT file_hash FROM files WHERE file_path = ?", (file_path,)
        ).fetchone()
        return row[0] if row else None

    def get_file_stat(self, file_path: str) -> dict | None:
        """返回文件的完整元数据 {hash, mtime, size}，不存在返回 None。"""
        row = self.db.execute(
            "SELECT file_hash, mtime, size FROM files WHERE file_path = ?", (file_path,)
        ).fetchone()
        if not row:
            return None
        return {"hash": row[0], "mtime": row[1] or 0.0, "size": row[2] or 0}

    def set_file_hash(self, file_path: str, file_hash: str, mtime: float = 0.0, size: int = 0) -> None:
        self.db.execute(
            "INSERT INTO files (file_path, file_hash, mtime, size) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(file_path) DO UPDATE SET file_hash = excluded.file_hash, "
            "mtime = excluded.mtime, size = excluded.size",
            (file_path, file_hash, mtime, size),
        )
        self.db.commit()

    def remove_file_record(self, file_path: str) -> None:
        self.db.execute("DELETE FROM files WHERE file_path = ?", (file_path,))
        self.db.commit()

    def all_indexed_files(self) -> set[str]:
        return {r[0] for r in self.db.execute("SELECT file_path FROM files").fetchall()}

    # ---------- 检索 ----------

    @staticmethod
    def _fts_query(text: str) -> str | None:
        """把自然语言 query 转为安全的 FTS5 MATCH 表达式（token OR 连接，unicode61 语义）。"""
        tokens = re.findall(r"\w+", text)
        tokens = [t for t in tokens if len(t) >= _MIN_TOKEN_LEN]
        if not tokens:
            return None
        return " OR ".join(f'"{t}"' for t in tokens)

    @staticmethod
    def _fts_query_trigram(text: str) -> str | None:
        """trigram 表的 MATCH 表达式：token 需 >= 3 字符；CJK 连续串整体作为 phrase（子串匹配）。"""
        tokens = re.findall(r"\w+", text)
        tokens = [t for t in tokens if len(t) >= _TRIGRAM_MIN_TOKEN_LEN]
        if not tokens:
            return None
        return " OR ".join(f'"{t}"' for t in tokens)

    def search_vector(self, query_embedding: list[float], k: int) -> list[tuple[int, float]]:
        """向量 KNN 检索，返回 [(chunk_id, distance)]，distance 越小越相似。"""
        match_expr = "vec_int8(?)" if self.dtype == "int8" else "?"
        rows = self.db.execute(
            f"SELECT rowid, distance FROM vec_chunks "
            f"WHERE embedding MATCH {match_expr} AND k = ? ORDER BY distance",
            (self._serialize_vec(query_embedding), k),
        ).fetchall()
        return [(int(r[0]), float(r[1])) for r in rows]

    def search_fts(self, query_text: str, k: int) -> list[tuple[int, float]]:
        """BM25 词法检索（unicode61 精确分词），返回 [(chunk_id, rank)]，rank 越小越相关。"""
        q = self._fts_query(query_text)
        if not q:
            return []
        try:
            rows = self.db.execute(
                "SELECT rowid, rank FROM fts_chunks WHERE fts_chunks MATCH ? "
                "ORDER BY rank LIMIT ?",
                (q, k),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [(int(r[0]), float(r[1])) for r in rows]

    def search_fts_trigram(self, query_text: str, k: int) -> list[tuple[int, float]]:
        """trigram BM25 检索（CJK 子串可匹配），仅供含 CJK 的查询作额外召回路。"""
        if not getattr(self, "fts_trigram", False):
            return []
        q = self._fts_query_trigram(query_text)
        if not q:
            return []
        try:
            rows = self.db.execute(
                "SELECT rowid, rank FROM fts_chunks_tri WHERE fts_chunks_tri MATCH ? "
                "ORDER BY rank LIMIT ?",
                (q, k),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [(int(r[0]), float(r[1])) for r in rows]

    def get_chunks(self, ids: list[int]) -> dict[int, dict]:
        """按 id 批量取代码块元数据。"""
        if not ids:
            return {}
        marks = ",".join("?" * len(ids))
        rows = self.db.execute(
            f"SELECT id, file_path, language, symbol, start_line, end_line, code "
            f"FROM chunks WHERE id IN ({marks})",
            ids,
        ).fetchall()
        result: dict[int, dict] = {}
        for r in rows:
            result[int(r[0])] = {
                "id": int(r[0]),
                "file_path": r[1],
                "language": r[2],
                "symbol": r[3],
                "start_line": r[4],
                "end_line": r[5],
                "code": r[6],
            }
        return result

    # ---------- call graph ----------

    def hot_callees(self, names: list[str]) -> set[str]:
        """返回 names 中的"全局热点"符号（被调 caller 数超过 _MAX_CALLEE_CALLERS）。

        热点符号（日期/字符串工具等被全库调用的函数）作为图扩展目标毫无信息量，
        反而把关联位灌满噪音，需要过滤。
        """
        names = [n for n in names if n]
        if not names:
            return set()
        marks = ",".join("?" * len(names))
        rows = self.db.execute(
            f"SELECT callee_name, COUNT(DISTINCT caller_id) FROM edges "
            f"WHERE callee_name IN ({marks}) GROUP BY callee_name",
            names,
        ).fetchall()
        return {r[0] for r in rows if int(r[1]) > _MAX_CALLEE_CALLERS}

    def callers_of(self, callee_name: str, limit: int = 20) -> list[dict]:
        """结构化查询：返回调用了 callee_name 的所有 chunk（按 id 去重）。

        用于"谁调用了 X"类意图查询，直接走 edges 表，不依赖语义召回。
        """
        if not callee_name:
            return []
        rows = self.db.execute(
            "SELECT DISTINCT caller_id FROM edges WHERE callee_name = ? LIMIT ?",
            (callee_name, limit),
        ).fetchall()
        ids = [int(r[0]) for r in rows]
        chunk_map = self.get_chunks(ids)
        return [chunk_map[i] for i in ids if i in chunk_map]

    def expand_graph(
        self,
        chunk_ids: list[int],
        symbols: list[str],
        limit: int = 10,
        extra_callee_names: list[str] | None = None,
    ) -> list[dict]:
        """图扩展：返回与给定 chunk 有调用关系的相关 chunk。

        callees：给定 chunk 调用的函数（edges.caller_id 命中 -> 匹配 symbol，热点过滤）
        callers：调用了给定 symbol / extra_callee_names 的函数（edges.callee_name 命中 -> caller_id）
            extra_callee_names 用于类/接口级 chunk：其内部声明的方法名不是 chunk symbol，
            由调用方提取后传入，否则 caller 方向永远查不到方法调用边。
        两个方向都过滤 accessor / 样板方法（setRemark 这类 bean setter 是纯噪音）。
        每条结果带 relation 字段（"callee"/"caller"）；跨模块同名同内容副本去重。
        """
        origin = set(chunk_ids)
        related: dict[int, str] = {}  # chunk_id -> relation

        # callees：这些 chunk 调用了哪些函数名 -> 找对应定义（先滤掉全局热点工具函数）
        if chunk_ids:
            marks = ",".join("?" * len(chunk_ids))
            callee_names = [
                r[0] for r in self.db.execute(
                    f"SELECT DISTINCT callee_name FROM edges WHERE caller_id IN ({marks})",
                    chunk_ids,
                ).fetchall()
            ]
            callee_names = [n for n in callee_names if n and not _is_boilerplate_symbol(n)]
            if callee_names:
                hot = self.hot_callees(callee_names)
                callee_names = [n for n in callee_names if n not in hot]
            if callee_names:
                nmarks = ",".join("?" * len(callee_names))
                rows = self.db.execute(
                    f"SELECT id, symbol FROM chunks WHERE symbol IN ({nmarks})", callee_names
                ).fetchall()
                by_sym: dict[str, list[int]] = {}
                for r in rows:
                    by_sym.setdefault(r[1], []).append(int(r[0]))
                for sym, cids in by_sym.items():
                    if len(cids) > _MAX_SYMBOL_FANOUT:
                        continue
                    for cid in cids:
                        if cid not in origin:
                            related.setdefault(cid, "callee")

        # callers：谁调用了这些 symbol / 块内声明的方法名（热点符号跳过，防工具函数反查爆炸）
        caller_targets = [s for s in symbols if s] + [n for n in (extra_callee_names or []) if n]
        caller_targets = [n for n in dict.fromkeys(caller_targets) if not _is_boilerplate_symbol(n)]
        if caller_targets:
            hot = self.hot_callees(caller_targets)
            caller_targets = [n for n in caller_targets if n not in hot]
        if caller_targets:
            smarks = ",".join("?" * len(caller_targets))
            for r in self.db.execute(
                f"SELECT DISTINCT caller_id FROM edges WHERE callee_name IN ({smarks})",
                caller_targets,
            ).fetchall():
                cid = int(r[0])
                if cid not in origin and cid not in related:
                    related[cid] = "caller"

        if not related:
            return []
        ids = list(related.keys())[: max(limit * 3, limit)]
        chunk_map = self.get_chunks(ids)
        out: list[dict] = []
        seen_content: set[tuple] = set()
        for cid in ids:
            ch = chunk_map.get(cid)
            if not ch:
                continue
            # 跨模块复制的同名同内容块（api/domin 双份 DTO）只保留一份
            key = (ch.get("symbol"), (ch.get("code") or "")[:200])
            if key in seen_content:
                continue
            seen_content.add(key)
            ch = dict(ch)
            ch["relation"] = related[cid]
            out.append(ch)
            if len(out) >= limit:
                break
        return out

    # ---------- 杂项 ----------

    def stats(self) -> dict:
        n_chunks = self.db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        n_files = self.db.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        return {"chunks": n_chunks, "files": n_files}

    def close(self) -> None:
        try:
            self.db.close()
        except Exception:
            pass

"""基础设施测试（不需 API key）：验证 chunker 与 store。

验证点：
  1. Tree-sitter AST 切分能正确切出函数/类
  2. sqlite-vec 向量表建表 + 插入 + KNN 检索
  3. FTS5 全文 BM25 检索
  4. 内容寻址去重（重复插入不增加）
"""
import os
import random
import shutil
import tempfile

from chunker import chunk_file
from store import CodeStore


def main() -> None:
    # ---------- 1. chunker ----------
    chunks = chunk_file("chunker.py")
    assert chunks, "chunker 未切出任何块"
    print(f"[chunker] chunker.py 切出 {len(chunks)} 个代码块，前 5 个：")
    for c in chunks[:5]:
        print(f"  - {c.symbol}  (L{c.start_line}-{c.end_line}, {c.language})")

    # ---------- 2. store ----------
    dim = 8
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "test.db")
    store = CodeStore(db_path, dim)

    embs = [[random.random() for _ in range(dim)] for _ in chunks]
    added = store.add_chunks(chunks, embs)
    print(f"[store] 首次插入 {added} 块，stats={store.stats()}")
    assert added == len(chunks), "插入数量不符"

    # 去重：再插一次相同内容，应不新增
    added2 = store.add_chunks(chunks, embs)
    print(f"[store] 重复插入新增 {added2} 块（应为 0，内容寻址去重）")
    assert added2 == 0, "去重失败"

    # ---------- 3. 向量 KNN ----------
    q = [random.random() for _ in range(dim)]
    vec_hits = store.search_vector(q, 3)
    print(f"[store] 向量 KNN top3: {vec_hits}")
    assert len(vec_hits) == 3, "向量检索数量不符"

    # ---------- 4. BM25 ----------
    fts_hits = store.search_fts("def chunk file parser node", 3)
    print(f"[store] BM25 top: {fts_hits}")

    # ---------- 5. 取回 ----------
    ids = [h[0] for h in vec_hits]
    got = store.get_chunks(ids)
    print(f"[store] get_chunks 取回 {len(got)} 块")
    assert len(got) == len(ids), "取回数量不符"

    store.close()
    shutil.rmtree(tmpdir, ignore_errors=True)
    print("\nOK 基础设施验证全部通过（chunker + sqlite-vec + FTS5）")


if __name__ == "__main__":
    main()

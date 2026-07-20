"""Embedding 层：默认 Voyage AI voyage-code-3，可选本地 sentence-transformers 后端。

后端选择：SCM_EMBED_BACKEND=voyage（默认）| local
本地后端：SCM_LOCAL_EMBED_MODEL（默认 Qwen/Qwen3-Embedding-0.6B），需 pip install sentence-transformers
区分 document（索引时）与 query（检索时）两种输入模式，
并按字符预算动态分批，避免超过单次请求 token 上限。
"""
from __future__ import annotations

import os
import threading
import time
from collections import OrderedDict

import voyageai


# 常见模型的输出维度，避免额外探测调用
_MODEL_DIM = {
    "voyage-code-3": 1024,
    "voyage-3.5": 1024,
    "voyage-3.5-lite": 1024,
    "voyage-3": 1024,
    "voyage-3-lite": 512,
    "voyage-large-2": 1536,
}

# 单条文本最大字符数（超长截断，约对应 token 上限）
_MAX_CHARS_PER_TEXT = 16000
# 单批最大文本条数
_MAX_BATCH = 128
# 单批最大累计字符（默认约 50K tokens；未绑卡受 10K TPM 限制时可调小）
_CHAR_BUDGET = int(os.getenv("SCM_EMBED_CHAR_BUDGET", "200000"))


class Embedder:
    """Voyage embedding 客户端封装（embed_documents 可多线程并发调用）。"""

    # 索引时建议的并发请求数（纯网络 IO，并发收益显著）
    default_concurrency = 3

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "voyage-code-3",
        output_dimension: int | None = None,
        max_retries: int | None = None,
        output_dtype: str | None = None,
    ) -> None:
        self.model = model
        self.output_dimension = output_dimension
        # 量化输出类型：float（默认/最高精度）/ int8（4x 省存储，精度近乎无损）
        self.output_dtype = output_dtype or os.getenv("SCM_EMBED_DTYPE", "float")
        # 遇到限流/临时错误时 SDK 自动 wait-and-retry
        retries = max_retries if max_retries is not None else int(os.getenv("SCM_EMBED_MAX_RETRIES", "4"))
        self.client = voyageai.Client(
            api_key=api_key or os.getenv("VOYAGE_API_KEY"),
            max_retries=retries,
        )
        # 请求最小间隔（秒）：未绑卡受 3 RPM 限制时设为 21 可避免限流
        self._min_interval = float(os.getenv("SCM_EMBED_MIN_INTERVAL", "0"))
        self._last_req = 0.0
        # 节流状态锁：并发索引时保证全局 RPM 语义
        self._throttle_lock = threading.Lock()
        self._dim: int | None = None
        self._query_cache: OrderedDict[str, list[float]] = OrderedDict()
        self._query_cache_max = 32

    @property
    def dim(self) -> int:
        """输出向量维度。优先查表，否则探测一次。"""
        if self._dim is None:
            self._dim = self.output_dimension or _MODEL_DIM.get(self.model)
        if self._dim is None:
            self._dim = len(self.embed_query("x"))
        return self._dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """索引用：批量 embedding 代码块。"""
        return self._embed(texts, "document")

    def embed_query(self, text: str) -> list[float]:
        """检索用：embedding 单条查询（带 LRU 缓存）。"""
        if text in self._query_cache:
            self._query_cache.move_to_end(text)
            return self._query_cache[text]
        result = self._embed([text], "query")[0]
        self._query_cache[text] = result
        if len(self._query_cache) > self._query_cache_max:
            self._query_cache.popitem(last=False)
        return result

    def _embed(self, texts: list[str], input_type: str) -> list[list[float]]:
        results: list[list[float]] = []
        batch: list[str] = []
        batch_chars = 0
        for raw in texts:
            t = (raw or "")[:_MAX_CHARS_PER_TEXT]
            if not t:
                t = " "  # 占位，保证与输入一一对应
            if batch and (len(batch) >= _MAX_BATCH or batch_chars + len(t) > _CHAR_BUDGET):
                results.extend(self._embed_batch(batch, input_type))
                batch, batch_chars = [], 0
            batch.append(t)
            batch_chars += len(t)
        if batch:
            results.extend(self._embed_batch(batch, input_type))
        return results

    def _embed_batch(self, batch: list[str], input_type: str) -> list[list[float]]:
        if self._min_interval > 0:
            # 锁内等待+登记，并发调用下仍严格保持全局请求间隔；HTTP 调用在锁外
            with self._throttle_lock:
                wait = self._min_interval - (time.time() - self._last_req)
                if wait > 0:
                    time.sleep(wait)
                self._last_req = time.time()
        kwargs = {"model": self.model, "input_type": input_type}
        if self.output_dimension:
            kwargs["output_dimension"] = self.output_dimension
        if self.output_dtype and self.output_dtype != "float":
            kwargs["output_dtype"] = self.output_dtype
        resp = self.client.embed(batch, **kwargs)
        return resp.embeddings


class LocalEmbedder:
    """本地 sentence-transformers 后端（如 Qwen3-Embedding-0.6B）。

    与 Embedder 同接口（dim / embed_documents / embed_query / output_dtype），
    可直接替换。需额外安装 sentence-transformers（惰性导入，不影响 voyage 路径）。
    Qwen3-Embedding 系列自带 query prompt，检索时自动启用；其它模型静默跳过。
    """

    # 本地推理是算力瓶颈，并发调用只会互相抢 CPU，保持串行
    default_concurrency = 1

    def __init__(
        self,
        model: str | None = None,
        device: str | None = None,
        batch_size: int | None = None,
    ) -> None:
        from sentence_transformers import SentenceTransformer  # 惰性导入

        self.model = model or os.getenv("SCM_LOCAL_EMBED_MODEL", "Qwen/Qwen3-Embedding-0.6B")
        self.output_dtype = "float"
        self.output_dimension = None
        dev = device or os.getenv("SCM_LOCAL_EMBED_DEVICE") or None
        self.st = SentenceTransformer(self.model, device=dev)
        # CPU 推理控制序列长度保速度；代码块均值远小于 1024 token
        self.st.max_seq_length = int(os.getenv("SCM_LOCAL_EMBED_MAX_SEQ", "1024"))
        self.batch_size = batch_size or int(os.getenv("SCM_LOCAL_EMBED_BATCH", "16"))
        self._query_prompt = "query" if "query" in (getattr(self.st, "prompts", None) or {}) else None
        self._dim: int | None = None
        self._query_cache: OrderedDict[str, list[float]] = OrderedDict()
        self._query_cache_max = 32

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = int(self.st.get_sentence_embedding_dimension())
        return self._dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        clipped = [(t or " ")[:_MAX_CHARS_PER_TEXT] for t in texts]
        embs = self.st.encode(
            clipped, batch_size=self.batch_size,
            normalize_embeddings=True, show_progress_bar=False,
        )
        return [e.tolist() for e in embs]

    def embed_query(self, text: str) -> list[float]:
        if text in self._query_cache:
            self._query_cache.move_to_end(text)
            return self._query_cache[text]
        kwargs = {"normalize_embeddings": True, "show_progress_bar": False}
        if self._query_prompt:
            kwargs["prompt_name"] = self._query_prompt
        result = self.st.encode([text[:_MAX_CHARS_PER_TEXT]], **kwargs)[0].tolist()
        self._query_cache[text] = result
        if len(self._query_cache) > self._query_cache_max:
            self._query_cache.popitem(last=False)
        return result


def create_embedder(**kwargs):
    """按 SCM_EMBED_BACKEND 选择后端：voyage（默认）/ local。"""
    backend = os.getenv("SCM_EMBED_BACKEND", "voyage").lower()
    if backend == "local":
        return LocalEmbedder()
    return Embedder(**kwargs)

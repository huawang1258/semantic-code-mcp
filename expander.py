"""查询扩展：HyDE 假想文档 + 多查询变体（可选，需 OpenAI 兼容 LLM 端点）。

单次 LLM 调用同时产出 2 条查询变体与 1 段假想代码，
变体走 query embedding + BM25，假想代码走 document embedding（标准 HyDE），
多路召回在 retriever 里 RRF 融合（原查询权重加倍防稀释）。

配置（全部可选，缺任一则禁用扩展，管线自动回退原样）：
  SCM_LLM_BASE_URL   OpenAI 兼容端点，如 https://api.example.com/v1
  SCM_LLM_API_KEY    端点 key
  SCM_LLM_MODEL      模型名
  SCM_LLM_TIMEOUT    超时秒（默认 12），超时/异常静默回退
  SCM_QUERY_EXPANSION=on 开启（默认 off）
"""
from __future__ import annotations

import json
import logging
import os
import re
from collections import OrderedDict

import requests

logger = logging.getLogger("semantic-code-mcp")

_PROMPT = """You are a code-search query expander for a codebase retrieval system.
Given a user query, return ONLY a JSON object (no markdown fence):
{"variants": ["<rewrite 1>", "<rewrite 2>"], "hypothetical_code": "<snippet>"}

Rules:
- variants: 2 alternative phrasings in English, using likely code identifiers,
  API names and terminology a developer would write in that codebase.
  If the query is not English, variant 1 must be an English translation.
- hypothetical_code: 5-15 lines of plausible code (signature + key lines + comment)
  that would exist in the codebase and directly answer the query.
- Keep it terse. JSON only.

Query: {query}"""


class QueryExpander:
    """LLM 查询扩展器。未配置端点时 enabled=False，调用方应跳过。"""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self.base_url = (base_url or os.getenv("SCM_LLM_BASE_URL") or "").rstrip("/")
        self.api_key = api_key or os.getenv("SCM_LLM_API_KEY") or ""
        self.model = model or os.getenv("SCM_LLM_MODEL") or ""
        self.timeout = timeout or float(os.getenv("SCM_LLM_TIMEOUT", "12"))
        self.enabled = bool(self.base_url and self.model)
        self._cache: OrderedDict[str, dict | None] = OrderedDict()
        self._cache_max = 64

    def expand(self, query: str) -> dict | None:
        """返回 {"variants": [str, ...], "hyde": str} 或 None（禁用/失败）。"""
        if not self.enabled:
            return None
        if query in self._cache:
            self._cache.move_to_end(query)
            return self._cache[query]
        result = self._call(query)
        self._cache[query] = result
        if len(self._cache) > self._cache_max:
            self._cache.popitem(last=False)
        return result

    def _call(self, query: str) -> dict | None:
        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": _PROMPT.replace("{query}", query)}],
                    "temperature": 0.2,
                    "max_tokens": 2000,
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            return self._parse(content)
        except Exception as e:  # 扩展失败不阻断检索
            logger.debug("查询扩展失败，回退原查询: %s", e)
            return None

    @staticmethod
    def _parse(content: str) -> dict | None:
        """从 LLM 输出中提取 JSON（容忍 markdown fence / 前后杂讯）。"""
        m = re.search(r"\{.*\}", content, re.S)
        if not m:
            return None
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
        variants = [v.strip() for v in data.get("variants", []) if isinstance(v, str) and v.strip()]
        hyde = data.get("hypothetical_code") or ""
        if not variants and not hyde:
            return None
        return {"variants": variants[:2], "hyde": hyde if isinstance(hyde, str) else ""}


def create_expander() -> QueryExpander | None:
    """按 SCM_QUERY_EXPANSION 开关创建扩展器，未开启或端点缺失返回 None。"""
    if os.getenv("SCM_QUERY_EXPANSION", "off").lower() not in ("on", "1", "true"):
        return None
    exp = QueryExpander()
    return exp if exp.enabled else None

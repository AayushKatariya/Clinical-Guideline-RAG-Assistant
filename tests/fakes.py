"""Deterministic, offline stand-ins for the embedding model, reranker and LLM."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from typing import Any

import numpy as np

from clinrag.llm.client import ChatResult, ToolCall
from clinrag.retrieval.tokenize import tokenize


class HashingEmbedder:
    """Bag-of-words hashed into a fixed-size vector. Lexical, but good enough to test plumbing."""

    name = "fake-hashing-embedder"

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in tokenize(text):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vec(t) for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._vec(text)


class OverlapReranker:
    """Fraction of query terms present in the passage, in [0, 1]."""

    name = "fake-overlap-reranker"

    def score(self, query: str, passages: list[str]) -> np.ndarray:
        q = set(tokenize(query))
        if not q:
            return np.zeros(len(passages), dtype=np.float32)
        return np.array([len(q & set(tokenize(p))) / len(q) for p in passages], dtype=np.float32)


class ScriptedLLM:
    """Returns pre-scripted ChatResults (or calls a function) and records every request."""

    def __init__(self, script: list[ChatResult | str] | Callable[[list[dict]], ChatResult | str]):
        self.script = script
        self.calls: list[dict[str, Any]] = []
        self.model = "scripted"

    def chat(self, messages, tools=None, temperature=None, max_tokens=None, json_mode=False):
        self.calls.append({"messages": [dict(m) for m in messages], "tools": tools})
        if callable(self.script):
            out = self.script(messages)
        else:
            if not self.script:
                raise AssertionError("ScriptedLLM ran out of responses")
            out = self.script.pop(0)
        return ChatResult(content=out) if isinstance(out, str) else out


def tool_call(name: str, call_id: str = "call_1", **arguments) -> ChatResult:
    return ChatResult(
        content=None,
        tool_calls=[ToolCall(id=call_id, name=name, arguments=json.dumps(arguments))],
    )

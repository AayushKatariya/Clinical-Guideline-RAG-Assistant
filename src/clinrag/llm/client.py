"""Thin chat-completions client with tool calling and LangSmith tracing.

Works with any OpenAI-compatible endpoint (OpenAI, Gemini, vLLM, Ollama, LiteLLM proxy) via
`OPENAI_BASE_URL`. Every call is traced to LangSmith when `LANGSMITH_TRACING=true`.

Provider extras: some endpoints add non-standard fields to the assistant message or its
tool calls and require them back on the next request. Gemini 3+ attaches
`extra_content.google.thought_signature` to tool calls and rejects the follow-up request
with HTTP 400 if it is missing. Those fields are kept in `ToolCall.extra` /
`ChatResult.extra`, and `ChatResult.as_message()` sends them back unchanged.
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from clinrag.config import Settings

log = logging.getLogger(__name__)

_REASONING_PREFIXES = ("o1", "o3", "o4", "gpt-5")

# Fields of the standard Chat Completions schema; anything else is a provider extra.
_STANDARD_MESSAGE_FIELDS = {
    "role",
    "content",
    "tool_calls",
    "refusal",
    "function_call",
    "audio",
    "annotations",
    "name",
}
_STANDARD_TOOL_CALL_FIELDS = {"id", "type", "function", "index"}


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON string, as returned by the API
    extra: dict[str, Any] = field(default_factory=dict)  # provider fields to echo back

    def parsed_arguments(self) -> dict[str, Any]:
        try:
            value = json.loads(self.arguments or "{}")
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}


@dataclass
class ChatResult:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)  # provider fields to echo back

    def as_message(self) -> dict[str, Any]:
        """The assistant turn, in the format the API expects back in `messages`.

        Always append assistant turns with this; rebuilding them by hand drops the
        provider extras (and Gemini then rejects the next request).
        """
        msg: dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": tc.arguments},
                    **tc.extra,
                }
                for tc in self.tool_calls
            ]
        msg.update(self.extra)
        return msg


class LLMClient(Protocol):
    model: str

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> ChatResult: ...


def _dump(obj: Any) -> dict[str, Any]:
    """Fields actually present in an SDK response object, including unknown extras."""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    return {**obj.model_dump(exclude_unset=True), **(getattr(obj, "model_extra", None) or {})}


def _to_result(resp: Any) -> ChatResult:
    message = resp.choices[0].message
    raw = _dump(message)
    # Read tool calls from the objects themselves so their own extras are kept.
    tool_calls = raw.get("tool_calls") if isinstance(message, dict) else message.tool_calls
    calls = []
    for tc in tool_calls or []:
        tc = _dump(tc)
        fn = tc.get("function") or {}
        calls.append(
            ToolCall(
                id=tc.get("id", ""),
                name=fn.get("name", ""),
                arguments=fn.get("arguments") or "{}",
                extra={
                    k: v
                    for k, v in tc.items()
                    if k not in _STANDARD_TOOL_CALL_FIELDS and v is not None
                },
            )
        )
    usage = {}
    if getattr(resp, "usage", None):
        usage = {
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
        }
    extra = {k: v for k, v in raw.items() if k not in _STANDARD_MESSAGE_FIELDS and v is not None}
    return ChatResult(content=raw.get("content"), tool_calls=calls, usage=usage, extra=extra)


class OpenAIChat:
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float | None = 0.0,
        max_tokens: int = 800,
        reasoning_effort: str | None = None,
        max_retries: int = 4,
        http_client: Any = None,
    ):
        from langsmith.wrappers import wrap_openai
        from openai import OpenAI

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.max_retries = max_retries
        # Retries are handled in `_create` (with back-off long enough for per-minute
        # quotas), so the SDK's own short retries are switched off.
        self._client = wrap_openai(
            OpenAI(api_key=api_key, base_url=base_url, max_retries=0, http_client=http_client)
        )
        self._reasoning = model.startswith(_REASONING_PREFIXES)
        self._json_mode_supported = True

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> ChatResult:
        kwargs: dict[str, Any] = {"model": self.model, "messages": messages}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        temp = self.temperature if temperature is None else temperature
        if temp is not None and not self._reasoning:
            kwargs["temperature"] = temp
        limit = max_tokens or self.max_tokens
        kwargs["max_completion_tokens" if self._reasoning else "max_tokens"] = limit
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        if json_mode and self._json_mode_supported:
            kwargs["response_format"] = {"type": "json_object"}
        return _to_result(self._create(kwargs))

    def _create(self, kwargs: dict[str, Any]) -> Any:
        from openai import (
            APIConnectionError,
            APITimeoutError,
            BadRequestError,
            InternalServerError,
            RateLimitError,
        )

        for attempt in range(self.max_retries + 1):
            try:
                return self._client.chat.completions.create(**kwargs)
            except BadRequestError as exc:
                if "response_format" in kwargs and "response_format" in str(exc):
                    # This endpoint has no JSON mode: ask in the prompt only, from now on.
                    log.info("%s rejected response_format; using plain JSON prompts", self.model)
                    self._json_mode_supported = False
                    kwargs.pop("response_format")
                    return self._create(kwargs)
                raise
            except RateLimitError as exc:
                if attempt >= self.max_retries or _is_daily_quota(exc):
                    raise
                delay = _retry_delay(exc, attempt)
            except (APIConnectionError, APITimeoutError, InternalServerError) as exc:
                if attempt >= self.max_retries:
                    raise
                delay = _retry_delay(exc, attempt)
            log.warning(
                "%s: retrying in %.0fs (attempt %d of %d)",
                self.model,
                delay,
                attempt + 1,
                self.max_retries,
            )
            time.sleep(delay)
        raise AssertionError("unreachable")


def _is_daily_quota(exc: Exception) -> bool:
    """Per-day quotas don't recover within a retry window, so fail fast on them."""
    return bool(re.search(r"per ?day|PerDay", str(exc), re.I))


def _retry_delay(exc: Exception, attempt: int) -> float:
    """Server's Retry-After if given (capped at 90 s), else 4, 8, 16, 32, 60, 60... s."""
    response = getattr(exc, "response", None)
    header = response.headers.get("retry-after") if response is not None else None
    try:
        if header:
            return min(float(header), 90.0)
    except ValueError:
        pass
    return min(4.0 * 2**attempt, 60.0) + random.uniform(0, 1)


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def loads_json_object(text: str | None) -> dict[str, Any]:
    """Parse a JSON object from model output, tolerating ```json fences and surrounding
    prose. Returns {} when there is no JSON object."""
    if not text:
        return {}
    candidates = [text.strip()]
    candidates += [m.strip() for m in _FENCE.findall(text)]
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        return value if isinstance(value, dict) else {}
    return {}


def make_llm(settings: Settings, model: str | None = None) -> OpenAIChat:
    return OpenAIChat(
        model=model or settings.llm_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        reasoning_effort=settings.llm_reasoning_effort,
        max_retries=settings.llm_max_retries,
    )

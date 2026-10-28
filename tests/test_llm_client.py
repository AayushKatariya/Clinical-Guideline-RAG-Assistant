"""OpenAIChat against a fake OpenAI-compatible server (httpx.MockTransport), including the
Gemini-style `extra_content` thought signatures that must be echoed back."""

from __future__ import annotations

import json

import httpx
import pytest

from clinrag.llm.client import OpenAIChat, loads_json_object

SIGNATURE = {"google": {"thought_signature": "sig-abc123"}}


def _completion(message: dict) -> dict:
    return {
        "id": "cmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "gemini-test",
        "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


class FakeServer:
    def __init__(self, replies: list[tuple[int, dict]]):
        self.replies = replies
        self.requests: list[dict] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content))
        status, body = self.replies.pop(0)
        return httpx.Response(status, json=body)


def _client(server: FakeServer, **kw) -> OpenAIChat:
    return OpenAIChat(
        model="gemini-test",
        api_key="test-key",
        base_url="https://example.test/v1beta/openai/",
        http_client=httpx.Client(transport=httpx.MockTransport(server)),
        max_retries=0,
        **kw,
    )


def test_thought_signature_round_trip():
    server = FakeServer(
        [
            (
                200,
                _completion(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {
                                    "name": "search_guidelines",
                                    "arguments": '{"query": "malaria"}',
                                },
                                "extra_content": SIGNATURE,
                            }
                        ],
                    }
                ),
            ),
            (200, _completion({"role": "assistant", "content": "Use an ACT [S1]."})),
        ]
    )
    llm = _client(server, temperature=1.0, max_tokens=4096, reasoning_effort="low")
    messages = [{"role": "user", "content": "malaria?"}]
    first = llm.chat(messages, tools=[{"type": "function", "function": {"name": "x"}}])
    assert first.tool_calls[0].extra == {"extra_content": SIGNATURE}
    assert first.tool_calls[0].parsed_arguments() == {"query": "malaria"}

    messages += [
        first.as_message(),
        {"role": "tool", "tool_call_id": "call-1", "content": "[S1] ACT for 3 days"},
    ]
    second = llm.chat(messages)
    assert second.content == "Use an ACT [S1]."

    echoed = server.requests[1]["messages"][1]["tool_calls"][0]
    assert echoed["extra_content"] == SIGNATURE  # signature sent back unchanged
    assert echoed["function"]["name"] == "search_guidelines"
    req = server.requests[0]
    assert req["reasoning_effort"] == "low"
    assert req["temperature"] == 1.0 and req["max_tokens"] == 4096


def test_plain_openai_response_has_no_extras():
    server = FakeServer([(200, _completion({"role": "assistant", "content": "hi"}))])
    result = _client(server).chat([{"role": "user", "content": "hello"}])
    assert result.extra == {} and result.as_message() == {"role": "assistant", "content": "hi"}
    assert "reasoning_effort" not in server.requests[0]


def test_json_mode_falls_back_when_response_format_is_rejected():
    error = {"error": {"message": "Unknown field: response_format", "type": "invalid_request"}}
    server = FakeServer(
        [
            (400, error),
            (200, _completion({"role": "assistant", "content": '```json\n{"a": 1}\n```'})),
            (200, _completion({"role": "assistant", "content": '{"b": 2}'})),
        ]
    )
    llm = _client(server)
    first = llm.chat([{"role": "user", "content": "json please"}], json_mode=True)
    assert loads_json_object(first.content) == {"a": 1}
    assert "response_format" in server.requests[0]
    assert "response_format" not in server.requests[1]
    llm.chat([{"role": "user", "content": "again"}], json_mode=True)
    assert "response_format" not in server.requests[2]  # remembered


def test_other_bad_requests_still_raise():
    from openai import BadRequestError

    server = FakeServer([(400, {"error": {"message": "invalid model"}})])
    with pytest.raises(BadRequestError):
        _client(server).chat([{"role": "user", "content": "x"}])


def test_rate_limits_back_off_and_retry(monkeypatch):
    slept: list[float] = []
    monkeypatch.setattr("clinrag.llm.client.time.sleep", slept.append)
    limited = {"error": {"message": "Resource exhausted: requests per minute"}}
    server = FakeServer(
        [(429, limited), (429, limited), (200, _completion({"role": "assistant", "content": "ok"}))]
    )
    llm = OpenAIChat(
        model="gemini-test",
        api_key="k",
        base_url="https://example.test/v1beta/openai/",
        http_client=httpx.Client(transport=httpx.MockTransport(server)),
        max_retries=3,
    )
    assert llm.chat([{"role": "user", "content": "x"}]).content == "ok"
    assert len(slept) == 2 and 4 <= slept[0] < 5 and 8 <= slept[1] < 9  # 4 s, then 8 s


def test_daily_quota_is_not_retried(monkeypatch):
    from openai import RateLimitError

    monkeypatch.setattr("clinrag.llm.client.time.sleep", lambda s: pytest.fail("slept"))
    daily = {"error": {"message": "Quota exceeded: GenerateRequestsPerDayPerProjectPerModel"}}
    server = FakeServer([(429, daily)])
    llm = OpenAIChat(
        model="gemini-test",
        api_key="k",
        base_url="https://example.test/v1beta/openai/",
        http_client=httpx.Client(transport=httpx.MockTransport(server)),
        max_retries=3,
    )
    with pytest.raises(RateLimitError):
        llm.chat([{"role": "user", "content": "x"}])
    assert len(server.requests) == 1


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"q": "a"}', {"q": "a"}),
        ('```json\n{"q": "a"}\n```', {"q": "a"}),
        ('Sure! Here it is: {"q": "a"} Hope that helps.', {"q": "a"}),
        ("[1, 2]", {}),
        ("not json", {}),
        (None, {}),
    ],
)
def test_loads_json_object(text, expected):
    assert loads_json_object(text) == expected

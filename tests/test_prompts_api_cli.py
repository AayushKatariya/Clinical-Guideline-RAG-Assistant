from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from clinrag.agent.assistant import ClinicalAssistant
from clinrag.agent.prompts import list_prompts, load_prompt
from clinrag.agent.refusal import REFUSE_MARKER
from clinrag.api.app import create_app
from clinrag.cli import app as cli_app
from clinrag.config import PROJECT_ROOT

from conftest import PROMPTS_DIR
from fakes import ScriptedLLM

# ---------------------------------------------------------------- prompts


def test_all_prompt_versions_load():
    prompts = list_prompts(PROMPTS_DIR)
    assert [p.version for p in prompts] == ["v1", "v2", "v3", "v4"]
    assert len({p.sha for p in prompts}) == 4
    for p in prompts:
        rendered = p.render_user("Q?", "")
        assert "Q?" in rendered and p.no_context_note in rendered


def test_refusal_protocol_introduced_in_v3():
    assert REFUSE_MARKER not in load_prompt(PROMPTS_DIR, "v1").system
    assert REFUSE_MARKER not in load_prompt(PROMPTS_DIR, "v2").system
    assert REFUSE_MARKER in load_prompt(PROMPTS_DIR, "v3").system
    assert REFUSE_MARKER in load_prompt(PROMPTS_DIR, "v4").system


def test_missing_prompt_version():
    with pytest.raises(FileNotFoundError, match="available"):
        load_prompt(PROMPTS_DIR, "v99")


def test_invalid_prompt_file(tmp_path):
    (tmp_path / "v1_bad.yaml").write_text(
        "version: v1\nname: bad\nsystem: s\nuser_template: '{question}'\n"
    )
    with pytest.raises(ValueError, match="context"):
        load_prompt(tmp_path, "v1")


# ---------------------------------------------------------------- API


@pytest.fixture
def client(retriever):
    assistant = ClinicalAssistant(
        llm=ScriptedLLM(lambda msgs: "Use an ACT for 3 days [S1]."),
        retriever=retriever,
        prompt=load_prompt(PROMPTS_DIR, "v4"),
        top_k=3,
        refusal_threshold=0.3,
    )
    return TestClient(create_app(assistant))


def test_api_health_and_prompts(client):
    health = client.get("/health").json()
    assert health["status"] == "ok" and health["prompt_version"] == "v4"
    assert [p["version"] for p in client.get("/prompts").json()] == ["v1", "v2", "v3", "v4"]


def test_api_ask(client):
    body = client.post("/ask", json={"question": "treatment of uncomplicated malaria"}).json()
    assert body["refused"] is False and body["citations"][0]["doc_id"] == "who-malaria"
    v2 = client.post("/ask", json={"question": "malaria treatment", "prompt_version": "v2"}).json()
    assert v2["prompt_version"] == "v2"
    assert (
        client.post("/ask", json={"question": "anything?", "prompt_version": "v9"}).status_code
        == 404
    )
    assert client.post("/ask", json={"question": ""}).status_code == 422


def test_api_serves_chat_page(client):
    page = client.get("/")
    assert page.status_code == 200 and page.headers["content-type"].startswith("text/html")
    assert 'id="ask-form"' in page.text
    assert all(route in page.text for route in ('"/ask"', '"/health"', '"/prompts"'))
    assert "/" not in client.get("/openapi.json").json()["paths"]  # page is not an API route


def test_api_search(client):
    hits = client.post("/search", json={"query": "HbA1c diabetes", "k": 2, "mode": "bm25"}).json()
    assert len(hits) <= 2 and hits[0]["chunk"]["doc_id"] == "ada-diabetes"


# ---------------------------------------------------------------- CLI


def test_cli_prompts_and_eval_summary():
    runner = CliRunner()
    out = runner.invoke(cli_app, ["prompts"])
    assert out.exit_code == 0 and "claim_check" in out.output
    example = PROJECT_ROOT / "data" / "eval" / "eval_set.example.jsonl"
    out = runner.invoke(cli_app, ["eval", "summary", "--dataset", str(example)])
    assert out.exit_code == 0, out.output
    assert "total" in out.output

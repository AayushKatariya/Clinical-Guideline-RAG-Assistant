from __future__ import annotations

import re
import uuid

import pytest

from clinrag.agent.assistant import ClinicalAssistant
from clinrag.agent.prompts import load_prompt
from clinrag.agent.refusal import DEFAULT_REFUSAL, is_refusal, strip_marker
from clinrag.llm.client import ChatResult
from clinrag.sql.text_to_sql import TextToSQL

from conftest import PROMPTS_DIR
from fakes import ScriptedLLM, tool_call


def make_assistant(retriever, llm, version="v4", **kw) -> ClinicalAssistant:
    return ClinicalAssistant(
        llm=llm,
        retriever=retriever,
        prompt=load_prompt(PROMPTS_DIR, version),
        top_k=3,
        refusal_threshold=0.3,
        **kw,
    )


def _first_source(messages, prefix="S") -> str:
    joined = "\n".join(str(m.get("content")) for m in messages)
    return re.search(rf"\[({prefix}\d+)\]", joined).group(1)


def test_grounded_answer_with_citations(retriever):
    llm = ScriptedLLM(lambda msgs: f"Offer ABPM to confirm hypertension [{_first_source(msgs)}].")
    resp = make_assistant(retriever, llm).answer(
        "How should a raised clinic blood pressure be confirmed with ambulatory monitoring?"
    )
    assert not resp.refused
    assert resp.citations and resp.citations[0].doc_id == "nice-hypertension"
    assert resp.contexts and resp.contexts[0].startswith("[S1]")
    assert resp.retrieved_doc_ids[0] == "nice-hypertension"
    assert resp.prompt_version == "v4" and resp.max_rerank_score is not None
    # The prompt sent to the model contains the numbered evidence and the question.
    user_msg = llm.calls[0]["messages"][1]["content"]
    assert "[S1]" in user_msg and "ambulatory" in user_msg


def test_evidence_gate_refuses_when_nothing_is_relevant(retriever):
    llm = ScriptedLLM(["Canberra is the capital of Australia."])  # model ignores the rules
    resp = make_assistant(retriever, llm).answer("What is the capital city of Australia?")
    assert resp.refused and resp.refusal_reason == "no_evidence"
    assert resp.answer == DEFAULT_REFUSAL and resp.citations == []
    assert resp.contexts == []
    assert "no guideline excerpt passed" in llm.calls[0]["messages"][1]["content"]


def test_model_refusal_marker(retriever):
    llm = ScriptedLLM(["[REFUSE] The corpus does not cover paediatric dosing."])
    resp = make_assistant(retriever, llm).answer("Artesunate dose in severe malaria?")
    assert resp.refused and resp.refusal_reason == "model"
    assert resp.answer == "The corpus does not cover paediatric dosing."


def test_search_tool_loop(retriever):
    def script(msgs):
        if not any(m["role"] == "tool" for m in msgs):
            return tool_call("search_guidelines", query="tuberculosis isoniazid rifampicin regimen")
        tool_out = [m for m in msgs if m["role"] == "tool"][-1]["content"]
        sid = re.search(r"\[(S\d+)\]", tool_out).group(1)
        return f"The 6-month 2HRZE/4HR regimen is recommended [{sid}]."

    resp = make_assistant(retriever, ScriptedLLM(script)).answer(
        "What is the regimen for pulmonary TB?"
    )
    assert [t.name for t in resp.tool_calls] == ["search_guidelines"]
    assert any(c.doc_id == "who-tuberculosis" for c in resp.citations)
    assert any("2HRZE/4HR" in c for c in resp.contexts)


def test_sql_tool_loop(retriever, sqlite_engine):
    sql_llm = ScriptedLLM(
        ["SELECT drug_name, dose, route FROM drug_dosing WHERE drug_name LIKE '%artesunate%'"]
    )
    text_to_sql = TextToSQL(sql_llm, sqlite_engine)

    def script(msgs):
        if not any(m["role"] == "tool" for m in msgs):
            return tool_call("query_clinical_database", question="artesunate dose severe malaria")
        return "Artesunate 2.4 mg/kg IV or IM [DB1]."

    resp = make_assistant(retriever, ScriptedLLM(script), text_to_sql=text_to_sql).answer(
        "What dose of artesunate is used for severe malaria?"
    )
    assert not resp.refused
    assert resp.sql_queries and "drug_dosing" in resp.sql_queries[0]
    assert [c.kind for c in resp.citations] == ["database"]
    assert any("2.4 mg/kg" in c for c in resp.contexts)


def test_sql_tool_hidden_when_unavailable(retriever):
    a = make_assistant(retriever, ScriptedLLM(["x"]))
    assert [t["function"]["name"] for t in a.tools] == ["search_guidelines"]


def test_last_step_forces_an_answer(retriever):
    def script(msgs):
        return tool_call("search_guidelines", call_id=f"c{len(msgs)}", query="malaria")

    llm = ScriptedLLM(script)
    a = make_assistant(retriever, llm, max_steps=2)
    a.answer("malaria treatment")
    assert len(llm.calls) == 3
    assert llm.calls[-1]["tools"] is None


def test_unknown_tool_is_reported_to_model(retriever):
    llm = ScriptedLLM([tool_call("delete_database"), "[REFUSE] cannot help"])
    resp = make_assistant(retriever, llm).answer("malaria")
    tool_msg = [m for m in llm.calls[1]["messages"] if m["role"] == "tool"][0]
    assert "unknown tool" in tool_msg["content"]
    assert resp.refused


def test_accepts_langsmith_run_id(retriever):
    llm = ScriptedLLM(["[REFUSE] no"])
    resp = make_assistant(retriever, llm).answer(
        "malaria", langsmith_extra={"run_id": str(uuid.uuid4()), "tags": ["test"]}
    )
    assert resp.refused


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("[REFUSE] Not covered.", True),
        ("I'm sorry, but I cannot find this in the provided guidelines.", True),
        ("The excerpts do not contain information about melanoma.", True),
        ("", True),
        ("Offer an ACE inhibitor [S1].", False),
        (
            "Offer ABPM to confirm the diagnosis [S1]. If ABPM is unsuitable, offer HBPM [S1]. "
            "The guideline does not cover children, so this applies to adults only, and the "
            "excerpts do not mention pregnancy; clinicians should follow local protocols for "
            "those groups and review the full recommendations for additional considerations.",
            False,
        ),
    ],
)
def test_is_refusal(text, expected):
    assert is_refusal(text) is expected


def test_strip_marker():
    assert strip_marker("[REFUSE]: out of scope") == "out of scope"
    assert strip_marker("[REFUSE]") == DEFAULT_REFUSAL


def test_chatresult_message_roundtrip():
    msg = tool_call("search_guidelines", query="x").as_message()
    assert (
        msg["role"] == "assistant"
        and msg["tool_calls"][0]["function"]["name"] == "search_guidelines"
    )
    assert ChatResult(content="hi").as_message() == {"role": "assistant", "content": "hi"}

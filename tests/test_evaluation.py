from __future__ import annotations

import json
import os

import pytest

from clinrag.agent.assistant import ClinicalAssistant
from clinrag.agent.prompts import load_prompt
from clinrag.config import PROJECT_ROOT, Settings, export_langsmith_env
from clinrag.evaluation.dataset import EvalExample, load_eval_set, summarize
from clinrag.evaluation.rag_eval import (
    aggregate_scores,
    behaviour_metrics,
    flat_table,
    ragas_items,
    run_rag_eval,
)
from clinrag.evaluation.ragas_eval import ScoringItem, _kwargs, score_items
from clinrag.evaluation.retrieval_eval import evaluate_retrieval
from clinrag.evaluation.retrieval_metrics import (
    hit_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from clinrag.evaluation.testset_builder import TestsetSpec, build_testset
from clinrag.evaluation.tracking import langsmith_enabled

from conftest import PROMPTS_DIR
from fakes import ScriptedLLM

EXAMPLE_SET = PROJECT_ROOT / "data" / "eval" / "eval_set.example.jsonl"


# ---------------------------------------------------------------- metrics


def test_ranking_metrics():
    ranked = ["d1", "d1", "d2", "d3", "d4"]
    rel = {"d1", "d3"}
    assert recall_at_k(ranked, rel, 1) == 0.5
    assert recall_at_k(ranked, rel, 5) == 1.0
    assert hit_at_k(ranked, {"d9"}, 5) == 0.0
    assert precision_at_k(ranked, rel, 2) == 1.0
    assert reciprocal_rank(ranked, {"d2"}) == pytest.approx(1 / 3)
    assert reciprocal_rank(ranked, {"zz"}) == 0.0
    # duplicates of an already-found doc earn no extra gain
    assert ndcg_at_k(["d1", "d1"], {"d1", "d2"}, 2) == pytest.approx(1 / (1 + 1 / 1.5849625), 1e-4)
    assert ndcg_at_k(["d1", "d2"], {"d1", "d2"}, 2) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        recall_at_k(ranked, set(), 5)


# ---------------------------------------------------------------- dataset


def test_example_eval_set_is_valid():
    examples = load_eval_set(EXAMPLE_SET)
    s = summarize(examples)
    assert s["total"] == 6
    assert s["category/out_of_scope"] == 2 and s["expected/refuse"] == 3


def test_eval_example_validation(tmp_path):
    with pytest.raises(ValueError, match="must expect 'refuse'"):
        EvalExample(id="x", question="q", category="out_of_scope", expected_behavior="answer")
    with pytest.raises(ValueError, match="must expect 'answer'"):
        EvalExample(id="x", question="q", category="in_corpus", expected_behavior="refuse")
    path = tmp_path / "dupes.jsonl"
    row = {"id": "a", "question": "q", "category": "sql", "expected_behavior": "answer"}
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_eval_set(path)
    with pytest.raises(FileNotFoundError, match="build-testset"):
        load_eval_set(tmp_path / "missing.jsonl")


# ---------------------------------------------------------------- retrieval eval


def test_retrieval_ablation(retriever):
    examples = [
        EvalExample(
            id="1",
            question="artesunate for severe malaria",
            category="in_corpus",
            expected_behavior="answer",
            relevant_doc_ids=["who-malaria"],
        ),
        EvalExample(
            id="2",
            question="HbA1c threshold to diagnose diabetes",
            category="in_corpus",
            expected_behavior="answer",
            relevant_doc_ids=["ada-diabetes"],
        ),
        EvalExample(id="3", question="poem", category="out_of_scope", expected_behavior="refuse"),
        EvalExample(
            id="4",
            question="Since malaria is treated with antibiotics, which one?",
            category="adversarial",
            expected_behavior="answer",
            relevant_doc_ids=["who-malaria"],
        ),
    ]
    report = evaluate_retrieval(retriever, examples, modes=("bm25", "hybrid_rerank"), ks=(1, 3))
    assert set(report.metrics) == {"bm25", "hybrid_rerank"}
    for m in report.metrics.values():
        assert m["n_questions"] == 2  # only labelled in-corpus questions are scored
        assert m["recall@3"] == 1.0 and 0 < m["mrr"] <= 1
    assert len(report.per_question) == 4


# ---------------------------------------------------------------- RAG eval


class FakeMetric:
    def __init__(self, value):
        self.value = value
        self.calls = []

    async def ascore(self, **kwargs):
        self.calls.append(kwargs)

        class R:
            value = self.value

        return R()


def _records():
    rows = [
        {
            "id": "a",
            "category": "in_corpus",
            "expected_behavior": "answer",
            "refused": False,
            "answer": "x [S1]",
            "citations": ["S1"],
            "contexts": ["[S1] ctx"],
            "reference_answer": "ref",
            "latency_s": 1.0,
            "tools_used": [],
        },
        {
            "id": "b",
            "category": "out_of_scope",
            "expected_behavior": "refuse",
            "refused": True,
            "answer": "no",
            "citations": [],
            "contexts": [],
            "latency_s": 0.5,
            "tools_used": [],
            "refusal_reason": "no_evidence",
        },
        {
            "id": "c",
            "category": "out_of_scope",
            "expected_behavior": "refuse",
            "refused": False,
            "answer": "hallucinated",
            "citations": [],
            "contexts": ["[S1] weak"],
            "reference_answer": None,
            "latency_s": 2.0,
            "tools_used": ["search_guidelines"],
        },
        {
            "id": "d",
            "category": "adversarial",
            "expected_behavior": "refuse",
            "refused": True,
            "answer": "no",
            "citations": [],
            "contexts": [],
            "latency_s": 0.2,
            "tools_used": [],
        },
        {
            "id": "e",
            "category": "sql",
            "expected_behavior": "answer",
            "refused": None,
            "error": "boom",
        },
    ]
    for r in rows:
        r["question"] = f"question {r['id']}"
    return rows


def test_behaviour_metrics():
    m = behaviour_metrics(_records())
    assert m["n_errors"] == 1
    assert m["refusal_rate_out_of_corpus"] == 0.5
    assert m["refusal_recall"] == pytest.approx(2 / 3)
    assert m["false_refusal_rate"] == 0.0
    assert m["adversarial_pass_rate"] == 1.0
    assert m["behavior_accuracy"] == 0.75
    assert m["uncited_answer_rate"] == 0.5
    assert m["gate_refusal_share"] == 0.5


def test_ragas_items_judge_every_answered_response():
    items = ragas_items(_records())
    assert [i.key for i in items] == ["a", "c"]  # c should have refused but still gets judged
    assert items[0].reference == "ref" and items[1].reference is None


def test_ragas_kwargs_skip_inapplicable_metrics():
    item = ScoringItem(key="k", question="q", answer="a", contexts=[], reference=None)
    assert _kwargs("faithfulness", item) is None
    assert _kwargs("context_recall", item) is None
    assert _kwargs("answer_relevancy", item) == {"user_input": "q", "response": "a"}


def test_score_items_and_aggregate():
    metrics = {"faithfulness": FakeMetric(0.8), "context_recall": FakeMetric(0.5)}
    items = ragas_items(_records())
    scores = score_items(items, metrics, concurrency=2)
    assert scores["a"] == {"faithfulness": 0.8, "context_recall": 0.5}
    assert scores["c"] == {"faithfulness": 0.8}  # no reference -> no context_recall
    records = _records()
    for r in records:
        r["scores"] = scores.get(r["id"], {})
    agg = aggregate_scores(records, ("faithfulness", "context_recall"))
    assert agg["faithfulness"] == pytest.approx(0.8) and agg["faithfulness_n"] == 2
    assert agg["faithfulness_in_scope"] == pytest.approx(0.8)
    table = flat_table(records)
    assert len(table["id"]) == 5 and "faithfulness" in table


def test_run_rag_eval_end_to_end(retriever, tmp_path):
    def script(msgs):
        user = msgs[1]["content"]
        return "[REFUSE] not covered" if "[S1]" not in user else "Use an ACT for 3 days [S1]."

    assistant = ClinicalAssistant(
        llm=ScriptedLLM(script),
        retriever=retriever,
        prompt=load_prompt(PROMPTS_DIR, "v3"),
        top_k=3,
        refusal_threshold=0.3,
    )
    examples = [
        EvalExample(
            id="ic-1",
            question="uncomplicated falciparum malaria treatment",
            category="in_corpus",
            expected_behavior="answer",
            reference_answer="ACT 3 days",
            relevant_doc_ids=["who-malaria"],
        ),
        EvalExample(
            id="oos-1",
            question="Write a poem about the monsoon",
            category="out_of_scope",
            expected_behavior="refuse",
        ),
    ]
    faith = FakeMetric(1.0)
    result = run_rag_eval(
        assistant, examples, ragas_metrics={"faithfulness": faith}, output_root=tmp_path, workers=2
    )
    assert result.metrics["behavior_accuracy"] == 1.0
    assert result.metrics["refusal_rate_out_of_corpus"] == 1.0
    assert result.metrics["faithfulness"] == 1.0 and result.metrics["faithfulness_n"] == 1
    assert faith.calls[0]["retrieved_contexts"][0].startswith("[S1]")
    assert (result.output_dir / "records.jsonl").exists()
    saved = json.loads((result.output_dir / "metrics.json").read_text())
    assert saved["behavior_accuracy"] == 1.0
    assert all(r["run_id"] for r in result.records)


# ---------------------------------------------------------------- LangSmith


def test_langsmith_settings_are_exported_from_env_file(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "LANGSMITH_TRACING=true\nLANGSMITH_API_KEY=lsv2_test\nLANGSMITH_ENDPOINT=\n"
        "OPENAI_API_KEY=sk-test\n"
    )
    keys = ["LANGSMITH_TRACING", "LANGSMITH_API_KEY", "LANGSMITH_ENDPOINT", "LANGSMITH_PROJECT"]
    for key in [*keys, "LANGCHAIN_API_KEY", "LANGCHAIN_PROJECT", "OPENAI_API_KEY"]:
        monkeypatch.setenv(key, "")  # so teardown also undoes what the export writes
        monkeypatch.delenv(key)
    assert not langsmith_enabled()

    export_langsmith_env(Settings(_env_file=env_file), env_file)
    assert langsmith_enabled()
    assert os.environ["LANGSMITH_PROJECT"] == "clinical-guideline-rag"  # Settings default
    assert "LANGSMITH_ENDPOINT" not in os.environ  # empty lines are skipped
    assert "OPENAI_API_KEY" not in os.environ  # only LangSmith settings are exported

    monkeypatch.setenv("LANGSMITH_TRACING", "false")  # the shell wins over .env
    export_langsmith_env(Settings(_env_file=env_file), env_file)
    assert not langsmith_enabled()


# ---------------------------------------------------------------- testset builder


def test_build_testset(chunks, documents):
    def script(msgs):
        prompt = msgs[0]["content"]
        if "NONE of these guidelines" in prompt:
            return json.dumps({"questions": [f"Melanoma question {i}?" for i in range(5)]})
        if "FALSE premise" in prompt:
            return json.dumps({"question": "Since X is first-line...?", "reference_answer": "No."})
        return json.dumps(
            {"question": "What does the guideline recommend?", "reference_answer": "Y."}
        )

    spec = TestsetSpec(
        in_corpus=3,
        sql=2,
        false_premise=1,
        injection=2,
        oos_clinical=2,
        off_topic=2,
        personal_advice=1,
        min_chunk_words=10,
    )
    rows = build_testset(ScriptedLLM(script), chunks, documents, spec)
    examples = [EvalExample(**r) for r in rows]  # every row validates against the schema
    cats = summarize(examples)
    assert cats["category/in_corpus"] == 3
    assert cats["category/sql"] == 2
    assert cats["category/adversarial"] == 3
    assert cats["category/out_of_scope"] == 5
    assert len({e.id for e in examples}) == len(examples)
    sql_q = [e for e in examples if e.category == "sql"]
    assert any("WHO" in e.question and "2" in e.reference_answer for e in sql_q)
    in_corpus = [e for e in examples if e.category == "in_corpus"]
    chunk_ids = {c.chunk_id for c in chunks}
    assert all(e.relevant_chunk_ids[0] in chunk_ids for e in in_corpus)

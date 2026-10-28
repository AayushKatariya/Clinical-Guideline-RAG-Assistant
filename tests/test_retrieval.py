from __future__ import annotations

import pytest

from clinrag.retrieval import BM25Index, DenseIndex, HybridRetriever, reciprocal_rank_fusion
from clinrag.retrieval.tokenize import tokenize

from fakes import HashingEmbedder


def test_tokenizer_keeps_clinical_terms():
    toks = tokenize("Type-2 diabetes: HbA1c >= 6.5% and COVID-19, not the 2.4 mg/kg dose")
    assert {"type-2", "type", "2", "hba1c", "6.5", "covid-19", "covid", "19", "mg/kg"} <= set(toks)
    assert "not" in toks  # negation is kept
    assert "the" not in toks and "and" not in toks


def test_bm25_ranks_lexical_match_first(chunks):
    idx = BM25Index.build([c.index_text for c in chunks], [c.chunk_id for c in chunks])
    hits = idx.search("rifampicin isoniazid regimen", k=3)
    assert chunks[hits[0][0]].doc_id == "who-tuberculosis"
    assert idx.search("the and of", k=3) == []  # only stopwords -> nothing


def test_rrf_combines_rankings():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "c", "d"]], k=60)
    # b: 1/62 + 1/61, c: 1/63 + 1/62, a: 1/61, d: 1/63
    assert [i for i, _ in fused] == ["b", "c", "a", "d"]
    weighted = reciprocal_rank_fusion([["a"], ["b"]], weights=[1.0, 2.0])
    assert weighted[0][0] == "b"
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"]], weights=[1, 2])


@pytest.mark.parametrize("mode", ["bm25", "dense", "hybrid", "hybrid_rerank"])
def test_all_modes_find_the_right_guideline(retriever, mode):
    hits = retriever.retrieve("severe malaria artesunate treatment", k=3, mode=mode)
    assert 0 < len(hits) <= 3
    assert hits[0].chunk.doc_id == "who-malaria"
    assert [h.rank for h in hits] == list(range(1, len(hits) + 1))
    assert len({h.chunk.chunk_id for h in hits}) == len(hits)


def test_rerank_scores_are_sorted_and_recorded(retriever):
    hits = retriever.retrieve("HbA1c threshold for diagnosing diabetes", k=3, mode="hybrid_rerank")
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)
    assert all({"rrf", "rerank"} <= set(h.scores) for h in hits)
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_filters_and_per_doc_cap(retriever):
    hits = retriever.retrieve("treatment", k=5, mode="hybrid", filters={"organization": "WHO"})
    assert hits and all(h.chunk.organization == "WHO" for h in hits)
    capped = retriever.retrieve("malaria", k=5, mode="hybrid", max_per_doc=1)
    docs = [h.chunk.doc_id for h in capped]
    assert len(docs) == len(set(docs))


def test_invalid_mode_and_missing_reranker(retriever):
    with pytest.raises(ValueError):
        retriever.retrieve("x", mode="magic")
    retriever.reranker = None
    with pytest.raises(ValueError):
        retriever.retrieve("x", mode="hybrid_rerank")


def test_index_roundtrip_and_model_mismatch(chunks, tmp_path):
    emb = HashingEmbedder()
    HybridRetriever.build_indexes(chunks, emb, tmp_path)
    bm25, dense = BM25Index.load(tmp_path), DenseIndex.load(tmp_path)
    assert bm25.chunk_ids == dense.chunk_ids == [c.chunk_id for c in chunks]
    assert (tmp_path / "manifest.json").exists()

    class OtherEmbedder(HashingEmbedder):
        name = "another-model"

    with pytest.raises(ValueError, match="built with"):
        HybridRetriever(chunks, bm25, dense, OtherEmbedder())
    with pytest.raises(ValueError, match="out of sync"):
        HybridRetriever(chunks[:-1], bm25, dense, emb)

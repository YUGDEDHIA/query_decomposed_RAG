from __future__ import annotations

from src.retriever import BROAD_TOP_K, DEFAULT_TOP_K, Retriever
from src.vector_store import VectorStore


def _store_with_n_chunks(n: int, section: str | None = "SECTION II – RISK FACTORS") -> VectorStore:
    metadata = [
        {"text": f"chunk {i}", "company": "Acme", "section": section, "chunk_index": i}
        for i in range(n)
    ]
    embeddings = [[1.0, 0.0]] * n
    return VectorStore.build(metadata, embeddings)


def _retriever(monkeypatch, n_chunks: int, section: str | None = "SECTION II – RISK FACTORS") -> Retriever:
    # embed_query hits Ollama; stub it so this test is deterministic and
    # doesn't need a live model -- the vector search/filtering logic below
    # it still runs for real.
    monkeypatch.setattr("src.retriever.embed_query", lambda text: [1.0, 0.0])
    return Retriever(_store_with_n_chunks(n_chunks, section=section))


def test_retrieve_uses_default_top_k_for_unrecognized_question(monkeypatch):
    retriever = _retriever(monkeypatch, n_chunks=30)
    results = retriever.retrieve("what raw materials does the company depend on?")
    assert len(results) == DEFAULT_TOP_K


def test_retrieve_widens_top_k_for_recognized_topic(monkeypatch):
    retriever = _retriever(monkeypatch, n_chunks=30)
    results = retriever.retrieve("what are the risk factors for Acme?")
    assert len(results) == BROAD_TOP_K


def test_retrieve_respects_explicit_top_k_override(monkeypatch):
    retriever = _retriever(monkeypatch, n_chunks=30)
    results = retriever.retrieve("what are the risk factors for Acme?", top_k=3)
    assert len(results) == 3


def test_retrieve_scopes_to_inferred_topic_section(monkeypatch):
    monkeypatch.setattr("src.retriever.embed_query", lambda text: [1.0, 0.0])
    metadata = [
        {"text": "risk chunk", "company": "Acme", "section": "SECTION II – RISK FACTORS", "chunk_index": 0},
        {"text": "unrelated chunk", "company": "Acme", "section": "CAPITAL STRUCTURE", "chunk_index": 1},
    ]
    store = VectorStore.build(metadata, [[1.0, 0.0], [1.0, 0.0]])
    retriever = Retriever(store)

    results = retriever.retrieve("what are the risk factors for Acme?")
    assert [r["text"] for r in results] == ["risk chunk"]


def test_retrieve_explicit_section_topic_overrides_inference(monkeypatch):
    monkeypatch.setattr("src.retriever.embed_query", lambda text: [1.0, 0.0])
    store = _store_with_n_chunks(30, section="CAPITAL STRUCTURE")
    retriever = Retriever(store)

    # Question text would normally infer RISK FACTORS, but explicit
    # section_topic should win -- if inference won instead, this would
    # return 0 results since none of these chunks are in Risk Factors.
    results = retriever.retrieve("what are the risk factors?", section_topic="CAPITAL STRUCTURE")
    assert len(results) == BROAD_TOP_K


def _multi_company_retriever(monkeypatch) -> Retriever:
    monkeypatch.setattr("src.retriever.embed_query", lambda text: [1.0, 0.0])
    metadata = [
        {"text": "acme chunk", "company": "Acme Corp", "section": None, "chunk_index": 0},
        {"text": "globex chunk", "company": "Globex Corp", "section": None, "chunk_index": 1},
    ]
    return Retriever(VectorStore.build(metadata, [[1.0, 0.0], [1.0, 0.0]]))


def test_retrieve_infers_company_named_in_query(monkeypatch):
    retriever = _multi_company_retriever(monkeypatch)
    results = retriever.retrieve("What is Globex Corp's revenue?")
    assert [r["text"] for r in results] == ["globex chunk"]


def test_retrieve_explicit_company_overrides_inference(monkeypatch):
    retriever = _multi_company_retriever(monkeypatch)
    # Query names Globex, but explicit company= should win.
    results = retriever.retrieve("What is Globex Corp's revenue?", company="Acme Corp")
    assert [r["text"] for r in results] == ["acme chunk"]


def test_retrieve_company_inference_prefers_longer_more_specific_match(monkeypatch):
    monkeypatch.setattr("src.retriever.embed_query", lambda text: [1.0, 0.0])
    metadata = [
        {"text": "acme chunk", "company": "Acme", "section": None, "chunk_index": 0},
        {"text": "acme fabrics chunk", "company": "Acme Fabrics Limited", "section": None, "chunk_index": 1},
    ]
    retriever = Retriever(VectorStore.build(metadata, [[1.0, 0.0], [1.0, 0.0]]))

    results = retriever.retrieve("What is the revenue of Acme Fabrics Limited?")
    assert [r["text"] for r in results] == ["acme fabrics chunk"]


def test_infer_company_matches_despite_punctuation_lost_in_filename_round_trip():
    # Real bug: "Hy-Tech Engineers Limited" round-trips through the
    # CompanyName_RHP.pdf filename convention as "HY TECH ENGINEERS
    # LIMITED" (hyphen -> underscore -> space), so a raw substring check
    # against a question that writes the real hyphenated name never
    # matched -- silently re-triggering on-demand ingestion for a company
    # already indexed, every single time.
    metadata = [{"text": "x", "company": "HY TECH ENGINEERS LIMITED", "section": None, "chunk_index": 0}]
    retriever = Retriever(VectorStore.build(metadata, [[1.0, 0.0]]))

    assert retriever.infer_company("What are the risk factors for Hy-Tech Engineers Limited?") == \
        "HY TECH ENGINEERS LIMITED"


def _company_only_retriever(*companies: str) -> Retriever:
    metadata = [
        {"text": f"chunk {i}", "company": c, "section": None, "chunk_index": 0}
        for i, c in enumerate(companies)
    ]
    return Retriever(VectorStore.build(metadata, [[1.0, 0.0]] * len(companies)))


def test_fuzzy_match_company_catches_misspelling():
    retriever = _company_only_retriever("AASTHA SPINTEX LIMITED")
    assert retriever.fuzzy_match_company("Aastha Spintx Limited") == "AASTHA SPINTEX LIMITED"


def test_fuzzy_match_company_catches_abbreviation():
    retriever = _company_only_retriever("SANSTAR LIMITED")
    assert retriever.fuzzy_match_company("Sanstar Ltd") == "SANSTAR LIMITED"


def test_fuzzy_match_company_does_not_conflate_distinct_similarly_named_companies():
    # "Sanstar Limited" vs "Sanstar Foods Limited" -- a real different
    # company, not a typo of each other. Should not silently answer about
    # the wrong one.
    retriever = _company_only_retriever("SANSTAR LIMITED", "SANSTAR FOODS LIMITED")
    assert retriever.fuzzy_match_company("Sanstar Foods Ltd") == "SANSTAR FOODS LIMITED"


def test_fuzzy_match_company_returns_none_for_an_unrelated_name():
    retriever = _company_only_retriever("AASTHA SPINTEX LIMITED")
    assert retriever.fuzzy_match_company("Reliance Industries Limited") is None

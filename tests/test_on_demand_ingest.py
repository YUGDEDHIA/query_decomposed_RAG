from __future__ import annotations

from src.on_demand_ingest import ingest_company_on_demand
from src.retriever import Retriever
from src.vector_store import VectorStore


def _retriever_with_one_company() -> Retriever:
    store = VectorStore.build(
        [{"text": "x", "company": "Acme", "section": None, "chunk_index": 0}], [[1.0, 0.0]],
    )
    return Retriever(store)


def test_returns_none_when_sebi_has_no_match(monkeypatch):
    monkeypatch.setattr("src.on_demand_ingest.new_session", lambda: object())
    monkeypatch.setattr("src.on_demand_ingest.search_listing", lambda session, query: "<html></html>")
    monkeypatch.setattr("src.on_demand_ingest.parse_listing_rows", lambda html: [])

    result = ingest_company_on_demand("Nonexistent Company", _retriever_with_one_company())
    assert result is None


def test_returns_none_on_unexpected_failure_rather_than_raising(monkeypatch):
    def _boom():
        raise RuntimeError("network down")

    monkeypatch.setattr("src.on_demand_ingest.new_session", _boom)

    result = ingest_company_on_demand("Some Company", _retriever_with_one_company())
    assert result is None


def test_retriever_untouched_when_no_match_found(monkeypatch):
    monkeypatch.setattr("src.on_demand_ingest.new_session", lambda: object())
    monkeypatch.setattr("src.on_demand_ingest.search_listing", lambda session, query: "<html></html>")
    monkeypatch.setattr("src.on_demand_ingest.parse_listing_rows", lambda html: [])

    retriever = _retriever_with_one_company()
    before = len(retriever.store.metadata)
    ingest_company_on_demand("Nonexistent Company", retriever)
    assert len(retriever.store.metadata) == before

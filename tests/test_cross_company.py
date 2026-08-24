from __future__ import annotations

from src.cross_company import _retrieve_financial_summary, compare_companies
from src.retriever import Retriever
from src.vector_store import VectorStore


def _chunk(company, section, text="financial data here"):
    return {
        "text": text, "company": company, "section": section, "chunk_index": 0,
        "page_range": [10, 10], "source_file": f"{company}.pdf", "is_table": True,
    }


def test_retrieve_financial_summary_prefers_first_topic_over_fallback(monkeypatch):
    metadata = [_chunk("Acme", "SUMMARY OF FINANCIAL INFORMATION")]
    store = VectorStore.build(metadata, [[1.0, 0.0]])
    retriever = Retriever(store)

    result = _retrieve_financial_summary(retriever, [1.0, 0.0], "Acme")
    assert result["section"] == "SUMMARY OF FINANCIAL INFORMATION"


def test_retrieve_financial_summary_falls_back_to_restated_financials(monkeypatch):
    metadata = [_chunk("Acme", "SECTION V – RESTATED FINANCIAL INFORMATION")]
    store = VectorStore.build(metadata, [[1.0, 0.0]])
    retriever = Retriever(store)

    result = _retrieve_financial_summary(retriever, [1.0, 0.0], "Acme")
    assert result is not None
    assert result["company"] == "Acme"


def test_retrieve_financial_summary_returns_none_when_company_has_no_financials():
    metadata = [_chunk("Acme", "RISK FACTORS")]
    store = VectorStore.build(metadata, [[1.0, 0.0]])
    retriever = Retriever(store)

    assert _retrieve_financial_summary(retriever, [1.0, 0.0], "Acme") is None


def test_compare_companies_no_summaries_short_circuits_without_llm_call(monkeypatch):
    # No company has financial-topic chunks -- must not call Ollama.
    monkeypatch.setattr("src.cross_company.embed_query", lambda q: [1.0, 0.0])
    metadata = [_chunk("Acme", "RISK FACTORS")]
    store = VectorStore.build(metadata, [[1.0, 0.0]])
    retriever = Retriever(store)

    answer, sources = compare_companies("which company has the best financials?", retriever)
    assert sources == []
    assert "couldn't find" in answer.lower()

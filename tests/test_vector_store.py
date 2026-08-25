from __future__ import annotations

import numpy as np
import pytest

from src.vector_store import VectorStore


def _meta(idx: int, company: str = "Acme", section: str | None = None) -> dict:
    return {"text": f"chunk {idx}", "company": company, "section": section, "chunk_index": idx}


def test_build_normalizes_embeddings():
    metadata = [_meta(0), _meta(1)]
    embeddings = [[3.0, 4.0], [1.0, 0.0]]  # norms 5 and 1
    store = VectorStore.build(metadata, embeddings)

    assert np.allclose(store.embeddings[0], [0.6, 0.8])
    assert np.allclose(store.embeddings[1], [1.0, 0.0])


def test_build_raises_on_length_mismatch():
    with pytest.raises(ValueError):
        VectorStore.build([_meta(0)], [[1.0, 0.0], [0.0, 1.0]])


def test_add_appends_and_normalizes_new_chunks():
    store = VectorStore.build([_meta(0)], [[1.0, 0.0]])

    store.add([_meta(1, company="Globex")], [[3.0, 4.0]])

    assert len(store.metadata) == 2
    assert store.embeddings.shape == (2, 2)
    assert np.allclose(store.embeddings[1], [0.6, 0.8])


def test_add_is_searchable_immediately():
    store = VectorStore.build([_meta(0, company="Acme")], [[1.0, 0.0]])
    store.add([_meta(1, company="Globex")], [[0.0, 1.0]])

    results = store.search([0.0, 1.0], top_k=1)
    assert results[0][0]["company"] == "Globex"


def test_add_raises_on_length_mismatch():
    store = VectorStore.build([_meta(0)], [[1.0, 0.0]])
    with pytest.raises(ValueError):
        store.add([_meta(1)], [[1.0, 0.0], [0.0, 1.0]])


def test_add_empty_is_a_noop():
    store = VectorStore.build([_meta(0)], [[1.0, 0.0]])
    store.add([], [])
    assert len(store.metadata) == 1
    assert store.embeddings.shape == (1, 2)


def test_search_ranks_by_cosine_similarity():
    # Three orthogonal-ish 2D vectors; query closest to index 0.
    metadata = [_meta(0), _meta(1), _meta(2)]
    embeddings = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]
    store = VectorStore.build(metadata, embeddings)

    results = store.search([0.9, 0.1], top_k=3)

    assert [m["chunk_index"] for m, _ in results] == [0, 1, 2]
    assert results[0][1] > results[1][1] > results[2][1]


def test_search_respects_top_k():
    metadata = [_meta(i) for i in range(5)]
    embeddings = [[1.0, 0.0]] * 5
    store = VectorStore.build(metadata, embeddings)

    results = store.search([1.0, 0.0], top_k=2)
    assert len(results) == 2


def test_search_filters_by_company():
    metadata = [_meta(0, company="Acme"), _meta(1, company="Globex")]
    embeddings = [[1.0, 0.0], [1.0, 0.0]]  # identical similarity
    store = VectorStore.build(metadata, embeddings)

    results = store.search([1.0, 0.0], top_k=5, company="Globex")
    assert [m["chunk_index"] for m, _ in results] == [1]


def test_search_filters_by_company_and_section():
    metadata = [
        _meta(0, company="Acme", section="Risk Factors"),
        _meta(1, company="Acme", section="Business"),
        _meta(2, company="Globex", section="Risk Factors"),
    ]
    embeddings = [[1.0, 0.0]] * 3
    store = VectorStore.build(metadata, embeddings)

    results = store.search([1.0, 0.0], top_k=5, company="Acme", section="Risk Factors")
    assert [m["chunk_index"] for m, _ in results] == [0]


def test_search_returns_empty_when_no_candidates_match_filter():
    metadata = [_meta(0, company="Acme")]
    embeddings = [[1.0, 0.0]]
    store = VectorStore.build(metadata, embeddings)

    assert store.search([1.0, 0.0], company="NoSuchCompany") == []


def test_search_filters_by_section_topic_across_issue_offer_synonyms():
    metadata = [
        _meta(0, company="Acme", section="SECTION III – OBJECTS OF THE ISSUE"),
        _meta(1, company="Globex", section="OBJECTS OF THE OFFER"),
        _meta(2, company="Initech", section="CAPITAL STRUCTURE"),
    ]
    embeddings = [[1.0, 0.0]] * 3
    store = VectorStore.build(metadata, embeddings)

    results = store.search([1.0, 0.0], top_k=5, section_topic="OBJECTS OF THE ISSUE")
    assert {m["chunk_index"] for m, _ in results} == {0, 1}


def test_save_and_load_round_trip(tmp_path):
    metadata = [_meta(0), _meta(1, section="Risk Factors")]
    embeddings = [[3.0, 4.0], [1.0, 1.0]]
    store = VectorStore.build(metadata, embeddings)

    store.save(tmp_path)
    loaded = VectorStore.load(tmp_path)

    assert np.allclose(loaded.embeddings, store.embeddings)
    assert loaded.metadata == store.metadata

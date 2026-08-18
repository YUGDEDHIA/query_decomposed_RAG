"""Embed chunk text via Ollama's nomic-embed-text, batched.

nomic-embed-text is trained with task-instruction prefixes; skipping them
measurably degrades retrieval quality (this isn't optional tuning, it's how
the model was trained). Indexed chunks get "search_document: ", queries at
retrieval time get "search_query: " -- callers should never prepend a
prefix themselves, use embed_documents/embed_query so it can't be forgotten
on one side.
"""
from __future__ import annotations

import ollama

MODEL = "nomic-embed-text"
DOCUMENT_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "
DEFAULT_BATCH_SIZE = 64


def embed_documents(texts: list[str], batch_size: int = DEFAULT_BATCH_SIZE, on_batch=None) -> list[list[float]]:
    """Embed chunk texts for indexing, one vector per input, in order.

    on_batch(num_done, total), if given, is called after each batch --
    useful for progress logging on large corpora.
    """
    embeddings: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        prefixed = [DOCUMENT_PREFIX + t for t in batch]
        response = ollama.embed(model=MODEL, input=prefixed)
        embeddings.extend(response.embeddings)
        if on_batch:
            on_batch(len(embeddings), len(texts))
    return embeddings


def embed_query(text: str) -> list[float]:
    """Embed a single user query for retrieval."""
    response = ollama.embed(model=MODEL, input=QUERY_PREFIX + text)
    return list(response.embeddings[0])

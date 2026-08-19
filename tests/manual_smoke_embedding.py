"""Manual sanity check for embed_documents/embed_query + VectorStore.search
against real processed chunks. Needs a live Ollama with nomic-embed-text
pulled (`ollama pull nomic-embed-text`). Run directly:

    python tests/manual_smoke_embedding.py [query text]

Defaults to embedding a small sample from data/processed/ and running one
query so you can eyeball whether the top results look right.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.build_index import load_all_chunks
from src.embedder import embed_documents, embed_query
from src.vector_store import VectorStore

SAMPLE_SIZE = 200


def main() -> None:
    processed_dir = Path(__file__).resolve().parent.parent / "data" / "processed"
    chunks = load_all_chunks(processed_dir)
    if not chunks:
        sys.exit("No chunks in data/processed/ -- run `python -m src.ingest` first.")

    sample = chunks[:SAMPLE_SIZE]
    print(f"Embedding {len(sample)} chunks (of {len(chunks)} total)...")
    embeddings = embed_documents([c["text"] for c in sample])
    store = VectorStore.build(sample, embeddings)
    print(f"Built store: {store.embeddings.shape}")

    query = " ".join(sys.argv[1:]) or "what are the risk factors related to raw material prices?"
    print(f"\nQuery: {query!r}")
    query_vec = embed_query(query)
    results = store.search(query_vec, top_k=5)

    for meta, score in results:
        preview = meta["text"][:150].replace("\n", " ")
        print(f"\n--- score={score:.4f} company={meta['company']!r} section={meta['section']!r} ---")
        print(preview + ("..." if len(meta["text"]) > 150 else ""))


if __name__ == "__main__":
    main()

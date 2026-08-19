"""Build the numpy vector-store index from data/processed/*.jsonl.

Reads every chunk written by src/ingest.py, embeds its text via
src/embedder.py (batched, nomic-embed-text with the search_document prefix),
and persists a VectorStore to data/index/ (gitignored -- regenerate by
re-running this rather than tracking it).

Usage:
    python -m src.build_index
    python -m src.build_index --limit 500   # smoke test on the first N chunks
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from src.embedder import DEFAULT_BATCH_SIZE, embed_documents
from src.vector_store import VectorStore

logger = logging.getLogger("build_index")


def load_all_chunks(processed_dir: Path) -> list[dict]:
    """All chunks across every data/processed/*.jsonl file, in file order."""
    chunks = []
    for path in sorted(processed_dir.glob("*.jsonl")):
        with open(path) as f:
            for line in f:
                chunks.append(json.loads(line))
    return chunks


def build_index(
    processed_dir: Path,
    out_dir: Path,
    limit: int | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> None:
    chunks = load_all_chunks(processed_dir)
    if limit is not None:
        chunks = chunks[:limit]
    if not chunks:
        logger.warning("No chunks found in %s -- run src.ingest first.", processed_dir)
        return

    logger.info("Embedding %d chunks (batch size %d)...", len(chunks), batch_size)
    texts = [c["text"] for c in chunks]
    embeddings = embed_documents(
        texts, batch_size=batch_size,
        on_batch=lambda done, total: logger.info("  %d/%d embedded", done, total),
    )

    store = VectorStore.build(chunks, embeddings)
    store.save(out_dir)
    logger.info("Wrote index for %d chunks to %s", len(chunks), out_dir)


def main() -> None:
    default_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--processed-dir", type=Path, default=default_root / "data" / "processed")
    parser.add_argument("--out-dir", type=Path, default=default_root / "data" / "index")
    parser.add_argument("--limit", type=int, default=None, help="Only embed the first N chunks (for testing)")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    build_index(args.processed_dir, args.out_dir, limit=args.limit, batch_size=args.batch_size)


if __name__ == "__main__":
    main()

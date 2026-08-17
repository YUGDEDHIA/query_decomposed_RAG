"""Chunk every RHP in data/raw/ into text + table Chunks, section-tagged, and persist them.

Ties together load_pdf, chunk_text, extract_table_chunks, and
detect_sections/assign_sections -- each independently testable and
runnable on its own (see their manual_smoke_*.py scripts) -- into one
per-document pass. Output is one JSONL file per company under
data/processed/ (gitignored -- regenerate by re-running this rather than
tracking it), so a later ingestion step can glob and embed them.

Text and table chunks share one continuous chunk_index per document
(table chunks appended after text chunks, then renumbered) rather than
each keeping its own independent 0-based sequence.

Note: JSON has no tuple type, so `page_range` round-trips through the
JSONL file as a 2-element list, not a tuple -- something a future reader
needs to account for.

Embeddings / vector-store indexing are a separate, later step; this only
produces chunks.

Usage:
    python -m src.ingest                # all PDFs in data/raw/
    python -m src.ingest --limit 3       # smoke test on a few
    python -m src.ingest --overwrite     # re-chunk documents already done
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from pathlib import Path

from src.chunker import Chunk, chunk_text
from src.loader import company_from_filename, load_pdf
from src.section_detector import assign_sections, detect_sections
from src.table_extractor import extract_table_chunks

logger = logging.getLogger("ingest")


def _combine_and_renumber(text_chunks: list[Chunk], table_chunks: list[Chunk]) -> list[Chunk]:
    combined = text_chunks + table_chunks
    for i, chunk in enumerate(combined):
        chunk.chunk_index = i
    return combined


def process_pdf(path: Path, chunk_size: int = 600, overlap: int = 80) -> list[Chunk]:
    """Load, chunk (text + tables), and section-tag one RHP."""
    company = company_from_filename(str(path))

    pages = load_pdf(str(path))
    text_chunks = chunk_text(
        pages, company=company, source_file=path.name, chunk_size=chunk_size, overlap=overlap
    )
    table_chunks = extract_table_chunks(str(path), company=company, source_file=path.name, max_words=chunk_size)
    chunks = _combine_and_renumber(text_chunks, table_chunks)

    sections = detect_sections(str(path))  # None is fine -- assign_sections no-ops
    assign_sections(chunks, sections)

    return chunks


def write_chunks(chunks: list[Chunk], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for chunk in chunks:
            f.write(json.dumps(dataclasses.asdict(chunk)) + "\n")


def ingest_corpus(
    raw_dir: Path,
    out_dir: Path,
    overwrite: bool = False,
    limit: int | None = None,
    chunk_size: int = 600,
    overlap: int = 80,
) -> None:
    pdf_paths = sorted(raw_dir.glob("*.pdf"))
    if limit is not None:
        pdf_paths = pdf_paths[:limit]

    processed, skipped, failed = 0, 0, 0
    total_chunks, total_tables, sectioned_docs = 0, 0, 0

    for i, path in enumerate(pdf_paths, start=1):
        out_path = out_dir / f"{path.stem}.jsonl"
        if out_path.exists() and not overwrite:
            logger.info("[%d/%d] %s already chunked, skipping", i, len(pdf_paths), path.name)
            skipped += 1
            continue

        logger.info("[%d/%d] chunking %s", i, len(pdf_paths), path.name)
        try:
            chunks = process_pdf(path, chunk_size=chunk_size, overlap=overlap)
        except Exception as exc:
            logger.warning("Failed to process %s: %s", path.name, exc)
            failed += 1
            continue

        write_chunks(chunks, out_path)
        n_tables = sum(1 for c in chunks if c.is_table)
        n_sectioned = sum(1 for c in chunks if c.section is not None)
        processed += 1
        total_chunks += len(chunks)
        total_tables += n_tables
        if n_sectioned:
            sectioned_docs += 1
        logger.info(
            "  -> %d chunks (%d table) written to %s (%d/%d section-tagged)",
            len(chunks), n_tables, out_path, n_sectioned, len(chunks),
        )

    logger.info(
        "Done. %d processed, %d skipped, %d failed (of %d PDFs). "
        "%d total chunks (%d table), %d/%d docs had sections detected.",
        processed, skipped, failed, len(pdf_paths),
        total_chunks, total_tables, sectioned_docs, processed,
    )


def main() -> None:
    default_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", type=Path, default=default_root / "data" / "raw")
    parser.add_argument("--out-dir", type=Path, default=default_root / "data" / "processed")
    parser.add_argument("--overwrite", action="store_true", help="Re-chunk documents that already have output")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N PDFs (for testing)")
    parser.add_argument("--chunk-size", type=int, default=600)
    parser.add_argument("--overlap", type=int, default=80)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    # pdfminer (pdfplumber's backend) logs every parsed PDF token at DEBUG --
    # shares the root logger, so -v would otherwise flood stdout with it.
    logging.getLogger("pdfminer").setLevel(logging.WARNING)

    ingest_corpus(
        args.raw_dir, args.out_dir,
        overwrite=args.overwrite, limit=args.limit,
        chunk_size=args.chunk_size, overlap=args.overlap,
    )


if __name__ == "__main__":
    main()

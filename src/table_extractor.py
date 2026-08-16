"""Detect tables in an RHP PDF and flatten each into readable text chunks.

pdfplumber's table detector also fires constantly on ordinary running text
that happens to be broken up by whitespace -- on a real RHP, most of what it
finds is noise (single-row, 2-4 cell fragments). _is_real_table filters that
out empirically: requiring at least MIN_COLS columns and MIN_CELLS total
cells roughly halves the false-positive count on real filings while keeping
genuine tables (glossaries, financial statements, shareholding tables).

Known limitations, not solved here:
- pdfplumber detects tables per page and sometimes fragments one borderless
  table into several adjacent single-row "tables" -- no fragment stitching.
- No cross-page table merging -- a table split across a page break comes
  back as two separate chunks.
- Flattening drops column alignment (empty cells are dropped rather than
  positionally preserved), trading exact structure for robustness against
  the inconsistent, multi-row wrapped headers these filings actually use.
  Good enough for an LLM to read a number in context; not a numeric-analysis
  data structure.
"""
from __future__ import annotations

import re

import pdfplumber

from src.chunker import Chunk

MIN_COLS = 2
MIN_CELLS = 8

_WHITESPACE_RE = re.compile(r"\s+")


def _clean_cell(cell) -> str:
    if cell is None:
        return ""
    return _WHITESPACE_RE.sub(" ", str(cell)).strip()


def _is_real_table(table: list[list]) -> bool:
    rows = len(table)
    cols = max((len(row) for row in table), default=0)
    return cols >= MIN_COLS and rows * cols >= MIN_CELLS


def _flatten_table_lines(table: list[list]) -> list[str]:
    """One string per row: non-empty cells joined with " | ", empty rows dropped."""
    lines = []
    for row in table:
        cells = [c for c in (_clean_cell(cell) for cell in row) if c]
        if cells:
            lines.append(" | ".join(cells))
    return lines


def _group_lines(lines: list[str], max_words: int) -> list[str]:
    """Group row-lines into text blocks of at most max_words words each.

    Splits only between rows, never mid-row -- unlike a run-on sentence, a
    table row cut in half loses its meaning. A single row bigger than
    max_words on its own still becomes its own (oversized) group; there's
    no meaningful way to split it further.
    """
    groups = []
    current: list[str] = []
    current_words = 0
    for line in lines:
        n_words = len(line.split())
        if current and current_words + n_words > max_words:
            groups.append("\n".join(current))
            current, current_words = [], 0
        current.append(line)
        current_words += n_words
    if current:
        groups.append("\n".join(current))
    return groups


def extract_table_chunks(path: str, company: str, source_file: str, max_words: int = 600) -> list[Chunk]:
    """Find real tables in the PDF at `path`, splitting any bigger than max_words.

    A table's flattened text has no inherent size limit -- on real RHPs
    these run up to ~2700 words, well past what an embedding model will
    accept without silently truncating. Splitting at row boundaries keeps
    each chunk under max_words (default matches chunk_text's default) while
    never cutting a row in half.
    """
    chunks: list[Chunk] = []
    with pdfplumber.open(path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            for table in page.extract_tables():
                if not _is_real_table(table):
                    continue
                lines = _flatten_table_lines(table)
                if not lines:
                    continue
                for group_text in _group_lines(lines, max_words):
                    chunks.append(Chunk(
                        text=group_text,
                        company=company,
                        source_file=source_file,
                        page_range=(page_number, page_number),
                        chunk_index=len(chunks),
                        is_table=True,
                    ))
    return chunks

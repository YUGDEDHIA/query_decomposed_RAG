"""Turn per-page RHP text into overlapping, sentence-boundary-aware chunks."""
from __future__ import annotations

import re
from dataclasses import dataclass

_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    text: str
    company: str
    source_file: str
    page_range: tuple[int, int]  # (start_page, end_page), 1-indexed, inclusive
    chunk_index: int
    section: str | None = None  # set by section_detector.assign_sections, if detected


def _split_into_units(pages: list[str]) -> list[tuple[str, int]]:
    """Break page text into (sentence, page_number) units, 1-indexed pages.

    Splitting on paragraphs first, then sentences, means chunk_text only
    ever needs to pack whole sentences -- it never has to cut mid-sentence
    itself.
    """
    units = []
    for page_number, page_text in enumerate(pages, start=1):
        if not page_text.strip():
            continue
        for paragraph in _PARAGRAPH_SPLIT_RE.split(page_text):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            for sentence in _SENTENCE_SPLIT_RE.split(paragraph):
                sentence = sentence.strip()
                if sentence:
                    units.append((sentence, page_number))
    return units


def _atomize(units: list[tuple[str, int]], chunk_size: int) -> list[tuple[str, int]]:
    """Hard-split any single sentence longer than chunk_size by word count.

    A run-on sentence with no punctuation (or a mis-split OCR artifact)
    could otherwise be longer than chunk_size on its own, which would
    violate the "no chunk exceeds chunk_size" guarantee.
    """
    atomic = []
    for sentence, page_number in units:
        words = sentence.split()
        if len(words) <= chunk_size:
            atomic.append((sentence, page_number))
            continue
        for i in range(0, len(words), chunk_size):
            atomic.append((" ".join(words[i:i + chunk_size]), page_number))
    return atomic


def _make_chunk(unit_group: list[tuple[str, int]], company: str, source_file: str, chunk_index: int) -> Chunk:
    text = " ".join(sentence for sentence, _ in unit_group)
    pages_in_chunk = [page_number for _, page_number in unit_group]
    return Chunk(
        text=text,
        company=company,
        source_file=source_file,
        page_range=(min(pages_in_chunk), max(pages_in_chunk)),
        chunk_index=chunk_index,
    )


def _carry_for_overlap(unit_group: list[tuple[str, int]], overlap: int) -> list[tuple[str, int]]:
    """Trailing whole sentences from unit_group totalling at most `overlap` words.

    A unit bigger than `overlap` on its own (e.g. a hard-split piece of an
    oversized sentence) is never carried, even as the sole item -- doing so
    would let the next chunk exceed chunk_size before a single new sentence
    is even added.
    """
    carry: list[tuple[str, int]] = []
    carry_words = 0
    for sentence, page_number in reversed(unit_group):
        n_words = len(sentence.split())
        if carry_words + n_words > overlap:
            break
        carry.insert(0, (sentence, page_number))
        carry_words += n_words
    return carry


def chunk_text(
    pages: list[str],
    company: str,
    source_file: str,
    chunk_size: int = 600,
    overlap: int = 80,
) -> list[Chunk]:
    """Greedily pack sentences into chunks of at most `chunk_size` words.

    Chunks flow across page boundaries -- RHP paragraphs don't respect page
    breaks -- with `page_range` recording which page(s) each chunk drew
    from. Each new chunk starts with up to `overlap` words carried over
    (whole trailing sentences, not a raw word slice) from the previous one.
    """
    assert overlap < chunk_size, "overlap must be smaller than chunk_size"

    units = _atomize(_split_into_units(pages), chunk_size)

    chunks: list[Chunk] = []
    current: list[tuple[str, int]] = []
    current_word_count = 0

    for sentence, page_number in units:
        n_words = len(sentence.split())
        if current and current_word_count + n_words > chunk_size:
            chunks.append(_make_chunk(current, company, source_file, len(chunks)))
            current = _carry_for_overlap(current, overlap)
            current_word_count = sum(len(s.split()) for s, _ in current)

        current.append((sentence, page_number))
        current_word_count += n_words

    if current:
        chunks.append(_make_chunk(current, company, source_file, len(chunks)))

    return chunks

"""Detect RHP section boundaries from the document's own table of contents.

RHPs are SEBI-templated: every filing has a dotted-leader ToC
("SECTION II - RISK FACTORS .......... 27") and a printed page number in
the footer of almost every page. Combining the two gives real,
page-accurate section boundaries without any per-template heading-style
tuning (font size, bold, etc. -- which varies a lot; the ToC/footer format
doesn't).

Verified against 37 real RHPs from different merchant-banker templates:
the ToC page is reliably found by dotted-leader line *density*, not by
matching specific heading text (some filings title it "CONTENTS", not
"TABLE OF CONTENTS"). Footer page numbers were found on 85-100% of pages
per document, in two formats ("5" or "Page 5 of 465").

detect_sections() returns None (not a crash) when a document doesn't fit
the pattern -- callers should treat section tagging as best-effort, not a
guarantee.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import pdfplumber

TOC_SCAN_PAGES = 20
MIN_TOC_LINES = 10

_TOC_LINE_RE = re.compile(r"^(.*?)[.…\s]{4,}(\d{1,4})\s*$")
_FOOTER_BARE_RE = re.compile(r"^(\d{1,4})$")
_FOOTER_OF_RE = re.compile(r"^Page\s+(\d{1,4})\s+of\s+\d{1,4}$", re.IGNORECASE)
_TOC_HEADING_TITLES = {"TABLE OF CONTENTS", "CONTENTS"}


@dataclass
class Section:
    title: str
    start_page: int  # PDF page index, 1-indexed
    end_page: int  # inclusive


def _toc_line_count(text: str) -> int:
    return sum(1 for line in text.split("\n") if _TOC_LINE_RE.match(line.strip()))


def _parse_toc_entries(text: str) -> list[tuple[str, int]]:
    """(title, printed_page_number) pairs from a ToC page's raw text.

    A title that wraps onto its own line before the dotted leader + page
    number shows up (common for long section names) is stitched onto the
    following matched line rather than dropped.
    """
    entries = []
    pending_prefix = ""
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        if line.upper() in _TOC_HEADING_TITLES:
            continue
        match = _TOC_LINE_RE.match(line)
        if not match:
            pending_prefix = f"{pending_prefix} {line}".strip()
            continue
        title = re.sub(r"[.…\s]+$", "", match.group(1)).strip()
        if pending_prefix:
            title = f"{pending_prefix} {title}".strip()
            pending_prefix = ""
        if title and title.upper() not in _TOC_HEADING_TITLES:
            entries.append((title, int(match.group(2))))
    return entries


def _find_toc_page(pdf) -> int | None:
    """Page with the densest dotted-leader pattern in the first TOC_SCAN_PAGES pages."""
    best_page, best_count = None, 0
    for pno in range(1, min(TOC_SCAN_PAGES, len(pdf.pages)) + 1):
        count = _toc_line_count(pdf.pages[pno - 1].extract_text() or "")
        if count > best_count:
            best_page, best_count = pno, count
    return best_page if best_count >= MIN_TOC_LINES else None


def _footer_page_number(text: str) -> int | None:
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    if not lines:
        return None
    last = lines[-1]
    match = _FOOTER_BARE_RE.match(last) or _FOOTER_OF_RE.match(last)
    return int(match.group(1)) if match else None


def _monotonic_anchors(printed_by_page: list[tuple[int, int | None]]) -> list[tuple[int, int]]:
    """(printed_page, pdf_page) anchors, keeping only strictly-increasing printed numbers.

    A footer read that goes backwards or repeats is most likely a table
    cell that happened to be the last text on the page, not a real page
    number -- RHP pagination runs as a single increasing sequence
    throughout the document, so it's dropped rather than trusted.
    """
    anchors = []
    last_printed = -1
    for pdf_page, printed in printed_by_page:
        if printed is not None and printed > last_printed:
            anchors.append((printed, pdf_page))
            last_printed = printed
    return anchors


def _map_printed_to_pdf_page(printed_page: int, anchors: list[tuple[int, int]]) -> int:
    """Nearest anchor at or before printed_page, offset by the gap between them."""
    anchor_printed, anchor_pdf = anchors[0]
    for a_printed, a_pdf in anchors:
        if a_printed > printed_page:
            break
        anchor_printed, anchor_pdf = a_printed, a_pdf
    return anchor_pdf + (printed_page - anchor_printed)


def _build_sections(
    entries: list[tuple[str, int]],
    anchors: list[tuple[int, int]],
    total_pages: int,
) -> list[Section]:
    starts = [_map_printed_to_pdf_page(printed, anchors) for _, printed in entries]
    sections = []
    for i, (title, _) in enumerate(entries):
        start = starts[i]
        end = starts[i + 1] - 1 if i + 1 < len(starts) else total_pages
        sections.append(Section(title=title, start_page=start, end_page=max(end, start)))
    return sections


def detect_sections(path: str) -> list[Section] | None:
    """Parse the ToC and return section page ranges, or None if the document doesn't fit the pattern."""
    with pdfplumber.open(path) as pdf:
        toc_page = _find_toc_page(pdf)
        if toc_page is None:
            return None

        entries = _parse_toc_entries(pdf.pages[toc_page - 1].extract_text() or "")
        if len(entries) < MIN_TOC_LINES:
            return None

        printed_by_page = [
            (pno, _footer_page_number(page.extract_text() or ""))
            for pno, page in enumerate(pdf.pages, start=1)
        ]
        anchors = _monotonic_anchors(printed_by_page)
        if not anchors:
            return None

        total_pages = len(pdf.pages)

    return _build_sections(entries, anchors, total_pages)


def assign_sections(chunks: list, sections: list[Section] | None) -> list:
    """Tag each chunk's `.section` from the range its page_range starts in.

    No-op (chunks left with section=None) if sections is None -- callers
    don't need to special-case documents where ToC detection failed.
    """
    if not sections:
        return chunks
    for chunk in chunks:
        page = chunk.page_range[0]
        match = None
        for section in sections:
            if section.start_page <= page:
                match = section
            else:
                break
        chunk.section = match.title if match else None
    return chunks

from src.chunker import Chunk
from src.section_detector import (
    Section,
    _build_sections,
    _footer_page_number,
    _map_printed_to_pdf_page,
    _monotonic_anchors,
    _parse_toc_entries,
    _toc_line_count,
    assign_sections,
)


def test_parse_toc_entries_basic():
    text = (
        "TABLE OF CONTENTS\n"
        "SECTION I – GENERAL .............................................. 1\n"
        "DEFINITIONS AND ABBREVIATIONS .................................... 1\n"
        "SECTION II – RISK FACTORS ......................................... 27\n"
    )
    entries = _parse_toc_entries(text)
    assert entries == [
        ("SECTION I – GENERAL", 1),
        ("DEFINITIONS AND ABBREVIATIONS", 1),
        ("SECTION II – RISK FACTORS", 27),
    ]


def test_parse_toc_entries_stitches_wrapped_titles():
    text = (
        "CERTAIN CONVENTIONS, PRESENTATION OF FINANCIAL, INDUSTRY AND MARKET DATA\n"
        "AND CURRENCY OF PRESENTATION .................................... 20\n"
    )
    entries = _parse_toc_entries(text)
    assert entries == [
        ("CERTAIN CONVENTIONS, PRESENTATION OF FINANCIAL, INDUSTRY AND MARKET DATA "
         "AND CURRENCY OF PRESENTATION", 20),
    ]


def test_parse_toc_entries_handles_alternate_leader_formats():
    # "Page N of M"-style docs also use inconsistent dot/ellipsis leaders
    # with stray spaces (seen on real filings).
    text = "FORWARD LOOKING STATEMENTS…………………………………… . ………………..85\n"
    entries = _parse_toc_entries(text)
    assert entries == [("FORWARD LOOKING STATEMENTS", 85)]


def test_toc_line_count():
    text = "not a toc line\nA .......... 1\nB .......... 2\nalso not one"
    assert _toc_line_count(text) == 2


def test_footer_page_number_bare():
    assert _footer_page_number("some text\nmore text\n42") == 42


def test_footer_page_number_page_of_format():
    assert _footer_page_number("some text\nPage 3 of 465") == 3


def test_footer_page_number_missing():
    assert _footer_page_number("just running text, no footer") is None
    assert _footer_page_number("") is None


def test_monotonic_anchors_drops_non_increasing_reads():
    # page 5 reports "3" (a stray table value, not a real footer) -- should
    # be dropped rather than corrupting the mapping.
    printed_by_page = [(1, 1), (2, 2), (3, 3), (4, None), (5, 3), (6, 6)]
    assert _monotonic_anchors(printed_by_page) == [(1, 1), (2, 2), (3, 3), (6, 6)]


def test_map_printed_to_pdf_page_uses_nearest_preceding_anchor():
    anchors = [(1, 5), (10, 14)]  # printed page 1 -> pdf page 5, printed 10 -> pdf 14
    assert _map_printed_to_pdf_page(1, anchors) == 5
    assert _map_printed_to_pdf_page(5, anchors) == 9  # offset by 4, same as anchor
    assert _map_printed_to_pdf_page(10, anchors) == 14


def test_build_sections_computes_ranges_from_next_entrys_start():
    entries = [("GENERAL", 1), ("RISK FACTORS", 27), ("INTRODUCTION", 63)]
    anchors = [(1, 5), (27, 31), (63, 67)]  # constant +4 offset
    sections = _build_sections(entries, anchors, total_pages=200)
    assert sections == [
        Section(title="GENERAL", start_page=5, end_page=30),
        Section(title="RISK FACTORS", start_page=31, end_page=66),
        Section(title="INTRODUCTION", start_page=67, end_page=200),
    ]


def _chunk(page: int) -> Chunk:
    return Chunk(text="x", company="Acme", source_file="acme.pdf", page_range=(page, page), chunk_index=0)


def test_assign_sections_tags_by_page_range_start():
    sections = [
        Section(title="GENERAL", start_page=5, end_page=30),
        Section(title="RISK FACTORS", start_page=31, end_page=66),
    ]
    chunks = [_chunk(5), _chunk(29), _chunk(31), _chunk(50)]
    assign_sections(chunks, sections)
    assert [c.section for c in chunks] == ["GENERAL", "GENERAL", "RISK FACTORS", "RISK FACTORS"]


def test_assign_sections_before_first_section_gets_none():
    sections = [Section(title="GENERAL", start_page=5, end_page=30)]
    chunks = [_chunk(1)]
    assign_sections(chunks, sections)
    assert chunks[0].section is None


def test_assign_sections_is_noop_when_sections_is_none():
    chunks = [_chunk(1)]
    assign_sections(chunks, None)
    assert chunks[0].section is None

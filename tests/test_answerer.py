from src.answerer import SubAnswer, _format_excerpt, _sources_from_chunks, answer


def _chunk(company="Acme", section="Risk Factors", page=5, is_table=False, text="some text"):
    return {
        "text": text,
        "company": company,
        "section": section,
        "source_file": "acme.pdf",
        "page_range": [page, page],
        "chunk_index": 0,
        "is_table": is_table,
    }


def test_format_excerpt_includes_company_section_page():
    chunk = _chunk(company="Acme", section="Risk Factors", page=5)
    result = _format_excerpt(1, chunk)
    assert result.split("\n")[0] == "[Excerpt 1] Acme - Risk Factors (page 5)"
    assert "some text" in result


def test_format_excerpt_omits_section_when_none():
    chunk = _chunk(section=None)
    result = _format_excerpt(1, chunk)
    assert result.split("\n")[0] == "[Excerpt 1] Acme (page 5)"


def test_format_excerpt_marks_table_chunks():
    chunk = _chunk(is_table=True)
    result = _format_excerpt(1, chunk)
    assert result.split("\n")[0] == "[Excerpt 1] Acme - Risk Factors (page 5) [table]"


def test_sources_from_chunks_extracts_expected_fields():
    chunks = [_chunk(company="Acme", section="Risk Factors", page=5)]
    assert _sources_from_chunks(chunks) == [
        {"company": "Acme", "section": "Risk Factors", "page_range": [5, 5], "source_file": "acme.pdf"}
    ]


def test_answer_with_no_chunks_short_circuits_without_llm_call():
    # No Ollama call happens here -- this must work without a live model.
    result = answer("What is the revenue?", [])
    assert isinstance(result, SubAnswer)
    assert result.grounded is False
    assert result.sources == []
    assert "No relevant excerpts" in result.answer

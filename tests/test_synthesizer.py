from __future__ import annotations

from src.answerer import SubAnswer
from src.synthesizer import _dedupe_sources, _format_sub_answer, synthesize


def _source(company="Acme", section="Risk Factors", page=5, source_file="acme.pdf") -> dict:
    return {"company": company, "section": section, "page_range": [page, page], "source_file": source_file}


def test_format_sub_answer_grounded():
    sa = SubAnswer(question="Q1", answer="A1", grounded=True, sources=[])
    assert _format_sub_answer(1, sa) == 'Sub-question 1: "Q1"\nAnswer (grounded): A1'


def test_format_sub_answer_not_grounded():
    sa = SubAnswer(question="Q1", answer="A1", grounded=False, sources=[])
    assert "NOT grounded" in _format_sub_answer(1, sa)


def test_dedupe_sources_removes_exact_duplicates_across_sub_answers():
    source = _source()
    sa1 = SubAnswer(question="Q1", answer="A1", grounded=True, sources=[source])
    sa2 = SubAnswer(question="Q2", answer="A2", grounded=True, sources=[source])
    assert _dedupe_sources([sa1, sa2]) == [source]


def test_dedupe_sources_keeps_distinct_sources():
    s1 = _source(section="Risk Factors")
    s2 = _source(section="Capital Structure", page=10)
    sa = SubAnswer(question="Q1", answer="A1", grounded=True, sources=[s1, s2])
    assert _dedupe_sources([sa]) == [s1, s2]


def test_synthesize_no_sub_answers_short_circuits_without_llm_call():
    result = synthesize("What is the revenue?", [])
    assert result.fully_grounded is False
    assert result.sub_answers == []
    assert result.sources == []


def test_synthesize_single_sub_answer_short_circuits_without_llm_call():
    sa = SubAnswer(question="What are the risk factors?", answer="Some risks.", grounded=True, sources=[_source()])
    result = synthesize("What are the risk factors?", [sa])
    assert result.answer == "Some risks."
    assert result.fully_grounded is True
    assert result.sources == sa.sources


def test_synthesize_single_ungrounded_sub_answer_propagates_grounding():
    sa = SubAnswer(question="What are the risk factors?", answer="Not found.", grounded=False, sources=[])
    result = synthesize("What are the risk factors?", [sa])
    assert result.fully_grounded is False

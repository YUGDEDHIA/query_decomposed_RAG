"""Combine per-sub-question answers into one coherent final answer, via qwen2.5:14b-instruct.

Takes the original user question and the SubAnswers already produced by
src/answerer.py -- kept decoupled from answering itself, same modular split
as the rest of the pipeline -- and asks the model to weave them into one
answer using only what's already in the sub-answers, not adding outside
knowledge. Grounding and sources are computed deterministically from the
sub-answers, not self-reported by the model, same reasoning as
src/answerer.py.

A single sub-answer has nothing to combine, so synthesize() returns it
directly without an LLM call -- mirrors the planner's own "don't force a
split" behavior for an already-atomic question.
"""
from __future__ import annotations

from dataclasses import dataclass

import ollama

from src.answerer import SubAnswer

MODEL = "qwen2.5:14b-instruct"

_SYSTEM_PROMPT = """You are the final synthesis step of a research assistant that answers questions about IPO Red Herring Prospectus (RHP) filings by decomposing them into sub-questions, answering each independently, and now combining those answers into one response.

You are given the user's original question and a list of sub-questions with their independently-researched answers. Write ONE coherent answer to the original question by combining this information.

Rules:
- Use ONLY the information in the provided sub-answers -- do not add outside knowledge or invent facts beyond what's stated.
- If a sub-answer indicates the information wasn't found, say so plainly in your response rather than omitting that part of the question or guessing.
- Write naturally, as a single well-organized answer -- not just the sub-answers pasted back-to-back.
"""


@dataclass
class FinalAnswer:
    question: str
    answer: str
    fully_grounded: bool
    sub_answers: list[SubAnswer]
    sources: list[dict]
    intent: str = "rhp_query"  # "casual" replies are built directly in src/pipeline.py, bypassing synthesize()


def _format_sub_answer(index: int, sub_answer: SubAnswer) -> str:
    status = "grounded" if sub_answer.grounded else "NOT grounded -- info not found"
    return f'Sub-question {index}: "{sub_answer.question}"\nAnswer ({status}): {sub_answer.answer}'


def _dedupe_sources(sub_answers: list[SubAnswer]) -> list[dict]:
    seen = set()
    sources = []
    for sa in sub_answers:
        for s in sa.sources:
            key = (s["company"], s["section"], tuple(s["page_range"]), s["source_file"])
            if key not in seen:
                seen.add(key)
                sources.append(s)
    return sources


def synthesize(question: str, sub_answers: list[SubAnswer]) -> FinalAnswer:
    """Combine `sub_answers` (from src.answerer.answer) into one final answer to `question`."""
    if not sub_answers:
        return FinalAnswer(question=question, answer="No sub-questions were answered.", fully_grounded=False, sub_answers=[], sources=[])

    if len(sub_answers) == 1:
        sa = sub_answers[0]
        return FinalAnswer(question=question, answer=sa.answer, fully_grounded=sa.grounded, sub_answers=sub_answers, sources=sa.sources)

    formatted = "\n\n".join(_format_sub_answer(i, sa) for i, sa in enumerate(sub_answers, start=1))
    user_message = f'Original question: "{question}"\n\n{formatted}'

    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )

    return FinalAnswer(
        question=question,
        answer=response.message.content,
        fully_grounded=all(sa.grounded for sa in sub_answers),
        sub_answers=sub_answers,
        sources=_dedupe_sources(sub_answers),
    )

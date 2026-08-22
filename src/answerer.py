"""Answer one sub-question grounded strictly in its retrieved chunks, via qwen2.5:14b-instruct.

Takes chunks already retrieved by src/retriever.py -- kept decoupled from
retrieval itself, same modular split as the rest of the pipeline -- and
asks the model to answer using only that context, explicitly refusing to
guess when the context doesn't contain the answer rather than falling back
to outside knowledge (this is a RAG system over regulatory financial
filings; a plausible-sounding hallucinated number is worse than "not
found").

Citations (`sources`) are computed directly from the input chunks, not
self-reported by the model -- that's fully known at the code level and more
reliable than trusting the model to accurately report which excerpts it
drew from.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import ollama

MODEL = "qwen2.5:14b-instruct"

_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "grounded": {"type": "boolean"},
    },
    "required": ["answer", "grounded"],
}

_SYSTEM_PROMPT = """You are answering one focused sub-question as part of a larger research question about IPO Red Herring Prospectus (RHP) filings. You are given a set of retrieved document excerpts -- answer using ONLY the information in these excerpts, not outside knowledge.

Rules:
- If the excerpts contain enough information to answer the question, answer it directly and concisely, grounded strictly in the excerpts. Set "grounded" to true.
- If the excerpts do NOT contain enough information, say so explicitly instead of guessing. Set "grounded" to false.
- Numbers, dates, and figures must come directly from the excerpts, never be estimated or inferred beyond what's stated.
"""


@dataclass
class SubAnswer:
    question: str
    answer: str
    grounded: bool
    sources: list[dict]  # [{"company", "section", "page_range", "source_file"}, ...]


def _format_excerpt(index: int, chunk: dict) -> str:
    label = f"[Excerpt {index}] {chunk['company']}"
    if chunk.get("section"):
        label += f" - {chunk['section']}"
    label += f" (page {chunk['page_range'][0]})"
    if chunk.get("is_table"):
        label += " [table]"
    return f"{label}\n{chunk['text']}"


def _sources_from_chunks(chunks: list[dict]) -> list[dict]:
    return [
        {
            "company": c["company"],
            "section": c.get("section"),
            "page_range": c["page_range"],
            "source_file": c["source_file"],
        }
        for c in chunks
    ]


def answer(sub_question: str, chunks: list[dict]) -> SubAnswer:
    """Answer `sub_question` using only `chunks` (already retrieved) as context."""
    if not chunks:
        return SubAnswer(
            question=sub_question,
            answer="No relevant excerpts were retrieved for this question.",
            grounded=False,
            sources=[],
        )

    excerpts = "\n\n---\n\n".join(_format_excerpt(i, c) for i, c in enumerate(chunks, start=1))
    user_message = f'Sub-question: "{sub_question}"\n\nExcerpts:\n\n{excerpts}'

    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        format=_SCHEMA,
    )
    parsed = json.loads(response.message.content)

    return SubAnswer(
        question=sub_question,
        answer=parsed["answer"],
        grounded=parsed["grounded"],
        sources=_sources_from_chunks(chunks),
    )

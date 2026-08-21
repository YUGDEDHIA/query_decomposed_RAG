"""Decompose a user question into independent sub-questions via qwen2.5:14b-instruct.

Plain RAG embeds the whole question and does one retrieval pass, which
falls apart on compound questions -- a single embedding of the whole thing
sits in a vector-space no-man's-land and retrieves mediocre chunks for
every part of it instead of great chunks for any one part. This planner is
the step before retrieval: split the question into independent,
self-contained sub-questions, each simple enough to retrieve well on its
own.

Uses Ollama's JSON-schema-constrained output (`format=` a JSON schema) so
parsing never has to guess at the model's formatting.
"""
from __future__ import annotations

import json

import ollama

MODEL = "qwen2.5:14b-instruct"

_SCHEMA = {
    "type": "object",
    "properties": {
        "sub_questions": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
        }
    },
    "required": ["sub_questions"],
}

_SYSTEM_PROMPT = """You are the planning step of a retrieval-augmented question answering system over IPO Red Herring Prospectus (RHP) filings.

Break the user's question into a list of independent sub-questions. Each sub-question will be answered separately by retrieving relevant document chunks, so each one must:
- Be self-contained (name the company explicitly if the original question does -- never use pronouns like "it" or "the company" that only make sense next to the other sub-questions)
- Cover exactly one focused topic (one company + one aspect, e.g. one risk category, one financial metric, one section of the filing)
- Stay as close to the user's original wording as possible -- don't invent scope the user didn't ask for

If the question is already a single, focused question, return it unchanged as the only sub-question -- do not force a split.

Examples:

Question: "What are the risk factors for Aastha Spintex Limited?"
{"sub_questions": ["What are the risk factors for Aastha Spintex Limited?"]}

Question: "Compare the objects of the issue for Aastha Spintex and Advit Jewels."
{"sub_questions": ["What are the objects of the issue for Aastha Spintex Limited?", "What are the objects of the issue for Advit Jewels Limited?"]}

Question: "What is Aastha Spintex's revenue growth and what raw materials does it depend on?"
{"sub_questions": ["What is Aastha Spintex Limited's revenue growth?", "What raw materials does Aastha Spintex Limited depend on?"]}
"""


def decompose(question: str) -> list[str]:
    """Split `question` into independent, self-contained sub-questions."""
    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f'Question: "{question}"'},
        ],
        format=_SCHEMA,
    )
    parsed = json.loads(response.message.content)
    sub_questions = [q.strip() for q in parsed["sub_questions"] if q.strip()]
    return sub_questions if sub_questions else [question]

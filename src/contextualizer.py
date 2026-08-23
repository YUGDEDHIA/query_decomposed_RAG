"""Resolve a follow-up message against recent conversation history into a
self-contained standalone question, before anything else in the pipeline runs.

Every downstream stage (src/router.py's classify_intent, src/planner.py's
decompose, src/retriever.py's infer_company, the three live-data answerers,
src/cross_company.py's compare_companies) takes a single question string with
no concept of prior turns, and works correctly as long as that string is
already self-contained -- a follow-up like "what about its financials?" or
"and its GMP?" breaks every one of them the same way (no company to resolve,
nothing meaningful to decompose or infer_company against). Rather than thread
a history parameter through each of those already-working, already-tested
modules individually, this resolves history into a self-contained question
exactly once, upfront, in src/pipeline.py -- everything below it is
unchanged.

Uses Ollama's JSON-schema-constrained output (same pattern as
src/router.py/src/planner.py) so the decision of whether to rewrite at all is
made explicitly by the model as a boolean field, not inferred from whether
the rewritten text "looks different" -- the caller trusts the boolean, not
the rewritten text, to decide whether to use the rewrite. This matters: a
genuinely fresh, unrelated question (a new company named, a casual aside)
must never be dragged into stale prior context, which would be worse than
doing nothing.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import ollama

MODEL = "qwen2.5:14b-instruct"

# Bounds prompt size/latency -- generous headroom for realistic follow-up
# chains without letting the prompt grow unbounded over a long session.
MAX_HISTORY_TURNS = 4
# The entity/topic needed for resolution is almost always in the lead
# sentence(s) of a prior answer -- this avoids ballooning the prompt with
# multi-paragraph GMP/RHP prose for signal that's already been captured.
MAX_ANSWER_CHARS = 500

_SCHEMA = {
    "type": "object",
    "properties": {
        "is_follow_up": {"type": "boolean"},
        "standalone_question": {"type": "string"},
    },
    "required": ["is_follow_up", "standalone_question"],
}

_SYSTEM_PROMPT = """You resolve a follow-up message in an ongoing conversation with a research assistant that answers questions about IPO Red Herring Prospectus (RHP) filings, plus live Grey Market Premium (GMP), subscription/anchor investor, and post-listing performance data.

Given the recent conversation and a new message, decide:
1. Is the new message a follow-up that depends on the conversation to make sense on its own -- e.g. it uses a pronoun ("it", "its", "that IPO", "the company") or an implicit reference ("what about the GMP?", "and the financials?") instead of naming a company/entity explicitly?
2. If so, rewrite it into a standalone question that names the specific company/entity explicitly (from the conversation), while otherwise keeping the new message's own wording and topic as close to the original as possible -- do not add topics, data types, or scope the new message didn't ask for, and do not change what KIND of question it is (a GMP question stays a GMP question, a risk-factors question stays a risk-factors question).

If the new message already names its own company/entity, or is a fresh/unrelated question or remark that doesn't depend on the conversation, it is NOT a follow-up -- leave it alone.

The company/entity needed to resolve a follow-up may appear in either the user's earlier questions or the assistant's earlier answers (e.g. a ranking question's answer might be the first place a specific company name appears).

Examples:

Conversation:
User: What are the risk factors for Acme Industries Limited?
Assistant: Acme Industries Limited faces risks including...
New message: "What about its financials?"
{"is_follow_up": true, "standalone_question": "What are Acme Industries Limited's financials?"}

Conversation:
User: What are the risk factors for Acme Industries Limited?
Assistant: Acme Industries Limited faces risks including...
New message: "And what's the GMP for it?"
{"is_follow_up": true, "standalone_question": "What is the GMP for Acme Industries Limited?"}

Conversation:
User: Which recent IPOs have the best financials?
Assistant: Based on the indexed filings, Beta Textiles Limited stands out with...
New message: "What's its subscription status?"
{"is_follow_up": true, "standalone_question": "What is Beta Textiles Limited's subscription status?"}

Conversation:
User: What are the risk factors for Acme Industries Limited?
Assistant: Acme Industries Limited faces risks including...
New message: "What are the objects of the issue for Advit Jewels Limited?"
{"is_follow_up": false, "standalone_question": "What are the objects of the issue for Advit Jewels Limited?"}

Conversation:
User: What's the GMP for Acme Industries Limited?
Assistant: Acme Industries Limited's GMP is...
New message: "Thanks! By the way, what's the capital of France?"
{"is_follow_up": false, "standalone_question": "What's the capital of France?"}
"""


@dataclass
class Turn:
    question: str
    answer: str


def _truncate(text: str, max_chars: int = MAX_ANSWER_CHARS) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def _format_history(history: list[Turn]) -> str:
    recent = history[-MAX_HISTORY_TURNS:]
    lines = []
    for turn in recent:
        lines.append(f"User: {turn.question}")
        lines.append(f"Assistant: {_truncate(turn.answer)}")
    return "\n".join(lines)


def contextualize(question: str, history: list[Turn]) -> str:
    """`question` unchanged if `history` is empty or the model says it's not a
    follow-up; otherwise a standalone rewrite that resolves pronouns/implicit
    references against `history`.
    """
    if not history:
        return question

    user_message = f"Conversation so far:\n{_format_history(history)}\n\nNew message: \"{question}\""

    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        format=_SCHEMA,
    )
    parsed = json.loads(response.message.content)

    if not parsed.get("is_follow_up"):
        return question  # never trust the model's rewritten text for a non-follow-up

    rewritten = parsed.get("standalone_question", "").strip()
    return rewritten if rewritten else question

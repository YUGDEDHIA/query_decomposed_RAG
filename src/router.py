"""Classify what kind of handling a question needs before running anything expensive.

Categories, each routed to a different part of the system:
- "casual": greetings/small talk/meta questions -- no retrieval, no
  research pipeline. Routing these through decompose/retrieve/answer/
  synthesize wastes several LLM calls and produces a broken-looking "not
  grounded" research answer for what should just be a normal reply.
- "gmp_query": Grey Market Premium / listing-gain questions, and IPO
  calendar/timeline questions (which IPOs are open/opening, when one
  closes/lists) -- both live in src/gmp_answerer.py's snapshot rows
  already (open_date/close_date/status alongside GMP), so both are the
  same intent, not RHP filing content at all.
- "subscription_query": subscription status (QIB/NII/Retail/Employee bid
  multiples) or anchor investor data -- src/subscription_answerer.py.
  Previously misclassified under "gmp_query" ("subscription buzz") despite
  gmp_answerer.py having no subscription fields at all -- split out once
  src/scrape_subscription.py actually started fetching this data, rather
  than continuing to silently produce a wrong/empty answer.
- "listing_performance_query": how an already-listed IPO has performed
  since listing (current price vs. issue price, day-by-day trend), or
  which recent IPOs had the best/worst listing gains --
  src/listing_performance_answerer.py. Also live market data, distinct
  from GMP (pre-listing sentiment) and subscription status (pre-listing
  bidding demand) -- this one is exclusively about already-listed IPOs.
- "cross_company_query": "which/best/top IPOs by X" comparisons that don't
  name specific companies -- needs src/cross_company.py's per-company
  retrieval + single ranking call, not the named-entity planner (verified:
  a "which IPOs have good financials" question with no company named
  produced an unscoped top-5 search across the whole 37-company corpus,
  i.e. near-random results, which is what this category exists to avoid).
- "rhp_query": research about one or more *named* companies -- the
  existing planner already handles multi-company comparisons fine when
  the companies are named (it decomposes per company), so this only
  needs cross_company_query when nothing is named.

This is a cheap classification call before anything else runs.
"""
from __future__ import annotations

import json

import ollama

MODEL = "qwen2.5:14b-instruct"

_INTENT_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {
            "type": "string",
            "enum": [
                "rhp_query", "cross_company_query", "gmp_query",
                "subscription_query", "listing_performance_query", "casual",
            ],
        },
    },
    "required": ["intent"],
}

_ROUTER_SYSTEM_PROMPT = """You classify incoming messages to a research assistant that answers questions about IPO Red Herring Prospectus (RHP) filings using a document retrieval pipeline, plus live Grey Market Premium (GMP), subscription, and anchor investor data.

Classify the message as one of:
- "rhp_query": a question about one or more specifically NAMED companies -- risk factors, financials, capital structure, objects of the issue, business, promoters, or any other RHP content. Includes comparisons between named companies (e.g. "compare X and Y's risk factors").
- "cross_company_query": asks to compare, rank, recommend, or find "the best/top/which" IPOs by some RHP-derived criterion (financials, risk, business quality, etc.) WITHOUT naming specific companies -- it wants the assistant to search/judge across many or all indexed companies.
- "gmp_query": asks about Grey Market Premium, expected listing gains/price, or IPO calendar/timeline questions (which IPOs are currently open, opening soon, or when one opens/closes/lists) -- this is live market data, separate from RHP filing content.
- "subscription_query": asks about IPO subscription status/bid multiples (QIB, NII, Retail, Employee, overall), or about anchor investors -- also live market/registrar data, but distinct from GMP.
- "listing_performance_query": asks how an already-listed IPO has performed since listing (current price vs. issue price, listing-day gain, trend since listing), or which recent IPOs had the best/worst listing gains -- live market-price data about companies that have ALREADY listed, distinct from GMP (pre-listing) and subscription status (pre-listing bidding).
- "casual": greetings, small talk, thanks, questions about what the assistant can do, or anything unrelated to IPOs.

When genuinely unsure between a research category and "casual", prefer the research category -- wrongly guessing "casual" silently skips real research the user wanted, which is worse than the reverse.
"""

_CASUAL_SYSTEM_PROMPT = """You are the conversational front-end of a research assistant for IPO Red Herring Prospectus (RHP) filings, Grey Market Premium (GMP) data, and subscription/anchor investor data. The user's message doesn't need document research -- respond briefly and naturally. If it fits, gently mention you can look up things like risk factors, capital structure, or objects of the issue for specific companies, compare/rank companies by a criterion, check current GMP or IPO calendar dates, or check subscription/anchor investor data.
"""


def classify_intent(question: str) -> str:
    """"rhp_query", "cross_company_query", "gmp_query", "subscription_query", "listing_performance_query", or "casual"."""
    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": _ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        format=_INTENT_SCHEMA,
    )
    return json.loads(response.message.content)["intent"]


def casual_reply(question: str) -> str:
    """A direct conversational reply for a non-research message, no retrieval involved."""
    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": _CASUAL_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
    )
    return response.message.content

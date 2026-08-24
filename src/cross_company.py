"""Compare/rank across many companies for questions that don't name specific ones.

"Which IPOs have good financials" can't be answered by the normal
pipeline: the planner decomposes by named entity, and with no company
named, retrieval does an unscoped top-k search across the whole corpus --
which surfaces near-random results (verified: this is exactly what
produced the original "no financial analysis" non-answer this module
exists to fix).

This retrieves one financial-summary chunk per known company, then asks
the LLM to compare/rank them in a single call. Not a numeric-extraction
system -- the model reads each company's retrieved excerpt (which may be a
flattened financial-statement table) and reasons qualitatively from it,
not from precise recomputed metrics.

Known limitation, not solved here: doesn't cross-reference "currently
open" IPO status (that lives in GMP data, under possibly different company
name spellings -- see src/gmp_answerer.py). This ranks across the whole
indexed corpus, not filtered to IPOs currently open for bidding. Matching
GMP-site company names to RHP-corpus company names is its own fuzzy-name
problem, left for later.
"""
from __future__ import annotations

import ollama

from src.embedder import embed_query
from src.retriever import Retriever

MODEL = "qwen2.5:14b-instruct"
# Caps context size (roughly N companies x <=600-word chunk) within NUM_CTX;
# comfortably covers the current ~37-company corpus. Revisit (map-reduce
# instead of one big call) if the corpus grows well past this.
MAX_COMPANIES = 40
NUM_CTX = 32768

_FINANCIAL_TOPICS = ["SUMMARY OF FINANCIAL INFORMATION", "RESTATED FINANCIAL INFORMATION"]

_SYSTEM_PROMPT = """You are comparing multiple IPO companies using excerpts from their Red Herring Prospectus filings, to answer a question that asks to compare, rank, or recommend across many companies rather than research one named company.

You are given one financial-information excerpt per company. Some excerpts are flattened tables (cells joined with " | " per row); read them as such.

Rules:
- Base your comparison ONLY on the provided excerpts -- do not use outside knowledge of these companies.
- Reason qualitatively (revenue trend, profitability signals, growth trajectory) from what's in the excerpts -- you do not have precise, independently verified financial ratios, so don't state figures with more precision or certainty than the excerpts actually support.
- If an excerpt doesn't contain enough financial detail to judge a company, say so for that company rather than guessing.
- Note explicitly that this compares companies in the indexed corpus, not necessarily every IPO currently open in the market.
"""


def _retrieve_financial_summary(retriever: Retriever, query_vec: list[float], company: str) -> dict | None:
    for topic in _FINANCIAL_TOPICS:
        results = retriever.store.search(query_vec, top_k=1, company=company, section_topic=topic)
        if results:
            return results[0][0]
    return None


def compare_companies(question: str, retriever: Retriever) -> tuple[str, list[dict]]:
    """Compare/rank across all indexed companies for a question that doesn't name specific ones.

    Returns (answer, sources) -- one source per company summary actually used.
    """
    companies = retriever.known_companies[:MAX_COMPANIES]
    query_vec = embed_query(question)

    summaries = []
    for company in companies:
        chunk = _retrieve_financial_summary(retriever, query_vec, company)
        if chunk is not None:
            summaries.append(chunk)

    if not summaries:
        return "I couldn't find financial-summary excerpts for any indexed company to compare.", []

    formatted = "\n\n---\n\n".join(
        f"[{c['company']}] (section: {c.get('section')}, page {c['page_range'][0]})\n{c['text']}"
        for c in summaries
    )
    user_message = f'Question: "{question}"\n\nCompany excerpts ({len(summaries)} companies):\n\n{formatted}'

    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        options={"num_ctx": NUM_CTX},
    )

    sources = [
        {
            "company": c["company"], "section": c.get("section"),
            "page_range": c["page_range"], "source_file": c["source_file"],
        }
        for c in summaries
    ]
    return response.message.content, sources

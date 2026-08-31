"""End-to-end query-decomposed RAG pipeline: one call from question to final answer.

Ties together src/router.py (classify into one of 5 intents), and then
one of:
- casual: src/router.py's casual_reply(), no retrieval at all.
- gmp_query: src/gmp_answerer.py, reads the live GMP snapshot (also covers
  IPO calendar/timeline questions -- open/close/listing dates are already
  in every GMP row), no RHP retrieval at all.
- subscription_query: src/subscription_answerer.py, reads the live
  subscription-status/anchor-investor snapshot, no RHP retrieval at all.
- listing_performance_query: src/listing_performance_answerer.py, reads
  the live post-listing performance snapshot plus tracked daily history,
  no RHP retrieval at all.
- cross_company_query: src/cross_company.py, compares/ranks across many
  companies in one call (for "which IPOs have good financials"-style
  questions that don't name specific companies).
- rhp_query: the full src/planner.py -> src/retriever.py -> src/answerer.py
  -> src/synthesizer.py chain, for questions about one or more *named*
  companies.

Each path was built and validated independently (see the corresponding
tests/manual_smoke_*.py scripts); this just wires them into a single
reusable function instead of hand-assembling the chain per script.

For rhp_query, each sub-question is also checked against the index before
retrieving: if it names a company that isn't indexed yet,
src/on_demand_ingest.py searches SEBI, downloads and processes its RHP,
embeds it, and adds it to `retriever` (and persists to disk) before
retrieval runs -- instead of requiring a manual scrape+ingest+build-index
run first for every company a user might ask about.

Before reaching for that, a misspelled/abbreviated company name is checked
against retriever.fuzzy_match_company -- a typo of an already-indexed
company ("Aastha Spintx", "Sanstar Ltd") would otherwise look exactly like
a brand-new one and trigger a multi-minute SEBI re-download/re-process for
a company that's already sitting in the index. The resolved company name
(from fuzzy match or fresh ingest) is passed into retrieve() explicitly --
letting retrieve() re-infer it from the sub-question's own (still
misspelled) text would just fail the same way infer_company originally
did.

`retriever` is a required argument rather than loaded internally -- the
index is ~110MB and callers (e.g. src/chat.py's REPL loop) should load it
once and reuse it across many questions, not reload it per call.

`history`, if given, is resolved into a self-contained question exactly
once via src/contextualizer.py before anything else runs -- a follow-up
like "what about its financials?" would otherwise break every stage below
(classify_intent, decompose, infer_company, the live-data answerers) the
same way, since none of them have any concept of prior turns. Once
resolved, every downstream call is unchanged -- still a single question
string, so this is the only place multi-turn logic lives.
"""
from __future__ import annotations

from typing import Callable

from src.answerer import answer
from src.contextualizer import Turn, contextualize
from src.cross_company import compare_companies
from src.gmp_answerer import answer_gmp_question
from src.listing_performance_answerer import answer_listing_performance_question
from src.on_demand_ingest import extract_company_name, ingest_company_on_demand
from src.planner import decompose
from src.retriever import Retriever
from src.router import casual_reply, classify_intent
from src.subscription_answerer import answer_subscription_question
from src.synthesizer import FinalAnswer, synthesize


def answer_question(
    question: str,
    retriever: Retriever,
    history: list[Turn] | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> FinalAnswer:
    """Route `question` to the right handler and return one FinalAnswer.

    on_progress(message), if given, is called with a short status string
    before each stage -- useful for a UI to show something during the
    several-LLM-call, tens-of-seconds round trip a compound question takes.
    """
    def progress(message: str) -> None:
        if on_progress:
            on_progress(message)

    if history:
        progress("Resolving follow-up context...")
        question = contextualize(question, history)

    progress("Classifying question...")
    intent = classify_intent(question)

    if intent == "casual":
        progress("Responding...")
        return FinalAnswer(
            question=question, answer=casual_reply(question),
            fully_grounded=False, sub_answers=[], sources=[], intent="casual",
        )

    if intent == "gmp_query":
        progress("Checking GMP data...")
        return FinalAnswer(
            question=question, answer=answer_gmp_question(question),
            fully_grounded=False, sub_answers=[], sources=[], intent="gmp_query",
        )

    if intent == "subscription_query":
        progress("Checking subscription/anchor investor data...")
        return FinalAnswer(
            question=question, answer=answer_subscription_question(question),
            fully_grounded=False, sub_answers=[], sources=[], intent="subscription_query",
        )

    if intent == "listing_performance_query":
        progress("Checking post-listing performance data...")
        return FinalAnswer(
            question=question, answer=answer_listing_performance_question(question),
            fully_grounded=False, sub_answers=[], sources=[], intent="listing_performance_query",
        )

    if intent == "cross_company_query":
        progress("Comparing across indexed companies...")
        answer_text, sources = compare_companies(question, retriever)
        return FinalAnswer(
            question=question, answer=answer_text,
            fully_grounded=bool(sources), sub_answers=[], sources=sources, intent="cross_company_query",
        )

    progress("Decomposing question...")
    sub_questions = decompose(question)

    sub_answers = []
    for sq in sub_questions:
        company = retriever.infer_company(sq)
        if company is None:
            candidate = extract_company_name(sq)
            if candidate:
                company = retriever.fuzzy_match_company(candidate)
                if company:
                    progress(f"Reading '{candidate}' as indexed company '{company}'...")
                else:
                    progress(f"'{candidate}' isn't indexed yet -- searching SEBI for its RHP...")
                    company = ingest_company_on_demand(candidate, retriever, on_progress=progress)
                    if not company:
                        progress(f"Couldn't find an RHP for '{candidate}' on SEBI.")

        progress(f"Retrieving: {sq}")
        chunks = retriever.retrieve(sq, company=company)
        progress(f"Answering: {sq}")
        sub_answers.append(answer(sq, chunks))

    progress("Synthesizing final answer...")
    return synthesize(question, sub_answers)

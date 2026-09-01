"""Interactive CLI chat over the RHP corpus.

Ask a question, see the synthesized answer plus its sources. Loads the
index and wires up src/pipeline.py once at startup, then reuses it across
the whole session rather than per question.

Keeps a local conversation history across the REPL loop so follow-up
questions ("what about its financials?") resolve correctly -- see
src/contextualizer.py, which src/pipeline.py's answer_question() calls
internally once `history` is passed.

Usage:
    python -m src.chat
"""
from __future__ import annotations

from src.contextualizer import Turn
from src.pipeline import answer_question
from src.retriever import Retriever

EXIT_COMMANDS = {"exit", "quit", "q"}


def main() -> None:
    print("Loading index...")
    retriever = Retriever.load()
    print(f"Ready -- {len(retriever.store.metadata)} chunks indexed. Ask a question, or 'exit' to quit.\n")

    history: list[Turn] = []

    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question:
            continue
        if question.lower() in EXIT_COMMANDS:
            break

        final = answer_question(
            question, retriever, history=history,
            on_progress=lambda msg: print(f"  ... {msg}"),
        )

        print(f"\n{final.answer}\n")
        status = "fully grounded" if final.fully_grounded else "partially grounded -- some info may be missing"
        print(f"[{status}, {len(final.sources)} source(s)]")

        if final.sources and input("Show sources? [y/N] ").strip().lower() == "y":
            for s in final.sources:
                print(f"  - {s['company']} / {s['section']} (page {s['page_range'][0]})")
        print()

        history.append(Turn(question=question, answer=final.answer))


if __name__ == "__main__":
    main()

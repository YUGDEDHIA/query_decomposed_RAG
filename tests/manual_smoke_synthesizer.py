"""Manual sanity check for the full planner -> retriever -> answerer -> synthesizer chain.

Needs data/index/ built and live Ollama with qwen2.5:14b-instruct and
nomic-embed-text pulled. Run directly:

    python tests/manual_smoke_synthesizer.py
    python tests/manual_smoke_synthesizer.py "your own question here"
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.answerer import answer
from src.planner import decompose
from src.retriever import Retriever
from src.synthesizer import synthesize

DEFAULT_QUESTION = (
    "Compare the risk factors of Aastha Spintex Limited and Advit Jewels Limited, "
    "and what are Aastha Spintex's objects of the issue?"
)


def main() -> None:
    question = " ".join(sys.argv[1:]) or DEFAULT_QUESTION
    print(f"Question: {question!r}\n")

    sub_questions = decompose(question)
    print(f"Decomposed into {len(sub_questions)} sub-question(s):")
    for sq in sub_questions:
        print(f"  - {sq}")

    print("\nLoading index...")
    retriever = Retriever.load()

    sub_answers = []
    for sq in sub_questions:
        chunks = retriever.retrieve(sq)
        result = answer(sq, chunks)
        sub_answers.append(result)
        print(f"\n[sub-answer] grounded={result.grounded} -- {sq}")

    final = synthesize(question, sub_answers)

    print(f"\n{'=' * 70}\nFINAL ANSWER (fully_grounded={final.fully_grounded}):\n")
    print(final.answer)
    print(f"\n{'-' * 70}\nSources ({len(final.sources)}):")
    for s in final.sources:
        print(f"  - {s['company']} / {s['section']} (page {s['page_range'][0]})")


if __name__ == "__main__":
    main()

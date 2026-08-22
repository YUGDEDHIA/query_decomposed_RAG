"""Manual sanity check for the planner -> retriever -> answerer chain.

Needs data/index/ built and live Ollama with qwen2.5:14b-instruct and
nomic-embed-text pulled. Run directly:

    python tests/manual_smoke_answerer.py
    python tests/manual_smoke_answerer.py "your own question here"
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.answerer import answer
from src.planner import decompose
from src.retriever import Retriever

DEFAULT_QUESTION = (
    "What are the risk factors for Aastha Spintex Limited and what are its objects of the issue?"
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

    for sq in sub_questions:
        print(f"\n{'=' * 70}\nSub-question: {sq}")
        chunks = retriever.retrieve(sq, top_k=5)
        result = answer(sq, chunks)
        print(f"Grounded: {result.grounded}")
        print(f"Answer: {result.answer}")
        print("Sources:")
        for s in result.sources:
            print(f"  - {s['company']} / {s['section']} (page {s['page_range'][0]})")


if __name__ == "__main__":
    main()

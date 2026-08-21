"""Manual sanity check for planner.decompose against real compound questions.

Needs live Ollama with qwen2.5:14b-instruct pulled. Run directly:

    python tests/manual_smoke_planner.py
    python tests/manual_smoke_planner.py "your own question here"
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.planner import decompose

DEFAULT_QUESTIONS = [
    # Should NOT split -- already a single focused question.
    "What are the risk factors for Aastha Spintex Limited?",
    # Should split into 2 -- same aspect, two companies.
    "Compare the capital structure of Aastha Spintex Limited and Advit Jewels Limited.",
    # Should split into 2 -- two aspects, one company.
    "What are the objects of the issue for CALIBER MINING AND LOGISTICS LIMITED and what risks does it face related to raw materials?",
    # Should split into 3 -- stress test.
    "What is the revenue growth of Fractal Analytics, what are its risk factors, and who are its promoters?",
]


def main() -> None:
    questions = sys.argv[1:] or DEFAULT_QUESTIONS
    for q in questions:
        print(f"Question: {q!r}")
        sub_qs = decompose(q)
        for i, sq in enumerate(sub_qs, start=1):
            print(f"  {i}. {sq}")
        print()


if __name__ == "__main__":
    main()

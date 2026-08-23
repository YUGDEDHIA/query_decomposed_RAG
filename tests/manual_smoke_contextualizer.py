"""Manual sanity check for contextualizer.contextualize against real follow-up questions.

Needs live Ollama with qwen2.5:14b-instruct pulled. Run directly:

    python tests/manual_smoke_contextualizer.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.contextualizer import Turn, contextualize

CASES = [
    (
        "Same-intent follow-up (RHP -> RHP)",
        [Turn(question="What are the risk factors for Acme Industries Limited?",
              answer="Acme Industries Limited faces risks including raw material price volatility and regulatory changes.")],
        "What about its financials?",
    ),
    (
        "Cross-intent follow-up (RHP -> GMP)",
        [Turn(question="What are the risk factors for Acme Industries Limited?",
              answer="Acme Industries Limited faces risks including raw material price volatility and regulatory changes.")],
        "And what's the GMP for it?",
    ),
    (
        "Entity named only in the assistant's prior answer",
        [Turn(question="Which recent IPOs have the best financials?",
              answer="Based on the indexed filings, Beta Textiles Limited stands out with strong revenue growth.")],
        "What's its subscription status?",
    ),
    (
        "Fresh question naming its own company -- must come back unchanged",
        [Turn(question="What are the risk factors for Acme Industries Limited?",
              answer="Acme Industries Limited faces risks including raw material price volatility.")],
        "What are the objects of the issue for Advit Jewels Limited?",
    ),
    (
        "Unrelated casual aside -- must come back unchanged",
        [Turn(question="What's the GMP for Acme Industries Limited?",
              answer="Acme Industries Limited's GMP is currently +15%.")],
        "Thanks! By the way, what's the capital of France?",
    ),
]


def main() -> None:
    for label, history, question in CASES:
        print(f"--- {label} ---")
        print(f"History: {[(t.question, t.answer[:60]) for t in history]}")
        print(f"New message: {question!r}")
        result = contextualize(question, history)
        print(f"Resolved: {result!r}")
        print()


if __name__ == "__main__":
    main()

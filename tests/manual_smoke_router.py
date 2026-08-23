"""Manual sanity check for classify_intent against diverse real messages.

Needs live Ollama with qwen2.5:14b-instruct pulled. Run directly:

    python tests/manual_smoke_router.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.router import casual_reply, classify_intent

CASES = [
    # (message, expected_intent)
    ("Hii", "casual"),
    ("hello", "casual"),
    ("thanks!", "casual"),
    ("what can you do?", "casual"),
    ("who are you?", "casual"),
    ("What are the risk factors for Aastha Spintex Limited?", "rhp_query"),
    ("Compare the capital structure of two companies in the corpus.", "rhp_query"),
    ("Which company has the highest revenue growth?", "rhp_query"),  # no company named, no obvious keyword
    ("Tell me about Advit Jewels", "rhp_query"),  # borderline phrasing, names a company
]


def main() -> None:
    correct = 0
    for message, expected in CASES:
        intent = classify_intent(message)
        ok = "OK" if intent == expected else "MISMATCH"
        if intent == expected:
            correct += 1
        print(f"[{ok}] {message!r} -> {intent} (expected {expected})")

    print(f"\n{correct}/{len(CASES)} correct")

    print("\n--- sample casual reply ---")
    print(casual_reply("Hii"))


if __name__ == "__main__":
    main()

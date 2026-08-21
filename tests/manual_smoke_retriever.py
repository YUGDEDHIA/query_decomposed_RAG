"""Manual sanity check for Retriever.retrieve against the real persisted index.

Needs data/index/ built (`python -m src.build_index`) and a live Ollama
with nomic-embed-text pulled. Run directly:

    python tests/manual_smoke_retriever.py [query text]
    python tests/manual_smoke_retriever.py --company "ALPINE TEXWORLD LIMITED" [query text]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retriever import Retriever


def main() -> None:
    args = sys.argv[1:]
    company = None
    if args and args[0] == "--company":
        company = args[1]
        args = args[2:]

    query = " ".join(args) or "what are the risk factors related to raw material prices?"

    print("Loading index...")
    retriever = Retriever.load()
    print(f"Query: {query!r} (company filter: {company!r})\n")

    results = retriever.retrieve(query, top_k=5, company=company)
    if not results:
        print("No results.")
        return

    for r in results:
        preview = r["text"][:150].replace("\n", " ")
        print(f"--- score={r['score']:.4f} company={r['company']!r} section={r['section']!r} ---")
        print(preview + ("..." if len(r["text"]) > 150 else ""))
        print()


if __name__ == "__main__":
    main()

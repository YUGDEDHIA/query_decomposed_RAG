"""Manual sanity check for load_pdf + chunk_text against a real RHP.

Not a pytest unit test (no assertions on real-world PDF content -- that's
the job of test_chunker.py's synthetic-input tests). Run directly:

    python tests/manual_smoke_loader_chunker.py [path/to/some_RHP.pdf]

Defaults to the first PDF found in data/raw/.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunker import chunk_text
from src.loader import company_from_filename, load_pdf


def main() -> None:
    if len(sys.argv) > 1:
        pdf_path = Path(sys.argv[1])
    else:
        candidates = sorted((Path(__file__).resolve().parent.parent / "data" / "raw").glob("*.pdf"))
        if not candidates:
            sys.exit("No PDFs in data/raw/ -- pass a path explicitly, or run data/scrape_sebi_rhps.py first.")
        pdf_path = candidates[0]

    print(f"Loading {pdf_path}")
    pages = load_pdf(str(pdf_path))
    non_empty = sum(1 for p in pages if p.strip())
    print(f"{len(pages)} pages, {non_empty} non-empty")

    company = company_from_filename(str(pdf_path))
    chunks = chunk_text(pages, company=company, source_file=pdf_path.name)
    print(f"company={company!r}, {len(chunks)} chunks")

    for c in chunks[:3]:
        word_count = len(c.text.split())
        preview = c.text[:200].replace("\n", " ")
        print(f"\n--- chunk {c.chunk_index} (pages {c.page_range}, {word_count} words) ---")
        print(preview + ("..." if len(c.text) > 200 else ""))


if __name__ == "__main__":
    main()

"""Manual sanity check for extract_table_chunks against a real RHP.

Not a pytest unit test -- table quality on real filings needs eyeballing,
not synthetic assertions (see test_table_extractor.py for the deterministic
logic tests). Run directly:

    python tests/manual_smoke_tables.py [path/to/some_RHP.pdf]

Defaults to the first PDF found in data/raw/.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.loader import company_from_filename
from src.section_detector import assign_sections, detect_sections
from src.table_extractor import extract_table_chunks


def main() -> None:
    if len(sys.argv) > 1:
        pdf_path = Path(sys.argv[1])
    else:
        candidates = sorted((Path(__file__).resolve().parent.parent / "data" / "raw").glob("*.pdf"))
        if not candidates:
            sys.exit("No PDFs in data/raw/ -- pass a path explicitly, or run data/scrape_sebi_rhps.py first.")
        pdf_path = candidates[0]

    print(f"Scanning {pdf_path}")
    company = company_from_filename(str(pdf_path))
    chunks = extract_table_chunks(str(pdf_path), company=company, source_file=pdf_path.name)
    print(f"company={company!r}, {len(chunks)} table chunks found")

    sections = detect_sections(str(pdf_path))
    if sections:
        assign_sections(chunks, sections)

    for c in chunks[:5]:
        print(f"\n--- table {c.chunk_index} (page {c.page_range[0]}, section={c.section!r}) ---")
        print(c.text[:500] + ("..." if len(c.text) > 500 else ""))


if __name__ == "__main__":
    main()

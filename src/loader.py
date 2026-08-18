"""Load a single RHP PDF into per-page text.

Table/financial-statement extraction is out of scope here (see
docs/commit-02-loader-chunker.md) -- this only pulls plain text.
"""
from pathlib import Path

import pdfplumber


def load_pdf(path: str) -> list[str]:
    """Return one string per page of the PDF at `path`.

    Pages that are blank or image-only (cover pages, signature pages,
    scanned annexures) come back as "" rather than raising.
    """
    pages = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            pages.append(text if text else "")
    return pages


def company_from_filename(path: str) -> str:
    """"data/raw/Aastha_Spintex_Limited_RHP.pdf" -> "Aastha Spintex Limited".

    Matches the CompanyName_RHP.pdf convention data/scrape_sebi_rhps.py
    writes files under.
    """
    stem = Path(path).stem
    if stem.endswith("_RHP"):
        stem = stem[: -len("_RHP")]
    return stem.replace("_", " ").strip()

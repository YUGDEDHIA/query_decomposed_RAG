"""When a question names a company not yet indexed, fetch and index it on the fly.

Without this, a question naming an un-indexed company falls through to an
unscoped search across whatever *is* indexed (the same class of bug fixed
in src/retriever.py's company inference) -- or, worse, just silently
answers about the wrong company. Instead: search SEBI for the name, download
its RHP, run it through the exact same pipeline data/scrape_sebi_rhps.py +
src/ingest.py + src/build_index.py already do as separate manual steps, and
add it to the running Retriever in place.

Verified live: SEBI's own search (data.scrape_sebi_rhps.search_listing)
finds both a company already in this project's corpus and one that wasn't
(a then-currently-open IPO), so this generalizes beyond re-finding what's
already known.

This mutates and persists data/raw/, data/processed/, and data/index/ --
unlike everything else in this project, triggered implicitly by a chat
question rather than an explicit script run. The corpus grows over time as
users ask about new companies, rather than requiring a manual re-scrape.

Cold-start latency for a brand-new company is real (SEBI fetch + full PDF
processing + embedding -- multiple minutes for a large filing, matching
the per-document times seen during the original bulk ingestion). Callers
should pass on_progress so this doesn't look hung.
"""
from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path

import ollama

from data.scrape_sebi_rhps import (
    company_filename,
    download_pdf,
    get_pdf_url,
    new_session,
    parse_listing_rows,
    search_listing,
)
from src.embedder import embed_documents
from src.ingest import process_pdf, write_chunks
from src.retriever import Retriever

MODEL = "qwen2.5:14b-instruct"
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_DIR = ROOT / "data" / "raw"
DEFAULT_PROCESSED_DIR = ROOT / "data" / "processed"
DEFAULT_INDEX_DIR = ROOT / "data" / "index"

logger = logging.getLogger("on_demand_ingest")

_EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "company_name": {"type": ["string", "null"]},
    },
    "required": ["company_name"],
}

_EXTRACT_SYSTEM_PROMPT = """Extract the specific company name a question is asking about, if any. Return it in a form close to a formal registered company name (e.g. include "Limited" if implied), suitable for searching a company/IPO filings database. If the question doesn't name or clearly imply one specific company, return null for company_name.
"""


def extract_company_name(question: str) -> str | None:
    """A candidate company name mentioned in `question`, or None."""
    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": _EXTRACT_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        format=_EXTRACT_SCHEMA,
    )
    return json.loads(response.message.content)["company_name"]


def ingest_company_on_demand(
    candidate_name: str,
    retriever: Retriever,
    on_progress=None,
    raw_dir: Path = DEFAULT_RAW_DIR,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    index_dir: Path = DEFAULT_INDEX_DIR,
) -> str | None:
    """Search SEBI for `candidate_name`, fetch+process+embed its RHP, and add it to `retriever` in place.

    Returns the actual indexed company name on success, None if no RHP was
    found or ingestion failed at any step (network, parsing, embedding --
    all treated as "couldn't add this one", not a crash).
    """
    def progress(message: str) -> None:
        if on_progress:
            on_progress(message)

    try:
        session = new_session()
        rows = parse_listing_rows(search_listing(session, candidate_name))
        if not rows:
            logger.info("No SEBI RHP found for %r", candidate_name)
            return None

        _date, title, detail_url = rows[0]
        dest = raw_dir / company_filename(title)

        if not dest.exists():
            progress(f"Downloading RHP for {title}...")
            pdf_url = get_pdf_url(session, detail_url)
            if pdf_url is None:
                logger.warning("No PDF link found on %s", detail_url)
                return None
            raw_dir.mkdir(parents=True, exist_ok=True)
            download_pdf(session, pdf_url, dest)

        progress(f"Processing {dest.name} (can take a few minutes for a new filing)...")
        chunks = process_pdf(dest)
        if not chunks:
            logger.warning("No chunks extracted from %s", dest)
            return None
        write_chunks(chunks, processed_dir / f"{dest.stem}.jsonl")

        progress(f"Embedding {len(chunks)} chunks...")
        texts = [c.text for c in chunks]
        embeddings = embed_documents(
            texts, on_batch=lambda done, total: progress(f"  {done}/{total} embedded"),
        )

        metadata = [dataclasses.asdict(c) for c in chunks]
        retriever.store.add(metadata, embeddings)
        retriever.store.save(index_dir)

        company = chunks[0].company
        if company not in retriever.known_companies:
            retriever.known_companies = retriever._sorted_companies(set(retriever.known_companies) | {company})

        progress(f"Indexed {company}.")
        return company

    except Exception as exc:
        logger.warning("On-demand ingestion failed for %r: %s", candidate_name, exc)
        return None

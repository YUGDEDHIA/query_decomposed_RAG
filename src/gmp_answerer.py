"""Answer Grey Market Premium (GMP) questions from the latest scraped snapshot.

Reads data/gmp/latest.json (see data/scrape_gmp.py) and answers directly
from it via qwen2.5:14b-instruct -- GMP is live market data, not RHP
content, so this bypasses retrieval/planner/synthesizer entirely, the same
shape as src/router.py's casual_reply.

Auto-refreshes the snapshot when it's missing or older than STALE_AFTER,
instead of requiring a manual `python data/scrape_gmp.py` run first --
GMP updates roughly every 30 min on the source site, so a question is a
reasonable trigger to check freshness. If a refresh fails (site down,
network issue) but a stale snapshot exists, that's used anyway with a
note, rather than failing the whole answer over a transient scrape error.

GMP is unofficial, unregulated grey-market chatter, not a regulatory
disclosure -- the prompt requires the answer to carry that caveat rather
than present it as fact.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import ollama

MODEL = "qwen2.5:14b-instruct"
DEFAULT_GMP_PATH = Path(__file__).resolve().parent.parent / "data" / "gmp" / "latest.json"
STALE_AFTER = timedelta(minutes=20)

_SYSTEM_PROMPT = """You answer questions about IPO Grey Market Premium (GMP) using a live-scraped snapshot of GMP data provided below. GMP is unofficial, unregulated grey-market trading activity, not a regulatory disclosure -- always frame it as informal market sentiment, not a guaranteed outcome.

Rules:
- Use ONLY the data provided -- do not invent GMP figures or IPO details not present in it.
- If the data doesn't cover what's asked (e.g. a company not in the snapshot), say so plainly.
- Mention that the data is a snapshot from a specific time and can change.
"""

logger = logging.getLogger("gmp_answerer")


def _format_gmp_row(row: dict) -> str:
    return (
        f"{row['company']} | Status: {row['status']} | GMP: {row['gmp_rupees']} ({row['gmp_percent']}) | "
        f"Price band: {row['price_band']} | Est. listing: {row['est_listing']} | "
        f"Open: {row['open_date']} | Close: {row['close_date']} | Updated: {row['updated']}"
    )


def load_gmp_snapshot(path: Path = DEFAULT_GMP_PATH) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _is_stale(snapshot: dict) -> bool:
    scraped_at = datetime.fromisoformat(snapshot["scraped_at"])
    return datetime.now(timezone.utc) - scraped_at > STALE_AFTER


def _save_snapshot(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def get_fresh_snapshot(path: Path = DEFAULT_GMP_PATH) -> dict | None:
    """The cached snapshot if fresh enough, otherwise a newly-scraped one.

    Falls back to a stale cached snapshot (rather than failing outright) if
    a refresh is needed but the scrape itself fails.
    """
    cached = load_gmp_snapshot(path)
    if cached is not None and not _is_stale(cached):
        return cached

    from data.scrape_gmp import fetch_gmp_data  # deferred: avoids importing playwright unless actually scraping

    try:
        logger.info("GMP snapshot missing or stale, scraping a fresh one...")
        fresh = fetch_gmp_data()
        fresh["scraped_at"] = datetime.now(timezone.utc).isoformat()
        _save_snapshot(fresh, path)
        return fresh
    except Exception as exc:
        if cached is not None:
            logger.warning("GMP refresh failed (%s), falling back to stale cached snapshot", exc)
            return cached
        logger.warning("GMP refresh failed and no cached snapshot exists: %s", exc)
        return None


def answer_gmp_question(question: str, snapshot_path: Path = DEFAULT_GMP_PATH) -> str:
    """Answer a GMP question from a fresh-as-possible snapshot, or explain if none is available."""
    snapshot = get_fresh_snapshot(snapshot_path)
    if snapshot is None:
        return (
            "I couldn't fetch or find any Grey Market Premium data -- the source site may be "
            "unreachable right now. Try again shortly."
        )

    rows = snapshot["mainboard"] + snapshot["sme"]
    context = "\n".join(_format_gmp_row(r) for r in rows) if rows else "No IPOs currently listed in the snapshot."

    user_message = (
        f'Snapshot scraped at: {snapshot["scraped_at"]}\n\n'
        f"GMP data:\n{context}\n\n"
        f'Question: "{question}"'
    )

    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )
    return response.message.content

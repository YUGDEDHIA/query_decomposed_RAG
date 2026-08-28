"""Answer IPO subscription-status and anchor-investor questions from the latest scraped snapshot.

Reads data/subscription/latest.json (see data/scrape_subscription.py) and
answers directly from it via qwen2.5:14b-instruct -- same shape as
src/gmp_answerer.py: this is live/registrar-reported market data, not RHP
content, so it bypasses retrieval/planner/synthesizer entirely.

Auto-refreshes the snapshot when it's missing or older than STALE_AFTER,
same reasoning as gmp_answerer.py -- subscription figures move through the
bidding day, so a question is a reasonable trigger to check freshness. A
failed refresh with a stale snapshot available uses the stale one with a
note, rather than failing outright.

Subscription multiples are provisional until the issue actually closes --
the prompt requires that caveat rather than presenting a mid-bidding number
as final. Anchor investor data (chittorgarh's anchor-investors-list report)
is aggregated *by anchor investor* across many issues, not a per-IPO
breakdown of who anchored one specific company -- the prompt says so
explicitly rather than letting the model imply a per-company answer the
data doesn't actually support.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import ollama

MODEL = "qwen2.5:14b-instruct"
DEFAULT_SUBSCRIPTION_PATH = Path(__file__).resolve().parent.parent / "data" / "subscription" / "latest.json"
STALE_AFTER = timedelta(minutes=20)

_SYSTEM_PROMPT = """You answer questions about IPO subscription status (QIB/NII/Retail/Employee bid multiples) and anchor investors, using a live-scraped snapshot provided below.

Rules:
- Use ONLY the data provided -- do not invent subscription figures, multiples, or investor names not present in it.
- Subscription multiples are provisional and change until the issue actually closes -- never present a mid-bidding number as final.
- The anchor investor data is aggregated per anchor investor across many past issues (total investments, average gains) -- it is NOT a per-IPO list of who anchored one specific company's issue. If asked which anchors backed a specific company, say that breakdown isn't in this data rather than guessing.
- If the data doesn't cover what's asked (e.g. a company not in the snapshot), say so plainly.
- Mention that the data is a snapshot from a specific time and can change.
"""

logger = logging.getLogger("subscription_answerer")


def _format_subscription_row(row: dict) -> str:
    return (
        f"{row['company']} | Status: {row['status_code'] or 'CLOSED'} | Closes: {row['closing_date']} | "
        f"QIB: {row['qib_x']}x | NII: {row['nii_x']}x | Retail: {row['retail_x']}x | "
        f"Total: {row['total_x']}x | Applications: {row['applications']} | As on: {row['subscription_as_on']}"
    )


def _format_anchor_row(row: dict) -> str:
    return (
        f"{row['anchor_investor']} | Issues: {row['num_issues']} | "
        f"Total invested: Rs.{row['total_investment_cr']} Cr | Avg listing gain: {row['avg_listing_gain_percent']}% | "
        f"Avg current gain: {row['avg_current_gain_percent']}%"
    )


def load_subscription_snapshot(path: Path = DEFAULT_SUBSCRIPTION_PATH) -> dict | None:
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


def get_fresh_snapshot(path: Path = DEFAULT_SUBSCRIPTION_PATH) -> dict | None:
    """The cached snapshot if fresh enough, otherwise a newly-scraped one.

    Falls back to a stale cached snapshot (rather than failing outright) if
    a refresh is needed but the scrape itself fails.
    """
    cached = load_subscription_snapshot(path)
    if cached is not None and not _is_stale(cached):
        return cached

    from data.scrape_subscription import fetch_subscription_data  # deferred: avoids importing playwright unless actually scraping

    try:
        logger.info("Subscription snapshot missing or stale, scraping a fresh one...")
        fresh = fetch_subscription_data()
        fresh["scraped_at"] = datetime.now(timezone.utc).isoformat()
        _save_snapshot(fresh, path)
        return fresh
    except Exception as exc:
        if cached is not None:
            logger.warning("Subscription refresh failed (%s), falling back to stale cached snapshot", exc)
            return cached
        logger.warning("Subscription refresh failed and no cached snapshot exists: %s", exc)
        return None


def answer_subscription_question(question: str, snapshot_path: Path = DEFAULT_SUBSCRIPTION_PATH) -> str:
    """Answer a subscription/anchor-investor question from a fresh-as-possible snapshot, or explain if none is available."""
    snapshot = get_fresh_snapshot(snapshot_path)
    if snapshot is None:
        return (
            "I couldn't fetch or find any subscription/anchor investor data -- the source site may be "
            "unreachable right now. Try again shortly."
        )

    subscription_context = (
        "\n".join(_format_subscription_row(r) for r in snapshot["subscription"])
        if snapshot["subscription"] else "No IPOs currently in the subscription snapshot."
    )
    anchor_context = (
        "\n".join(_format_anchor_row(r) for r in snapshot["anchors"])
        if snapshot["anchors"] else "No anchor investor data in the snapshot."
    )

    user_message = (
        f'Snapshot scraped at: {snapshot["scraped_at"]}\n\n'
        f"Subscription status:\n{subscription_context}\n\n"
        f"Anchor investors (aggregated across issues):\n{anchor_context}\n\n"
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

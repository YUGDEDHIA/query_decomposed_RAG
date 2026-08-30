"""Answer post-listing IPO performance questions from a scraped snapshot plus tracked history.

Reads data/listing_performance/latest.json (see
data/scrape_listing_performance.py) for "how is X doing right now" /
"best listing gains recently" questions, same live-market-data shape as
src/gmp_answerer.py and src/subscription_answerer.py.

Unlike those two, this also builds real historical trend data over time:
chittorgarh's tracker page only ever shows the current market price for
~25 recent IPOs, not a multi-year history endpoint -- so each time this
snapshot is refreshed (at most once per UTC calendar day, see _is_stale),
every already-listed row is appended as one dated line to
data/listing_performance/history.jsonl. Depth of the trend is therefore
bounded by how long/often this feature has been in use, same tradeoff as
src/on_demand_ingest.py growing the RHP corpus one asked-about company at a
time -- stated explicitly to the model (see _SYSTEM_PROMPT) rather than
implying more history exists than actually does.

Ranking ("best/worst listing gains") and trend numbers are computed here in
plain Python (current_gain, top_performers, trend_for_company) and handed
to the LLM already-correct -- it only phrases them in prose, the same
"compute in code, don't let the model eyeball numbers" split as
src/answerer.py's citations being computed from the input chunks rather
than self-reported.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import ollama

MODEL = "qwen2.5:14b-instruct"
DEFAULT_LATEST_PATH = Path(__file__).resolve().parent.parent / "data" / "listing_performance" / "latest.json"
DEFAULT_HISTORY_PATH = Path(__file__).resolve().parent.parent / "data" / "listing_performance" / "history.jsonl"

_SYSTEM_PROMPT = """You answer questions about how IPOs have performed since listing (listing-day gains, current market price vs. issue price, and day-by-day trend where available), using data computed and provided below.

Rules:
- Use ONLY the numbers provided -- they are already computed (gains, rankings, trend), do not recompute or second-guess them, and do not invent figures for a company not present in the data.
- Tracked daily trend data only goes back as far as this system has actually scraped it -- if trend history for a company is short or missing, say so plainly rather than implying a longer history exists.
- Current market price is a point-in-time snapshot, not investment advice -- frame it as informational, not a recommendation.
- If the data doesn't cover what's asked (e.g. a company not in the snapshot, or one that hasn't listed yet), say so plainly.
"""

logger = logging.getLogger("listing_performance_answerer")

_NON_ALNUM_RE = re.compile(r"[^a-z0-9 ]")
_SUFFIX_WORDS = {"ltd", "limited"}


def current_gain(row: dict) -> float | None:
    """Best available gain% vs. issue price for `row`: current market gain if the
    company has traded since, else its one-time listing-day gain, else None if
    it hasn't listed yet."""
    if row.get("current_gain_percent") is not None:
        return row["current_gain_percent"]
    return row.get("listing_day_gain_percent")


def top_performers(rows: list[dict], n: int = 5) -> list[dict]:
    """The `n` rows with the highest current_gain, excluding not-yet-listed rows."""
    scored = [r for r in rows if current_gain(r) is not None]
    return sorted(scored, key=current_gain, reverse=True)[:n]


def trend_for_company(history: list[dict], company: str) -> list[dict]:
    """Tracked daily history rows for `company`, oldest first."""
    matches = [r for r in history if r.get("company") == company]
    return sorted(matches, key=lambda r: r["date"])


def _normalize_company_name(name: str) -> str:
    words = _NON_ALNUM_RE.sub(" ", name.lower()).split()
    return " ".join(w for w in words if w not in _SUFFIX_WORDS)


def _find_company_in_question(question: str, companies: list[str]) -> str | None:
    normalized_question = _NON_ALNUM_RE.sub(" ", question.lower())
    for company in sorted(companies, key=len, reverse=True):
        core = _normalize_company_name(company)
        if core and core in normalized_question:
            return company
    return None


def _format_performance_row(row: dict) -> str:
    gain = current_gain(row)
    gain_str = f"{gain:+.2f}%" if gain is not None else "not listed yet"
    return (
        f"{row['company']} ({row['issue_category']}) | Listed: {row['listing_date'] or 'not yet'} | "
        f"Issue price: Rs.{row['issue_price']} | Current price: Rs.{row['market_price']} | "
        f"Current gain vs. issue price: {gain_str}"
    )


def _format_trend(company: str, trend_rows: list[dict]) -> str:
    if not trend_rows:
        return f"No tracked daily history yet for {company} -- history builds up over time as this data is queried."
    return "\n".join(f"{r['date']}: {current_gain(r):+.2f}%" for r in trend_rows)


def load_listing_snapshot(path: Path = DEFAULT_LATEST_PATH) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_history(path: Path = DEFAULT_HISTORY_PATH) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _is_stale(snapshot: dict) -> bool:
    """Stale once the UTC calendar date has rolled over since the last scrape --
    daily granularity, not a rolling time window, so at most one history row
    is appended per company per day regardless of what hour it's first asked."""
    scraped_at = datetime.fromisoformat(snapshot["scraped_at"])
    return scraped_at.astimezone(timezone.utc).date() < datetime.now(timezone.utc).date()


def _save_snapshot(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _append_history(rows: list[dict], history_path: Path) -> None:
    listed_rows = [r for r in rows if r.get("market_price") is not None]
    if not listed_rows:
        return
    today = datetime.now(timezone.utc).date().isoformat()
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "a") as f:
        for row in listed_rows:
            f.write(json.dumps({"date": today, **row}) + "\n")


def get_fresh_listing_snapshot(
    path: Path = DEFAULT_LATEST_PATH, history_path: Path = DEFAULT_HISTORY_PATH,
) -> dict | None:
    """The cached snapshot if it's from today (UTC), otherwise a newly-scraped
    one -- which also appends today's already-listed rows to history_path.

    Falls back to a stale cached snapshot (rather than failing outright) if
    a refresh is needed but the scrape itself fails.
    """
    cached = load_listing_snapshot(path)
    if cached is not None and not _is_stale(cached):
        return cached

    from data.scrape_listing_performance import fetch_listing_performance_data  # deferred: avoids importing playwright unless actually scraping

    try:
        logger.info("Listing performance snapshot missing or stale, scraping a fresh one...")
        fresh = fetch_listing_performance_data()
        fresh["scraped_at"] = datetime.now(timezone.utc).isoformat()
        _save_snapshot(fresh, path)
        _append_history(fresh["rows"], history_path)
        return fresh
    except Exception as exc:
        if cached is not None:
            logger.warning("Listing performance refresh failed (%s), falling back to stale cached snapshot", exc)
            return cached
        logger.warning("Listing performance refresh failed and no cached snapshot exists: %s", exc)
        return None


def answer_listing_performance_question(
    question: str, snapshot_path: Path = DEFAULT_LATEST_PATH, history_path: Path = DEFAULT_HISTORY_PATH,
) -> str:
    """Answer a post-listing-performance question from a fresh-as-possible snapshot plus tracked history."""
    snapshot = get_fresh_listing_snapshot(snapshot_path, history_path)
    if snapshot is None:
        return (
            "I couldn't fetch or find any IPO listing-performance data -- the source site may be "
            "unreachable right now. Try again shortly."
        )

    rows = snapshot["rows"]
    ranked_context = (
        "\n".join(_format_performance_row(r) for r in top_performers(rows, n=10))
        or "No already-listed IPOs with current price data in the snapshot."
    )

    trend_block = ""
    named_company = _find_company_in_question(question, [r["company"] for r in rows])
    if named_company:
        history = load_history(history_path)
        trend_block = (
            f"\n\nTracked daily trend for {named_company}:\n"
            f"{_format_trend(named_company, trend_for_company(history, named_company))}"
        )

    user_message = (
        f'Snapshot scraped at: {snapshot["scraped_at"]}\n\n'
        f"Top current performers (best gain vs. issue price first):\n{ranked_context}"
        f"{trend_block}\n\n"
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

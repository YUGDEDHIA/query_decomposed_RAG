"""Scrape live IPO listing-performance data (issue price vs. listing price vs.
current market price) into data/listing_performance/latest.json.

Source: chittorgarh.com's IPO performance tracker,
/report/ipo-performance-report-listing-current-gain/125/all/ -- same site
and same robots.txt-permissive shape as data/scrape_subscription.py.

Verified directly (not guessed): the table (one row per IPO, mainboard and
SME mixed together via an "Issue Category" column) shows ~25 of the most
recent IPOs, mixing not-yet-listed rows (blank listing/price columns) with
already-listed ones -- there's no separate "give me full multi-year
history" endpoint on this report, which is exactly why the historical
trend in this project is built by *this scraper's own repeated runs*
(src/listing_performance_answerer.py appends one dated row per already-
listed company to data/listing_performance/history.jsonl each time it
scrapes), not by backfilling from the source.

"Close Price On Listing" and "Market Price" cells combine a price and a
parenthesized gain-vs-issue-price percentage in one string, e.g.
"248.44 (23.6%)" -- split into separate numeric fields by
_parse_price_and_gain. "52 Week High/Low" cells instead pair a price with a
date ("100.80 (21-Aug-2026)"), a different shape, and aren't needed for
gain tracking, so they're left as raw text.

Usage:
    python data/scrape_listing_performance.py   # writes data/listing_performance/latest.json
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from playwright.sync_api import sync_playwright

PERFORMANCE_URL = "https://www.chittorgarh.com/report/ipo-performance-report-listing-current-gain/125/all/"
USER_AGENT = (
    "query-decomposed-RAG-research-scraper/1.0 "
    "(non-commercial educational RAG project; fetches public IPO performance data)"
)

PERFORMANCE_COLUMNS = [
    "company", "issue_category", "opening_date", "listing_date", "issue_amount_cr",
    "subscription_x", "issue_price", "listing_close_price", "market_price",
    "week_52_high", "week_52_low",
]

_NUMBER_RE = re.compile(r"[,\s]")
_PRICE_GAIN_RE = re.compile(r"^([\d,]+\.?\d*)\s*\(([-\d.]+)%\)$")

logger = logging.getLogger("listing_performance_scraper")


def _parse_number(text: str) -> float | None:
    text = (text or "").strip()
    if not text or text == "—":
        return None
    try:
        return float(_NUMBER_RE.sub("", text))
    except ValueError:
        return None


def _parse_price_and_gain(text: str) -> tuple[float | None, float | None]:
    """"248.44 (23.6%)" -> (248.44, 23.6). Not-yet-listed / unparseable -> (None, None)."""
    text = (text or "").strip()
    if not text or text == "—":
        return None, None
    match = _PRICE_GAIN_RE.match(text)
    if not match:
        return None, None
    return float(match.group(1).replace(",", "")), float(match.group(2))


def _table_rows(table, columns: list[str]) -> list[dict]:
    rows = []
    for tr in table.query_selector_all("tbody tr"):
        cells = [td.inner_text().strip() for td in tr.query_selector_all("td")]
        if len(cells) != len(columns):
            continue
        rows.append({col: val for col, val in zip(columns, cells)})
    return rows


def fetch_listing_performance_data() -> dict:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(user_agent=USER_AGENT)
        page.goto(PERFORMANCE_URL, wait_until="domcontentloaded", timeout=30_000)
        page.wait_for_selector("table tbody tr", timeout=15_000)

        tables = page.query_selector_all("table")
        if not tables:
            browser.close()
            raise RuntimeError(f"No table found on {PERFORMANCE_URL}")

        rows = _table_rows(tables[0], PERFORMANCE_COLUMNS)
        browser.close()

    for row in rows:
        row["issue_amount_cr"] = _parse_number(row["issue_amount_cr"])
        row["subscription_x"] = _parse_number(row["subscription_x"])
        row["issue_price"] = _parse_number(row["issue_price"])
        row["listing_close_price"], row["listing_day_gain_percent"] = _parse_price_and_gain(row["listing_close_price"])
        row["market_price"], row["current_gain_percent"] = _parse_price_and_gain(row["market_price"])

    return {"rows": rows}


def main() -> None:
    default_out = Path(__file__).resolve().parent / "listing_performance" / "latest.json"
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=default_out)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    logger.info("Fetching %s", PERFORMANCE_URL)
    data = fetch_listing_performance_data()
    data["scraped_at"] = datetime.now(timezone.utc).isoformat()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(data, f, indent=2)

    listed = sum(1 for r in data["rows"] if r["market_price"] is not None)
    logger.info("Wrote %d rows (%d already listed) to %s", len(data["rows"]), listed, args.out)


if __name__ == "__main__":
    main()

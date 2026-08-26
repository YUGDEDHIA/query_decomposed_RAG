"""Scrape live IPO Grey Market Premium (GMP) data into data/gmp/latest.json.

Source: https://www.ipomarket.in/gmp -- robots.txt allows /gmp (only
disallows /api/, /admin/, /dashboard/). Chosen after checking several
candidate GMP trackers: this one is server-reachable, updates every ~30
min, and -- usefully -- already carries OPEN/UPCOMING/CLOSED status with
open/close dates, not just the GMP number itself.

Verified directly (not guessed): the Mainboard and SME tables are
NOT present in the raw server HTML -- only a small "today's top GMPs"
highlights table is. The real Mainboard/SME tables render client-side
after hydration, so this needs a real browser (Playwright), not
requests+BeautifulSoup like data/scrape_sebi_rhps.py.

GMP is unofficial, unregulated grey-market chatter, not a regulatory
disclosure -- treat it as informal market sentiment, not fact, same
caveat the site itself carries.

Usage:
    python data/scrape_gmp.py                # writes data/gmp/latest.json
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from playwright.sync_api import sync_playwright

GMP_URL = "https://www.ipomarket.in/gmp"
USER_AGENT = (
    "query-decomposed-RAG-research-scraper/1.0 "
    "(non-commercial educational RAG project; fetches public GMP listings)"
)
LIVE_COLUMNS = [
    "company", "open_date", "price_band", "gmp_rupees", "gmp_percent",
    "est_listing", "_trend", "status", "close_date", "updated", "score", "_apply",
]
RECENT_COLUMNS = ["company", "listed_date", "issue_price", "final_gmp_percent", "listing_price", "actual_gain"]

logger = logging.getLogger("gmp_scraper")

_RUPEE_RE = re.compile(r"[₹,]")


def _parse_gmp_rupees(text: str) -> int | None:
    text = text.strip()
    if not text or text == "—":
        return None
    try:
        return int(_RUPEE_RE.sub("", text))
    except ValueError:
        return None


def _table_rows(table, columns: list[str]) -> list[dict]:
    rows = []
    for tr in table.query_selector_all("tbody tr"):
        cells = [td.inner_text().strip() for td in tr.query_selector_all("td")]
        if len(cells) != len(columns):
            continue
        row = {col: val for col, val in zip(columns, cells) if not col.startswith("_")}
        rows.append(row)
    return rows


def fetch_gmp_data() -> dict:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(user_agent=USER_AGENT)
        page.goto(GMP_URL, wait_until="networkidle", timeout=30_000)

        tables = page.query_selector_all("table")
        if len(tables) < 3:
            browser.close()
            raise RuntimeError(f"Expected at least 3 tables (mainboard/SME/recently-listed), found {len(tables)}")

        mainboard = _table_rows(tables[0], LIVE_COLUMNS)
        sme = _table_rows(tables[1], LIVE_COLUMNS)
        recently_listed = _table_rows(tables[2], RECENT_COLUMNS)

        browser.close()

    for row in mainboard + sme:
        row["gmp_rupees_numeric"] = _parse_gmp_rupees(row.get("gmp_rupees", ""))

    return {"mainboard": mainboard, "sme": sme, "recently_listed": recently_listed}


def main() -> None:
    default_out = Path(__file__).resolve().parent / "gmp" / "latest.json"
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=default_out)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    logger.info("Fetching %s", GMP_URL)
    data = fetch_gmp_data()
    data["scraped_at"] = datetime.now(timezone.utc).isoformat()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(
        "Wrote %d mainboard, %d SME, %d recently-listed rows to %s",
        len(data["mainboard"]), len(data["sme"]), len(data["recently_listed"]), args.out,
    )


if __name__ == "__main__":
    main()

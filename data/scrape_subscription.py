"""Scrape live IPO subscription status and anchor investor data into data/subscription/latest.json.

Source: chittorgarh.com -- robots.txt is `Allow: /` site-wide (only a
handful of specific bot user-agents and one unrelated discussion page are
disallowed), so both report pages used here are fair game:
- /report/ipo-subscription-status-live-bidding-data-bse-nse/21/ (QIB/NII/
  Retail/Employee/Shareholder subscription multiples, per open or recently
  closed IPO)
- /report/anchor-investors-list/133/all/ (anchor investors ranked by total
  investment across issues, with average listing/current gain per investor
  -- this is aggregated *by anchor investor*, not a per-company breakdown
  of who anchored one specific IPO; chittorgarh doesn't expose that on a
  single report page the way it does subscription status)

Verified directly (not guessed): a plain requests-style fetch of either
page 403s, and even a Playwright `wait_until="networkidle"` goto times out
(ongoing background trackers never go idle) -- but `domcontentloaded` +
`wait_for_selector("table tbody tr")` reliably gets the fully-populated
table, same real-browser-required shape as data/scrape_gmp.py.

The subscription table's first column ("Company") carries a trailing
single-letter status suffix ("O" = currently open for bidding, "P" =
closed but not yet listed, no suffix = fully closed/listed) -- split off
into a separate `status_code` field rather than left in `company`, so
company-name matching elsewhere doesn't have to know about it.

Usage:
    python data/scrape_subscription.py     # writes data/subscription/latest.json
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from playwright.sync_api import sync_playwright

SUBSCRIPTION_URL = "https://www.chittorgarh.com/report/ipo-subscription-status-live-bidding-data-bse-nse/21/"
ANCHOR_URL = "https://www.chittorgarh.com/report/anchor-investors-list/133/all/"
USER_AGENT = (
    "query-decomposed-RAG-research-scraper/1.0 "
    "(non-commercial educational RAG project; fetches public IPO subscription/anchor data)"
)

SUBSCRIPTION_COLUMNS = [
    "company", "closing_date", "total_issue_amount_cr", "qib_x", "snii_x", "bnii_x",
    "nii_x", "retail_x", "employee_x", "shareholder_x", "others_x", "total_x",
    "applications", "subscription_as_on",
]
ANCHOR_COLUMNS = [
    "anchor_investor", "num_issues", "total_investment_cr", "total_issue_amount_cr",
    "avg_issue_amount_cr", "avg_pe", "avg_listing_gain_percent", "avg_current_gain_percent",
    "avg_subscription_x",
]

_STATUS_SUFFIX_RE = re.compile(r"\s([A-Z])$")
_NUMBER_RE = re.compile(r"[,\s]")

logger = logging.getLogger("subscription_scraper")


def _parse_number(text: str) -> float | None:
    text = (text or "").strip()
    if not text or text == "—":
        return None
    try:
        return float(_NUMBER_RE.sub("", text))
    except ValueError:
        return None


def _split_company_status(text: str) -> tuple[str, str | None]:
    """"Hy-Tech Engineers Ltd. O" -> ("Hy-Tech Engineers Ltd.", "O")."""
    match = _STATUS_SUFFIX_RE.search(text.strip())
    if match:
        return text[: match.start()].strip(), match.group(1)
    return text.strip(), None


def _table_rows(table, columns: list[str]) -> list[dict]:
    rows = []
    for tr in table.query_selector_all("tbody tr"):
        cells = [td.inner_text().strip() for td in tr.query_selector_all("td")]
        if len(cells) != len(columns):
            continue
        rows.append({col: val for col, val in zip(columns, cells)})
    return rows


def _load_table(page, url: str):
    page.goto(url, wait_until="domcontentloaded", timeout=30_000)
    page.wait_for_selector("table tbody tr", timeout=15_000)
    tables = page.query_selector_all("table")
    if not tables:
        raise RuntimeError(f"No table found on {url}")
    return tables[0]


def fetch_subscription_data() -> dict:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(user_agent=USER_AGENT)

        subscription = _table_rows(_load_table(page, SUBSCRIPTION_URL), SUBSCRIPTION_COLUMNS)
        anchors = _table_rows(_load_table(page, ANCHOR_URL), ANCHOR_COLUMNS)

        browser.close()

    for row in subscription:
        row["company"], row["status_code"] = _split_company_status(row["company"])
        for key in (
            "total_issue_amount_cr", "qib_x", "snii_x", "bnii_x", "nii_x",
            "retail_x", "employee_x", "shareholder_x", "others_x", "total_x", "applications",
        ):
            row[key] = _parse_number(row[key])

    for row in anchors:
        for key in (
            "num_issues", "total_investment_cr", "total_issue_amount_cr", "avg_issue_amount_cr",
            "avg_pe", "avg_listing_gain_percent", "avg_current_gain_percent", "avg_subscription_x",
        ):
            row[key] = _parse_number(row[key])

    return {"subscription": subscription, "anchors": anchors}


def main() -> None:
    default_out = Path(__file__).resolve().parent / "subscription" / "latest.json"
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=default_out)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    logger.info("Fetching %s and %s", SUBSCRIPTION_URL, ANCHOR_URL)
    data = fetch_subscription_data()
    data["scraped_at"] = datetime.now(timezone.utc).isoformat()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(
        "Wrote %d subscription rows, %d anchor investor rows to %s",
        len(data["subscription"]), len(data["anchors"]), args.out,
    )


if __name__ == "__main__":
    main()

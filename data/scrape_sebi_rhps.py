"""Download RHP PDFs from SEBI's public filings listing into data/raw/.

Source (Filings > Public Issues > Red Herring Documents filed with ROC):
https://www.sebi.gov.in/sebiweb/home/HomeAction.do?doListing=yes&sid=3&ssid=15&smid=11

robots.txt only disallows /js and /css, so this is fair game -- but pagination
past page 1 is a JS call (searchFormNewsList), not a URL you can increment.
Reverse-engineered from https://www.sebi.gov.in/js/entry.js: it POSTs to an
AJAX endpoint with the section/subsection IDs and a 0-indexed page number.
See fetch_listing_page().

Usage:
    python data/scrape_sebi_rhps.py                # all pages, into data/raw/
    python data/scrape_sebi_rhps.py --limit 5       # smoke test
    python data/scrape_sebi_rhps.py --start-page 10 --pages 5
"""
from __future__ import annotations

import argparse
import logging
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

LISTING_URL = "https://www.sebi.gov.in/sebiweb/home/HomeAction.do?doListing=yes&sid=3&ssid=15&smid=11"
AJAX_URL = "https://www.sebi.gov.in/sebiweb/ajax/home/getnewslistinfo.jsp"
USER_AGENT = (
    "query-decomposed-RAG-research-scraper/1.0 "
    "(non-commercial educational RAG project; fetches public RHP filings)"
)
RETRIES = 3

TITLE_IS_RHP_RE = re.compile(r"-\s*RHP\s*$", re.IGNORECASE)

logger = logging.getLogger("sebi_scraper")


def new_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    # The AJAX pagination endpoint 403s without cookies from this first GET.
    _request(session, "get", LISTING_URL)
    return session


def _request(session: requests.Session, method: str, url: str, **kwargs) -> requests.Response:
    last_exc = None
    for attempt in range(1, RETRIES + 1):
        try:
            resp = getattr(session, method)(url, timeout=30, **kwargs)
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            last_exc = exc
            logger.debug("Attempt %d/%d failed for %s: %s", attempt, RETRIES, url, exc)
            time.sleep(2 * attempt)
    raise last_exc


def fetch_listing_page(session: requests.Session, page_index: int) -> str:
    """Fetch one page (0-indexed, 25 rows/page) of the RHP listing.

    The response is an HTML fragment followed by "#@#" and a breadcrumb
    fragment (per entry.js's response.split("#@#")); only the first half
    is the listing table.
    """
    payload = {
        "nextValue": "1",
        "next": "n",
        "search": "",
        "fromDate": "",
        "toDate": "",
        "fromYear": "",
        "toYear": "",
        "deptId": "",
        "sid": "3",
        "ssid": "15",
        "smid": "11",
        "ssidhidden": "15",
        "intmid": "-1",
        "sText": "Filings",
        "ssText": "Public Issues",
        "smText": "Red Herring Documents filed with ROC",
        "doDirect": str(page_index),
    }
    headers = {"X-Requested-With": "XMLHttpRequest", "Referer": LISTING_URL}
    resp = _request(session, "post", AJAX_URL, data=payload, headers=headers)
    return resp.text.split("#@#")[0]


def search_listing(session: requests.Session, query: str) -> str:
    """Search the RHP listing for `query` (e.g. a company name), same response shape as fetch_listing_page.

    Reverse-engineered from the site's own search "GO" button
    (searchFormNewsList('s','-1')): same AJAX endpoint, `next="s"` instead
    of `"n"`, and a non-empty `search` field. Verified directly: finds an
    exact-title match ("Aastha Spintex" -> "Aastha Spintex Limited - RHP")
    and generalizes to companies not previously seen (a then-currently-open
    IPO not in this project's corpus).
    """
    payload = {
        "nextValue": "1",
        "next": "s",
        "search": query,
        "fromDate": "",
        "toDate": "",
        "fromYear": "",
        "toYear": "",
        "deptId": "",
        "sid": "3",
        "ssid": "15",
        "smid": "11",
        "ssidhidden": "15",
        "intmid": "-1",
        "sText": "Filings",
        "ssText": "Public Issues",
        "smText": "Red Herring Documents filed with ROC",
        "doDirect": "-1",
    }
    headers = {"X-Requested-With": "XMLHttpRequest", "Referer": LISTING_URL}
    resp = _request(session, "post", AJAX_URL, data=payload, headers=headers)
    return resp.text.split("#@#")[0]


def parse_listing_rows(html: str) -> list[tuple[str, str, str]]:
    """Extract (date, title, detail_url) for genuine RHP rows on one page.

    Skips corrigenda/addenda to an RHP -- their titles don't end in "- RHP".
    Also skips the abridged-prospectus link SEBI nests in the same cell: it's
    a separate <a> after (or inside, depending on parser leniency) the real
    one, so taking the cell's *first* link always lands on the RHP detail
    page rather than the short-form PDF.
    """
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", id="sample_1")
    if table is None:
        return []
    rows = []
    for tr in table.find_all("tr"):
        cells = tr.find_all("td")
        if len(cells) != 2:
            continue
        date_text = cells[0].get_text(strip=True)
        link = cells[1].find("a")
        if link is None or not link.get("href"):
            continue
        # The title attribute holds the clean row title even when the link's
        # own text is polluted by a nested abridged-prospectus anchor.
        title = (link.get("title") or link.get_text(strip=True)).split("<br")[0].strip()
        if not TITLE_IS_RHP_RE.search(title):
            continue
        rows.append((date_text, title, link["href"]))
    return rows


def get_pdf_url(session: requests.Session, detail_url: str) -> str | None:
    """Resolve a filing detail page to the real RHP PDF under sebi_data/attachdocs/."""
    resp = _request(session, "get", detail_url)
    soup = BeautifulSoup(resp.text, "html.parser")
    iframe = soup.find("iframe", src=re.compile(r"file="))
    if iframe is None:
        return None
    return iframe["src"].split("file=", 1)[1]


def company_filename(title: str) -> str:
    """"KNACK PACKAGING LIMITED - RHP" -> "KNACK_PACKAGING_LIMITED_RHP.pdf".

    Matches the CompanyName_RHP.pdf convention the commit-2 loader expects
    (company name derived from the filename, not parsed from PDF text).
    """
    company = TITLE_IS_RHP_RE.sub("", title).strip()
    company = re.sub(r"[^A-Za-z0-9]+", "_", company).strip("_")
    return f"{company}_RHP.pdf"


def download_pdf(session: requests.Session, pdf_url: str, dest: Path) -> None:
    resp = _request(session, "get", pdf_url, stream=True)
    tmp = dest.with_suffix(".pdf.part")
    with open(tmp, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 16):
            f.write(chunk)
    tmp.rename(dest)


def run(out_dir: Path, start_page: int, pages: int | None, limit: int | None, delay: float) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    session = new_session()
    downloaded = 0
    page = start_page
    end_page = None if pages is None else start_page + pages

    while end_page is None or page < end_page:
        logger.info("Fetching listing page %d", page)
        rows = parse_listing_rows(fetch_listing_page(session, page))
        if not rows:
            logger.info("No more rows at page %d, stopping.", page)
            break

        for date_text, title, detail_url in rows:
            if limit is not None and downloaded >= limit:
                logger.info("Reached --limit=%d, stopping.", limit)
                return downloaded

            dest = out_dir / company_filename(title)
            if dest.exists():
                logger.debug("Already have %s, skipping", dest.name)
                continue

            time.sleep(delay)
            try:
                pdf_url = get_pdf_url(session, detail_url)
                if pdf_url is None:
                    logger.warning("No PDF link found on %s, skipping", detail_url)
                    continue
                time.sleep(delay)
                download_pdf(session, pdf_url, dest)
            except requests.RequestException as exc:
                logger.warning("Failed to fetch %s (%s): %s", title, date_text, exc)
                continue

            downloaded += 1
            logger.info("Saved %s (%s)", dest.name, date_text)

        page += 1
        time.sleep(delay)

    logger.info("Done. Downloaded %d RHPs to %s", downloaded, out_dir)
    return downloaded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "raw",
                         help="Destination directory (default: data/raw)")
    parser.add_argument("--start-page", type=int, default=0, help="0-indexed listing page to start from")
    parser.add_argument("--pages", type=int, default=None, help="Number of listing pages to walk (default: all)")
    parser.add_argument("--limit", type=int, default=None, help="Stop after downloading this many RHPs")
    parser.add_argument("--delay", type=float, default=1.5, help="Seconds to sleep between requests")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                         format="%(asctime)s %(levelname)s %(message)s")

    run(args.out_dir, args.start_page, args.pages, args.limit, args.delay)


if __name__ == "__main__":
    main()

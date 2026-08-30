from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from src.listing_performance_answerer import (
    _find_company_in_question,
    _is_stale,
    answer_listing_performance_question,
    current_gain,
    get_fresh_listing_snapshot,
    load_history,
    load_listing_snapshot,
    top_performers,
    trend_for_company,
)


def _row(company="Acme Ltd.", market_price=248.44, current_gain_percent=23.58, listing_day_gain_percent=23.6):
    return {
        "company": company, "issue_category": "Mainboard", "opening_date": "17-Aug-2026",
        "listing_date": "24-Aug-2026", "issue_amount_cr": 1700.0, "subscription_x": None,
        "issue_price": 201.0, "listing_close_price": 248.44, "market_price": market_price,
        "week_52_high": "", "week_52_low": "",
        "listing_day_gain_percent": listing_day_gain_percent, "current_gain_percent": current_gain_percent,
    }


def _unlisted_row(company="Not Yet Ltd."):
    return _row(company=company, market_price=None, current_gain_percent=None, listing_day_gain_percent=None)


def _snapshot(age: timedelta, company="Acme Ltd.") -> dict:
    scraped_at = datetime.now(timezone.utc) - age
    return {"scraped_at": scraped_at.isoformat(), "rows": [_row(company)]}


# -- pure computation functions --

def test_current_gain_prefers_market_gain_over_listing_day_gain():
    row = _row(current_gain_percent=23.58, listing_day_gain_percent=23.6)
    assert current_gain(row) == 23.58


def test_current_gain_falls_back_to_listing_day_gain_when_no_market_price():
    row = _row(market_price=None, current_gain_percent=None, listing_day_gain_percent=23.6)
    assert current_gain(row) == 23.6


def test_current_gain_none_for_not_yet_listed_company():
    assert current_gain(_unlisted_row()) is None


def test_top_performers_sorts_by_current_gain_descending():
    rows = [_row("Low Gain", current_gain_percent=1.0), _row("High Gain", current_gain_percent=50.0)]
    result = top_performers(rows, n=5)
    assert [r["company"] for r in result] == ["High Gain", "Low Gain"]


def test_top_performers_excludes_not_yet_listed_rows():
    rows = [_row("Listed"), _unlisted_row("Pending")]
    result = top_performers(rows, n=5)
    assert [r["company"] for r in result] == ["Listed"]


def test_top_performers_respects_n():
    rows = [_row(f"Co {i}", current_gain_percent=float(i)) for i in range(10)]
    assert len(top_performers(rows, n=3)) == 3


def test_trend_for_company_filters_and_sorts_chronologically():
    history = [
        {"date": "2026-08-24", "company": "Acme Ltd.", "current_gain_percent": 5.0},
        {"date": "2026-08-22", "company": "Acme Ltd.", "current_gain_percent": 1.0},
        {"date": "2026-08-23", "company": "Other Ltd.", "current_gain_percent": 9.0},
        {"date": "2026-08-23", "company": "Acme Ltd.", "current_gain_percent": 3.0},
    ]
    result = trend_for_company(history, "Acme Ltd.")
    assert [r["date"] for r in result] == ["2026-08-22", "2026-08-23", "2026-08-24"]


def test_trend_for_company_returns_empty_for_untracked_company():
    assert trend_for_company([{"date": "2026-08-24", "company": "Other Ltd."}], "Acme Ltd.") == []


def test_find_company_in_question_matches_despite_missing_ltd_suffix():
    # Real gap this exists to close: chittorgarh names companies "Hy-Tech
    # Engineers Ltd." but a question naturally omits the suffix entirely.
    companies = ["Hy-Tech Engineers Ltd."]
    assert _find_company_in_question("How has Hy-Tech Engineers performed since listing?", companies) == "Hy-Tech Engineers Ltd."


def test_find_company_in_question_returns_none_for_unrelated_question():
    assert _find_company_in_question("Which IPOs had the best listing gains?", ["Acme Ltd."]) is None


# -- snapshot lifecycle (mirrors src/gmp_answerer.py's tests) --

def test_load_listing_snapshot_returns_none_when_missing(tmp_path):
    assert load_listing_snapshot(tmp_path / "nope.json") is None


def test_is_stale_uses_calendar_date_not_a_rolling_window():
    # A snapshot from 2 hours ago today is fresh; one from yesterday (even
    # if less than 24h old close to midnight) is stale -- daily, not hourly.
    assert _is_stale(_snapshot(timedelta(hours=2))) is False
    assert _is_stale(_snapshot(timedelta(days=1, hours=1))) is True


def test_get_fresh_listing_snapshot_reuses_same_day_cache_without_scraping(tmp_path, monkeypatch):
    path = tmp_path / "latest.json"
    history_path = tmp_path / "history.jsonl"
    path.write_text(json.dumps(_snapshot(timedelta(hours=1))))

    def _fail(*args, **kwargs):
        raise AssertionError("should not scrape when cache is from today")

    monkeypatch.setattr("data.scrape_listing_performance.fetch_listing_performance_data", _fail)

    result = get_fresh_listing_snapshot(path, history_path)
    assert result["rows"][0]["company"] == "Acme Ltd."


def test_get_fresh_listing_snapshot_rescrapes_and_appends_history_when_stale(tmp_path, monkeypatch):
    path = tmp_path / "latest.json"
    history_path = tmp_path / "history.jsonl"
    path.write_text(json.dumps(_snapshot(timedelta(days=1, hours=1), company="Old Corp")))

    monkeypatch.setattr(
        "data.scrape_listing_performance.fetch_listing_performance_data",
        lambda: {"rows": [_row("New Corp"), _unlisted_row("Pending Corp")]},
    )

    result = get_fresh_listing_snapshot(path, history_path)
    assert result["rows"][0]["company"] == "New Corp"
    assert json.loads(path.read_text())["rows"][0]["company"] == "New Corp"

    history = load_history(history_path)
    # Only the already-listed row gets a history entry -- the not-yet-listed
    # one has no market_price to track yet.
    assert [r["company"] for r in history] == ["New Corp"]
    assert history[0]["date"] == datetime.now(timezone.utc).date().isoformat()


def test_get_fresh_listing_snapshot_falls_back_to_stale_cache_on_scrape_failure(tmp_path, monkeypatch):
    path = tmp_path / "latest.json"
    history_path = tmp_path / "history.jsonl"
    path.write_text(json.dumps(_snapshot(timedelta(days=1, hours=1), company="Old Corp")))

    def _fail(*args, **kwargs):
        raise RuntimeError("site unreachable")

    monkeypatch.setattr("data.scrape_listing_performance.fetch_listing_performance_data", _fail)

    result = get_fresh_listing_snapshot(path, history_path)
    assert result["rows"][0]["company"] == "Old Corp"


def test_get_fresh_listing_snapshot_returns_none_when_no_cache_and_scrape_fails(tmp_path, monkeypatch):
    def _fail(*args, **kwargs):
        raise RuntimeError("site unreachable")

    monkeypatch.setattr("data.scrape_listing_performance.fetch_listing_performance_data", _fail)

    assert get_fresh_listing_snapshot(tmp_path / "missing.json", tmp_path / "history.jsonl") is None


def test_answer_listing_performance_question_with_no_snapshot_and_failed_scrape_short_circuits_without_llm_call(tmp_path, monkeypatch):
    # No Ollama call happens here -- this must work without a live model.
    def _fail(*args, **kwargs):
        raise RuntimeError("site unreachable")

    monkeypatch.setattr("data.scrape_listing_performance.fetch_listing_performance_data", _fail)

    answer = answer_listing_performance_question(
        "how has Acme performed since listing?",
        snapshot_path=tmp_path / "missing.json", history_path=tmp_path / "history.jsonl",
    )
    assert "couldn't fetch" in answer.lower()

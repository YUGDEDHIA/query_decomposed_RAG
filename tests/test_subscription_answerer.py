from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from src.subscription_answerer import (
    _format_anchor_row,
    _format_subscription_row,
    _is_stale,
    answer_subscription_question,
    get_fresh_snapshot,
    load_subscription_snapshot,
)


def _sub_row(company="Acme Ltd.", status_code="O", total_x=1.5):
    return {
        "company": company, "closing_date": "28-Aug-2026", "total_issue_amount_cr": 175.06,
        "qib_x": 0.29, "snii_x": 0.26, "bnii_x": 0.07, "nii_x": 0.13, "retail_x": 0.21,
        "employee_x": None, "shareholder_x": None, "others_x": None, "total_x": total_x,
        "applications": 10699.0, "subscription_as_on": "25-Aug-2026 14:43", "status_code": status_code,
    }


def _anchor_row(name="ICICI PRUDENTIAL MUTUAL FUND"):
    return {
        "anchor_investor": name, "num_issues": 23.0, "total_investment_cr": 2704.75,
        "total_issue_amount_cr": 60031.31, "avg_issue_amount_cr": 2610.06, "avg_pe": 85.18,
        "avg_listing_gain_percent": 11.81, "avg_current_gain_percent": 44.52, "avg_subscription_x": 20.33,
    }


def _snapshot(age: timedelta, company="Acme Ltd.") -> dict:
    scraped_at = datetime.now(timezone.utc) - age
    return {
        "scraped_at": scraped_at.isoformat(),
        "subscription": [_sub_row(company)],
        "anchors": [_anchor_row()],
    }


def test_format_subscription_row_includes_key_fields():
    text = _format_subscription_row(_sub_row())
    assert "Acme Ltd." in text
    assert "1.5x" in text
    assert "O" in text


def test_format_subscription_row_labels_no_suffix_as_closed():
    text = _format_subscription_row(_sub_row(status_code=None))
    assert "CLOSED" in text


def test_format_anchor_row_includes_key_fields():
    text = _format_anchor_row(_anchor_row())
    assert "ICICI PRUDENTIAL MUTUAL FUND" in text
    assert "23" in text


def test_load_subscription_snapshot_returns_none_when_missing(tmp_path):
    assert load_subscription_snapshot(tmp_path / "nope.json") is None


def test_load_subscription_snapshot_reads_real_file(tmp_path):
    path = tmp_path / "latest.json"
    path.write_text(json.dumps(_snapshot(timedelta(minutes=1))))

    loaded = load_subscription_snapshot(path)
    assert loaded["subscription"][0]["company"] == "Acme Ltd."


def test_is_stale():
    assert _is_stale(_snapshot(timedelta(minutes=5))) is False
    assert _is_stale(_snapshot(timedelta(minutes=30))) is True


def test_get_fresh_snapshot_reuses_fresh_cache_without_scraping(tmp_path, monkeypatch):
    path = tmp_path / "latest.json"
    path.write_text(json.dumps(_snapshot(timedelta(minutes=2))))

    def _fail(*args, **kwargs):
        raise AssertionError("should not scrape when cache is fresh")

    monkeypatch.setattr("data.scrape_subscription.fetch_subscription_data", _fail)

    result = get_fresh_snapshot(path)
    assert result["subscription"][0]["company"] == "Acme Ltd."


def test_get_fresh_snapshot_rescrapes_when_stale(tmp_path, monkeypatch):
    path = tmp_path / "latest.json"
    path.write_text(json.dumps(_snapshot(timedelta(minutes=30), company="Old Corp")))

    monkeypatch.setattr(
        "data.scrape_subscription.fetch_subscription_data",
        lambda: {"subscription": [_sub_row("New Corp")], "anchors": [_anchor_row()]},
    )

    result = get_fresh_snapshot(path)
    assert result["subscription"][0]["company"] == "New Corp"
    # The refreshed snapshot should also be persisted back to disk.
    assert json.loads(path.read_text())["subscription"][0]["company"] == "New Corp"


def test_get_fresh_snapshot_falls_back_to_stale_cache_on_scrape_failure(tmp_path, monkeypatch):
    path = tmp_path / "latest.json"
    path.write_text(json.dumps(_snapshot(timedelta(minutes=30), company="Old Corp")))

    def _fail(*args, **kwargs):
        raise RuntimeError("site unreachable")

    monkeypatch.setattr("data.scrape_subscription.fetch_subscription_data", _fail)

    result = get_fresh_snapshot(path)
    assert result["subscription"][0]["company"] == "Old Corp"  # stale, but still usable


def test_get_fresh_snapshot_returns_none_when_no_cache_and_scrape_fails(tmp_path, monkeypatch):
    def _fail(*args, **kwargs):
        raise RuntimeError("site unreachable")

    monkeypatch.setattr("data.scrape_subscription.fetch_subscription_data", _fail)

    assert get_fresh_snapshot(tmp_path / "missing.json") is None


def test_answer_subscription_question_with_no_snapshot_and_failed_scrape_short_circuits_without_llm_call(tmp_path, monkeypatch):
    # No Ollama call happens here -- this must work without a live model.
    def _fail(*args, **kwargs):
        raise RuntimeError("site unreachable")

    monkeypatch.setattr("data.scrape_subscription.fetch_subscription_data", _fail)

    answer = answer_subscription_question("what is the subscription for Acme?", snapshot_path=tmp_path / "missing.json")
    assert "couldn't fetch" in answer.lower()

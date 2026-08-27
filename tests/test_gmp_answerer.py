from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from src.gmp_answerer import (
    _format_gmp_row,
    _is_stale,
    answer_gmp_question,
    get_fresh_snapshot,
    load_gmp_snapshot,
)


def _row(company="Acme Ltd", status="OPEN", gmp="₹100"):
    return {
        "company": company, "status": status, "gmp_rupees": gmp, "gmp_percent": "+10.00%",
        "price_band": "₹90 – ₹95", "est_listing": "₹105", "open_date": "20 Aug",
        "close_date": "22 Aug 2026", "updated": "20 Aug, 12:00",
    }


def _snapshot(age: timedelta, company="Acme Ltd") -> dict:
    scraped_at = datetime.now(timezone.utc) - age
    return {"scraped_at": scraped_at.isoformat(), "mainboard": [_row(company)], "sme": [], "recently_listed": []}


def test_format_gmp_row_includes_key_fields():
    text = _format_gmp_row(_row())
    assert "Acme Ltd" in text
    assert "OPEN" in text
    assert "₹100" in text


def test_load_gmp_snapshot_returns_none_when_missing(tmp_path):
    assert load_gmp_snapshot(tmp_path / "nope.json") is None


def test_load_gmp_snapshot_reads_real_file(tmp_path):
    path = tmp_path / "latest.json"
    path.write_text(json.dumps(_snapshot(timedelta(minutes=1))))

    loaded = load_gmp_snapshot(path)
    assert loaded["mainboard"][0]["company"] == "Acme Ltd"


def test_is_stale():
    assert _is_stale(_snapshot(timedelta(minutes=5))) is False
    assert _is_stale(_snapshot(timedelta(minutes=30))) is True


def test_get_fresh_snapshot_reuses_fresh_cache_without_scraping(tmp_path, monkeypatch):
    path = tmp_path / "latest.json"
    path.write_text(json.dumps(_snapshot(timedelta(minutes=2))))

    def _fail(*args, **kwargs):
        raise AssertionError("should not scrape when cache is fresh")

    monkeypatch.setattr("data.scrape_gmp.fetch_gmp_data", _fail)

    result = get_fresh_snapshot(path)
    assert result["mainboard"][0]["company"] == "Acme Ltd"


def test_get_fresh_snapshot_rescrapes_when_stale(tmp_path, monkeypatch):
    path = tmp_path / "latest.json"
    path.write_text(json.dumps(_snapshot(timedelta(minutes=30), company="Old Corp")))

    monkeypatch.setattr(
        "data.scrape_gmp.fetch_gmp_data",
        lambda: {"mainboard": [_row("New Corp")], "sme": [], "recently_listed": []},
    )

    result = get_fresh_snapshot(path)
    assert result["mainboard"][0]["company"] == "New Corp"
    # The refreshed snapshot should also be persisted back to disk.
    assert json.loads(path.read_text())["mainboard"][0]["company"] == "New Corp"


def test_get_fresh_snapshot_falls_back_to_stale_cache_on_scrape_failure(tmp_path, monkeypatch):
    path = tmp_path / "latest.json"
    path.write_text(json.dumps(_snapshot(timedelta(minutes=30), company="Old Corp")))

    def _fail(*args, **kwargs):
        raise RuntimeError("site unreachable")

    monkeypatch.setattr("data.scrape_gmp.fetch_gmp_data", _fail)

    result = get_fresh_snapshot(path)
    assert result["mainboard"][0]["company"] == "Old Corp"  # stale, but still usable


def test_get_fresh_snapshot_returns_none_when_no_cache_and_scrape_fails(tmp_path, monkeypatch):
    def _fail(*args, **kwargs):
        raise RuntimeError("site unreachable")

    monkeypatch.setattr("data.scrape_gmp.fetch_gmp_data", _fail)

    assert get_fresh_snapshot(tmp_path / "missing.json") is None


def test_answer_gmp_question_with_no_snapshot_and_failed_scrape_short_circuits_without_llm_call(tmp_path, monkeypatch):
    # No Ollama call happens here -- this must work without a live model.
    def _fail(*args, **kwargs):
        raise RuntimeError("site unreachable")

    monkeypatch.setattr("data.scrape_gmp.fetch_gmp_data", _fail)

    answer = answer_gmp_question("what is the GMP for Acme?", snapshot_path=tmp_path / "missing.json")
    assert "couldn't fetch" in answer.lower()

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.scrape_listing_performance import _parse_number, _parse_price_and_gain


def test_parse_number_strips_commas():
    assert _parse_number("1,700.00") == 1700.0
    assert _parse_number("201") == 201.0


def test_parse_number_missing_value():
    assert _parse_number("—") is None
    assert _parse_number("") is None


def test_parse_price_and_gain_positive():
    assert _parse_price_and_gain("248.44 (23.6%)") == (248.44, 23.6)


def test_parse_price_and_gain_negative():
    assert _parse_price_and_gain("58.16 (-3.07%)") == (58.16, -3.07)


def test_parse_price_and_gain_strips_commas_in_price():
    assert _parse_price_and_gain("1,248.44 (23.6%)") == (1248.44, 23.6)


def test_parse_price_and_gain_missing_value_means_not_listed_yet():
    assert _parse_price_and_gain("") == (None, None)
    assert _parse_price_and_gain("—") == (None, None)


def test_parse_price_and_gain_ignores_unrelated_shape():
    # 52-week high/low cells pair a price with a date, not a gain% -- must
    # not be misparsed as a gain.
    assert _parse_price_and_gain("100.80 (21-Aug-2026)") == (None, None)

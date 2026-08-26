import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.scrape_gmp import _parse_gmp_rupees


def test_parse_gmp_rupees_strips_currency_and_commas():
    assert _parse_gmp_rupees("₹327") == 327
    assert _parse_gmp_rupees("₹1,200") == 1200


def test_parse_gmp_rupees_missing_value():
    assert _parse_gmp_rupees("—") is None
    assert _parse_gmp_rupees("") is None


def test_parse_gmp_rupees_handles_whitespace():
    assert _parse_gmp_rupees("  ₹50  ") == 50

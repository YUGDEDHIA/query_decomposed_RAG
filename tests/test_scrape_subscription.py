import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.scrape_subscription import _parse_number, _split_company_status


def test_parse_number_strips_commas():
    assert _parse_number("9,275.22") == 9275.22
    assert _parse_number("175.06") == 175.06


def test_parse_number_missing_value():
    assert _parse_number("—") is None
    assert _parse_number("") is None


def test_parse_number_handles_whitespace():
    assert _parse_number("  10,699  ") == 10699.0


def test_split_company_status_extracts_trailing_letter():
    assert _split_company_status("Hy-Tech Engineers Ltd. O") == ("Hy-Tech Engineers Ltd.", "O")
    assert _split_company_status("Symbiotec Pharmalab Ltd. P") == ("Symbiotec Pharmalab Ltd.", "P")


def test_split_company_status_no_suffix_means_closed():
    assert _split_company_status("Lohia Corp Ltd.") == ("Lohia Corp Ltd.", None)

from src.table_extractor import (
    MIN_CELLS,
    MIN_COLS,
    _clean_cell,
    _flatten_table_lines,
    _group_lines,
    _is_real_table,
)


def test_clean_cell_normalizes_whitespace_and_none():
    assert _clean_cell(None) == ""
    assert _clean_cell("Amount\n(₹ in\nmillion)") == "Amount (₹ in million)"
    assert _clean_cell("  padded  ") == "padded"
    assert _clean_cell("") == ""


def test_flatten_table_lines_drops_empty_cells():
    table = [
        ["Particulars", None, None, "Amount of Assets as at", None],
        [None, None, None, "March 31, 2026", None],
        ["Insured Assets", None, "1,400.06", None, None],
    ]
    assert _flatten_table_lines(table) == [
        "Particulars | Amount of Assets as at",
        "March 31, 2026",
        "Insured Assets | 1,400.06",
    ]


def test_flatten_table_lines_skips_fully_empty_rows():
    table = [["A", "B"], [None, None], ["C", "D"]]
    assert _flatten_table_lines(table) == ["A | B", "C | D"]


def test_group_lines_keeps_small_table_as_one_group():
    lines = ["A | B", "C | D"]
    assert _group_lines(lines, max_words=100) == ["A | B\nC | D"]


def test_group_lines_splits_at_row_boundaries_not_mid_row():
    # Each line is 3 words; max_words=6 fits exactly 2 rows per group.
    lines = ["one two three", "four five six", "seven eight nine", "ten eleven twelve"]
    groups = _group_lines(lines, max_words=6)
    assert groups == [
        "one two three\nfour five six",
        "seven eight nine\nten eleven twelve",
    ]


def test_group_lines_oversized_single_row_becomes_its_own_group():
    lines = ["short row", " ".join(f"w{i}" for i in range(50))]
    groups = _group_lines(lines, max_words=10)
    assert groups == ["short row", lines[1]]


def test_group_lines_empty_input():
    assert _group_lines([], max_words=100) == []


def test_is_real_table_filters_small_fragments():
    # Actual false positives observed on real RHPs: a 1x2 or 3x1 fragment
    # from ordinary body text split by whitespace.
    assert _is_real_table([["a", "b"]]) is False  # 1x2 = 2 cells
    assert _is_real_table([["a"], ["b"], ["c"]]) is False  # 3x1, cols < MIN_COLS

    # A real small table, e.g. a 5-row, 2-col glossary excerpt.
    assert _is_real_table([["Term", "Description"]] * 5) is True


def test_is_real_table_boundary_values():
    # Exactly at the threshold should pass; one cell short should not.
    assert MIN_COLS == 2 and MIN_CELLS == 8
    at_threshold = [["a", "b", "c", "d"], ["e", "f", "g", "h"]]  # 2x4 = 8 cells
    assert _is_real_table(at_threshold) is True

    below_threshold = [["a", "b", "c"], ["d", "e", "f"]]  # 2x3 = 6 cells
    assert _is_real_table(below_threshold) is False


def test_is_real_table_handles_ragged_rows():
    # extract_tables() can return rows of uneven length; cols is the max.
    ragged = [["a", "b", "c", "d", "e"], ["f", "g"]]  # cols=5, rows=2 -> 10 cells
    assert _is_real_table(ragged) is True

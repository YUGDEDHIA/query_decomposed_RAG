import dataclasses
import json

from src.chunker import Chunk
from src.ingest import _combine_and_renumber, write_chunks


def _chunk(idx: int, is_table: bool = False) -> Chunk:
    return Chunk(
        text=f"text {idx}",
        company="Acme",
        source_file="acme.pdf",
        page_range=(idx, idx),
        chunk_index=idx,
        is_table=is_table,
    )


def test_combine_and_renumber_appends_tables_after_text_and_renumbers():
    text_chunks = [_chunk(0), _chunk(1)]
    table_chunks = [_chunk(0, is_table=True), _chunk(1, is_table=True)]

    combined = _combine_and_renumber(text_chunks, table_chunks)

    assert [c.chunk_index for c in combined] == [0, 1, 2, 3]
    assert [c.is_table for c in combined] == [False, False, True, True]
    # Original text chunks come first, in order, followed by table chunks.
    assert combined[0].text == "text 0"
    assert combined[2].text == "text 0"  # first table chunk, renumbered to index 2


def test_combine_and_renumber_handles_empty_lists():
    assert _combine_and_renumber([], []) == []
    assert [c.chunk_index for c in _combine_and_renumber([_chunk(5)], [])] == [0]


def test_write_chunks_round_trips_through_jsonl(tmp_path):
    chunks = [_chunk(0), _chunk(1, is_table=True)]
    out_path = tmp_path / "acme.jsonl"

    write_chunks(chunks, out_path)

    lines = out_path.read_text().strip().split("\n")
    assert len(lines) == 2

    loaded = [json.loads(line) for line in lines]
    for original, record in zip(chunks, loaded):
        expected = dataclasses.asdict(original)
        expected["page_range"] = list(expected["page_range"])  # JSON has no tuple type
        assert record == expected


def test_write_chunks_creates_parent_directory(tmp_path):
    out_path = tmp_path / "nested" / "dir" / "acme.jsonl"
    write_chunks([_chunk(0)], out_path)
    assert out_path.exists()

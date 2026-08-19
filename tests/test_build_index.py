import json

from src.build_index import load_all_chunks


def _write_jsonl(path, records):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_load_all_chunks_reads_across_files_in_sorted_order(tmp_path):
    _write_jsonl(tmp_path / "b_company.jsonl", [{"text": "b0"}, {"text": "b1"}])
    _write_jsonl(tmp_path / "a_company.jsonl", [{"text": "a0"}])

    chunks = load_all_chunks(tmp_path)

    assert [c["text"] for c in chunks] == ["a0", "b0", "b1"]


def test_load_all_chunks_empty_dir(tmp_path):
    assert load_all_chunks(tmp_path) == []

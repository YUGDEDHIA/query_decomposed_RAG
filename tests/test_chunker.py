from src.chunker import chunk_text


def _sentence(n_words: int, tag: str) -> str:
    """A sentence of exactly n_words words, tagged so tests can identify it."""
    return " ".join(f"{tag}w{i}" for i in range(n_words)) + "."


def test_sliding_window_chunk_count_and_overlap():
    # 6 sentences of 10 words each. chunk_size=20 fits exactly 2 sentences
    # (a 3rd would push to 30 > 20); overlap=10 fits exactly 1 trailing
    # sentence (2 would be 20 > 10). This should produce a clean 1-sentence
    # sliding window: [s1,s2], [s2,s3], [s3,s4], [s4,s5], [s5,s6].
    sentences = [_sentence(10, f"s{i}") for i in range(1, 7)]
    page = " ".join(sentences)

    chunks = chunk_text([page], company="Acme", source_file="acme.pdf", chunk_size=20, overlap=10)

    assert len(chunks) == 5
    assert [c.chunk_index for c in chunks] == [0, 1, 2, 3, 4]
    for c in chunks:
        assert len(c.text.split()) == 20
        assert c.company == "Acme"
        assert c.source_file == "acme.pdf"

    # Each chunk overlaps the next by exactly the second sentence / first
    # sentence of the pair (the shared middle sentence in the window).
    for i in range(len(chunks) - 1):
        this_second_sentence = chunks[i].text.split(". ")[1]
        next_first_sentence = chunks[i + 1].text.split(". ")[0]
        assert this_second_sentence.rstrip(".") == next_first_sentence


def test_no_chunk_exceeds_max_size():
    sentences = [_sentence(7, f"s{i}") for i in range(1, 15)]
    page = " ".join(sentences)

    chunks = chunk_text([page], company="Acme", source_file="acme.pdf", chunk_size=25, overlap=5)

    assert len(chunks) > 1
    for c in chunks:
        assert len(c.text.split()) <= 25


def test_oversized_sentence_is_hard_split_not_dropped():
    # A single "sentence" (no punctuation) longer than chunk_size on its own.
    huge = " ".join(f"w{i}" for i in range(50))
    chunks = chunk_text([huge], company="Acme", source_file="acme.pdf", chunk_size=20, overlap=5)

    assert len(chunks) == 3  # 50 words -> 20 + 20 + 10
    for c in chunks:
        assert len(c.text.split()) <= 20
    # No words lost or duplicated across the hard split.
    rejoined = " ".join(c.text for c in chunks)
    assert rejoined.split() == huge.split()


def test_chunks_flow_across_page_boundaries_and_track_page_range():
    page1 = _sentence(5, "p1a") + " " + _sentence(5, "p1b")
    page2 = _sentence(5, "p2a") + " " + _sentence(5, "p2b")

    chunks = chunk_text([page1, page2], company="Acme", source_file="acme.pdf", chunk_size=15, overlap=0)

    # chunk_size 15 fits 3 of the 5-word sentences before overflowing, so
    # the boundary between page 1 (2 sentences) and page 2 (2 sentences)
    # falls inside a chunk -- that chunk's page_range should span both pages.
    spanning = [c for c in chunks if c.page_range[0] != c.page_range[1]]
    assert spanning, "expected at least one chunk spanning both pages"
    assert spanning[0].page_range == (1, 2)


def test_empty_pages_produce_no_chunks():
    assert chunk_text(["", "   ", ""], company="Acme", source_file="acme.pdf") == []
    assert chunk_text([], company="Acme", source_file="acme.pdf") == []

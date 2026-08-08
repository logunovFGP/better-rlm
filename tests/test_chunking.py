from src.chunking import chunk_text


def _reconstructs(text, chunks):
    return "".join(text[c.start:c.end] for c in chunks) == text


def test_lines_strategy_partitions_and_reconstructs():
    text = "".join(f"line {i}\n" for i in range(100))
    chunks = chunk_text(text, "lines", chunk_lines=10, chunk_chars=120000, overlap=0)
    assert len(chunks) == 10
    assert _reconstructs(text, chunks)
    assert chunks[0].start_line == 1
    assert all(c.est_tokens >= 1 for c in chunks)


def test_files_strategy_splits_per_file_marker():
    text = (
        "\n\n===== FILE: a.txt (3 bytes) =====\nAAA"
        "\n\n===== FILE: b.txt (3 bytes) =====\nBBB"
    )
    chunks = chunk_text(text, "files", chunk_lines=2000, chunk_chars=120000, overlap=0)
    assert len(chunks) == 2
    assert chunks[0].label == "a.txt" and chunks[1].label == "b.txt"


def test_functions_strategy_splits_on_defs():
    text = "import os\n\ndef a():\n    return 1\n\ndef b():\n    return 2\n"
    chunks = chunk_text(text, "functions", chunk_lines=2000, chunk_chars=120000, overlap=0)
    assert len(chunks) >= 2
    assert _reconstructs(text, chunks)


def test_lines_strategy_honors_chunk_chars_on_long_lines():
    # 20 lines x ~500 chars: chunk_lines alone would emit ONE 10 KB chunk. A JSON-per-line
    # log scaled up this way blows past the sub-model's window, so chunk_chars must bound it.
    text = "".join(f"{'x' * 499}\n" for _ in range(20))
    chunks = chunk_text(text, "lines", chunk_lines=2000, chunk_chars=1000, overlap=0)
    assert len(chunks) > 1
    assert all(c.n_chars <= 1000 for c in chunks)
    assert _reconstructs(text, chunks)


def test_files_strategy_honors_chunk_chars_and_keeps_label():
    big = "B" * 2500
    text = f"\n\n===== FILE: big.txt (2500 bytes) =====\n{big}"
    chunks = chunk_text(text, "files", chunk_lines=2000, chunk_chars=1000, overlap=0)
    assert len(chunks) > 1                              # one oversized file -> several chunks
    assert all(c.n_chars <= 1000 for c in chunks)
    assert all(c.label == "big.txt" for c in chunks)    # provenance survives the split
    # No _reconstructs here: "files" spans start at the first FILE marker by design, so
    # any preamble before it is intentionally not covered. Assert contiguity instead.
    assert [c.start for c in chunks[1:]] == [c.end for c in chunks[:-1]]


def test_unknown_strategy_raises():
    import pytest
    with pytest.raises(ValueError):
        chunk_text("x", "nope", chunk_lines=10, chunk_chars=10, overlap=0)

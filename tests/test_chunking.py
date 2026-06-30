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


def test_unknown_strategy_raises():
    import pytest
    with pytest.raises(ValueError):
        chunk_text("x", "nope", chunk_lines=10, chunk_chars=10, overlap=0)

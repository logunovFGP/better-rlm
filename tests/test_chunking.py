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


def test_capped_window_does_not_emit_a_sliver_chunk():
    """A line window just over chunk_chars must split evenly, not leave a remainder.

    Regression: the default 2000-line window is ~124 KB on a typical log, so capping at
    120 KB emitted 120 KB + a 4 KB sliver for EVERY window -- 401 chunks on a 400k-line
    log where 201 is right, doubling rlm_sub_query_batch's fan-out onto chunks holding a
    few dozen lines.
    """
    # ~62 chars/line * 2000 lines = ~124 KB per window, i.e. just over chunk_chars.
    line = "2026-09-01 13:00:00 svc-0 INFO request id=000000 latency=000ms padding\n"
    text = line * 20_000
    chunks = chunk_text(text, "lines", chunk_lines=2000, chunk_chars=120_000, overlap=0)

    assert all(c.n_chars <= 120_000 for c in chunks), "cap violated"
    # No sliver: the smallest chunk is not a rounding remainder of the largest. The old
    # code emitted a ~120 KB chunk followed by a ~4 KB one for every window.
    assert min(c.n_chars for c in chunks) * 2 >= max(c.n_chars for c in chunks)
    # And each chunk is actually FULL, not half-full: capping after the fact split every
    # 124 KB window into two ~62 KB halves, doubling the count for the same content.
    assert max(c.n_chars for c in chunks) > 0.9 * 120_000
    # ~1.42 MB of text at ~120 KB per chunk -> 12, not the 20 a blind post-hoc cap gave.
    assert len(chunks) == 12, f"expected 12 full chunks, got {len(chunks)}"
    # Adjacency: the chunks must still reconstruct the source exactly.
    assert "".join(text[c.start:c.end] for c in chunks) == text


def test_zero_chunk_chars_terminates():
    # A 0 in config.yaml used to make _cap_spans loop forever appending empty spans.
    chunks = chunk_text("a\nb\nc\n", "lines", chunk_lines=2, chunk_chars=0, overlap=0)
    assert chunks and all(c.n_chars >= 1 for c in chunks)

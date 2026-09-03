import json

import pytest

from src.chunking import chunk_text
from src.context_store import ContextStore, _text_digest


def _lines(store, meta, text, *, chunk_lines=1, **kw):
    """Chunk `text` by lines and record it on `meta`. Returns the Chunk objects."""
    chunks = chunk_text(text, "lines", chunk_lines=chunk_lines, chunk_chars=120000, overlap=0)
    store.set_chunks(meta.ctx_id, "lines", [c.as_dict() for c in chunks], **kw)
    return chunks


def test_load_text_metadata_and_read(cfg):
    store = ContextStore(cfg)
    meta = store.load_text("a\nb\nc\n", source="t")
    assert meta.lines == 3
    assert meta.bytes == 6
    assert meta.est_tokens >= 1
    assert store.read_text(meta.ctx_id) == "a\nb\nc\n"
    assert store.get(meta.ctx_id).ctx_id == meta.ctx_id


def test_load_dir_skips_platform_venvs(cfg, tmp_path):
    # Measured on this repo before the fix: load_dir pulled in 7,383 files / 48.2 MB, of
    # which 47.7 MB was .venv_windows - the project itself was 1% of its own context.
    d = tmp_path / "proj"
    d.mkdir()
    (d / "real.py").write_text("print('real')\n")
    for venv in (".venv_windows", ".venv_sh", ".venv", "venv", "node_modules"):
        sub = d / venv / "site-packages"
        sub.mkdir(parents=True)
        (sub / "dep.py").write_text("print('dependency source, not the project')\n")
    store = ContextStore(cfg)
    meta = store.load_dir(str(d))
    assert meta.file_count == 1, f"expected only real.py, got {meta.files}"
    assert "dependency source" not in store.read_text(meta.ctx_id)


def test_load_dir_concatenates_and_chunks_per_file(cfg, tmp_path):
    d = tmp_path / "proj"
    d.mkdir()
    (d / "x.py").write_text("print('x')\n")
    (d / "y.py").write_text("print('y')\n")
    (d / ".env").write_text("SECRET=1\n")  # must be skipped
    store = ContextStore(cfg)
    meta = store.load_dir(str(d))
    assert meta.file_count == 2  # .env skipped
    text = store.read_text(meta.ctx_id)
    assert "x.py" in text and "y.py" in text and "SECRET" not in text

    chunks = chunk_text(text, "files", chunk_lines=2000, chunk_chars=120000, overlap=0)
    store.set_chunks(meta.ctx_id, "files", [c.as_dict() for c in chunks])
    assert len(store.get(meta.ctx_id).chunks) == 2
    assert "print(" in store.read_chunk(meta.ctx_id, 0)


def test_set_chunks_computes_byte_offsets_for_ascii(cfg):
    store = ContextStore(cfg)
    text = "line one\nline two\nline three\n"
    meta = store.load_text(text)
    chunks = chunk_text(text, "lines", chunk_lines=1, chunk_chars=120000, overlap=0)
    store.set_chunks(meta.ctx_id, "lines", [c.as_dict() for c in chunks])
    for c in store.get(meta.ctx_id).chunks:
        assert c["byte_start"] == c["start"]   # 1 byte per char, ASCII only
        assert c["byte_end"] == c["end"]


def test_set_chunks_computes_byte_offsets_for_multibyte(cfg):
    store = ContextStore(cfg)
    text = "café\n字字字\nplain\n"
    meta = store.load_text(text)
    chunks = chunk_text(text, "lines", chunk_lines=1, chunk_chars=120000, overlap=0)
    store.set_chunks(meta.ctx_id, "lines", [c.as_dict() for c in chunks])
    raw = meta.content_path
    with open(raw, "rb") as fh:
        data = fh.read()
    stored = store.get(meta.ctx_id).chunks
    assert any(c["byte_end"] - c["byte_start"] != c["end"] - c["start"] for c in stored)
    for c in stored:
        assert data[c["byte_start"]:c["byte_end"]].decode("utf-8") == text[c["start"]:c["end"]]


def test_crlf_file_gets_byte_offsets_that_account_for_translation(cfg, tmp_path):
    """A pure-CRLF file still earns the seek path: each "\\n" in the translated text cost
    two bytes on disk, which the walk can price exactly. This is content-driven, not
    host-driven -- Python translates "\\r\\n" on every platform, so this test asserts the
    same thing on Windows and on Linux. Without it, every command-sourced context on
    Windows (whose child writes "\\r\\n" to a text-mode stdout) silently lost the
    optimization while the identical command on Linux kept it."""
    store = ContextStore(cfg)
    p = tmp_path / "crlf.txt"
    p.write_bytes(b"one\r\ntwo\r\nthree\r\n")
    meta = store.load_file(str(p))
    text = store.read_text(meta.ctx_id)
    assert "\r" not in text, "read_text must keep translating newlines"
    chunks = chunk_text(text, "lines", chunk_lines=1, chunk_chars=120000, overlap=0)
    store.set_chunks(meta.ctx_id, "lines", [c.as_dict() for c in chunks])
    stored = store.get(meta.ctx_id).chunks
    assert all(c["byte_start"] != -1 for c in stored), "CRLF must reach the fast path"
    assert stored[0]["byte_end"] == 5, 'one\\r\\n is 5 bytes for 4 chars'
    for i, c in enumerate(stored):
        assert store.read_chunk(meta.ctx_id, i) == text[c["start"]:c["end"]]


def test_mixed_line_endings_decline_byte_offsets(cfg, tmp_path):
    """Half CRLF, half LF: neither walk prices it, so it must be refused rather than
    guessed at. The byte total is what catches it."""
    store = ContextStore(cfg)
    p = tmp_path / "mixed.txt"
    p.write_bytes(b"one\r\ntwo\nthree\r\nfour\n")
    meta = store.load_file(str(p))
    text = store.read_text(meta.ctx_id)
    chunks = chunk_text(text, "lines", chunk_lines=1, chunk_chars=120000, overlap=0)
    store.set_chunks(meta.ctx_id, "lines", [c.as_dict() for c in chunks])
    for i, c in enumerate(store.get(meta.ctx_id).chunks):
        assert (c["byte_start"], c["byte_end"]) == (-1, -1)
        assert store.read_chunk(meta.ctx_id, i) == text[c["start"]:c["end"]]


def test_lone_cr_file_uses_the_fast_path_because_reads_translate(cfg, tmp_path):
    """A lone "\\r" is one byte for one char, so char offsets already sit on byte
    boundaries and read_chunk's own translation turns the "\\r" it seeks into the "\\n"
    read_text reports. No separate carriage-return scan is needed to see this."""
    store = ContextStore(cfg)
    p = tmp_path / "cr.txt"
    p.write_bytes(b"a\rb\rc\n")
    meta = store.load_file(str(p))
    text = store.read_text(meta.ctx_id)
    assert text == "a\nb\nc\n" and len(text.encode()) == meta.bytes   # counts agree
    chunks = chunk_text(text, "lines", chunk_lines=1, chunk_chars=120000, overlap=0)
    store.set_chunks(meta.ctx_id, "lines", [c.as_dict() for c in chunks])
    for i, c in enumerate(store.get(meta.ctx_id).chunks):
        assert c["byte_start"] != -1
        assert store.read_chunk(meta.ctx_id, i) == text[c["start"]:c["end"]]


def test_set_chunks_does_not_mutate_the_callers_dicts(cfg):
    """set_chunks used to rewrite byte_start/byte_end on the dicts handed to it. A caller
    that had set those deliberately saw them silently replaced -- which is exactly how an
    earlier fallback test ended up asserting nothing about the fallback."""
    store = ContextStore(cfg)
    text = "a\nb\nc\n"
    meta = store.load_text(text)
    dicts = [c.as_dict() for c in
             chunk_text(text, "lines", chunk_lines=1, chunk_chars=120000, overlap=0)]
    before = [dict(d) for d in dicts]
    store.set_chunks(meta.ctx_id, "lines", dicts)
    assert dicts == before, "the caller's dicts were modified in place"
    assert all(c["byte_start"] != -1 for c in store.get(meta.ctx_id).chunks)


def test_chunk_metas_written_before_byte_offsets_read_back_intact(cfg, tmp_path):
    """The regression that made the offsets worth guarding: a context chunked before
    byte offsets existed has char offsets computed on TRANSLATED text. Reading it must
    return exactly what it returned then -- not a slice shifted by the \\r bytes."""
    store = ContextStore(cfg)
    p = tmp_path / "old.log"
    p.write_bytes(b"alpha\r\nbravo\r\ncharlie\r\ndelta\r\n")
    meta = store.load_file(str(p))
    text = store.read_text(meta.ctx_id)
    legacy = [c.as_dict() for c in
              chunk_text(text, "lines", chunk_lines=2, chunk_chars=120000, overlap=0)]
    for c in legacy:                      # a meta from before the fields existed
        c.pop("byte_start"), c.pop("byte_end")
    stored = store.get(meta.ctx_id)
    stored.chunks, stored.chunk_strategy = legacy, "lines"
    store._save(stored)

    for i, c in enumerate(legacy):
        assert store.read_chunk(meta.ctx_id, i) == text[c["start"]:c["end"]]
    assert store.read_chunk(meta.ctx_id, 0) == "alpha\nbravo\n"
    assert store.read_chunk(meta.ctx_id, 1) == "charlie\ndelta\n"


def test_set_chunks_leaves_offsets_unset_on_invalid_utf8(cfg, tmp_path):
    store = ContextStore(cfg)
    p = tmp_path / "bad.txt"
    p.write_bytes(b"good line\n" + b"\xff\xff" + b"\nmore\n")
    meta = store.load_file(str(p))
    text = store.read_text(meta.ctx_id)   # decodes with errors="replace" -> U+FFFD
    chunks = chunk_text(text, "lines", chunk_lines=1, chunk_chars=120000, overlap=0)
    store.set_chunks(meta.ctx_id, "lines", [c.as_dict() for c in chunks])
    for c in store.get(meta.ctx_id).chunks:
        assert c["byte_start"] == -1 and c["byte_end"] == -1
    # readers must still work via the char-offset fallback
    assert store.read_chunk(meta.ctx_id, 0) == text[chunks[0].start:chunks[0].end]


def test_read_chunk_matches_read_text_slice_via_seek_path(cfg, tmp_path):
    store = ContextStore(cfg)
    # All three take the fast path: LF maps 1:1, CRLF is priced at +1 byte per newline.
    ascii_meta = store.load_text("line one\nline two\nline three\n")
    multibyte_meta = store.load_text("café\n字字字\nplain\n")
    p = tmp_path / "crlf.txt"
    p.write_bytes("héllo\r\n字字\r\nplain\r\n".encode("utf-8"))
    crlf_meta = store.load_file(str(p))

    for meta in (ascii_meta, multibyte_meta, crlf_meta):
        text = store.read_text(meta.ctx_id)
        chunks = chunk_text(text, "lines", chunk_lines=1, chunk_chars=120000, overlap=0)
        store.set_chunks(meta.ctx_id, "lines", [c.as_dict() for c in chunks])
        stored = store.get(meta.ctx_id).chunks
        assert all(c["byte_start"] != -1 for c in stored), "expected the validated fast path"
        for i, c in enumerate(stored):
            assert store.read_chunk(meta.ctx_id, i) == text[c["start"]:c["end"]]


def test_read_chunk_falls_back_when_byte_offsets_missing(cfg):
    store = ContextStore(cfg)
    text = "a\nb\nc\n"
    meta = store.load_text(text)
    chunks = chunk_text(text, "lines", chunk_lines=1, chunk_chars=120000, overlap=0)
    store.set_chunks(meta.ctx_id, "lines", [c.as_dict() for c in chunks])

    # Strip the offsets from the STORED meta, after set_chunks. Seeding -1 into the dicts
    # beforehand does NOT work: set_chunks recomputes them, so the test silently ran the
    # seek path and asserted nothing about the fallback it is named for.
    meta_path = store._meta_path(meta.ctx_id)
    raw = json.loads(meta_path.read_text())
    for c in raw["chunks"]:
        c.pop("byte_start"), c.pop("byte_end")
    meta_path.write_text(json.dumps(raw))
    assert all("byte_start" not in c for c in store.get(meta.ctx_id).chunks)

    for i, c in enumerate(chunks):
        assert store.read_chunk(meta.ctx_id, i) == text[c.start:c.end]


def test_read_chunk_ignores_stale_byte_offsets_after_the_file_changes(cfg, tmp_path):
    """load_file references the user's file IN PLACE, so it can change after chunking.
    Byte offsets frozen at chunk time must not be seeked into a file that no longer
    matches them: the seek returned shifted, truncated content ('字alpha\\nbra') while
    read_text sliced '字alpha\\nbravo'. Stale is survivable; the two paths disagreeing
    silently is not."""
    store = ContextStore(cfg)
    p = tmp_path / "live.log"
    p.write_bytes(b"alpha\nbravo\ncharlie\ndelta\n")
    meta = store.load_file(str(p))
    text = store.read_text(meta.ctx_id)
    chunks = chunk_text(text, "lines", chunk_lines=2, chunk_chars=120000, overlap=0)
    store.set_chunks(meta.ctx_id, "lines", [c.as_dict() for c in chunks], text=text)
    assert all(c["byte_start"] != -1 for c in store.get(meta.ctx_id).chunks)

    p.write_bytes("字".encode("utf-8") + b"alpha\nbravo\ncharlie\ndelta\n")  # +1 char, +3 bytes
    now = store.read_text(meta.ctx_id)
    for i, c in enumerate(chunks):
        assert store.read_chunk(meta.ctx_id, i) == now[c.start:c.end]


def test_set_chunks_does_not_decode_again_when_given_the_text(cfg, monkeypatch):
    """Both server.py callers have just decoded the whole context to chunk it. Without
    text=, the byte-offset pass decoded it a second time while that copy was still live
    -- two full copies of a multi-GB context at once."""
    store = ContextStore(cfg)
    text = "line one\nline two\nline three\n"
    meta = store.load_text(text)
    chunks = [c.as_dict() for c in
              chunk_text(text, "lines", chunk_lines=1, chunk_chars=120000, overlap=0)]

    def boom(self, ctx_id):
        raise AssertionError("set_chunks re-decoded the context the caller handed it")
    monkeypatch.setattr(ContextStore, "read_text", boom)
    store.set_chunks(meta.ctx_id, "lines", chunks, text=text)
    assert all(c["byte_start"] != -1 for c in store.get(meta.ctx_id).chunks)


def test_paragraph_chunking_still_works_on_a_crlf_context(cfg, tmp_path):
    """The symptom that made the newline change untenable: chunk_text's paragraph regex
    (r"\\n[ \\t]*\\n") cannot match "\\r\\n\\r\\n", so an untranslated CRLF context found
    ZERO paragraph boundaries and "paragraphs"/"semantic" silently became blind cuts."""
    store = ContextStore(cfg)
    p = tmp_path / "doc.txt"
    body = b"para one\r\n\r\npara two\r\n\r\npara three\r\n"
    p.write_bytes(body)
    meta = store.load_file(str(p))
    text = store.read_text(meta.ctx_id)
    chunks = chunk_text(text, "paragraphs", chunk_lines=2000, chunk_chars=12, overlap=0)
    # Boundaries land on the blank lines, not on arbitrary 12-char cuts.
    assert [text[c.start:c.end] for c in chunks][0].startswith("para one")
    assert any(c.end < len(text) and text[c.end - 1] == "\n" for c in chunks)


def test_read_chunk_fast_path_never_calls_read_text(cfg, monkeypatch):
    store = ContextStore(cfg)
    text = "a\nb\nc\n"
    meta = store.load_text(text)
    chunks = chunk_text(text, "lines", chunk_lines=1, chunk_chars=120000, overlap=0)
    store.set_chunks(meta.ctx_id, "lines", [c.as_dict() for c in chunks])
    assert all(c["byte_start"] != -1 for c in store.get(meta.ctx_id).chunks)

    def boom(self, ctx_id):
        raise AssertionError("read_chunk should not fall back to read_text here")
    monkeypatch.setattr(ContextStore, "read_text", boom)
    assert store.read_chunk(meta.ctx_id, 0) == "a\n"


def test_old_meta_without_byte_offset_keys_still_loads(cfg):
    store = ContextStore(cfg)
    meta = store.load_text("a\nb\n")
    legacy_chunk = {"index": 0, "start": 0, "end": 4, "start_line": 1, "end_line": 2,
                     "n_chars": 4, "est_tokens": 1, "label": ""}   # no byte_start/byte_end
    meta.chunks = [legacy_chunk]
    meta.chunk_strategy = "lines"
    store._save(meta)
    reloaded = store.get(meta.ctx_id)
    assert reloaded.chunks == [legacy_chunk]


# --- read_chunk guard rails -------------------------------------------------
# Error paths, which the coverage report showed as the least-covered part of the
# feature. Each asserts the exception TYPE and MESSAGE, not merely that something threw:
# the message is what a caller sees and what tells them which tool to call next.

def test_read_chunk_rejects_a_context_that_was_never_chunked(cfg):
    store = ContextStore(cfg)
    meta = store.load_text("a\nb\n")
    with pytest.raises(ValueError, match="has not been chunked; call rlm_chunk_context"):
        store.read_chunk(meta.ctx_id, 0)


@pytest.mark.parametrize("index", [-1, 3, 99], ids=["negative", "one-past-end", "far-past-end"])
def test_read_chunk_rejects_an_out_of_range_index(cfg, index):
    store = ContextStore(cfg)
    text = "a\nb\nc\n"
    meta = store.load_text(text)
    _lines(store, meta, text)                      # 3 chunks -> valid indices 0..2
    with pytest.raises(IndexError, match=r"out of range \(0\.\.2\)"):
        store.read_chunk(meta.ctx_id, index)


def test_read_chunk_returns_empty_string_for_a_zero_length_chunk(cfg, monkeypatch):
    """A chunk whose start equals its end is degenerate but not an error: the byte span is
    empty and the read must return "".

    Asserting it never reaches read_text is not gold-plating. Tightening the guard to
    `byte_start < byte_end` still returns "" -- via the fallback, which decodes the WHOLE
    file to slice nothing out of it. The output is identical and the cost is O(file), so
    the empty answer alone cannot tell the two apart.
    """
    store = ContextStore(cfg)
    text = "alpha\nbravo\n"
    meta = store.load_text(text)
    dicts = [c.as_dict() for c in
             chunk_text(text, "lines", chunk_lines=1, chunk_chars=120000, overlap=0)]
    dicts.append({**dicts[0], "index": len(dicts), "start": 6, "end": 6, "n_chars": 0})
    store.set_chunks(meta.ctx_id, "lines", dicts)
    stored = store.get(meta.ctx_id).chunks
    assert stored[-1]["byte_start"] == stored[-1]["byte_end"] == 6

    def boom(self, ctx_id):
        raise AssertionError("a zero-length chunk decoded the whole file to return ''")
    monkeypatch.setattr(ContextStore, "read_text", boom)
    assert store.read_chunk(meta.ctx_id, len(stored) - 1) == ""


def test_read_chunk_surfaces_the_real_error_when_the_content_file_disappears(cfg, tmp_path):
    """load_file references the user's file in place, so it can be deleted after chunking.
    The size guard cannot stat a missing file; rather than fail mid-seek it declines the
    fast path and lets read_text raise the error that names the actual problem."""
    store = ContextStore(cfg)
    p = tmp_path / "gone.log"
    p.write_text("alpha\nbravo\n", encoding="utf-8")
    meta = store.load_file(str(p))
    _lines(store, meta, store.read_text(meta.ctx_id))
    p.unlink()
    with pytest.raises(FileNotFoundError):
        store.read_chunk(meta.ctx_id, 0)


# --- set_chunks: degenerate input and untrusted callers ---------------------

def test_set_chunks_accepts_an_empty_chunk_list(cfg):
    """A strategy can legitimately produce no chunks (an empty context). Recording that
    must not blow up on the offset walk, and read_chunk then reports "not chunked"."""
    store = ContextStore(cfg)
    meta = store.load_text("")
    store.set_chunks(meta.ctx_id, "lines", [])
    assert store.get(meta.ctx_id).chunks == []
    with pytest.raises(ValueError, match="has not been chunked"):
        store.read_chunk(meta.ctx_id, 0)


def test_set_chunks_refuses_offsets_when_the_callers_text_does_not_match_the_file(cfg):
    """`text=` is an optimization: the caller hands over a decode it already holds, and
    set_chunks trusts it for the arithmetic. That trust is what _probe_mapping exists to
    check -- it re-reads the LAST chunk from disk and compares.

    The divergence here is deliberately LATE and byte-length preserving: the first line is
    identical and the totals match, so neither the byte-count check nor a probe of the
    first chunk would notice. Only reading the last chunk does.
    """
    store = ContextStore(cfg)
    real = "aaaa\nbbbb\ncccc\n"
    meta = store.load_text(real)
    lying = "aaaa\nXXXX\nYYYY\n"
    assert len(lying.encode()) == meta.bytes, "the totals must agree, or the walk refuses first"

    chunks = chunk_text(lying, "lines", chunk_lines=1, chunk_chars=120000, overlap=0)
    store.set_chunks(meta.ctx_id, "lines", [c.as_dict() for c in chunks], text=lying)
    for c in store.get(meta.ctx_id).chunks:
        assert (c["byte_start"], c["byte_end"]) == (-1, -1)
    # and the reads still come from the file, not the caller's fiction
    assert store.read_chunk(meta.ctx_id, 0) == "aaaa\n"


def test_set_chunks_declines_offsets_when_the_file_vanished_before_the_probe(cfg, tmp_path):
    """With `text=` supplied, set_chunks never reads the file for the walk -- so a file
    deleted between load and chunk is only discovered by the probe. It must decline the
    offsets rather than raise, leaving a meta that still records the chunk boundaries."""
    store = ContextStore(cfg)
    p = tmp_path / "vanishing.log"
    p.write_text("alpha\nbravo\n", encoding="utf-8")
    meta = store.load_file(str(p))
    text = store.read_text(meta.ctx_id)
    p.unlink()
    chunks = chunk_text(text, "lines", chunk_lines=1, chunk_chars=120000, overlap=0)
    store.set_chunks(meta.ctx_id, "lines", [c.as_dict() for c in chunks], text=text)
    stored = store.get(meta.ctx_id).chunks
    assert len(stored) == 2
    assert all((c["byte_start"], c["byte_end"]) == (-1, -1) for c in stored)


# --- per-chunk digests: the answer cache's identity, stamped once at chunk time -------

def test_set_chunks_stamps_a_digest_matching_every_chunks_own_text(cfg):
    """The digest IS the answer cache's identity, so it has to equal what a reader gets
    back. Stamped at chunk time because this is the one moment the whole decoded text and
    the chunk boundaries are both in hand."""
    store = ContextStore(cfg)
    meta = store.load_text("a\nbb\nccc\n", source="t")
    _lines(store, meta, store.read_text(meta.ctx_id))

    for i, c in enumerate(store.get(meta.ctx_id).chunks):
        assert c["sha256"] == _text_digest(store.read_chunk(meta.ctx_id, i))


def test_a_digest_is_stamped_even_when_byte_offsets_are_refused(cfg, tmp_path):
    """Mixed line endings price under neither walk, so byte offsets are refused. The
    digest must survive that: it is of text[start:end], which is what BOTH read paths
    return, so it never depended on the offsets. These files are exactly the ones that
    would otherwise get no speedup at all."""
    store = ContextStore(cfg)
    p = tmp_path / "mixed.txt"
    p.write_bytes(b"one\r\ntwo\nthree\r\nfour\n")
    meta = store.load_file(str(p))
    _lines(store, meta, store.read_text(meta.ctx_id))

    for i, c in enumerate(store.get(meta.ctx_id).chunks):
        assert (c["byte_start"], c["byte_end"]) == (-1, -1), "offsets should be refused here"
        assert c["sha256"] == _text_digest(store.read_chunk(meta.ctx_id, i))


def test_chunk_digests_serves_stored_hashes_with_one_probe_read(cfg, monkeypatch):
    """The point of the change: the cache scan used to read every selected chunk purely to
    hash it. One probe read buys the proof, the rest come from meta."""
    store = ContextStore(cfg)
    meta = store.load_text("".join(f"line{i}\n" for i in range(6)), source="t")
    _lines(store, meta, store.read_text(meta.ctx_id))

    reads: list[int] = []
    real = ContextStore.read_chunk
    monkeypatch.setattr(ContextStore, "read_chunk",
                        lambda self, c, i: (reads.append(i), real(self, c, i))[1])
    digests = store.chunk_digests(meta.ctx_id, range(6))

    assert len(reads) == 1, f"read {len(reads)} chunks to hash 6"
    assert digests == {i: c["sha256"]
                       for i, c in enumerate(store.get(meta.ctx_id).chunks)}


def test_chunk_digests_falls_back_when_the_file_changed_size(cfg, tmp_path):
    """load_file references user files IN PLACE, so a file can change under a chunk index.
    Serving a stored digest for content that is no longer there is a FALSE cache HIT -- an
    answer handed back for bytes the model never saw.

    The first chunk is deliberately left BYTE-IDENTICAL across the rewrite, because the
    probe read only ever proves the chunk it reads. A rewrite that preserves the opening
    chunk and changes a later one sails past the probe, and the size check is the only
    thing left standing between it and a stale digest. Written the obvious way -- change
    everything -- this test passed with the size check deleted.
    """
    store = ContextStore(cfg)
    p = tmp_path / "log.txt"
    p.write_text("aaa\nbbb\nccc\n", encoding="utf-8")
    meta = store.load_file(str(p))
    _lines(store, meta, store.read_text(meta.ctx_id))
    stored = [c["sha256"] for c in store.get(meta.ctx_id).chunks]

    p.write_text("aaa\nZZZ\n", encoding="utf-8")   # chunk 0 survives, chunk 1 does not
    fresh = store.chunk_digests(meta.ctx_id, [0, 1])

    assert fresh[0] == stored[0], "chunk 0 really is unchanged, so the probe cannot help"
    assert fresh[1] != stored[1], "served a digest for content that is no longer there"
    assert fresh[1] == _text_digest(store.read_chunk(meta.ctx_id, 1))


def test_chunk_digests_probe_catches_a_size_preserving_edit(cfg, tmp_path):
    """_offsets_still_apply cannot see an edit that preserves the byte count -- its own
    docstring says so. The probe read is the second half of the trust: without it, a
    same-size edit would serve every stored digest for content that changed."""
    store = ContextStore(cfg)
    p = tmp_path / "log.txt"
    p.write_text("aaa\nbbb\n", encoding="utf-8")
    meta = store.load_file(str(p))
    _lines(store, meta, store.read_text(meta.ctx_id))

    p.write_text("AAA\nBBB\n", encoding="utf-8")   # same byte count, different bytes
    fresh = store.chunk_digests(meta.ctx_id, [0, 1])

    assert fresh[0] == _text_digest("AAA\n"), "the probe did not catch a same-size edit"
    assert fresh[1] == _text_digest("BBB\n")

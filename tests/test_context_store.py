import json

from src.chunking import chunk_text
from src.context_store import ContextStore


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

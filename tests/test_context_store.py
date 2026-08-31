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


def test_crlf_file_declines_byte_offsets_and_still_reads_correctly(cfg, tmp_path):
    """read_text translates newlines, so char offsets on a CRLF file do NOT map to raw
    byte positions. The offsets must therefore refuse to exist rather than be stored
    wrong: an earlier attempt disabled translation globally to make them fit, which
    shifted every previously-stored chunk and broke paragraph chunking."""
    store = ContextStore(cfg)
    p = tmp_path / "crlf.txt"
    p.write_bytes(b"one\r\ntwo\r\nthree\r\n")
    meta = store.load_file(str(p))
    text = store.read_text(meta.ctx_id)
    assert "\r" not in text, "read_text must keep translating newlines"
    chunks = chunk_text(text, "lines", chunk_lines=1, chunk_chars=120000, overlap=0)
    store.set_chunks(meta.ctx_id, "lines", [c.as_dict() for c in chunks])
    for i, c in enumerate(store.get(meta.ctx_id).chunks):
        assert (c["byte_start"], c["byte_end"]) == (-1, -1)
        assert store.read_chunk(meta.ctx_id, i) == text[c["start"]:c["end"]]


def test_lone_cr_file_declines_byte_offsets(cfg, tmp_path):
    """A lone "\\r" translates to "\\n" one-for-one, so the byte-count check alone cannot
    see it: the totals match while a seek would return "\\r" where a char slice returns
    "\\n". Only the carriage-return scan catches this one."""
    store = ContextStore(cfg)
    p = tmp_path / "cr.txt"
    p.write_bytes(b"a\rb\rc\n")
    meta = store.load_file(str(p))
    text = store.read_text(meta.ctx_id)
    assert text == "a\nb\nc\n" and len(text.encode()) == meta.bytes   # counts agree
    chunks = chunk_text(text, "lines", chunk_lines=1, chunk_chars=120000, overlap=0)
    store.set_chunks(meta.ctx_id, "lines", [c.as_dict() for c in chunks])
    for i, c in enumerate(store.get(meta.ctx_id).chunks):
        assert (c["byte_start"], c["byte_end"]) == (-1, -1)
        assert store.read_chunk(meta.ctx_id, i) == text[c["start"]:c["end"]]


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


def test_read_chunk_matches_read_text_slice_via_seek_path(cfg):
    store = ContextStore(cfg)
    # Both take the fast path: no "\r" on disk, so char offsets map 1:1 to bytes.
    ascii_meta = store.load_text("line one\nline two\nline three\n")
    multibyte_meta = store.load_text("café\n字字字\nplain\n")

    for meta in (ascii_meta, multibyte_meta):
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

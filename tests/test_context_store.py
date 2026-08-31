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


def test_set_chunks_byte_offsets_survive_crlf_file_loaded_in_place(cfg, tmp_path):
    store = ContextStore(cfg)
    p = tmp_path / "crlf.txt"
    p.write_bytes(b"one\r\ntwo\r\nthree\r\n")
    meta = store.load_file(str(p))
    text = store.read_text(meta.ctx_id)
    chunks = chunk_text(text, "lines", chunk_lines=1, chunk_chars=120000, overlap=0)
    store.set_chunks(meta.ctx_id, "lines", [c.as_dict() for c in chunks])
    raw = p.read_bytes()
    for c in store.get(meta.ctx_id).chunks:
        assert c["byte_start"] != -1   # validated, not fallen back
        assert raw[c["byte_start"]:c["byte_end"]].decode("utf-8") == text[c["start"]:c["end"]]


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

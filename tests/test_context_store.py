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

    from src.chunking import chunk_text
    chunks = chunk_text(text, "files", chunk_lines=2000, chunk_chars=120000, overlap=0)
    store.set_chunks(meta.ctx_id, "files", [c.as_dict() for c in chunks])
    assert len(store.get(meta.ctx_id).chunks) == 2
    assert "print(" in store.read_chunk(meta.ctx_id, 0)

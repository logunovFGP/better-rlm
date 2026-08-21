"""A dir load must ingest everything it can, and say what it could not.

Reported from the field: rlm_load_context on a source tree reported success having
loaded 173 of 184 files. The 11 missing ones were the exact subject of the question
(tenancy.ts, the tenant plugin, 4 of 5 tenant migrations). Nothing in the output
flagged it, so the answer was wrong by omission with no way to tell.

Two defects, both covered here: a valid UTF-8 file misread as binary, and the fact
that ANY skip was silent.
"""

import tempfile
from pathlib import Path

import pytest

import src.server as srv
from src.context_store import ContextStore, _looks_binary


# --- root cause: a fixed-size byte read can end mid-character ------------------
@pytest.mark.parametrize("offset", range(1018, 1030))
def test_a_multibyte_char_near_the_read_boundary_is_not_binary(tmp_path, offset):
    """The old code did chunk.decode() on exactly 1024 bytes. A 3-byte character
    straddling that boundary raised UnicodeDecodeError, so the file was classified
    binary and load_dir dropped it in silence. Measured at offsets 1022 and 1023."""
    p = tmp_path / f"straddle_{offset}.ts"
    p.write_bytes((("a" * offset) + "— export const tenancy = 1\n").encode("utf-8"))
    assert p.read_bytes().decode("utf-8"), "fixture must be valid UTF-8"
    assert _looks_binary(p) is False, f"valid UTF-8 misread as binary at offset {offset}"


def test_a_real_binary_file_is_still_detected(tmp_path):
    p = tmp_path / "logo.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
    assert _looks_binary(p) is True


def test_bytes_that_are_not_utf8_at_all_are_still_binary(tmp_path):
    p = tmp_path / "latin1.bin"
    p.write_bytes(b"\xff\xfe\xfd" * 64)
    assert _looks_binary(p) is True


# --- and no skip may be silent -------------------------------------------------
def _store():
    import dataclasses

    from src.config import load_config
    return ContextStore(dataclasses.replace(
        load_config(), store_dir=Path(tempfile.mkdtemp()) / "contexts"))


def test_load_dir_records_every_skip_with_its_reason(tmp_path):
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "dep.ts").write_text("export const dep = 1;\n")
    (tmp_path / ".env").write_text("SECRET=x\n")
    (tmp_path / "logo.png").write_bytes(b"\x89PNG" + b"\x00" * 40)
    (tmp_path / "keep.ts").write_text("export const keep = 1;\n")
    # the file the old heuristic would have dropped in silence
    (tmp_path / "tenancy.ts").write_bytes((("a" * 1022) + "— t\n").encode("utf-8"))

    meta = _store().load_dir(str(tmp_path))
    assert meta.file_count == 2, f"expected keep.ts + tenancy.ts, got {meta.files}"
    assert "tenancy.ts" in meta.files, "the straddling file must now load"

    assert meta.skipped_counts == {"skip-dir": 1, "skip-name": 1, "binary": 1}, \
        meta.skipped_counts
    reasons = {e.split(":", 1)[0] for e in meta.skipped}
    assert reasons == {"skip-dir", "skip-name", "binary"}, meta.skipped
    assert any("logo.png" in e for e in meta.skipped)
    assert any(".env" in e for e in meta.skipped)


class _Meta:
    ctx_id, source, source_type, data_type = "ctx_x", "/s", "dir", "dir"
    bytes, lines, est_tokens, sha256 = 1, 1, 1, "a" * 64
    file_count = 173
    files: list = []
    skipped_counts: dict = {}
    skipped: list = []


def test_the_load_output_shouts_when_the_context_is_incomplete():
    m = _Meta()
    m.skipped_counts = {"binary": 11}
    m.skipped = [f"binary: f{i}.ts" for i in range(11)]

    out = srv._meta_block(m)
    assert "INCOMPLETE" in out
    assert "of 184 found" in out, "must state how many of how many"
    assert "wrong by omission" in out
    assert "binary x11" in out


def test_intentional_exclusions_do_not_raise_the_alarm():
    """node_modules being skipped is the POINT of _SKIP_DIRS. An alarm that fires on
    every JS project is one nobody reads -- and then it is missing when 11 source
    files really do vanish."""
    m = _Meta()
    m.skipped_counts = {"skip-dir": 3000}
    m.skipped = ["skip-dir: node_modules/x.js"]

    out = srv._meta_block(m)
    assert "INCOMPLETE" not in out, "an intentional exclusion must not cry wolf"
    assert "excluded by policy: 3,000" in out, "but it must still be stated"


def test_a_surprising_skip_is_still_loud_among_intentional_ones():
    m = _Meta()
    m.skipped_counts = {"skip-dir": 3000, "binary": 1}
    m.skipped = ["binary: src/tenancy.ts", "skip-dir: node_modules/x.js"]

    out = srv._meta_block(m)
    assert "excluded by policy: 3,000" in out
    assert "INCOMPLETE" in out and "binary x1" in out
    assert "src/tenancy.ts" in out, "the surprising path must be named"


def test_the_skip_sample_is_bounded_so_meta_json_cannot_balloon(tmp_path):
    """Unbounded, a node_modules tree wrote 3,000 entries / 128 KB of meta.json for a
    context holding one 20-byte file -- and list_metas() parses every meta.json."""
    (tmp_path / "keep.ts").write_text("x")
    nm = tmp_path / "node_modules" / "pkg"
    nm.mkdir(parents=True)
    for i in range(300):
        (nm / f"f{i}.js").write_text("x")

    meta = _store().load_dir(str(tmp_path))
    assert meta.skipped_counts == {"skip-dir": 300}, "the COUNT must stay exact"
    assert len(meta.skipped) <= 50, "the stored sample must be bounded"


def test_a_complete_load_says_nothing_extra():
    class _Meta:
        ctx_id, source, source_type, data_type = "ctx_x", "/s", "dir", "dir"
        bytes, lines, est_tokens, sha256 = 1, 1, 1, "a" * 64
        file_count, files, skipped = 3, [], []

    assert "INCOMPLETE" not in srv._meta_block(_Meta())

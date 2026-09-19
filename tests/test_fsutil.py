"""One writer, and the hardening every caller now inherits.

Five of six writers had none of this: a predictable `<name>.<pid>.tmp`, no fsync, no
mode, and no cleanup -- a failure left the temp behind with nothing to collect it.
`engine`'s was the worst: its temp name carried no pid at all, and the checkpoint path
is derived from the question, so two processes resuming the same query wrote the same
temp file. None of it had a test, which is why the gap survived six months.
"""

from __future__ import annotations

import os
import sys

import pytest

from better_rlm import fsutil


def test_a_text_write_lands_whole_and_leaves_no_temp(tmp_path):
    target = tmp_path / "sub" / "out.txt"
    fsutil.atomic_write(target, "hello")
    assert target.read_text(encoding="utf-8") == "hello"
    assert list(tmp_path.rglob("*.tmp")) == []


def test_bytes_go_through_untranslated(tmp_path):
    """The checkpoint's .state.dill is raw pickled bytes. Text mode would corrupt it."""
    blob = bytes(range(256))
    target = tmp_path / "state.dill"
    fsutil.atomic_write(target, blob)
    assert target.read_bytes() == blob


def test_text_keeps_lf_on_every_platform(tmp_path):
    """.gitattributes pins this repo to eol=lf, and these files are tracked. Text mode's
    platform default rewrites every line ending on Windows, turning a one-token change
    into a whole-file diff."""
    target = tmp_path / "out.yaml"
    fsutil.atomic_write(target, "a: 1\nb: 2\n")
    assert bytes([13]) not in target.read_bytes()


def test_a_failed_write_removes_its_temp_and_re_raises(tmp_path, monkeypatch):
    """The four naive writers swallowed OSError and left the temp behind. Nothing globs
    for those, so they accumulate -- one was found stuck in ~/.rlm/logs for days."""
    target = tmp_path / "out.txt"

    def boom(_fd):
        raise OSError("disk full")

    monkeypatch.setattr(fsutil.os, "fsync", boom)
    with pytest.raises(OSError, match="disk full"):
        fsutil.atomic_write(target, "never lands")
    assert not target.exists()
    assert list(tmp_path.rglob("*.tmp")) == [], "the temp was orphaned"


def test_the_bytes_reach_the_disk_before_the_rename(tmp_path, monkeypatch):
    """fsync is what stops a host crash leaving the renamed file holding unflushed
    bytes -- and the ledger and the checkpoint are the files whose loss costs the work
    they exist to protect. No writer had it except config_writer and envfile."""
    order: list[str] = []
    real_fsync, real_replace = fsutil.os.fsync, fsutil.os.replace
    monkeypatch.setattr(fsutil.os, "fsync", lambda fd: (order.append("fsync"), real_fsync(fd))[1])
    monkeypatch.setattr(fsutil.os, "replace",
                        lambda a, b: (order.append("replace"), real_replace(a, b))[1])
    fsutil.atomic_write(tmp_path / "out.txt", "x")
    assert order == ["fsync", "replace"]


def test_the_temp_name_is_unpredictable(tmp_path, monkeypatch):
    """engine's temp carried no pid and its path is derived from the question, so two
    processes resuming the same query collided on one temp file."""
    seen: list[str] = []
    real = fsutil.tempfile.mkstemp
    monkeypatch.setattr(fsutil.tempfile, "mkstemp",
                        lambda **kw: (lambda r: (seen.append(r[1]), r)[1])(real(**kw)))
    fsutil.atomic_write(tmp_path / "ck.json", "{}")
    fsutil.atomic_write(tmp_path / "ck.json", "{}")
    assert len(set(seen)) == 2, "two writes shared a temp name"
    assert "ck.json" not in os.path.basename(seen[0]), "the temp is named after the target"


@pytest.mark.skipif(sys.platform == "win32",
                    reason="POSIX mode bits; Windows is asserted by the ACL path")
def test_a_secret_is_owner_only_from_creation(tmp_path):
    """Applied to the TEMP before the rename, so the content is never group-readable
    even briefly."""
    target = tmp_path / ".env"
    fsutil.atomic_write(target, "MINIMAX_API_KEY=shh\n", secret=True)
    assert target.stat().st_mode & 0o077 == 0


def test_an_ordinary_write_is_not_forced_private(tmp_path):
    """config.yaml is meant to be diffed and committed; only .env must be locked down."""
    target = tmp_path / "config.yaml"
    fsutil.atomic_write(target, "mode: api\n")
    assert target.exists()


# --- the sweep claim ---------------------------------------------------------
def test_the_claim_is_taken_once_inside_the_cooldown(tmp_path):
    """The cooldown is measured against the sentinel's MTIME, not the timestamp written
    into it -- so a test has to move the mtime, not just pass a later t. Pinned upstream
    by test_logsetup.py::test_the_sweep_cooldown_is_measured_against_the_sentinel_mtime.
    """
    import time

    assert fsutil.claim_sweep(tmp_path, 60.0, time.time()) is True
    assert fsutil.claim_sweep(tmp_path, 60.0, time.time()) is False, "swept twice"

    sentinel = tmp_path / ".sweep"
    old_mtime = time.time() - 61
    os.utime(sentinel, (old_mtime, old_mtime))
    assert fsutil.claim_sweep(tmp_path, 60.0, time.time()) is True


def test_an_absent_directory_is_not_swept(tmp_path):
    assert fsutil.claim_sweep(tmp_path / "nope", 60.0, 1000.0) is False


def test_a_failed_claim_leaves_no_temp(tmp_path):
    """A directory where the sentinel cannot be replaced: the claim fails and cleans up.
    Found in the wild -- a .sweep.<pid>.tmp stuck in ~/.rlm/logs for days."""
    (tmp_path / ".sweep").mkdir()          # os.replace onto a directory raises
    assert fsutil.claim_sweep(tmp_path, 60.0, 1000.0) is False
    assert list(tmp_path.glob(".sweep.*.tmp")) == []

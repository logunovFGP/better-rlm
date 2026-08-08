"""End-to-end check of the hardened sandbox harness.

Runs the real generated script with the local interpreter (no Docker, no proxy)
after retargeting its STATE path, so the three regressions the patch fixes are
each exercised: bounded locals, a detectable result marker, and state that
survives — or loudly reports — a corrupt pickle.
"""

import json
import os
import subprocess
import sys

import pytest
from rlm.environments.docker_repl import _build_exec_script

import src.sandbox_patch as sp
from src.sandbox_patch import REPR_CAP, SENTINEL, harden_script


def _run(code: str, state_path, tmp_path) -> dict:
    """Build + harden the harness for ``code``, point STATE at tmp, run it."""
    script = harden_script(_build_exec_script(code, proxy_port=1, depth=1))
    script = script.replace(
        'STATE = "/workspace/state.dill"', f"STATE = {str(state_path)!r}"
    )
    path = tmp_path / "harness.py"
    path.write_text(script, encoding="utf-8", newline="\n")
    proc = subprocess.run(
        [sys.executable, str(path)], capture_output=True, text=True, timeout=120
    )
    assert proc.returncode == 0, proc.stderr
    marked = [ln for ln in proc.stdout.splitlines() if ln.startswith(SENTINEL)]
    assert len(marked) == 1, f"no result marker in: {proc.stdout[:400]!r}"
    return json.loads(marked[0][len(SENTINEL) :])


def test_locals_are_capped_and_result_is_marked(tmp_path):
    state = tmp_path / "state.dill"
    data = _run('big = "x" * 5000\nprint("rows", len(big))', state, tmp_path)

    assert data["stdout"] == "rows 5000\n"
    assert data["stderr"] == ""
    # The whole point: a 5 KB variable must not come back as 5 KB of repr.
    assert len(data["locals"]["big"]) <= REPR_CAP
    assert state.exists() and not (tmp_path / "state.dill.tmp").exists()


def test_state_survives_between_runs(tmp_path):
    state = tmp_path / "state.dill"
    _run('kept = "value"', state, tmp_path)
    data = _run("print(kept)", state, tmp_path)
    assert data["stdout"] == "value\n"


def test_corrupt_state_is_reported_not_silently_reset(tmp_path):
    state = tmp_path / "state.dill"
    _run('kept = "value"', state, tmp_path)
    state.write_bytes(b"\x80\x04 truncated garbage")

    data = _run("print(SHOW_VARS())", state, tmp_path)
    assert "state.dill unreadable" in data["stderr"]
    assert "No variables created yet" in data["stdout"]


def test_utf8_open_forces_encoding_for_text_but_not_binary(tmp_path):
    # docker_repl writes host files the Linux guest reads as UTF-8. On a non-UTF-8
    # locale (measured: cp1251) the default encoding raised UnicodeEncodeError on "→".
    # Binary must stay untouched — passing encoding= to a binary open is a TypeError.
    text = tmp_path / "ctx.txt"
    with sp._utf8_open(text, "w") as f:
        f.write("→ ✓ é")
    assert text.read_bytes() == "→ ✓ é".encode("utf-8")

    binary = tmp_path / "state.dill"
    with sp._utf8_open(binary, "wb") as f:
        f.write(b"\x80\x04not-utf8")
    assert binary.read_bytes() == b"\x80\x04not-utf8"


def test_reap_removes_dead_owners_and_keeps_live_ones(tmp_path, monkeypatch):
    dead_pid = 999_999_999  # assert it is really absent rather than assume
    # POSIX signals ESRCH -> ProcessLookupError; Windows has no such mapping and
    # OpenProcess on an out-of-range pid surfaces as OSError (WinError 87). Keep the
    # narrow assertion where the platform supports it rather than widening both.
    absent = OSError if sys.platform == "win32" else ProcessLookupError
    with pytest.raises(absent):
        os.kill(dead_pid, 0)

    live = tmp_path / "docker_repl_live"
    dead = tmp_path / "docker_repl_dead"
    fresh = tmp_path / "docker_repl_nomarker"
    for d in (live, dead, fresh):
        d.mkdir()
    (live / "owner").write_text(f"{os.getpid()}\nlive-container\n")
    (dead / "owner").write_text(f"{dead_pid}\nno-such-container\n")

    calls = []
    monkeypatch.setattr(sp.subprocess, "run", lambda *a, **k: calls.append(a[0]))

    assert sp.reap_orphans(str(tmp_path)) == ["docker_repl_dead"]
    assert not dead.exists()
    assert live.exists()          # owned by this very process
    assert fresh.exists()         # no marker, but too young to call an orphan
    assert calls == [["docker", "rm", "-f", "no-such-container"]]


def test_setup_marker_records_pid_container_and_host_workspace(tmp_path, monkeypatch):
    """The 3rd line is how remotely-fetched data reaches the store: the container
    writes to /workspace, the host then loads it by this path."""
    monkeypatch.setattr(sp, "_orig_setup", lambda self: "ok")

    class Env:
        temp_dir = str(tmp_path)
        container_id = "abc123"

    assert sp._setup(Env()) == "ok"
    pid, container, host = (tmp_path / "owner").read_text().split()
    assert int(pid) == os.getpid()
    assert container == "abc123"
    assert host == str(tmp_path)


def test_harden_fails_loudly_if_upstream_template_moves():
    with pytest.raises(RuntimeError, match="rlms internals changed"):
        harden_script("print('not the harness')")

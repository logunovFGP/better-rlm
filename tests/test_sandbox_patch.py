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


def test_reap_removes_dead_owners_and_keeps_live_ones(tmp_path, monkeypatch):
    dead_pid = 999_999_999  # assert it is really absent rather than assume
    with pytest.raises(ProcessLookupError):
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


def test_harden_fails_loudly_if_upstream_template_moves():
    with pytest.raises(RuntimeError, match="rlms internals changed"):
        harden_script("print('not the harness')")

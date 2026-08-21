"""The hardened sandbox harness, now that the engine is vendored source.

These checks used to prove that ``src/sandbox_patch.py`` rewrote the vendored
template correctly. The hardening now IS the template
(``rlm/environments/docker_repl.py``), so they prove the template itself: bounded
locals, a detectable result marker, and state that survives — or loudly reports —
a corrupt pickle. Runs the real generated script under the local interpreter
(no Docker, no proxy) after retargeting its STATE path.

test_template_still_carries_every_hardening is the regression guard that replaced
the old fail-loudly string match: an upstream merge that reverts the template now
fails here instead of silently costing megabytes per call.
"""

import inspect
import json
import os
import subprocess
import sys

import pytest
from rlm.environments import docker_repl as dr
from rlm.environments.docker_repl import (
    LOCALS_REPR_CAP,
    RLM_RESULT_SENTINEL,
    DockerREPL,
    _build_exec_script,
)

import src.sandbox_reap as reap


def _run(code: str, state_path, tmp_path) -> dict:
    """Build the harness for ``code``, point STATE at tmp, run it."""
    script = _build_exec_script(code, proxy_port=1, depth=1).replace(
        'STATE = "/workspace/state.dill"', f"STATE = {str(state_path)!r}"
    )
    path = tmp_path / "harness.py"
    path.write_text(script, encoding="utf-8", newline="\n")
    proc = subprocess.run(
        [sys.executable, str(path)], capture_output=True, text=True, timeout=120
    )
    assert proc.returncode == 0, proc.stderr
    marked = [ln for ln in proc.stdout.splitlines() if ln.startswith(RLM_RESULT_SENTINEL)]
    assert len(marked) == 1, f"no result marker in: {proc.stdout[:400]!r}"
    return json.loads(marked[0][len(RLM_RESULT_SENTINEL):])


def test_locals_are_capped_and_result_is_marked(tmp_path):
    state = tmp_path / "state.dill"
    data = _run('big = "x" * 5000\nprint("rows", len(big))', state, tmp_path)

    assert data["stdout"] == "rows 5000\n"
    assert data["stderr"] == ""
    # The whole point: a 5 KB variable must not come back as 5 KB of repr.
    assert len(data["locals"]["big"]) <= LOCALS_REPR_CAP
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


def test_template_still_carries_every_hardening():
    """An upstream merge that restores the original template must fail here."""
    script = _build_exec_script("pass", proxy_port=1, depth=1)
    assert f'print("{RLM_RESULT_SENTINEL}"' in script, "result marker gone"
    assert f"repr(v)[:{LOCALS_REPR_CAP}]" in script, "locals repr cap gone"
    assert "os.replace(_tmp, STATE)" in script, "atomic state write gone"
    assert "_STATE_LOAD_ERR" in script, "state-load guard gone"
    assert 'print(json.dumps({' not in script, "unmarked result print is back"


def test_host_writes_the_guest_reads_are_utf8():
    # docker_repl writes host files the Linux guest reads as UTF-8. On a non-UTF-8
    # locale (measured: cp1251) the default encoding raised UnicodeEncodeError on "→".
    # This replaced a module-wide `open` shadow once the source became editable.
    for method in (DockerREPL.load_context, DockerREPL.execute_code):
        src = inspect.getsource(method)
        for line in src.splitlines():
            if "open(" in line and '"w"' in line:
                assert 'encoding="utf-8"' in line or line.rstrip().endswith('"w",'), (
                    f"unencoded host text write in {method.__name__}: {line.strip()}"
                )
    # state.dill must stay binary — passing encoding= to a binary open is a TypeError.
    # (the atomic write goes to _tmp then os.replace's it over STATE)
    script = _build_exec_script("pass", proxy_port=1, depth=1)
    assert 'open(_tmp, "wb")' in script and 'open(STATE, "rb")' in script


def test_exec_timeout_is_configurable_and_defaults(monkeypatch):
    # config.yaml documented sandbox_timeout_s long before anything enforced it.
    # __init__ assigns timeout_s and then calls setup(), which starts a real container:
    # without stubbing it, this assertion about attribute wiring fails with "failed to
    # connect to the docker API" wherever the daemon is not running, taking the repo's
    # whole verify command — and so the pre-push hook — down with it.
    monkeypatch.setattr(dr.DockerREPL, "setup", lambda self: None)
    assert DockerREPL(timeout_s=7).timeout_s == 7
    assert DockerREPL().timeout_s == dr.DEFAULT_EXEC_TIMEOUT_S


def test_reap_removes_dead_owners_and_keeps_live_ones(tmp_path, monkeypatch):
    # Absence is asserted through the module's own probe, never os.kill(pid, 0): on
    # Windows signal 0 is CTRL_C_EVENT, so that call delivers a console Ctrl+C to the
    # process group instead of answering the question. It queued a KeyboardInterrupt
    # that landed thirty tests later, inside an unrelated subprocess launch.
    dead_pid = 999_999_999
    assert not reap._pid_alive(dead_pid), "pid must really be absent, not assumed"
    assert reap._pid_alive(os.getpid()), "the probe must recognise a live process"

    live = tmp_path / "docker_repl_live"
    dead = tmp_path / "docker_repl_dead"
    fresh = tmp_path / "docker_repl_nomarker"
    for d in (live, dead, fresh):
        d.mkdir()
    (live / "owner").write_text(f"{os.getpid()}\nlive-container\n")
    (dead / "owner").write_text(f"{dead_pid}\nno-such-container\n")

    calls = []
    monkeypatch.setattr(reap.subprocess, "run", lambda *a, **k: calls.append(a[0]))

    assert reap.reap_orphans(str(tmp_path)) == ["docker_repl_dead"]
    assert not dead.exists()
    assert live.exists()          # owned by this very process
    assert fresh.exists()         # no marker, but too young to call an orphan
    assert calls == [["docker", "rm", "-f", "no-such-container"]]


def test_liveness_probe_never_signals_the_process_it_asks_about(monkeypatch):
    """os.kill(pid, 0) is a liveness probe on POSIX and a console Ctrl+C on Windows,
    where signal.CTRL_C_EVENT == 0. reap_orphans calls the probe once per owner marker
    at startup, so on Windows the old implementation interrupted process groups it read
    out of stale markers. Anything that reintroduces os.kill there must fail here."""
    if os.name != "nt":
        return  # os.kill IS the right call on POSIX; nothing to guard

    signalled = []
    monkeypatch.setattr(reap.os, "kill", lambda *a: signalled.append(a))
    assert reap._pid_alive(os.getpid())
    assert not reap._pid_alive(999_999_999)
    assert signalled == [], "the probe signalled instead of asking"


def test_reap_sweep_never_raises(monkeypatch, tmp_path):
    # A janitor must never break the startup it runs in.
    monkeypatch.setattr(reap, "reap_orphans", lambda root: (_ for _ in ()).throw(OSError("boom")))
    monkeypatch.setenv("RLM_DOCKER_WORKSPACE_DIR", str(tmp_path))
    reap.reap_stale_sandboxes()  # must not raise


def test_owner_marker_records_pid_container_and_host_workspace(tmp_path):
    """The 3rd line is how remotely-fetched data reaches a store: the container
    writes to /workspace, the host then loads it by this path."""
    env = DockerREPL.__new__(DockerREPL)   # no __init__: nothing was started here
    env._cleaned_up = True                 # ...so __del__ must not go `docker exec abc123`
    env.temp_dir = str(tmp_path)
    env.container_id = "abc123"
    env._write_owner_marker()

    pid, container, host = (tmp_path / "owner").read_text().split()
    assert int(pid) == os.getpid()
    assert container == "abc123"
    assert host == str(tmp_path)


def test_del_never_shells_out_while_the_interpreter_is_finalizing(monkeypatch, tmp_path):
    """A __del__ that runs `docker exec` at shutdown fails the process without failing a
    test: the exception escapes finalization and sets the exit code, so CI goes red while
    the summary still reads all-passed. Cheap to get wrong again, so pin it."""
    env = DockerREPL.__new__(DockerREPL)
    env.temp_dir = str(tmp_path)
    env.container_id = "abc123"
    ran = []
    monkeypatch.setattr(dr.subprocess, "run", lambda *a, **k: ran.append(a))
    monkeypatch.setattr(dr.shutil, "rmtree", lambda *a, **k: ran.append(a))
    monkeypatch.setattr(dr.sys, "is_finalizing", lambda: True)

    env.__del__()
    assert ran == [], "touched docker or the filesystem during interpreter shutdown"
    # Marked only now: cleanup() has to be live for the assertion above to mean anything,
    # but once monkeypatch lifts the stubs a collection would run the real `docker exec
    # abc123`, which is what stalled the Windows job for the run that added this test.
    env._cleaned_up = True


def test_stale_rlms_install_is_refused_loudly(monkeypatch):
    """A leftover `rlms` shadowing ./rlm must fail at import, not silently drop
    every engine fix. pip does not uninstall a dependency merely dropped from
    pyproject, so this is the state an upgrader lands in.
    """
    import importlib
    import sys

    from rlm.environments import docker_repl

    monkeypatch.delattr(docker_repl, "RLM_RESULT_SENTINEL")
    monkeypatch.delitem(sys.modules, "src.engine")
    try:
        with pytest.raises(ImportError, match="stale `rlms`"):
            importlib.import_module("src.engine")
    finally:
        # leave a working src.engine behind for whatever runs next
        monkeypatch.undo()
        sys.modules.pop("src.engine", None)
        importlib.import_module("src.engine")

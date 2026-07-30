"""Harden ``rlms==0.1.3``'s DockerREPL exec protocol.

Three upstream defects let ``rlm_exec`` fail silently:

1. Every call shipped ``repr()`` of *every* REPL variable back as one JSON line.
   With a 7 MB context loaded that measured 30,013,251 bytes per call — and the
   ``locals`` field is read nowhere in this server, while tool output is capped
   at ``output_cap_bytes`` (4 KB) anyway. Each derived variable that slices the
   context (``parts``, ``files``, …) added another ~7 MB to *every subsequent*
   call, so multi-print exploration degraded as it went.
2. The exec script was rewritten at a fixed path on the bind mount and run
   immediately — no fsync, no unique name — and ``docker exec``'s exit code was
   never checked. A partial or killed run yielded empty stdout, which the host
   read as ``{}``: a 19-byte "### stdout" block reported as success, with no
   error anywhere.
3. ``state.dill`` was written non-atomically and load failures were swallowed,
   so a crash mid-write silently reset every REPL variable.

Patched from ``src/`` rather than in ``site-packages`` so the fix survives
``pip install`` and applies to every venv (.venv / .venv_sh / .venv_windows) on
every platform. Only DockerREPL is affected — LocalREPL runs in-process and has
none of these paths.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path

from rlm.environments import docker_repl as _dr

from .config import load_config

_LOG = logging.getLogger("rlm-mcp")

#: Per-variable cap for the ``locals`` echo — enough to identify a variable,
#: small enough that a loaded context costs bytes instead of megabytes.
REPR_CAP = 200

#: Marks the harness's result line, so a truncated or killed run is detectable
#: instead of being read as "the cell printed nothing".
SENTINEL = "__RLM_RESULT__"

#: Wall-clock ceiling for one sandbox call. Reads the ``sandbox_timeout_s`` knob
#: that config.yaml already documents — until now nothing enforced it.
TIMEOUT_S = load_config().sandbox_timeout_s


def _sub(script: str, old: str, new: str, what: str) -> str:
    """Single targeted replacement, loud if the vendored template moved."""
    if old not in script:
        raise RuntimeError(
            f"sandbox_patch: cannot harden {what} — rlms internals changed. "
            "Re-check rlm/environments/docker_repl.py against pinned rlms==0.1.3."
        )
    return script.replace(old, new, 1)


def harden_script(script: str) -> str:
    """Rewrite the generated in-container harness to bound its payload, mark its
    result line, and make state writes atomic and state loss visible."""
    script = _sub(
        script,
        '"locals": {k: repr(v) for k, v in _locals.items() if not k.startswith("_")},',
        '"locals": {k: repr(v)[:%d] for k, v in _locals.items() if not k.startswith("_")},'
        % REPR_CAP,
        "the locals echo",
    )
    script = _sub(
        script,
        "print(json.dumps({",
        f'print("{SENTINEL}" + json.dumps({{',
        "the result marker",
    )
    # A crash between truncate and flush used to leave a half-written pickle that
    # the next call silently read as "no variables".
    script = _sub(
        script,
        '    with open(STATE, "wb") as f:\n        dill.dump(clean, f)',
        '    _tmp = STATE + ".tmp"\n'
        '    with open(_tmp, "wb") as f:\n'
        "        dill.dump(clean, f)\n"
        "        f.flush()\n"
        "        os.fsync(f.fileno())\n"
        "    os.replace(_tmp, STATE)",
        "the atomic state write",
    )
    script = _sub(
        script,
        '            with open(STATE, "rb") as f:\n'
        "                return dill.load(f)\n"
        "        except Exception:\n"
        "            pass",
        '            with open(STATE, "rb") as f:\n'
        "                return dill.load(f)\n"
        "        except Exception as _e:\n"
        '            globals()["_STATE_LOAD_ERR"] = (\n'
        '                "WARNING: state.dill unreadable (%r) — REPL variables were reset\\n" % (_e,)\n'
        "            )",
        "the state-load guard",
    )
    return _sub(
        script,
        '    "stderr": stderr_buf.getvalue(),',
        '    "stderr": globals().get("_STATE_LOAD_ERR", "") + stderr_buf.getvalue(),',
        "the stderr passthrough",
    )


def _unlink(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


def _execute_code(self, code: str):  # noqa: ANN001 — bound as a method
    """Hardened ``DockerREPL.execute_code``: unique fsynced script, bounded
    timeout, and an explicit exit-code / result-marker check so a dead sandbox
    reports an error instead of an empty cell."""
    start = time.perf_counter()
    script = harden_script(
        _dr._build_exec_script(
            code,
            self.proxy_port,
            self.depth,
            custom_tools=self.custom_tools,
            compaction=self.compaction,
        )
    )

    # Unique name + fsync: the container only ever opens a file the host has
    # already flushed and closed, so there is no rewrite-in-place race (the
    # macOS virtiofs / Docker Desktop bind-mount failure mode). encoding and
    # newline are pinned because the guest is Linux whatever the host is —
    # without them a Windows host writes cp1252 + CRLF into a UTF-8 reader.
    name = f"_exec_{uuid.uuid4().hex}.py"
    host_path = os.path.join(self.temp_dir, name)
    with open(host_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(script)
        f.flush()
        os.fsync(f.fileno())

    timed_out = False
    try:
        proc = subprocess.run(
            ["docker", "exec", self.container_id, "python", f"/workspace/{name}"],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_S,
        )
        out, err, rc = (proc.stdout or ""), (proc.stderr or ""), proc.returncode
    except subprocess.TimeoutExpired:
        timed_out, out, err, rc = True, "", "", -1

    with self._calls_lock:
        calls = self.pending_calls.copy()
        self.pending_calls.clear()

    data = None
    for line in reversed(out.splitlines()):
        if line.startswith(SENTINEL):
            try:
                data = json.loads(line[len(SENTINEL) :])
            except json.JSONDecodeError as exc:
                err = f"{err}\nresult line unparseable ({exc})".strip()
            break

    if data is None:
        # The script is kept on disk for post-mortem: a silent empty result was
        # the whole bug, so surface where to look.
        reason = (
            f"timed out after {TIMEOUT_S}s"
            if timed_out
            else f"rc={rc}, no result marker in {len(out)} B of stdout"
        )
        _LOG.warning("sandbox exec failed (%s), script kept at %s", reason, host_path)
        return _dr.REPLResult(
            stdout=out,
            stderr=f"sandbox exec failed: {reason}; script kept at {host_path}\n{err}".strip(),
            locals={},
            execution_time=time.perf_counter() - start,
            rlm_calls=calls,
        )

    _unlink(host_path)
    return _dr.REPLResult(
        stdout=data.get("stdout", ""),
        stderr=data.get("stderr", "") + err,
        locals=data.get("locals", {}),
        execution_time=time.perf_counter() - start,
        rlm_calls=calls,
        final_answer=data.get("final_answer"),
    )


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # exists, owned by another user
        return True
    return True


def workspace_root() -> str:
    """Where the vendored env puts sandbox dirs — mirrors docker_repl.setup()."""
    return os.environ.get(
        "RLM_DOCKER_WORKSPACE_DIR", os.path.join(os.getcwd(), ".rlm_workspace"))


def reap_orphans(root: str) -> list[str]:
    """Remove sandboxes whose owning server is gone; return what was reaped.

    SIGKILL and hard client teardowns bypass every exit hook, so a container
    outlives its server: measured 11 of 19 processes exiting with no shutdown
    record, containers alive for hours, workspace dirs left behind. A sweep at
    startup catches what atexit structurally cannot.
    """
    reaped: list[str] = []
    for d in sorted(Path(root).glob("docker_repl_*")):
        marker = d / "owner"
        pid: int | None = None
        container = ""
        if marker.exists():
            try:
                fields = marker.read_text(encoding="utf-8").split()
                pid, container = int(fields[0]), (fields[1] if len(fields) > 1 else "")
            except (ValueError, IndexError, OSError):
                pid = None
        elif time.time() - d.stat().st_mtime < 3600:
            continue  # pre-patch or mid-setup: too young to call an orphan
        if pid is not None and _pid_alive(pid):
            continue
        if container:
            subprocess.run(["docker", "rm", "-f", container],
                           capture_output=True, timeout=60)
        shutil.rmtree(d, ignore_errors=True)
        reaped.append(d.name)
    if reaped:
        _LOG.warning("evt=reaped_orphans count=%d dirs=%s", len(reaped), ",".join(reaped))
    return reaped


def _setup(self):
    """Record who owns this sandbox so a later startup sweep can reap it if this
    process dies without running any exit hook."""
    result = _orig_setup(self)
    try:
        (Path(self.temp_dir) / "owner").write_text(
            f"{os.getpid()}\n{self.container_id}\n", encoding="utf-8")
    except OSError as exc:  # ownership is best-effort; never block the sandbox
        _LOG.warning("evt=owner_marker_failed err=%r", exc)
    return result


_orig_setup = _dr.DockerREPL.setup


def patch_sandbox() -> None:
    """Rebind ``DockerREPL.execute_code`` / ``setup`` with the hardened versions and
    reap sandboxes abandoned by dead servers. Idempotent."""
    if getattr(_dr, "_rlmmcp_patched", False):
        return
    _dr.DockerREPL.execute_code = _execute_code
    _dr.DockerREPL.setup = _setup
    _dr._rlmmcp_patched = True
    try:
        reap_orphans(workspace_root())
    except Exception as exc:  # a janitor must never break startup
        _LOG.warning("evt=reap_failed err=%r", exc)

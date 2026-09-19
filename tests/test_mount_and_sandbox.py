"""Auto-mounting to Claude Code, and what happens when the sandbox is gone.

Two features sharing one theme: finishing a step and being silently wrong about
it. Setup that reports Ready while nothing is mounted, and a sandbox failure that
reaches an agent as a docker socket path it can do nothing with.
"""

from __future__ import annotations

import dataclasses

from better_rlm import engine, mcpreg
from better_rlm.config import load_config


@dataclasses.dataclass
class _Proc:
    """The fields of subprocess.CompletedProcess these paths actually read."""

    returncode: int
    stdout: str = ""
    stderr: str = ""


# --- mounting -----------------------------------------------------------------


def test_a_wheel_registers_its_own_interpreter_not_the_bare_name(monkeypatch):
    """install.ps1 writes a better-rlm.cmd shim that repoints the bare word at a
    checkout, so a name-based registration can silently flip to a different
    install -- exactly the failure this feature exists to stop. `sys.executable
    -m` cannot be repointed by anything on PATH."""
    monkeypatch.setattr(mcpreg, "IS_CHECKOUT", False)
    cmd = mcpreg.server_command()
    assert cmd[1:] == ["-m", "better_rlm.cli", "server"]
    assert "better-rlm" not in cmd[0], cmd[0]


def test_a_checkout_registers_its_own_launcher(monkeypatch, tmp_path):
    """A checkout has a venv its launcher knows about; a bare interpreter does not."""
    monkeypatch.setattr(mcpreg, "IS_CHECKOUT", True)
    monkeypatch.setattr(mcpreg, "PKG_ROOT", tmp_path)
    joined = " ".join(mcpreg.server_command())
    assert str(tmp_path) in joined and "run_server" in joined


def test_no_claude_cli_is_skipped_with_the_command_to_paste(monkeypatch):
    """Setup must not die because an unrelated CLI is missing -- and must not
    claim to have mounted anything either."""
    monkeypatch.setattr(mcpreg, "claude_cli", lambda: None)
    outcome, msg = mcpreg.ensure_registered()
    assert outcome == "skipped"
    assert "claude mcp add" in msg and "-s user" in msg


def test_an_identical_registration_is_left_alone(monkeypatch):
    """Re-running setup must not churn a registration that is already correct."""
    calls: list[list[str]] = []
    monkeypatch.setattr(mcpreg, "claude_cli", lambda: "claude")
    monkeypatch.setattr(mcpreg, "registered_command", lambda c: mcpreg.server_command())
    monkeypatch.setattr(mcpreg, "_run", lambda a: (calls.append(a), (0, ""))[1])

    outcome, _ = mcpreg.ensure_registered()
    assert outcome == "ok"
    assert calls == [], "it rewrote a registration that already matched"


def test_a_registration_pointing_elsewhere_is_repointed_and_reported(monkeypatch):
    """The reported bug: a checkout registration kept serving stale models long
    after the wizard wrote good ones to ~/.rlm, and nothing on screen said so.
    Leaving a foreign command in place IS the failure, so it is replaced and named.
    """
    calls: list[list[str]] = []
    stale = ["cmd", "/c", "G:" + chr(92) + "old" + chr(92) + "run_server.cmd"]
    monkeypatch.setattr(mcpreg, "claude_cli", lambda: "claude")
    monkeypatch.setattr(mcpreg, "registered_command", lambda c: stale)
    monkeypatch.setattr(mcpreg, "_run", lambda a: (calls.append(a), (0, ""))[1])

    outcome, msg = mcpreg.ensure_registered()
    assert outcome == "repointed"
    assert "run_server.cmd" in msg, "it did not say what it replaced"
    verbs = [a[1:3] for a in calls]
    assert ["mcp", "remove"] in verbs, "`add` alone exits 1 on an existing name"
    assert ["mcp", "add"] in verbs


def test_a_failed_add_is_reported_not_swallowed(monkeypatch):
    monkeypatch.setattr(mcpreg, "claude_cli", lambda: "claude")
    monkeypatch.setattr(mcpreg, "registered_command", lambda c: None)
    monkeypatch.setattr(mcpreg, "_run", lambda a: (1, "boom"))
    outcome, msg = mcpreg.ensure_registered()
    assert outcome == "failed" and "boom" in msg


def test_registration_survives_a_claude_cli_that_hangs(monkeypatch):
    """_run turns OSError/timeout into a return code, so a wedged CLI ends setup
    with a message rather than an exception escaping commit()."""
    import subprocess as sp

    def boom(*a, **k):
        raise sp.TimeoutExpired("claude", 30)

    monkeypatch.setattr(mcpreg.subprocess, "run", boom)
    rc, out = mcpreg._run(["claude", "mcp", "get", "rlm"])
    assert rc == 1 and "TimeoutExpired" in out


def test_the_wizard_mounts_when_it_finishes(tmp_path, monkeypatch):
    """Mounting is part of finishing setup. An operator who sees Ready while
    nothing is mounted has been told something false."""
    import io

    from rich.console import Console

    from better_rlm import onboard, tui

    seen: list[str] = []
    monkeypatch.setattr(mcpreg, "ensure_registered",
                        lambda **k: (seen.append("called"), ("added", "mounted as rlm"))[1])
    monkeypatch.setattr(tui, "_save", lambda *a, **k: None)
    monkeypatch.setattr(tui, "render_status", lambda st: "")

    buf = io.StringIO()
    w = onboard.Wizard(mode="api", base_url="", models=(("root_model", "x"),))
    onboard.commit(Console(file=buf, width=200), tmp_path / "config.yaml", w)

    assert seen == ["called"], "the wizard finished without mounting"
    assert "Mounted" in buf.getvalue()


# --- sandbox unavailable ------------------------------------------------------


def _cfg(**kw):
    return dataclasses.replace(load_config(), sandbox="docker", **kw)


def test_docker_missing_and_docker_stopped_are_told_apart(monkeypatch):
    """Two causes, two different fixes. Collapsing them into "docker unavailable"
    sends an operator to install something they already have."""
    monkeypatch.setattr(engine.shutil, "which", lambda n: None)
    assert "not installed" in (engine.sandbox_diagnosis(_cfg()) or "")

    monkeypatch.setattr(engine.shutil, "which", lambda n: "/usr/bin/docker")
    monkeypatch.setattr(engine.subprocess, "run", lambda *a, **k: _Proc(1))
    assert "daemon is not running" in (engine.sandbox_diagnosis(_cfg()) or "")


def test_a_built_image_is_required_before_the_sandbox_is_called_healthy(monkeypatch):
    """A pip install ships no image, so the daemon can be up and the sandbox still
    unusable -- reported before the call rather than as a docker pull error."""
    monkeypatch.setattr(engine.shutil, "which", lambda n: "/usr/bin/docker")
    seq = iter([_Proc(0, "29.7.2"), _Proc(1, "", "No such image")])
    monkeypatch.setattr(engine.subprocess, "run", lambda *a, **k: next(seq))
    got = engine.sandbox_diagnosis(_cfg())
    assert got and "has not been built" in got, got


def test_a_healthy_docker_diagnoses_nothing(monkeypatch):
    monkeypatch.setattr(engine.shutil, "which", lambda n: "/usr/bin/docker")
    monkeypatch.setattr(engine.subprocess, "run", lambda *a, **k: _Proc(0, "29.7.2"))
    assert engine.sandbox_diagnosis(_cfg()) is None


def test_a_local_sandbox_is_never_diagnosed_as_broken():
    """`sandbox: local` needs no daemon; reporting Docker trouble there is noise."""
    assert engine.sandbox_diagnosis(
        dataclasses.replace(load_config(), sandbox="local")) is None


def test_the_guidance_names_the_tools_that_still_work():
    """The whole point: an agent handed a docker socket path retries the same call;
    one handed this reroutes to the tools that need no sandbox."""
    msg = engine.sandbox_guidance(_cfg(), "the daemon is not running")
    for alt in ("rlm_grep", "rlm_sub_query", "rlm_sub_query_batch", "rlm_estimate"):
        assert alt in msg, f"{alt} is not offered as an alternative"
    assert "do not retry" in msg.lower()
    assert "has not enabled" in msg


def test_the_guidance_never_silently_downgrades_isolation():
    """`sandbox: local` runs generated code ON THE HOST. Falling back to it because
    a daemon happens to be down turns an outage into a privilege escalation, so it
    is labelled and offered as a choice, never taken automatically."""
    msg = engine.sandbox_guidance(_cfg(), "the daemon is not running")
    assert "NO isolation" in msg and "explicit choice" in msg


def test_a_missing_sandbox_reaches_the_agent_as_guidance_not_a_traceback(monkeypatch):
    """rlm_exec returns the message instead of raising, so the tool result an agent
    reads is a routing instruction rather than a docker npipe path."""
    import better_rlm.server as srv

    class _Dead:
        loaded_ctx = None

        def _boom(self, *a, **k):
            raise engine.SandboxUnavailable(
                "Sandbox unavailable: test.\nUSE INSTEAD:\n  - rlm_grep(ctx_id, pattern)")

        load_context = _boom
        execute = _boom

    monkeypatch.setattr(srv, "_get_repl", lambda: _Dead())
    fn = getattr(srv.rlm_exec, "fn", srv.rlm_exec)
    out = fn("print(1)")
    assert "USE INSTEAD" in out and "rlm_grep" in out
    assert "Traceback" not in out

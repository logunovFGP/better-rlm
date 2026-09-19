"""Auto-mounting to Claude Code, and what happens when the sandbox is gone.

Two features sharing one theme: finishing a step and being silently wrong about
it. Setup that reports Ready while nothing is mounted, and a sandbox failure that
reaches an agent as a docker socket path it can do nothing with.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

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
    assert cmd[1:] == ["-P", "-m", "better_rlm.cli", "server"]
    # The property is "argv[0] is an interpreter", asserted on the BASENAME. A
    # substring check for "better-rlm" over the whole path passed locally and failed
    # on CI, where the repo itself lives in .../better-rlm/better-rlm/.venv/bin/python
    # -- the project name was in the directory, not in the executable.
    exe = Path(cmd[0]).name.lower()
    assert exe.startswith(("python", "pypy")), cmd[0]


def test_a_wheel_registration_cannot_be_shadowed_by_the_servers_working_directory(monkeypatch):
    """-P, and it is load-bearing.

    `-m` puts the CURRENT DIRECTORY first on sys.path, and Claude Code starts an MCP
    server with cwd set to the project directory. Working inside a checkout of this
    repo, that directory holds `better_rlm/` -- so the wheel's own interpreter imported
    the CHECKOUT and served the checkout's config.yaml: Claude model ids against a
    MiniMax endpoint, with a registration that named the right interpreter the whole
    time. Measured from the repo root: `where` said `mode: checkout` without -P and
    `installed (pip)` with it.

    Not -I: that also drops user site-packages, and `pip install --user better-rlm`
    puts the package exactly there.
    """
    monkeypatch.setattr(mcpreg, "IS_CHECKOUT", False)
    cmd = mcpreg.server_command()
    assert "-P" in cmd, "cwd can shadow the installed package again"
    assert cmd.index("-P") < cmd.index("-m"), "-P must precede -m to apply"
    assert "-I" not in cmd, "-I would hide a --user install"


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
    # Stale on the first read, correct on the read-back after the add -- the real
    # sequence. A stub that returns stale forever now reports "failed", which is
    # the read-back doing its job rather than a regression.
    reads = iter([stale, mcpreg.server_command()])
    monkeypatch.setattr(mcpreg, "registered_command", lambda c: next(reads))
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
    out = srv.rlm_exec("print(1)")

    assert "USE INSTEAD" in out and "rlm_grep" in out
    # The load-bearing half. logged_tool turns an escaping exception into
    # "ERROR in rlm_exec: <same text>" and records outcome=error -- so asserting
    # only on the guidance text passed even when the tool re-raised, which is how
    # the first version of this test survived its own mutant. A machine without
    # Docker is a fact about the machine, not a failed call.
    assert not out.startswith("ERROR in"), out[:120]


def test_rlm_query_also_degrades_instead_of_raising(monkeypatch, tmp_path):
    """The recursive loop writes Python into the sandbox, so it needs one too.

    Added because a mis-aimed mutation revealed this path was uncovered: the
    suite stayed green with rlm_query re-raising, which is the docker-npipe
    traceback all over again on the more expensive tool.
    """
    import better_rlm.server as srv

    def boom(*a, **k):
        raise engine.SandboxUnavailable(
            "Sandbox unavailable: test." + chr(10) + "USE INSTEAD:" + chr(10)
            + "  - rlm_sub_query_batch(ctx_id, ...)")

    # Everything before the sandbox has to be stubbed or the call dies at model
    # resolution instead, which is a different failure and not the one under test.
    monkeypatch.setattr(srv, "_resolve_root_model", lambda o: "m-root")
    monkeypatch.setattr(srv.models, "select", lambda cfg, role: "m-sub")
    monkeypatch.setattr(srv.DEPS.store, "read_text", lambda c: "body")
    monkeypatch.setattr(srv, "query_checkpoint_path", lambda *a, **k: tmp_path / "ck")
    monkeypatch.setattr(srv, "run_query", boom)

    out = srv.rlm_query("ctx_x", "why?")
    assert "USE INSTEAD" in out and "rlm_sub_query_batch" in out
    assert not out.startswith("ERROR in"), out[:120]


# --- setup must not lie about what it did ---------------------------------------


def test_no_code_path_tells_the_operator_to_run_claude_mcp_restart():
    """`claude mcp restart` DOES NOT EXIST.

    The CLI has add / add-json / get / list / login / logout / remove /
    reset-project-choices / serve -- and no restart. Setup, /mode, /provider and the
    model pickers all closed by telling the operator to run it, so following the
    instructions literally could not work and the server stayed stale. Verified
    against `claude mcp --help`; this pins it so the phrase cannot drift back in.
    """
    root = Path(__file__).resolve().parent.parent
    offenders = []
    for py in (root / "better_rlm").glob("*.py"):
        for n, line in enumerate(py.read_text(encoding="utf-8").splitlines(), 1):
            if "mcp restart" in line and not line.lstrip().startswith("#"):
                offenders.append(f"{py.name}:{n}")
    assert not offenders, f"a non-existent command is printed at {offenders}"


def test_setup_names_the_build_it_just_ran(tmp_path, monkeypatch):
    """Setup was completed twice against a stale install with no mounting step, and
    nothing on screen told the operator which build they were on. A version and an
    install root make that visible in one line."""
    import io

    from rich.console import Console

    from better_rlm import onboard, tui

    monkeypatch.setattr(mcpreg, "ensure_registered", lambda **k: ("ok", "already mounted"))
    monkeypatch.setattr(tui, "_save", lambda *a, **k: None)
    monkeypatch.setattr(tui, "render_status", lambda st: "")

    buf = io.StringIO()
    onboard.commit(Console(file=buf, width=200), tmp_path / "config.yaml",
                   onboard.Wizard(mode="api", models=(("root_model", "x"),)))
    out = buf.getvalue()

    version, active, _ = mcpreg.install_identity()
    assert version in out, "setup did not say which version ran"
    assert active.name in out, "setup did not say which install ran"


def test_a_second_installed_copy_is_called_out(tmp_path, monkeypatch):
    """Two site directories can each hold a better_rlm, and sys.path order decides
    which one runs -- not which was installed last. That is exactly how two setup
    runs landed on a stale build, and `pip uninstall` makes it worse by removing the
    newer copy first."""
    import io

    from rich.console import Console

    from better_rlm import onboard, tui

    ghost = tmp_path / "other-site-packages"
    monkeypatch.setattr(mcpreg, "install_identity",
                        lambda: ("9.9.9", tmp_path / "active", [ghost]))
    monkeypatch.setattr(mcpreg, "ensure_registered", lambda **k: ("ok", "already mounted"))
    monkeypatch.setattr(tui, "_save", lambda *a, **k: None)
    monkeypatch.setattr(tui, "render_status", lambda st: "")

    buf = io.StringIO()
    onboard.commit(Console(file=buf, width=200), tmp_path / "config.yaml",
                   onboard.Wizard(mode="api", models=(("root_model", "x"),)))
    out = buf.getvalue()

    assert "other-site-packages" in out, "a shadowing copy was not mentioned"
    assert "uninstall" in out, "the operator was not warned off the command that downgrades them"


def test_install_identity_never_lists_the_active_root_as_a_duplicate():
    """A one-install machine must produce no warning at all, or the warning becomes
    noise everybody learns to skip."""
    _version, active, others = mcpreg.install_identity()
    assert active not in others


def test_a_registration_that_reads_back_wrong_is_reported_as_failed(monkeypatch):
    """`claude mcp add` exiting 0 is not proof the entry landed as asked. A quoting
    slip once registered `C:Python314Scripts...` with a zero exit, and only reading
    it back afterwards showed it."""
    monkeypatch.setattr(mcpreg, "claude_cli", lambda: "claude")
    monkeypatch.setattr(mcpreg, "_run", lambda a: (0, ""))
    # Not registered before; afterwards it reads back as something else entirely.
    reads = iter([None, ["cmd", "/c", "mangled"]])
    monkeypatch.setattr(mcpreg, "registered_command", lambda c: next(reads))

    outcome, msg = mcpreg.ensure_registered()
    assert outcome == "failed", outcome
    assert "reads back as" in msg


def test_the_suite_never_shells_out_to_the_claude_cli(monkeypatch, tmp_path):
    """The guard, asserted rather than assumed.

    Seven tests in test_provider.py drive onboard.run() end to end, and setup
    mounts itself as its last step -- so without a guard every pytest run issued a
    real `claude mcp remove` + `claude mcp add` and repointed the operator's `rlm`
    server at the checkout. Two rounds of "fixed" / "broken again" went by before
    the suite turned out to be the thing undoing the fix.

    Asserts the effect (nothing is spawned), not the mechanism, so a different
    neutralisation still passes.
    """
    import io

    from rich.console import Console

    from better_rlm import onboard, tui

    spawned: list[list[str]] = []
    monkeypatch.setattr(mcpreg.subprocess, "run",
                        lambda *a, **k: spawned.append(list(a[0]) if a else []))
    monkeypatch.setattr(tui, "_save", lambda *a, **k: None)
    monkeypatch.setattr(tui, "render_status", lambda st: "")

    onboard.commit(Console(file=io.StringIO(), quiet=True), tmp_path / "config.yaml",
                   onboard.Wizard(mode="api", models=(("root_model", "x"),)))

    # Only the MUTATING calls matter. load_status separately runs
    # `claude auth status --json`, which is read-only and fine; an assertion of
    # "no subprocess at all" flagged that and would have sent the next reader
    # chasing a non-problem.
    mutating = [c for c in spawned if "mcp" in c and ("add" in c or "remove" in c)]
    assert mutating == [], f"the suite rewrote a real registration: {mutating}"

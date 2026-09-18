"""Tests for ``better_rlm.cli`` -- the ``better-rlm`` entry point.

The dispatcher decides whether an operator's words reach a shell script, the
in-process server, or the TUI, and it behaves differently depending on whether
it is running from a checkout or from a wheel. Both halves are asserted here; no
subprocess is spawned (that would run the real installer).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from better_rlm import cli

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def checkout(monkeypatch):
    monkeypatch.setattr(cli, "IS_CHECKOUT", True)
    monkeypatch.setattr(cli, "PKG_ROOT", ROOT)


@pytest.fixture
def installed(monkeypatch):
    """Simulate site-packages: no launchers, no repo-root config."""
    monkeypatch.setattr(cli, "IS_CHECKOUT", False)


# --- checkout behaviour -------------------------------------------------------


@pytest.mark.parametrize(
    "cmd,expect",
    [
        ("auth", ["bash", str(ROOT / "install.sh"), "--auth"]),
        ("install", ["bash", str(ROOT / "install.sh")]),
        ("server", ["bash", str(ROOT / "run_server.sh")]),
    ],
)
def test_checkout_subcommand_runs_the_posix_launcher(checkout, cmd, expect):
    with patch.object(cli.subprocess, "call", return_value=0) as call, \
            patch.object(cli.sys, "platform", "darwin"):
        assert cli.main([cmd]) == 0
    assert call.call_args.args[0] == expect
    assert call.call_args.kwargs["cwd"] == str(ROOT)


def test_checkout_subcommand_forwards_extra_args(checkout):
    """`better-rlm install --register` must reach install.sh, or the command
    silently drops the flag that does the work."""
    with patch.object(cli.subprocess, "call", return_value=0) as call, \
            patch.object(cli.sys, "platform", "darwin"):
        cli.main(["install", "--register", "--hook"])
    assert call.call_args.args[0][-2:] == ["--register", "--hook"]


def test_checkout_subcommand_propagates_exit_status(checkout):
    with patch.object(cli.subprocess, "call", return_value=3), \
            patch.object(cli.sys, "platform", "darwin"):
        assert cli.main(["auth"]) == 3


def test_missing_launcher_fails_loudly(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "IS_CHECKOUT", True)
    monkeypatch.setattr(cli, "PKG_ROOT", tmp_path)
    with patch.object(cli.sys, "platform", "darwin"), \
            patch.object(cli.subprocess, "call") as call:
        assert cli.main(["auth"]) == 1
    call.assert_not_called()
    assert "missing launcher" in capsys.readouterr().err


def test_windows_auth_uses_powershell_and_the_ps1_switch(checkout):
    """install.ps1 takes -Auth, not --auth, and needs an ExecutionPolicy bypass."""
    with patch.object(cli.subprocess, "call", return_value=0) as call, \
            patch.object(cli.sys, "platform", "win32"), \
            patch.object(Path, "is_file", lambda self: True):
        cli.main(["auth"])
    argv = call.call_args.args[0]
    assert argv[0] == "powershell" and argv[-1] == "-Auth"


# --- installed (pip) behaviour ------------------------------------------------


def test_installed_server_runs_in_process_not_through_a_script(installed):
    """`claude mcp add rlm -- better-rlm server` is the whole point of publishing:
    it must not depend on run_server.sh, which a wheel does not ship."""
    with patch("better_rlm.server.main", return_value=None) as server_main, \
            patch.object(cli.subprocess, "call") as call:
        assert cli.main(["server"]) == 0
    server_main.assert_called_once()
    call.assert_not_called()


def test_installed_auth_prints_the_flow_instead_of_failing(installed, capsys):
    with patch.object(cli.subprocess, "call") as call:
        assert cli.main(["auth"]) == 0
    call.assert_not_called()
    out = capsys.readouterr().out
    assert "claude setup-token" in out
    assert "CLAUDE_CODE_OAUTH_TOKEN" in out


def test_installed_install_refuses_and_says_why(installed, capsys):
    """Failing on a missing install.sh would read as a bug. It is a real
    limitation and the message has to name the fix."""
    assert cli.main(["install"]) == 1
    err = capsys.readouterr().err
    assert "needs a checkout" in err
    assert "git clone" in err


def test_where_reports_the_checkout_shape(checkout, capsys):
    assert cli.main(["where"]) == 0
    out = capsys.readouterr().out
    assert "checkout" in out
    assert "config:" in out and "env:" in out


def test_where_reports_the_installed_shape(installed, capsys):
    """Pinned explicitly rather than left to ambient state: run the suite against a
    non-editable install and an unfixtured test would flip to the other branch and
    fail for environmental reasons."""
    assert cli.main(["where"]) == 0
    out = capsys.readouterr().out
    assert "installed (pip)" in out


def test_installed_server_rejects_arguments_instead_of_dropping_them(installed, capsys):
    """A checkout forwards these to run_server.sh. Silently voiding them here would
    make one command mean two different things."""
    with patch("better_rlm.server.main") as server_main:
        assert cli.main(["server", "--port", "9"]) == 2
    server_main.assert_not_called()
    assert "takes no arguments" in capsys.readouterr().err


def test_no_checkout_message_names_the_subcommand_that_was_typed(installed, capsys, monkeypatch):
    """Hardcoding 'install' meant a future _SCRIPTS entry would report a command
    nobody ran."""
    monkeypatch.setitem(cli._SCRIPTS, "hook", ("install.sh", ["--hook"], "install.ps1", ["-Hook"]))
    assert cli.main(["hook"]) == 1
    assert "better-rlm hook needs a checkout" in capsys.readouterr().err


# --- shared -------------------------------------------------------------------


@pytest.mark.parametrize("argv,forwarded", [
    ([], []),
    (["config"], ["--menu"]),
    (["--one-shot", "/status"], ["--one-shot", "/status"]),
])
def test_non_subcommand_argv_falls_through_to_the_tui(argv, forwarded):
    """Bare, `config`, and raw TUI flags all land in the TUI -- the last one keeps
    `python -m better_rlm.cli --one-shot /status` working for CI.

    `config` becomes --menu rather than being dropped. Bare `better-rlm` is the setup
    surface: it gates on needs_onboarding and, once setup finishes or is cancelled,
    exits instead of falling through to the maintenance menu. Asking for `config` by
    name asks for that menu, configured or not, so the two cannot be the same argv."""
    with patch("better_rlm.tui.main", return_value=0) as tui_main:
        assert cli.main(argv) == 0
    assert tui_main.call_args.args[0] == forwarded


@pytest.mark.parametrize("flag", ["-h", "--help", "help"])
def test_help_prints_usage_without_importing_the_tui(flag, capsys):
    with patch("better_rlm.tui.main") as tui_main:
        assert cli.main([flag]) == 0
    tui_main.assert_not_called()
    out = capsys.readouterr().out
    for cmd in ("auth", "config", "install", "server", "where"):
        assert cmd in out

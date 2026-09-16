"""Tests for ``src.cli`` -- the ``better-rlm`` entry point.

The dispatcher is the one place that decides whether an operator's words reach
a shell script or the TUI, and it is reachable from a global PATH name, so a
wrong branch here runs the wrong thing from an unexpected directory. Every
subcommand is asserted against the argv it builds -- no subprocess is actually
spawned (that would run the real installer).
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from unittest.mock import patch

import pytest

from src import cli

ROOT = Path(__file__).resolve().parent.parent


def test_console_script_is_declared_and_points_here():
    """The PATH name is only real if pyproject declares it. install.sh links
    .venv_sh/bin/better-rlm, which setuptools creates from this entry alone."""
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert data["project"]["scripts"]["better-rlm"] == "src.cli:main"


def test_pkg_root_is_the_checkout_root():
    # Every subcommand resolves its launcher relative to this.
    assert cli.PKG_ROOT == ROOT
    assert (cli.PKG_ROOT / "pyproject.toml").is_file()


@pytest.mark.parametrize(
    "cmd,expect",
    [
        ("auth", ["bash", str(ROOT / "install.sh"), "--auth"]),
        ("install", ["bash", str(ROOT / "install.sh")]),
        ("server", ["bash", str(ROOT / "run_server.sh")]),
    ],
)
def test_subcommand_runs_the_posix_launcher(cmd, expect):
    with patch.object(cli.subprocess, "call", return_value=0) as call, \
            patch.object(cli.sys, "platform", "darwin"):
        assert cli.main([cmd]) == 0
    assert call.call_args.args[0] == expect
    assert call.call_args.kwargs["cwd"] == str(ROOT)


def test_subcommand_forwards_extra_args():
    """`better-rlm install --register` must reach install.sh, or the global
    command silently drops the flag that does the work."""
    with patch.object(cli.subprocess, "call", return_value=0) as call, \
            patch.object(cli.sys, "platform", "darwin"):
        cli.main(["install", "--register", "--hook"])
    assert call.call_args.args[0][-2:] == ["--register", "--hook"]


def test_subcommand_propagates_exit_status():
    # `better-rlm auth` failing must not look like success to a calling script.
    with patch.object(cli.subprocess, "call", return_value=3), \
            patch.object(cli.sys, "platform", "darwin"):
        assert cli.main(["auth"]) == 3


def test_missing_launcher_fails_loudly(tmp_path, capsys):
    with patch.object(cli, "PKG_ROOT", tmp_path), \
            patch.object(cli.sys, "platform", "darwin"), \
            patch.object(cli.subprocess, "call") as call:
        assert cli.main(["auth"]) == 1
    call.assert_not_called()
    assert "missing launcher" in capsys.readouterr().err


@pytest.mark.parametrize("argv", [[], ["config"], ["--one-shot", "/status"]])
def test_non_subcommand_argv_falls_through_to_the_tui(argv):
    """Bare, `config`, and raw TUI flags all land in the TUI -- the last one
    keeps `python -m src.cli --one-shot /status` working for CI."""
    with patch("src.tui.main", return_value=0) as tui_main:
        assert cli.main(argv) == 0
    forwarded = tui_main.call_args.args[0]
    assert "config" not in forwarded
    assert forwarded == [a for a in argv if a != "config"]


@pytest.mark.parametrize("flag", ["-h", "--help", "help"])
def test_help_prints_usage_without_importing_the_tui(flag, capsys):
    with patch("src.tui.main") as tui_main:
        assert cli.main([flag]) == 0
    tui_main.assert_not_called()
    out = capsys.readouterr().out
    for cmd in ("auth", "config", "install", "server"):
        assert cmd in out


def test_windows_auth_uses_powershell_and_the_ps1_switch():
    """install.ps1 takes -Auth, not --auth, and needs an ExecutionPolicy bypass.
    Asserted here because CI's Windows leg never runs the installer itself."""
    with patch.object(cli.subprocess, "call", return_value=0) as call, \
            patch.object(cli.sys, "platform", "win32"), \
            patch.object(Path, "is_file", lambda self: True):
        cli.main(["auth"])
    argv = call.call_args.args[0]
    assert argv[0] == "powershell" and "-Auth" in argv
    assert argv[-1].endswith("-Auth")

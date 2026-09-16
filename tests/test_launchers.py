"""The TUI launchers have to work in both shapes that ship them.

`install.sh` builds `.venv_sh`; `/plugin` never runs the installer and launches the
server with `uv run --directory ${CLAUDE_PLUGIN_ROOT}`, which resolves into `.venv`.
run_tui.sh pointed unconditionally at `.venv_sh`, so under a plugin install it died
with a bare "no such file or directory" -- making the TUI look unavailable when it
was shipped and working. Nothing executes these scripts in CI, so the drift is
invisible until someone tries it.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PLUGIN = json.loads((ROOT / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8"))


def _plugin_server_args() -> list[str]:
    return PLUGIN["mcpServers"]["rlm"]["args"]


@pytest.mark.parametrize(
    "script,installer_venv",
    [("run_tui.sh", ".venv_sh"), ("run_tui.cmd", ".venv_windows")],
)
def test_launcher_prefers_the_installer_venv_then_falls_back_to_uv(script, installer_venv):
    body = (ROOT / script).read_text(encoding="utf-8")
    assert installer_venv in body, f"{script} no longer uses the installer venv"
    assert "uv run" in body, f"{script} has no uv fallback — it breaks under /plugin"
    # Order matters: a checkout must keep using the interpreter run_server.sh uses,
    # so the TUI and the server never resolve two different dependency graphs.
    assert body.index(installer_venv) < body.index("uv run"), (
        f"{script} reaches for uv before the installer venv"
    )


@pytest.mark.parametrize("script", ["run_tui.sh", "run_tui.cmd"])
def test_uv_fallback_matches_the_plugin_manifests_own_invocation(script):
    """If the manifest gains an extra or changes how it resolves the project, the
    fallback must follow — otherwise uv builds a SECOND environment beside the one
    the server already paid to resolve, and the TUI reports a different dependency
    set than the server is running."""
    body = (ROOT / script).read_text(encoding="utf-8")
    args = _plugin_server_args()
    assert "--directory" in args, "plugin manifest stopped resolving by directory"
    for i, a in enumerate(args):
        if a == "--extra":
            assert f"--extra {args[i + 1]}" in body, (
                f"{script} omits `--extra {args[i + 1]}` that plugin.json passes"
            )
    assert "--directory" in body


@pytest.mark.parametrize("script", ["run_tui.sh", "run_tui.cmd"])
def test_launcher_fails_with_a_fix_not_a_stack_trace(script):
    # The original failure was `.venv_sh/bin/python: no such file or directory`,
    # which names neither cause nor cure.
    body = (ROOT / script).read_text(encoding="utf-8")
    assert "no interpreter found" in body
    assert "plugin" in body.lower() and "checkout" in body.lower()


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv not on PATH")
def test_the_uv_fallback_command_actually_runs():
    """Assert the invocation works, not just that the string is present."""
    args = _plugin_server_args()
    extras = [args[i : i + 2] for i, a in enumerate(args) if a == "--extra"]
    cmd = ["uv", "run", "--directory", str(ROOT)]
    for pair in extras:
        cmd += pair
    cmd += ["python", "-m", "better_rlm.cli", "--one-shot", "/status"]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    assert res.returncode == 0, res.stderr[-800:]
    assert "mode:" in res.stdout, res.stdout[:400]

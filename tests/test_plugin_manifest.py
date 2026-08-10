"""The plugin manifest is the launch command for anyone who installs via /plugin.

Nothing else in the repo reads it, so a stale module path or a dropped extra shows up
only as an MCP server that silently fails to start on someone else's machine. These
assertions tie the manifest back to the things it names.
"""
import json
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PLUGIN = ROOT / ".claude-plugin" / "plugin.json"
MARKETPLACE = ROOT / ".claude-plugin" / "marketplace.json"


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads(PLUGIN.read_text(encoding="utf-8"))


def test_launch_args_name_a_real_module_and_extra(manifest):
    server = manifest["mcpServers"]["rlm"]
    args = server["args"]
    assert server["command"] == "uv"
    # -m <module> must be importable, and --directory must be what makes that true:
    # config.py resolves config.yaml/.env from the package root, not the caller's cwd.
    assert args[args.index("-m") + 1] == "src.server"
    assert (ROOT / "src" / "server.py").is_file()
    assert args[args.index("--directory") + 1] == "${CLAUDE_PLUGIN_ROOT}"

    extra = args[args.index("--extra") + 1]
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert extra in pyproject["project"]["optional-dependencies"]


def test_utf8_mode_is_set(manifest):
    # Same reason run_server.cmd sets it: the Linux sandbox guest reads every
    # host-written file as UTF-8, and Windows' default locale encoding is cp1252.
    assert manifest["mcpServers"]["rlm"]["env"]["PYTHONUTF8"] == "1"


def test_marketplace_points_at_this_plugin(manifest):
    entries = json.loads(MARKETPLACE.read_text(encoding="utf-8"))["plugins"]
    # `/plugin install <name>@<marketplace>` resolves through this pairing; a mismatch
    # makes the documented install command fail.
    assert [e["name"] for e in entries] == [manifest["name"]]
    assert entries[0]["source"] == "./"


def test_bundled_skill_is_present():
    # Plugins auto-discover skills/; this is what a plugin user gets instead of the
    # symlink/junction the installers create.
    assert (ROOT / "skills" / "rlm-large-context" / "SKILL.md").is_file()

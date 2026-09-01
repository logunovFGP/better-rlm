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


def test_manifest_version_matches_the_version_file(manifest):
    # Three copies used to disagree (pyproject 0.2.0, manifest 0.1.0, handshake 1.28.1).
    # VERSION is now the only place a version is typed; the manifest is static JSON that
    # cannot read it, so scripts/sync_version.py writes it and this holds the two together.
    assert manifest["version"] == (ROOT / "VERSION").read_text(encoding="utf-8").strip()


def test_pyproject_derives_its_version_from_the_file():
    # Not a second literal: setuptools reads VERSION, so a built wheel and the running
    # server cannot report different numbers.
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert "version" in pyproject["project"]["dynamic"]
    assert "version" not in pyproject["project"], "version is hardcoded again"
    assert pyproject["tool"]["setuptools"]["dynamic"]["version"]["file"] == "VERSION"


def test_runtime_version_tracks_the_file_without_a_reinstall(tmp_path, monkeypatch):
    """A bump must take effect immediately, not at the next pip install.

    Regression: reading importlib.metadata instead looked right and was silently wrong.
    This repo installs editable, so the version is frozen into rlm_mcp-<v>.dist-info at
    install time -- pyproject could say 0.9.9 while the handshake still said 0.2.0.
    """
    import src.version as v

    bumped = tmp_path / "VERSION"
    bumped.write_text("9.9.9\n", encoding="utf-8")
    monkeypatch.setattr(v, "VERSION_FILE", bumped)
    assert v._read() == "9.9.9"


def test_sync_version_detects_drift(tmp_path, monkeypatch):
    import scripts.sync_version as sv

    plugin = tmp_path / "plugin.json"
    plugin.write_text('{\n  "name": "x",\n  "version": "0.0.1"\n}\n', encoding="utf-8")
    ver = tmp_path / "VERSION"
    ver.write_text("1.2.3\n", encoding="utf-8")
    monkeypatch.setattr(sv, "PLUGIN", plugin)
    monkeypatch.setattr(sv, "VERSION", ver)

    assert sv.main(["--check"]) == 1                       # drift reported
    assert json.loads(plugin.read_text())["version"] == "0.0.1"   # --check changed nothing
    assert sv.main([]) == 0                                # and the fix applies it
    assert json.loads(plugin.read_text())["version"] == "1.2.3"
    assert sv.main(["--check"]) == 0


def test_server_advertises_its_own_version_not_the_sdks():
    """The MCP handshake must report rlm's version, not the mcp SDK's.

    The lowlevel server falls back to pkg_version("mcp") when its version is None, so
    forgetting to set it makes every client see the transport library's number.
    """
    from importlib.metadata import version as pkg_version

    from src import server
    from src.version import __version__

    advertised = server.mcp._mcp_server.create_initialization_options().server_version
    assert advertised == __version__
    assert advertised != pkg_version("mcp")

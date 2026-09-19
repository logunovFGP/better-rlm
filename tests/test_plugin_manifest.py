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


@pytest.fixture
def sv(tmp_path, monkeypatch):
    """scripts.sync_version with ALL THREE of its files redirected into tmp_path.

    Patching them one test at a time is how the checkout's own better_rlm/version.py
    got rewritten to 1.2.3 mid-suite: ``main`` writes two files, so a test that
    redirected only plugin.json edited the real one and still passed, because an
    imported module is not read from disk again. Taking them as a set is the only
    form a future test cannot half-apply.

    Both are pre-seeded with a version that differs from VERSION, so a test that
    cares about drift has drift and one that does not still gets a working run.
    """
    import scripts.sync_version as module

    monkeypatch.setattr(module, "VERSION", tmp_path / "VERSION")
    monkeypatch.setattr(module, "PLUGIN", tmp_path / "plugin.json")
    monkeypatch.setattr(module, "VERSION_PY", tmp_path / "version.py")
    module.VERSION.write_bytes(b"1.2.3\n")
    module.VERSION_PY.write_bytes(b'_BAKED = "0.0.1"\n')
    module.PLUGIN.write_bytes(b'{\n  "name": "x",\n  "version": "0.0.1"\n}\n')
    return module


def test_launch_args_name_a_real_module_and_extra(manifest):
    server = manifest["mcpServers"]["rlm"]
    args = server["args"]
    assert server["command"] == "uv"
    # -m <module> must be importable, and --directory must be what makes that true:
    # config.py resolves config.yaml/.env from the package root, not the caller's cwd.
    assert args[args.index("-m") + 1] == "better_rlm.server"
    assert (ROOT / "better_rlm" / "server.py").is_file()
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
    import better_rlm.version as v

    bumped = tmp_path / "VERSION"
    bumped.write_text("9.9.9\n", encoding="utf-8")
    monkeypatch.setattr(v, "VERSION_FILE", bumped)
    assert v._read() == "9.9.9"


def test_the_baked_version_matches_the_version_file():
    """The literal a wheel reports, held to VERSION.

    It is the ONLY answer outside a checkout now: metadata described a .dist-info
    directory rather than the running code, and an operator's site-packages held two
    of them, so `better-rlm --version` reported 0.7.0 while executing 0.9.2's code.
    A build whose literal has drifted fails here, which is before it can publish.
    """
    import better_rlm.version as v

    assert v._BAKED == (ROOT / "VERSION").read_text(encoding="utf-8").strip()


def test_a_wheel_reports_the_baked_literal_and_asks_no_metadata(tmp_path, monkeypatch):
    """site-packages has no pyproject.toml beside the package.

    Also pins that nothing consults importlib.metadata: that lookup returns whichever
    .dist-info sorts first, which is how a stale 0.7.0 directory outvoted the 0.9.2
    one beside it. And a VERSION file in site-packages is somebody else's, so the
    checkout probe -- not the file's existence -- decides whether to read it.
    """
    import importlib.metadata as md

    import better_rlm.version as v

    def explode(_name):                       # pragma: no cover - must never run
        raise AssertionError("_read consulted importlib.metadata")

    monkeypatch.setattr(md, "version", explode)
    monkeypatch.setattr(v, "_ROOT", tmp_path)          # no pyproject.toml here
    monkeypatch.setattr(v, "VERSION_FILE", tmp_path / "VERSION")
    (tmp_path / "VERSION").write_text("6.6.6\n", encoding="utf-8")   # not ours: ignored

    assert v._read() == v._BAKED


def test_sync_version_detects_drift(sv):
    assert sv.main(["--check"]) == 1                       # drift reported
    assert json.loads(sv.PLUGIN.read_text())["version"] == "0.0.1"  # --check wrote nothing
    assert '_BAKED = "0.0.1"' in sv.VERSION_PY.read_text()
    assert sv.main([]) == 0                                # and the fix applies it
    assert json.loads(sv.PLUGIN.read_text())["version"] == "1.2.3"
    assert '_BAKED = "1.2.3"' in sv.VERSION_PY.read_text()
    assert sv.main(["--check"]) == 0


def test_sync_version_reports_drift_in_either_file_alone(sv):
    """plugin.json already in sync and version.py not must still fail --check.

    The two syncs share one exit status, so an early `return 0` taken because the
    first file agreed would hide drift in the second -- which is a wrong version
    number shipping while the check that exists to prevent it reports success.
    """
    sv.PLUGIN.write_bytes(b'{\n  "version": "1.2.3"\n}\n')   # this one is fine

    assert sv.main(["--check"]) == 1
    assert sv.main([]) == 0
    assert '_BAKED = "1.2.3"' in sv.VERSION_PY.read_text()


def test_sync_version_keeps_the_files_line_endings(sv):
    """A bump changes one token, not every line ending.

    plugin.json is tracked and .gitattributes pins this repo to ``eol=lf``. Path
    .write_text in text mode translates every LF to CRLF on Windows: a real run here
    measured 0 CR bytes before and 30 after, so a Windows maintainer bumping VERSION
    turned a one-line diff into a whole-file rewrite -- against sync_version's own
    "byte-for-byte otherwise" promise. Asserted on raw bytes, because read_text's
    universal-newline translation would hide the regression.
    """
    assert sv.main([]) == 0
    raw = sv.PLUGIN.read_bytes()
    assert bytes([13]) not in raw, "line endings were rewritten as CRLF"
    assert json.loads(raw.decode("utf-8"))["version"] == "1.2.3"
    # version.py is tracked and eol=lf too, and it is the file a release ships.
    assert bytes([13]) not in sv.VERSION_PY.read_bytes()


def test_sync_version_refuses_to_rewrite_a_nested_version_key(sv):
    """count=1 hits the first match in the file, top-level or not.

    The old comment claimed the pattern was anchored to the top-level key; it was not.
    A nested "version" declared above it would be rewritten while json.loads kept
    reading the real one, so --check would report drift forever. Now the result is
    re-parsed and the mismatch is loud.
    """
    sv.PLUGIN.write_bytes(
        b'{\n  "mcpServers": {"rlm": {"version": "nested"}},\n'
        b'  "version": "0.0.1"\n}\n'
    )

    assert sv.main([]) == 1, "rewriting the nested key must fail, not pass silently"
    assert json.loads(sv.PLUGIN.read_text(encoding="utf-8"))["version"] == "0.0.1"


def test_server_advertises_its_own_version_not_the_sdks():
    """The MCP handshake must report rlm's version, not the mcp SDK's.

    The lowlevel server falls back to pkg_version("mcp") when its version is None, so
    forgetting to set it makes every client see the transport library's number.
    """
    from importlib.metadata import version as pkg_version

    from better_rlm import server
    from better_rlm.version import __version__

    advertised = server.mcp._mcp_server.create_initialization_options().server_version
    assert advertised == __version__
    assert advertised != pkg_version("mcp")

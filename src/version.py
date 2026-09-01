"""The version this server advertises. Single source: the VERSION file at the repo root.

Three places used to disagree: pyproject said 0.2.0, .claude-plugin/plugin.json said
0.1.0, and the MCP handshake advertised 1.28.1 -- the *mcp SDK's* version, because the
lowlevel server falls back to ``pkg_version("mcp")`` when nobody sets its ``version``
(Server.create_initialization_options). A client asking "what rlm is this?" got the
transport library's version number.

Why the file and not importlib.metadata: metadata is written at INSTALL time. This repo
is installed editable, so the version is frozen into ``rlm_mcp-<v>.dist-info`` and
bumping VERSION leaves the runtime reporting the old number until someone reinstalls --
exactly the silent drift this module exists to prevent. The file is read first for that
reason; metadata is the fallback for a wheel install, where VERSION is not on disk
beside the package.

pyproject reads the same file via ``[tool.setuptools.dynamic] version = {file =
"VERSION"}``, so the built distribution and the running server cannot disagree.
.claude-plugin/plugin.json is static JSON that cannot read anything; it is written from
VERSION by scripts/sync_version.py and held to it by tests/test_plugin_manifest.py.
"""
from pathlib import Path

VERSION_FILE = Path(__file__).resolve().parent.parent / "VERSION"


def _read() -> str:
    try:
        text = VERSION_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        text = ""
    if text:
        return text
    # Installed as a wheel: no VERSION beside the package, so fall back to the metadata
    # setuptools generated FROM that same file at build time.
    from importlib.metadata import PackageNotFoundError, version as pkg_version
    try:
        return pkg_version("rlm-mcp")
    except PackageNotFoundError:
        return "0+unknown"


__version__ = _read()

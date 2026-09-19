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
reason. A wheel has no VERSION beside the package and uses ``_BAKED`` -- NOT metadata,
which describes a .dist-info directory rather than the code that is running, and reads
a stale one when two of them sit side by side. See ``_BAKED``.

pyproject reads the same file via ``[tool.setuptools.dynamic] version = {file =
"VERSION"}``, so the built distribution and the running server cannot disagree.
.claude-plugin/plugin.json is static JSON that cannot read anything; it is written from
VERSION by scripts/sync_version.py and held to it by tests/test_plugin_manifest.py.
"""
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
VERSION_FILE = _ROOT / "VERSION"

#: The version, baked in as a literal at release time. Written by
#: scripts/sync_version.py and held to VERSION by
#: tests/test_plugin_manifest.py::test_the_baked_version_matches_the_version_file,
#: so a build whose literal has drifted cannot pass verify, let alone publish.
#:
#: This replaced an ``importlib.metadata`` lookup, which is not a fact about the code
#: that is running -- it is a fact about a .dist-info directory sitting nearby, and
#: those outlive the code they describe. Measured on an operator's machine:
#: site-packages held 0.9.2's code beside BOTH `better_rlm-0.9.2.dist-info` and a
#: `better_rlm-0.7.0.dist-info` left by an install pip never cleaned up.
#: ``importlib.metadata`` returns the first match in directory order, so 0.7.0 won and
#: `better-rlm --version` reported a version whose code was long gone. pip's own
#: "Successfully installed better-rlm-0.7.0" came from the same lookup, on the same run
#: that had just unpacked the 0.9.2 wheel.
_BAKED = "0.9.3"


def _read() -> str:
    """The running code's version.

    A checkout reads VERSION, so a bump is live immediately rather than at the next
    reinstall -- this repo installs editable, and metadata freezes at install time.
    Anywhere else the literal is the answer: nothing outside the package can make it
    wrong, which is the entire point of it.
    """
    # _ROOT is site-packages in a wheel, and a file named VERSION there belongs to
    # whoever dropped it, not to us. pyproject.toml beside it is the checkout probe
    # (config.IS_CHECKOUT asks the same question; this module stays import-free).
    if (_ROOT / "pyproject.toml").is_file():
        try:
            text = VERSION_FILE.read_text(encoding="utf-8").strip()
        except OSError:
            text = ""
        if text:
            return text
    return _BAKED


__version__ = _read()

"""The uninstallers are the only thing that reverses install.sh / install.ps1.

Nothing in the repo executes them, so drift is invisible: add an artefact to an installer
and the matching uninstaller silently stops being complete, leaving an orphaned MCP
registration or a dangling skill link on someone else's machine long after they think they
removed this checkout. These assertions tie each installer's artefacts back to its
uninstaller, and pin the two removals that are actively dangerous to get wrong.
"""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# Every artefact an installer creates, and the token that proves its uninstaller handles it.
# Keyed by artefact so a failure names the thing that would be left behind.
POSIX_ARTEFACTS = {
    "venv": ".venv_sh",
    "editable install": "rlm_mcp.egg-info",
    "env file": ".env.example",
    "skill link": ".claude/skills",
    "mcp registration": "mcp remove",
    "checkout ownership probe": "run_server.sh",
    "verify gate": "core.hooksPath",
    "sandbox image": "rlm-sandbox",
    "context store": ".rlm",
}
WINDOWS_ARTEFACTS = {
    "venv": ".venv_windows",
    "editable install": "rlm_mcp.egg-info",
    "env file": ".env.example",
    "skill junction": "rlm-large-context",
    "mcp registration": "mcp' 'remove",
    "checkout ownership probe": "run_server.cmd",
    "verify gate": "core.hooksPath",
    "sandbox image": "rlm-sandbox",
    "context store": ".rlm",
}


@pytest.fixture(scope="module")
def sh() -> str:
    return (ROOT / "uninstall.sh").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def ps1() -> str:
    return (ROOT / "uninstall.ps1").read_text(encoding="utf-8")


def test_both_uninstallers_exist():
    # One platform silently missing an uninstaller is the whole failure mode this file guards.
    assert (ROOT / "uninstall.sh").is_file()
    assert (ROOT / "uninstall.ps1").is_file()


def test_posix_uninstaller_is_executable():
    # install.sh ships 0755 and is invoked as ./install.sh; a 0644 uninstall.sh would fail
    # the same way for anyone following the README.
    assert (ROOT / "uninstall.sh").stat().st_mode & 0o111, "uninstall.sh is not executable"


@pytest.mark.parametrize("artefact,token", sorted(POSIX_ARTEFACTS.items()))
def test_posix_uninstaller_covers_every_installed_artefact(sh, artefact, token):
    assert token in sh, f"uninstall.sh never mentions {token!r} — {artefact} would be orphaned"


@pytest.mark.parametrize("artefact,token", sorted(WINDOWS_ARTEFACTS.items()))
def test_windows_uninstaller_covers_every_installed_artefact(ps1, artefact, token):
    assert token in ps1, f"uninstall.ps1 never mentions {token!r} — {artefact} would be orphaned"


def test_junction_is_deleted_as_a_reparse_point_only(ps1):
    # THE Windows footgun. Remove-Item -Recurse can follow a junction and delete the files it
    # points at, which here is the repo's own skills\rlm-large-context (or another
    # checkout's). install.ps1 carries the same warning where it re-points the link.
    assert "[System.IO.Directory]::Delete($link)" in ps1
    assert "Remove-Item -Recurse -Force $link" not in ps1


def test_shared_state_is_opt_in_on_both_platforms(sh, ps1):
    # ~/.rlm holds the only copy of every loaded context and is shared by every checkout;
    # rlm-sandbox is shared too. Removing either by default turns "uninstall" into data loss.
    assert "--purge-data" in sh and "--image" in sh
    assert "-PurgeData" in ps1 and "-Image" in ps1


def test_registration_removal_is_guarded_by_the_checkout_path(sh, ps1):
    # Both installers refuse to hijack a registration owned by another checkout, which is what
    # lets checkouts coexist. An unguarded `claude mcp remove` here would uninstall whichever
    # checkout currently owns the name, so the launcher path must be tested first.
    # Anchor on the mutation itself ("run" is the dry-run choke point), not on the first
    # mention of the command — the header comment names it too.
    before_remove = sh.split("run claude mcp remove", 1)[0]
    assert "run_server.sh" in before_remove, "uninstall.sh removes 'rlm' without proving it owns it"
    before_remove_ps1 = ps1.split("'mcp' 'remove'", 1)[0]
    assert "run_server.cmd" in before_remove_ps1, "uninstall.ps1 removes 'rlm' without proving it owns it"


# --- line endings: Windows-only scripts must stay CRLF ------------------------
@pytest.mark.parametrize("name", ["run_server.cmd", "install.ps1", "uninstall.ps1"])
def test_windows_only_scripts_keep_crlf(name):
    """`.gitattributes` pins these to eol=crlf, so a checkout produces CRLF on every
    platform. An editor (or a script) that rewrites one with LF silently undoes that —
    it has happened twice — and cmd.exe/PowerShell are the ones that pay.
    """
    data = (ROOT / name).read_bytes()
    lf_total = data.count(b"\n")
    crlf = data.count(b"\r\n")
    assert lf_total > 0, f"{name} is empty?"
    assert crlf == lf_total, (
        f"{name} has {lf_total - crlf} bare LF line ending(s) — must be CRLF "
        "(see .gitattributes)"
    )

"""The uninstallers are the only thing that reverses install.sh / install.ps1.

Nothing in the repo executes them, so drift is invisible: add an artefact to an installer
and the matching uninstaller silently stops being complete, leaving an orphaned MCP
registration or a dangling skill link on someone else's machine long after they think they
removed this checkout. These assertions tie each installer's artefacts back to its
uninstaller, and pin the two removals that are actively dangerous to get wrong.
"""
import re
import subprocess
import sys
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
    # the same way for anyone following the README. Ask git, not the filesystem: Windows
    # cannot represent the exec bit, so os.stat reports 0o666 even for install.sh, which is
    # committed 100755 — a stat() check is red on every Windows checkout no matter what
    # actually landed in the tree. The committed mode is what a cloner gets, so pin that.
    mode = subprocess.run(
        ["git", "ls-files", "-s", "--", "uninstall.sh"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout.split()[0]
    assert mode == "100755", f"uninstall.sh is committed as {mode}, not 100755"


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


# --- install.sh stores the OAuth token itself, and never leaks it -------------
@pytest.mark.skipif(sys.platform == "win32", reason="install.sh is the POSIX installer")
def test_installer_stores_the_token_without_leaking_it(tmp_path):
    """`--auth` reads the token at a hidden prompt and writes it to .env.

    An `export` in the user's shell never reaches the server, so the installer has to
    persist it. The value must not reach stdout: an installer's output is logged, and a
    setup-token credential is valid for a year.
    """
    m = re.search(r"^rlm_write_token\(\) \{.*?^\}", (ROOT / "install.sh").read_text(),
                  re.S | re.M)
    assert m, "rlm_write_token() is gone from install.sh"

    venv = tmp_path / ".venv_sh" / "bin"
    venv.mkdir(parents=True)
    (venv / "python").symlink_to(sys.executable)

    env_file = tmp_path / ".env"
    env_file.write_text("# keep me\nCLAUDE_CODE_OAUTH_TOKEN=\n")   # the empty-slot shape
    env_file.chmod(0o644)                                            # the loose mode too
    (tmp_path / "fn.sh").write_text(m.group(0) + '\nrlm_write_token "$1"\n')

    secret = "SEKRIT-TOKEN-abcdef0123456789"
    r = subprocess.run(["bash", "fn.sh", secret], cwd=tmp_path,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert secret not in r.stdout + r.stderr, "the installer printed the token"
    assert f"{len(secret)} bytes" in r.stdout, "should report the length instead"

    body = env_file.read_text()
    assert body.count("CLAUDE_CODE_OAUTH_TOKEN=") == 1, "left a duplicate/empty slot"
    assert f"CLAUDE_CODE_OAUTH_TOKEN={secret}" in body
    assert "# keep me" in body, "clobbered the rest of .env"
    assert env_file.stat().st_mode & 0o777 == 0o600, ".env left readable by others"


# --- installer parity: a capability on one platform must exist on both ---------
@pytest.mark.parametrize("capability,sh_token,ps1_token", [
    ("auth opt-in flag",        "--auth",              "$Auth"),
    ("token writer",            "rlm_write_token",     "Write-RlmToken"),
    ("free login check",        "claude auth status",  "claude auth status"),
    ("adopts an exported token", "CLAUDE_CODE_OAUTH_TOKEN:-", "$env:CLAUDE_CODE_OAUTH_TOKEN"),
    ("hidden prompt",           "read -rs",            "-AsSecureString"),
])
def test_installers_stay_in_step(capability, sh_token, ps1_token):
    """install.sh and install.ps1 are maintained as a pair (--register / -Register).
    A credential flow that exists on only one platform is how that pair rots.
    """
    sh = (ROOT / "install.sh").read_text(encoding="utf-8")
    ps1 = (ROOT / "install.ps1").read_text(encoding="utf-8")
    assert sh_token in sh, f"install.sh lost {capability!r} ({sh_token})"
    assert ps1_token in ps1, f"install.ps1 lost {capability!r} ({ps1_token})"


def test_powershell_writer_avoids_the_bom_trap():
    """Set-Content -Encoding utf8 emits a BOM on PowerShell 5.1 (this script's floor).
    A BOM becomes part of the first key python-dotenv parses."""
    ps1 = (ROOT / "install.ps1").read_text(encoding="utf-8")
    assert "UTF8Encoding" in ps1 and "WriteAllText" in ps1
    writer = ps1[ps1.index("function Write-RlmToken"):]
    writer = writer[:writer.index("\n}")]
    # Drop the <# .. #> doc block first: it names Set-Content precisely to say
    # "not this", and a naive negative match reads that as the defect.
    code = re.sub(r"<#.*?#>", "", writer, flags=re.S)
    assert "Set-Content" not in code, "BOM risk: use WriteAllText + UTF8Encoding($false)"
    assert "WriteAllText" in code and "UTF8Encoding" in code
    assert "$tok.Length" in code, "must report the length, not the token"

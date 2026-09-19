"""Register this install as Claude Code's ``rlm`` MCP server.

Finishing setup and then not being mounted is the gap this closes: the operator
configures a provider, sees "Ready", and the agent still talks to whatever was
registered months ago. Worse, a registration that points at a DIFFERENT install
reads a different config.yaml entirely -- a checkout registration kept serving a
stale endpoint and stale model ids long after the wizard had written good ones to
``~/.rlm``, and nothing on screen said so.

Only Claude Code for now. cline is a separate target with a separate config file
and is deliberately not attempted here.

Mirrors what install.sh and install.ps1 already do, because two registration
schemes for one server is how they drift:

  * scope ``-s user`` -- the server is a machine-level tool, not a per-project one;
  * probe with ``claude mcp get`` first, because ``claude mcp add`` exits 1 on a
    name that already exists;
  * a DIFFERENT command under the same name is replaced, not left alone. That is
    the stale-pointer case, and silently leaving it is the bug.
"""

from __future__ import annotations

import shutil
import subprocess
import sys

from .config import IS_CHECKOUT, PKG_ROOT

#: The name the agent addresses. Matches both installers and the docs.
SERVER_NAME = "rlm"

_TIMEOUT_S = 30


def server_command() -> list[str]:
    """The argv that starts THIS install's server.

    A checkout uses its launcher script, which knows about the venv. A wheel is
    started through its own interpreter rather than the ``better-rlm`` console
    script on PATH: install.ps1 writes a ``better-rlm.cmd`` shim that repoints the
    bare word at a checkout, so a name-based registration can silently flip to a
    different install. ``sys.executable -m`` cannot.
    """
    if IS_CHECKOUT:
        if sys.platform == "win32":
            return ["cmd", "/c", str(PKG_ROOT / "run_server.cmd")]
        return ["bash", str(PKG_ROOT / "run_server.sh")]
    return [sys.executable, "-m", "better_rlm.cli", "server"]


def claude_cli() -> str | None:
    """Path to the ``claude`` CLI, or None when it is not installed."""
    return shutil.which("claude")


def _run(args: list[str]) -> tuple[int, str]:
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=_TIMEOUT_S)
    except (OSError, subprocess.SubprocessError) as exc:
        return 1, f"{type(exc).__name__}: {exc}"
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def registered_command(claude: str) -> list[str] | None:
    """The argv currently registered under ``rlm``, or None when nothing is.

    Parsed from ``claude mcp get``, whose output is a small labelled block. A parse
    miss returns [] rather than None -- "registered, shape unknown" and "not
    registered" lead to different actions, so they must not collapse.
    """
    rc, out = _run([claude, "mcp", "get", SERVER_NAME])
    if rc != 0:
        return None
    command, args = "", ""
    for line in out.splitlines():
        s = line.strip()
        if s.startswith("Command:"):
            command = s.split(":", 1)[1].strip()
        elif s.startswith("Args:"):
            args = s.split(":", 1)[1].strip()
    if not command:
        return []
    return [command, *args.split()] if args else [command]


def manual_command() -> str:
    """The line to paste when we cannot do it ourselves."""
    return " ".join(["claude", "mcp", "add", "-s", "user", SERVER_NAME, "--",
                     *server_command()])


def ensure_registered(*, force: bool = False) -> tuple[str, str]:
    """Point Claude Code's ``rlm`` at this install. Returns (outcome, message).

    Outcomes: ``ok`` (already correct), ``added``, ``repointed``, ``skipped``
    (no claude CLI), ``failed``. Never raises -- a setup run must not die because
    an unrelated CLI is missing or slow.
    """
    claude = claude_cli()
    if not claude:
        return "skipped", ("the `claude` CLI is not on PATH, so nothing was mounted. "
                           "Install Claude Code and re-run setup, or register by hand:\n  "
                           + manual_command())

    want = server_command()
    have = registered_command(claude)

    if have == want and not force:
        return "ok", f"already mounted: {' '.join(want)}"

    if have is not None:
        # `claude mcp add` exits 1 on an existing name, so replace rather than add.
        rc, out = _run([claude, "mcp", "remove", "-s", "user", SERVER_NAME])
        if rc != 0:
            return "failed", f"could not remove the existing registration: {out.strip()[:200]}"

    rc, out = _run([claude, "mcp", "add", "-s", "user", SERVER_NAME, "--", *want])
    if rc != 0:
        return "failed", f"`claude mcp add` failed: {out.strip()[:200]}"

    if have:
        return "repointed", (f"was pointing at `{' '.join(have)}`, which reads a "
                             f"different config; now `{' '.join(want)}`")
    return "added", f"mounted as `{SERVER_NAME}`: {' '.join(want)}"

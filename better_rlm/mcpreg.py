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
from pathlib import Path

from .config import IS_CHECKOUT, PKG_ROOT

#: The name the agent addresses. Matches both installers and the docs.
SERVER_NAME = "rlm"

_TIMEOUT_S = 30


#: What to tell an operator whose server is already running. `claude mcp restart`
#: DOES NOT EXIST -- the CLI has add / add-json / get / list / login / logout /
#: remove / reset-project-choices / serve, and no restart -- so setup spent several
#: releases closing with an instruction that could not be followed. A running stdio
#: server holds the config it started with; only a new session launches a new one.
RESTART_HINT = ("Start a new Claude Code session to pick this up -- a running "
                "server keeps the config it started with.")


def install_identity() -> tuple[str, Path, list[Path]]:
    """(version, the active install root, any OTHER copies on this machine).

    The duplicate list is the important half. Two site directories can each hold a
    better_rlm, and which one wins is decided by sys.path order, not by which was
    installed last. That is how setup was completed twice against a stale build with
    nothing on screen saying so -- and why `pip uninstall` makes it worse, removing
    the newer copy first and silently downgrading the machine.
    """
    import site

    from .version import __version__

    active = Path(__file__).resolve().parent.parent
    candidates = list(sys.path)
    for name in ("getusersitepackages", "getsitepackages"):
        try:
            got = getattr(site, name, lambda: None)()
        except Exception:                      # noqa: BLE001 - a probe must not raise
            got = None
        if isinstance(got, str):
            candidates.append(got)
        elif isinstance(got, (list, tuple)):
            candidates.extend(got)

    roots: set[Path] = set()
    for d in candidates:
        try:
            root = Path(d).resolve()
        except (OSError, ValueError):
            continue
        if root != active and (root / "better_rlm" / "version.py").is_file():
            roots.add(root)
    return __version__, active, sorted(roots)


def stale_metadata() -> list[Path]:
    """.dist-info directories beside this install that describe a version it is not.

    Two of them for one distribution is always broken, and it is not cosmetic: every
    ``importlib.metadata`` lookup -- pip's own "Successfully installed" line included
    -- takes the FIRST in directory order. `better_rlm-0.7.0.dist-info` sitting next
    to `better_rlm-0.9.2.dist-info` made pip announce it had installed 0.7.0 on the
    run where it unpacked the 0.9.2 wheel. Nothing warned; it simply read as the
    upgrade having failed.

    Skipped for a checkout, where an editable install's metadata legitimately lags
    VERSION -- that drift is this repo's normal state, not a fault.
    """
    if IS_CHECKOUT:
        return []
    from .version import __version__

    active = Path(__file__).resolve().parent.parent
    try:
        found = sorted(active.glob("better_rlm-*.dist-info"))
    except OSError:                            # noqa: BLE001 - a probe must not raise
        return []
    return [d for d in found if d.name != f"better_rlm-{__version__}.dist-info"]


def server_command() -> list[str]:
    """The argv that starts the served install. Always the pip one.

    There is exactly one servable shape now. A checkout is for development and for
    the test suite; it is never registered, because two installs of the same server
    on one machine is the whole family of failures this module exists to prevent --
    each reads a different config.yaml, and which one an agent is talking to is
    invisible from inside the session. ``ensure_registered`` refuses from a checkout
    rather than guessing at another interpreter's path.

    Started through its own interpreter rather than the ``better-rlm`` console script
    on PATH: install.ps1 writes a ``better-rlm.cmd`` shim that repoints the bare word
    at a checkout, so a name-based registration can silently flip install.
    ``sys.executable -m`` cannot.

    **-P is load-bearing, not hygiene.** ``-m`` puts the CURRENT DIRECTORY first on
    sys.path, and Claude Code launches an MCP server with cwd set to the project
    directory. In a checkout of this repo that directory contains ``better_rlm/``,
    so the wheel's own interpreter imported the CHECKOUT and read the checkout's
    config.yaml -- Claude model ids at a MiniMax endpoint, exactly the stale-pointer
    failure this module exists to prevent, reached through a correct command.
    Measured from the repo root: without -P ``better-rlm where`` says
    ``mode: checkout``, with it ``installed (pip)``. -P and not -I, because -I also
    drops user site-packages, which is where a ``pip install --user`` lives.
    """
    return [sys.executable, "-P", "-m", "better_rlm.cli", "server"]


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
    if IS_CHECKOUT:
        # Registering a checkout is how a machine ends up serving two rlms that read
        # two config.yamls, with nothing on screen saying which one answered. The pip
        # install is the only servable one; this one is for development and tests.
        return "skipped", ("this is a checkout, and only a pip install is served.\n  "
                           "pip install --upgrade better-rlm\n  "
                           "then run `better-rlm` from outside this directory -- it "
                           "mounts itself.")

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

    # Read it back. `claude mcp add` exiting 0 is not proof the entry landed the way
    # we asked -- a quoting slip once registered `C:Python314Scripts...` with a zero
    # exit, and only a read-back revealed it.
    landed = registered_command(claude)
    if landed != want:
        return "failed", (f"registered, but it reads back as "
                          f"`{' '.join(landed or [])}` instead of `{' '.join(want)}`")

    if have:
        return "repointed", (f"was pointing at `{' '.join(have)}`, which reads a "
                             f"different config; now `{' '.join(want)}`")
    return "added", f"mounted as `{SERVER_NAME}`: {' '.join(want)}"

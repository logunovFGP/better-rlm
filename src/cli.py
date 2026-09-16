"""``better-rlm`` -- the operator entry point (also ``python -m src.cli``).

One command for the three things an operator does outside a Claude session:
sign in (``auth``), pick a transport / provider / model (``config``), and run
the server by hand (``server``). Each one delegates to the script that already
did that job -- this module dispatches, it does not reimplement.

    better-rlm              open the config TUI (same as `config`)
    better-rlm auth         sign the `claude` CLI in, store the token in .env
    better-rlm config       transport / provider / model picker -> config.yaml
    better-rlm install      re-run the full installer
    better-rlm server       run the MCP server in this terminal (stdio)
    better-rlm --help       this text

**This command is bound to the checkout it was installed from.** The console
script's shebang points at that checkout's ``.venv_sh/bin/python`` and
``PKG_ROOT`` resolves next to this file, so ``better-rlm`` always acts on one
checkout -- the last one whose installer linked the name. That is the cost of
a global name, and it is why the per-checkout scripts (``./install.sh``,
``./run_tui.sh``, ``./run_server.sh``) remain the addressable way to reach a
*specific* checkout. See CLAUDE.md, "Design boundary".

Flags that are not subcommands (``--config PATH``, ``--one-shot CMD``) fall
through to the TUI unchanged, so ``python -m src.cli --one-shot /status``
keeps working for CI.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parent.parent

# subcommand -> (POSIX script, its args, Windows script, its args).
# The Windows column is not a translation of the POSIX one: install.ps1 takes
# PowerShell switches, and run_server.cmd is a batch file, so each side names
# the launcher that platform actually ships.
_COMMANDS: dict[str, tuple[str, list[str], str, list[str]]] = {
    "auth": ("install.sh", ["--auth"], "install.ps1", ["-Auth"]),
    "install": ("install.sh", [], "install.ps1", []),
    "server": ("run_server.sh", [], "run_server.cmd", []),
}


def _run_script(cmd: str, extra: list[str]) -> int:
    """Run the platform's launcher for ``cmd`` and return its exit status.

    subprocess rather than os.exec*: the child inherits this process's stdin,
    so install.sh's hidden token prompt (``read -rs``) still gets the tty,
    and one code path covers both platforms.
    """
    posix_script, posix_args, win_script, win_args = _COMMANDS[cmd]
    if sys.platform == "win32":
        script = PKG_ROOT / win_script
        argv = (
            [str(script), *win_args, *extra]
            if win_script.endswith(".cmd")
            else ["powershell", "-ExecutionPolicy", "Bypass", "-File", str(script), *win_args, *extra]
        )
    else:
        script = PKG_ROOT / posix_script
        argv = ["bash", str(script), *posix_args, *extra]
    if not script.is_file():
        print(f"better-rlm: missing launcher {script}", file=sys.stderr)
        return 1
    return subprocess.call(argv, cwd=str(PKG_ROOT))


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)

    if args and args[0] in ("-h", "--help", "help"):
        print(__doc__ or "")
        return 0

    if args and args[0] in _COMMANDS:
        return _run_script(args[0], args[1:])

    # Bare invocation, `config`, or any TUI flag (--config / --one-shot).
    if args and args[0] == "config":
        args = args[1:]
    # Lazy import: src.tui pulls rich and the engine's import graph, which a
    # bare `better-rlm --help` or `better-rlm auth` must not pay for.
    from .tui import main as tui_main

    return tui_main(args)


if __name__ == "__main__":
    raise SystemExit(main())

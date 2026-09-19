"""``better-rlm`` -- the operator entry point (also ``python -m better_rlm.cli``).

One command for the things an operator does outside a Claude session: sign in
(``auth``), pick a transport / provider / model (``config``), and run the MCP
server (``server``).

    better-rlm              open the config TUI (same as `config`)
    better-rlm auth         sign the `claude` CLI in, store the token
    better-rlm config       transport / provider / model picker -> config.yaml
    better-rlm install      re-run the full installer          (checkout only)
    better-rlm server       run the MCP server on stdio
    better-rlm where        print the paths in use, and which mode you are in
    better-rlm --version    print the version, and warn about a second copy
    better-rlm --help       this text

**Two ways this package gets used, and they are not the same.**

*From a checkout* (``git clone`` + ``./install.sh``) everything works as it
always has: ``auth`` and ``install`` shell out to ``install.sh``, config lives
at the repo root, and the launcher scripts are right there.

*From ``pip install better-rlm``* there is no checkout -- no ``install.sh``, no
repo-root ``config.yaml``. So ``server`` runs the server in-process instead of
through ``run_server.sh`` (which makes ``claude mcp add rlm -- better-rlm
server`` work), config falls back to ``~/.rlm/config.yaml``, and ``install``
says plainly that it needs a checkout rather than failing on a missing file.
``auth`` still works: it is the ``claude`` CLI's own flow either way.

Subcommands dispatch to the thing that already does that job; they do not
reimplement it. See CLAUDE.md, "Design boundary".

Flags that are not subcommands (``--config PATH``, ``--one-shot CMD``) fall
through to the TUI unchanged, so ``python -m better_rlm.cli --one-shot /status``
keeps working for CI.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from .config import IS_CHECKOUT, PKG_ROOT, config_file, env_file

# subcommand -> (POSIX launcher, its args, Windows launcher, its args).
# Not a translation of one another: install.ps1 takes PowerShell switches and
# run_server.cmd is a batch file, so each column names the launcher that
# platform actually ships. Checkout-only -- none of these exist in site-packages.
_SCRIPTS: dict[str, tuple[str, list[str], str, list[str]]] = {
    "auth": ("install.sh", ["--auth"], "install.ps1", ["-Auth"]),
    "install": ("install.sh", [], "install.ps1", []),
    "server": ("run_server.sh", [], "run_server.cmd", []),
}


def _run_script(cmd: str, extra: list[str]) -> int:
    """Run the platform's launcher for ``cmd`` and return its exit status.

    subprocess rather than os.exec*: the child inherits this process's stdin, so
    install.sh's hidden token prompt (``read -rs``) still gets the tty, and one
    code path covers both platforms.
    """
    posix_script, posix_args, win_script, win_args = _SCRIPTS[cmd]
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


def _serve() -> int:
    """Run the MCP server in this process -- the pip-install path.

    A checkout goes through run_server.sh because that script also pins
    PYTHONUTF8 and the venv interpreter. Installed, there is no script and no
    venv to pick: this interpreter is already the right one.
    """
    from .server import main as server_main

    return server_main() or 0


def _auth_without_a_checkout() -> int:
    """``claude setup-token`` is the whole auth flow; install.sh only wraps it.

    Printing the two steps beats spawning a token prompt we would then have to
    write somewhere -- and it keeps this command out of the business of handling
    a year-long credential, which install.sh is careful never to do either.
    """
    env = env_file()
    print(
        "better-rlm auth (installed, no checkout)\n"
        "\n"
        "  1. claude setup-token        # prints a long-lived token, once\n"
        f"  2. put it in {env} as:\n"
        "         CLAUDE_CODE_OAUTH_TOKEN=<token>\n"
        "\n"
        "Then `chmod 600` that file. An `export` in your shell does NOT reach the\n"
        "server: Claude Code launches it with its own environment.\n"
        "\n"
        "Already signed in to the `claude` CLI interactively? That works too, but the\n"
        "login expires and cannot self-refresh -- prefer the token for a server you\n"
        "leave running.\n"
        "\n"
        "For API keys instead (mode: api), put ANTHROPIC_API_KEY in the same file."
    )
    return 0


def _other_copies_warning(others: list[Path]) -> str:
    """The `pip uninstall` footgun, or "" when this machine has one copy.

    A version number on its own can be a lie about what runs next: which copy
    Python loads is decided by sys.path order, not by which was installed last.
    Setup already says this (onboard._mount), but an operator checking whether an
    upgrade landed types `--version` or `where` -- and both used to answer without
    mentioning the second install that may be the one actually serving.
    """
    if not others:
        return ""
    return (f"warning: another copy of better_rlm is installed at {others[0]}.\n"
            "         Whichever one Python finds first wins. Use `pip install\n"
            "         --upgrade better-rlm`; do NOT run `pip uninstall`, which\n"
            "         removes the newer copy first.")


def _version() -> int:
    """``better-rlm --version``.

    It answered `unknown flag: --version` and exited 2, because the flag fell
    through to the TUI's parser. Reported by an operator running two installs,
    for whom this number is the only way to tell which one an upgrade landed on.
    """
    from .mcpreg import install_identity

    version, active, others = install_identity()
    print(f"better-rlm {version}")
    print(f"root: {active}")
    if warning := _other_copies_warning(others):
        print(warning, file=sys.stderr)
    return 0


def _where() -> int:
    """Answer 'which config is this thing actually reading?' without a guess."""
    from .mcpreg import install_identity

    version, _active, others = install_identity()
    cfg = config_file()
    print(f"version: {version}")
    print(f"mode:    {'checkout' if IS_CHECKOUT else 'installed (pip)'}")
    print(f"root:    {PKG_ROOT}")
    print(f"config:  {cfg}{'' if cfg.is_file() else '   (absent -- baked-in defaults apply)'}")
    env = env_file()
    print(f"env:     {env}{'' if env.is_file() else '   (absent)'}")
    if warning := _other_copies_warning(others):
        print(warning, file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)

    if args and args[0] in ("-h", "--help", "help"):
        print(__doc__ or "")
        return 0

    if args and args[0] in ("-V", "--version", "version"):
        return _version()

    if args and args[0] == "where":
        return _where()

    if args and args[0] == "server":
        if IS_CHECKOUT:
            return _run_script("server", args[1:])
        if args[1:]:
            # Silently dropping these would make the same command mean two things:
            # forwarded to run_server.sh from a checkout, void from a wheel.
            print(
                f"better-rlm server takes no arguments when installed from a wheel "
                f"(got {' '.join(args[1:])!r}). The server reads config.yaml and the "
                f"environment; see `better-rlm where` for which files those are.",
                file=sys.stderr,
            )
            return 2
        return _serve()

    if args and args[0] in _SCRIPTS:
        if IS_CHECKOUT:
            return _run_script(args[0], args[1:])
        if args[0] == "auth":
            return _auth_without_a_checkout()
        # Names the subcommand the user actually typed. Hardcoding "install" here
        # meant a future _SCRIPTS entry would report a command nobody ran.
        print(
            f"better-rlm {args[0]} needs a checkout -- it drives install.sh, which\n"
            "builds a venv, the Docker sandbox image and the skill links, none of\n"
            "which exist in a pip install.\n"
            "  git clone https://github.com/logunovFGP/better-rlm && cd better-rlm\n"
            "  ./install.sh",
            file=sys.stderr,
        )
        return 1

    # Bare invocation, `config`, or any TUI flag (--config / --one-shot).
    if args and args[0] == "config":
        # --menu, so tui.main can tell `better-rlm config` from bare `better-rlm`:
        # the first is the menu by name, the second is setup and may finish and exit.
        args = ["--menu", *args[1:]]
    # Lazy import: tui pulls rich and the engine's import graph -- seconds of it --
    # which `--help`, `where` and `auth` must not pay for. Note that .config above
    # IS imported at module scope and does read .env: that cost is milliseconds, and
    # hoisting it into each function would break the module-attribute seams
    # (cli.IS_CHECKOUT, cli.PKG_ROOT) that the tests patch. The engine graph is the
    # cost worth deferring; config is not.
    from .tui import main as tui_main

    return tui_main(args)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Install or remove the oversized-read PreToolUse hook.

Owns the whole lifecycle - copying hooks/big-read-to-rlm.py into ~/.claude/hooks
and merging the settings.json entry - so install.sh and uninstall.sh each hold one
call instead of two copies of the same paths, python3 probe and message text. The
shell scripts decide *whether*; this decides *how*.

Idempotent in both directions: registering twice leaves one entry, removing twice
is a no-op. Without that, every `./install.sh --hook` would append a duplicate and
the hook would run N times per Read.

That file is the user's entire Claude Code configuration - model, permissions,
plugins, other people's hooks. So: parse, mutate one list, write atomically via a
temp file and os.replace, and refuse outright if it does not parse. A partial write
here would cost far more than this feature is worth.

Paths are arguments rather than constants so the tests drive real files under
tmp_path instead of monkeypatching the module.

usage: install_hook.py [--remove] [--dry-run] [--hook PATH] [--settings PATH]
"""
import argparse
import json
import os
import shutil
import sys
import tempfile

#: Shipped copy, resolved from this file so it works from any cwd and from a
#: worktree. install.sh must not have to know the layout.
SOURCE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "hooks", "big-read-to-rlm.py")
DEFAULT_HOOK = os.path.expanduser("~/.claude/hooks/big-read-to-rlm.py")
DEFAULT_SETTINGS = os.path.expanduser("~/.claude/settings.json")
EVENT = "PreToolUse"
MATCHER = "Read"


def _command(hook_path: str) -> str:
    return f"python3 {hook_path}"


def _owns(entry: dict, hook_path: str) -> bool:
    """True when this PreToolUse entry is ours.

    Matched on the hook path appearing in the command, not on equality with
    _command(): an entry hand-edited to `python3.12 <path>` or wrapped in `timeout`
    is still ours, and leaving it behind on uninstall would strand a hook pointing
    at a file we just deleted - which breaks every Read on the machine.
    """
    return any(
        hook_path in (h or {}).get("command", "")
        for h in (entry.get("hooks") or [])
    )


def _write_atomic(path: str, data: dict) -> None:
    parent = os.path.dirname(path) or "."
    os.makedirs(parent, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def apply(settings_path: str, hook_path: str, remove: bool = False) -> str:
    """Merge (or unmerge) the hook. Returns a one-line status for the installer."""
    try:
        with open(settings_path, encoding="utf-8") as fh:
            settings = json.load(fh)
    except FileNotFoundError:
        settings = {}
    except ValueError as exc:
        # Never overwrite a settings.json we could not parse: that would replace a
        # file full of the user's configuration with one holding only our hook.
        raise SystemExit(
            f"  ERROR: {settings_path} is not valid JSON ({exc}). Left untouched."
        )

    hooks = settings.setdefault("hooks", {})
    entries = hooks.setdefault(EVENT, [])
    mine = [e for e in entries if _owns(e, hook_path)]

    if remove:
        if not mine:
            # Leave the empty scaffolding setdefault just created rather than
            # rewriting the file to say nothing changed.
            return "  not registered - nothing to remove"
        hooks[EVENT] = [e for e in entries if not _owns(e, hook_path)]
        if not hooks[EVENT]:
            del hooks[EVENT]
        if not hooks:
            del settings["hooks"]
        _write_atomic(settings_path, settings)
        return f"  removed {len(mine)} hook entr{'y' if len(mine) == 1 else 'ies'}"

    if len(mine) == 1:
        return "  already registered - nothing to do"
    if len(mine) > 1:
        # Only reachable from a hand-edited settings.json, but collapsing beats
        # leaving the hook to fire N times per Read.
        hooks[EVENT] = [e for e in entries if not _owns(e, hook_path)] + [mine[0]]
        _write_atomic(settings_path, settings)
        return f"  collapsed {len(mine)} duplicate entries to 1"

    entries.append(
        {"matcher": MATCHER, "hooks": [{"type": "command", "command": _command(hook_path)}]}
    )
    _write_atomic(settings_path, settings)
    return "  registered"


def place(hook_path: str, remove: bool = False) -> str:
    """Copy the shipped hook into place, or delete the installed copy."""
    if remove:
        if not os.path.isfile(hook_path):
            return "  hook script not installed - nothing to remove"
        os.unlink(hook_path)
        return f"  removed {hook_path}"
    os.makedirs(os.path.dirname(hook_path), exist_ok=True)
    shutil.copy2(SOURCE, hook_path)
    os.chmod(hook_path, 0o755)
    return f"  installed {hook_path}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Install or remove the oversized-read hook.")
    ap.add_argument("--remove", action="store_true", help="uninstall instead of install")
    ap.add_argument("--dry-run", action="store_true", help="print what would happen, change nothing")
    ap.add_argument("--hook", default=DEFAULT_HOOK, help="where the hook script lives")
    ap.add_argument("--settings", default=DEFAULT_SETTINGS, help="settings.json to edit")
    args = ap.parse_args(argv)

    if args.dry_run:
        verb = "unregister and delete" if args.remove else "install and register"
        print(f"  [dry-run] would {verb} {args.hook}")
        return 0

    # Order matters in both directions: register only a file that exists, and
    # unregister before deleting - a settings entry pointing at a missing file
    # fails every Read on the machine.
    if args.remove:
        print(apply(args.settings, args.hook, remove=True))
        print(place(args.hook, remove=True))
    else:
        print(place(args.hook))
        print(apply(args.settings, args.hook))
    return 0


if __name__ == "__main__":
    sys.exit(main())

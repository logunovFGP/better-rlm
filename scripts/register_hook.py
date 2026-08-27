#!/usr/bin/env python3
"""Add or remove the oversized-read PreToolUse hook in ~/.claude/settings.json.

Idempotent in both directions: registering twice leaves one entry, removing twice
is a no-op. Without that, every `./install.sh --hook` would append a duplicate and
the hook would run N times per Read.

That file is the user's entire Claude Code configuration - model, permissions,
plugins, other people's hooks. So: parse, mutate one list, write atomically via a
temp file and os.replace, and refuse outright if it does not parse. A partial write
here would cost far more than this feature is worth.

Paths are arguments rather than constants so the tests drive real files under
tmp_path instead of monkeypatching the module.

usage: register_hook.py [--remove] [--hook PATH] [--settings PATH]
"""
import argparse
import json
import os
import sys
import tempfile

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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Register the oversized-read hook.")
    ap.add_argument("--remove", action="store_true", help="unregister instead of register")
    ap.add_argument("--hook", default=DEFAULT_HOOK, help="path to the installed hook script")
    ap.add_argument("--settings", default=DEFAULT_SETTINGS, help="settings.json to edit")
    args = ap.parse_args(argv)
    print(apply(args.settings, args.hook, remove=args.remove))
    return 0


if __name__ == "__main__":
    sys.exit(main())

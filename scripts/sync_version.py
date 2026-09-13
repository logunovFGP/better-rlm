#!/usr/bin/env python3
"""Write VERSION into .claude-plugin/plugin.json. Run after bumping VERSION.

plugin.json is static JSON read by Claude Code's plugin loader before any of this
project's code runs, so it cannot import src.version the way everything else does. This
is the one place a second copy is unavoidable -- so it is generated rather than typed,
and tests/test_plugin_manifest.py fails if the two ever drift.

Rewrites only the top-level "version" value, byte-for-byte otherwise: the file also
carries the mcpServers launch command, and reserializing the whole dict would reorder
keys and lose formatting for a one-token change. The rewrite is verified by re-parsing
the result, so it cannot silently land on a nested "version" instead.

usage: sync_version.py [--check]     (--check exits 1 on drift, changes nothing)
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VERSION = ROOT / "VERSION"
PLUGIN = ROOT / ".claude-plugin" / "plugin.json"


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    want = VERSION.read_text(encoding="utf-8").strip()
    raw = PLUGIN.read_text(encoding="utf-8")
    have = json.loads(raw)["version"]

    if have == want:
        print(f"  plugin.json already at {want}")
        return 0
    if "--check" in argv:
        print(f"  DRIFT: VERSION={want} but plugin.json={have}."
              f"  Fix: python3 scripts/sync_version.py")
        return 1

    new, n = re.subn(r'("version":\s*)"[^"]*"',
                     lambda m: m.group(1) + json.dumps(want), raw, count=1)
    if n != 1:
        print("  ERROR: no \"version\" key in plugin.json")
        return 1
    # count=1 hits the FIRST "version" in the file, which is the top-level one only
    # because plugin.json happens to declare it before mcpServers. Check the result
    # rather than trust the ordering: a nested key placed above it would be rewritten
    # instead, while the json.loads read above kept returning the old top-level value
    # -- so --check would report drift forever and every "fix" would miss.
    if json.loads(new)["version"] != want:
        print("  ERROR: rewrote a nested \"version\", not the top-level one")
        return 1
    # newline="" so the string is written through untranslated: without it Windows
    # text mode turns every LF into CRLF, and .gitattributes pins this repo to
    # eol=lf -- a one-token bump would land as a whole-file rewrite.
    PLUGIN.write_text(new, encoding="utf-8", newline="")
    print(f"  plugin.json {have} -> {want}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

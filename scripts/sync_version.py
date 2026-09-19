#!/usr/bin/env python3
"""Write VERSION into .claude-plugin/plugin.json and better_rlm/version.py.

Run after bumping VERSION.

plugin.json is static JSON read by Claude Code's plugin loader before any of this
project's code runs, so it cannot import better_rlm.version the way everything else does.

better_rlm/version.py carries the second copy, ``_BAKED``, for the opposite reason: a
wheel ships no VERSION file, and the metadata that used to answer there described a
.dist-info directory rather than the running code -- so a stale one next door made an
install report a version whose code was gone. A literal inside the package cannot.

Neither copy is typed by hand; both are generated here, and
tests/test_plugin_manifest.py fails if either ever drifts from VERSION.

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
VERSION_PY = ROOT / "better_rlm" / "version.py"

_BAKED_RE = re.compile(r'^(_BAKED\s*=\s*)"[^"]*"', re.MULTILINE)


def _sync_baked(want: str, check: bool) -> int:
    """Hold ``better_rlm.version._BAKED`` to VERSION."""
    raw = VERSION_PY.read_text(encoding="utf-8")
    m = _BAKED_RE.search(raw)
    if not m:
        print("  ERROR: no _BAKED assignment in better_rlm/version.py")
        return 1
    have = m.group(0).split('"')[1]

    if have == want:
        print(f"  version.py already at {want}")
        return 0
    if check:
        print(f"  DRIFT: VERSION={want} but version.py _BAKED={have}."
              f"  Fix: python3 scripts/sync_version.py")
        return 1

    # Same newline="" reasoning as plugin.json below: .gitattributes pins eol=lf.
    VERSION_PY.write_text(_BAKED_RE.sub(lambda _m: _m.group(1) + json.dumps(want), raw,
                                        count=1),
                          encoding="utf-8", newline="")
    print(f"  version.py {have} -> {want}")
    return 0


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    want = VERSION.read_text(encoding="utf-8").strip()
    rc = _sync_baked(want, "--check" in argv)
    raw = PLUGIN.read_text(encoding="utf-8")
    have = json.loads(raw)["version"]

    if have == want:
        print(f"  plugin.json already at {want}")
        return rc
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
    return rc


if __name__ == "__main__":
    sys.exit(main())

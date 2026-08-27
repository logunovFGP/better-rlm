#!/usr/bin/env python3
"""PreToolUse:Read - route oversized reads to RLM, but only when RLM can serve them.

Why a hook and not a better skill description
---------------------------------------------
Skill selection matches a description against *conversation text*. This skill's
trigger is a property of the *data*: "the file is larger than ~200 KB". The user
types "analyze this log"; nothing in that sentence says 2 GB, so the model cannot
evaluate the predicate until it has already called Read - by which point the read
was truncated and the context is spent. No rewrite closes that gap, because it
would have to enumerate every phrasing of "read a file that happens to be large".

A PreToolUse hook evaluates the predicate at the one moment it is knowable: after
the tool call is formed, before it runs. Deterministic where matching is
probabilistic.

Fail-open ladder - anything unknown, unavailable or legitimately bounded falls
through to a normal Read. The hook blocks only when redirecting is strictly better:

    stdin unparseable            -> 0   normal Read
    no file_path / not a file    -> 0   normal Read
    explicit `limit` requested   -> 0   normal Read  (already bounded, see below)
    extension RLM cannot serve   -> 0   normal Read  (images, PDFs, archives)
    size <= LIMIT                -> 0   normal Read
    rlm MCP server not present   -> 0   normal Read  (never block without an out)
    big AND rlm present          -> 2   block, redirect to rlm_load_file
"""
import json
import os
import sys

#: Matches the threshold rlm-large-context states in its own description. Below it,
#: an inline Read is cheaper than a round trip through the context store.
LIMIT = 200_000

#: A bounded Read is the *correct* end of the RLM workflow: rlm_grep hands back line
#: numbers, then Read(offset=N, limit=40) fetches just those lines. Blocking it would
#: make the hook fight the very tool it is promoting, so an explicit `limit` is a
#: statement that the caller already knows how much they are asking for.
BOUNDED_ARGS = ("limit",)

#: Read renders these natively (images, PDFs, notebooks with outputs). The context
#: store holds text, so redirecting them sends the caller somewhere that cannot help.
#: Archives are here for the same reason: rlm_load_file would store the bytes, not
#: the contents. Everything else - .log, .json, .csv, .ndjson, source - is fair game.
OPAQUE_SUFFIXES = frozenset(
    {
        ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg", ".ico", ".pdf",
        ".ipynb", ".zip", ".gz", ".tgz", ".bz2", ".xz", ".zst", ".7z", ".tar",
        ".woff", ".woff2", ".ttf", ".otf", ".mp4", ".mov", ".mp3", ".wav",
        ".so", ".dylib", ".dll", ".exe", ".bin", ".wasm", ".parquet",
    }
)


def rlm_configured(cwd: str = "") -> bool:
    """True when an 'rlm' MCP server is reachable from this project.

    Checks every scope `claude mcp add` can write, not just the user one. A hook
    that only knew about `--scope user` would silently never fire for anyone who
    installed with `--scope local` or `--scope project` - the failure mode being a
    feature that looks installed and does nothing, which is the exact class of
    quiet non-event this hook exists to prevent.

    Only key presence is tested. Server values can hold env vars and tokens; they
    are never read, logged or echoed.
    """
    cwd = cwd or os.getcwd()

    try:
        with open(os.path.expanduser("~/.claude.json"), encoding="utf-8") as fh:
            cfg = json.load(fh)
    except (OSError, ValueError):
        cfg = {}

    # --scope user: top level.
    if "rlm" in (cfg.get("mcpServers") or {}):
        return True

    # --scope local: keyed by project directory.
    project = (cfg.get("projects") or {}).get(cwd) or {}
    if "rlm" in (project.get("mcpServers") or {}):
        return True

    # --scope project: committed .mcp.json, walking up to the filesystem root.
    probe = os.path.abspath(cwd)
    while True:
        try:
            with open(os.path.join(probe, ".mcp.json"), encoding="utf-8") as fh:
                if "rlm" in (json.load(fh).get("mcpServers") or {}):
                    return True
        except (OSError, ValueError):
            pass
        parent = os.path.dirname(probe)
        if parent == probe:
            return False
        probe = parent


def decide(payload: dict, cwd: str = "") -> tuple[int, str]:
    """Return (exit_code, message).

    Split out of main() so tests drive the ladder directly instead of shelling out
    and parsing stderr - the branch that matters is which rung fires, not the IO.
    """
    args = payload.get("tool_input") or {}
    path = args.get("file_path") or ""

    if any(args.get(k) is not None for k in BOUNDED_ARGS):
        return 0, ""

    if os.path.splitext(path)[1].lower() in OPAQUE_SUFFIXES:
        return 0, ""

    try:
        size = os.path.getsize(path)
    except OSError:
        return 0, ""

    if size <= LIMIT:
        return 0, ""
    if not rlm_configured(cwd):
        return 0, ""

    return 2, (
        f"[hook] {path} is {size // 1024:,} KB - too large to read inline; the read "
        f"would be truncated and the context spent.\n"
        f"Use the rlm MCP server instead (skill: rlm-large-context):\n"
        f"  rlm_load_file({path!r}) -> ctx_id\n"
        f"  then rlm_grep / rlm_exec / rlm_query against that ctx_id\n"
        f"To read a known slice inline, pass an explicit `limit` - that is allowed."
    )


def main() -> int:
    try:
        payload = json.loads(sys.stdin.read())
    except (ValueError, OSError):
        return 0

    code, message = decide(payload)
    if message:
        sys.stderr.write(message)
    return code


if __name__ == "__main__":
    sys.exit(main())

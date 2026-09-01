"""Output bounding and context-metadata rendering: cap every tool return so raw context
never floods the root model, and render a stored context's metadata for a tool reply.

The renderers live here rather than in server.py because they are pure presentation over
a ContextMeta -- no config, no store, no transport -- and server.py sits against its
800-line ceiling.
"""

from __future__ import annotations

_NOTICE = (
    "\n…[truncated: {hidden} of {total} bytes withheld — "
    "narrow your query or use rlm_inspect_context]"
)


def bound_output(text: str, cap_bytes: int) -> str:
    """Return ``text`` unchanged if within ``cap_bytes`` (UTF-8), else a capped
    head with a truncation notice. Never returns more than ~cap_bytes."""
    raw = text.encode("utf-8", errors="replace")
    if len(raw) <= cap_bytes:
        return text
    notice = _NOTICE.format(hidden=len(raw) - cap_bytes, total=len(raw))
    keep = max(0, cap_bytes - len(notice.encode("utf-8")))
    return raw[:keep].decode("utf-8", errors="ignore") + notice


#: Skips that are the POINT of the skip lists, not a surprise (node_modules, .env).
EXPECTED_SKIPS = frozenset({"skip-dir", "skip-name"})


def skipped_block(meta) -> str:
    """Report what a dir load did not ingest, loudly ONLY when it is surprising.

    A partial load that reports success is worse than a failure: an answer over the
    remaining files is wrong BY OMISSION and nothing contradicts it. Observed: 173 of
    184 files loaded, the 11 missing ones the exact subject of the question.

    But node_modules being skipped is the intent, not an incident. Screaming INCOMPLETE
    on every JS project trains the reader to skip the line -- and then it is not there
    when 11 source files really do vanish. So policy exclusions get one quiet count and
    the surprising ones get the alarm.
    """
    counts = dict(getattr(meta, "skipped_counts", None) or {})
    if not counts:
        return ""
    expected = {k: v for k, v in counts.items() if k in EXPECTED_SKIPS}
    surprising = {k: v for k, v in counts.items() if k not in EXPECTED_SKIPS}
    out = ""
    if expected:
        detail = ", ".join(f"{k} x{v:,}" for k, v in sorted(expected.items()))
        out += f"\n- excluded by policy: {sum(expected.values()):,} ({detail})"
    if surprising:
        n = sum(surprising.values())
        total = meta.file_count + sum(counts.values())
        detail = ", ".join(f"{k} x{v:,}" for k, v in sorted(surprising.items()))
        out += (f"\n- ⚠ **INCOMPLETE — {n:,} readable file(s) were NOT loaded** "
                f"(of {total:,} found; {detail}).\n"
                "  Any answer from this context is wrong by omission if it needed one:\n")
        sample = [e for e in (getattr(meta, "skipped", None) or [])
                  if e.split(":", 1)[0] not in EXPECTED_SKIPS][:20]
        out += "".join(f"    {e}\n" for e in sample).rstrip("\n")
        if n > len(sample):
            out += f"\n    … and {n - len(sample):,} more"
    return out


def meta_block(meta) -> str:
    """One context's metadata as the markdown block every load/inspect tool returns."""
    return (
        f"**ctx_id:** `{meta.ctx_id}`\n"
        f"- source: {meta.source}  ({meta.source_type}/{meta.data_type})\n"
        f"- size: {meta.bytes:,} bytes, {meta.lines:,} lines, ~{meta.est_tokens:,} tokens\n"
        f"- files: {meta.file_count}\n"
        f"- sha256: {meta.sha256[:16]}…"
        + skipped_block(meta)
    )

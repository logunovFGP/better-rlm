"""Output bounding: cap every tool return so raw context never floods the root model."""

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

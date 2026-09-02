"""Chunking strategies over materialized context text.

Returns boundary metadata only (char offsets, line ranges, est_tokens, label);
chunk content is read on demand from the store. Defaults keep chunks well under
Haiku's 200K-token ceiling.
"""

from __future__ import annotations

import bisect
import re
from dataclasses import asdict, dataclass

from .config import estimate_tokens

STRATEGIES = ("lines", "paragraphs", "functions", "headings", "semantic", "files")

_FILE_MARK = re.compile(r"^===== FILE: (.*?) \(\d+ bytes\) =====$", re.M)
_HEADING = re.compile(r"^#{1,6}[ \t]+\S.*$", re.M)
_FUNC = re.compile(
    r"^[ \t]*(?:async def |def |class |func |function |fn |public |private |protected )\S",
    re.M,
)


@dataclass
class Chunk:
    index: int
    start: int
    end: int
    start_line: int
    end_line: int
    n_chars: int
    est_tokens: int
    label: str = ""
    byte_start: int = -1   # -1 = unknown; set by ContextStore.set_chunks, readers must fall back
    byte_end: int = -1

    def as_dict(self) -> dict:
        return asdict(self)


def _line_starts(text: str) -> list[int]:
    """Char offset of every line start. re.finditer beats a per-character loop on
    the multi-MB contexts this runs over."""
    return [0, *(m.end() for m in re.finditer("\n", text))]


def _mk(text: str, starts: list[int], spans: list[tuple[int, int, str]]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for idx, (s, e, label) in enumerate(spans):
        if e <= s:
            continue
        body = text[s:e]
        chunks.append(Chunk(
            index=idx, start=s, end=e,
            start_line=bisect.bisect_right(starts, s),
            end_line=bisect.bisect_right(starts, max(s, e - 1)),
            n_chars=len(body), est_tokens=estimate_tokens(body), label=label,
        ))
    for new_idx, ch in enumerate(chunks):  # reindex after dropping empties
        ch.index = new_idx
    return chunks


def _cap_spans(spans: list[tuple[int, int, str]], max_chars: int) -> list[tuple[int, int, str]]:
    """Split any span longer than max_chars, preserving its label and order.

    ``chunk_chars`` is the sub-model context ceiling this module's docstring promises.
    Strategies that derive spans from split *points* get it via
    ``_split_points_to_spans``; strategies that build spans directly (``lines``,
    ``files``) must route through here, or one oversized chunk silently exceeds the
    sub-model's window and every sub-query over it fails with "prompt is too long".
    Sub-spans stay adjacent, so concatenation still reconstructs the source exactly.
    """
    out: list[tuple[int, int, str]] = []
    for s, e, label in spans:
        while e - s > max_chars:
            out.append((s, s + max_chars, label))
            s += max_chars
        if e > s:
            out.append((s, e, label))
    return out


def _split_points_to_spans(text: str, points: list[int], max_chars: int) -> list[tuple[int, int, str]]:
    """Given sorted split offsets, build spans, splitting any span exceeding max_chars."""
    points = sorted(set([0, *points, len(text)]))
    return _cap_spans([(s, e, "") for s, e in zip(points, points[1:])], max_chars)


def looks_like_file_bundle(head: str) -> bool:
    """True if ``head`` (the first few KB of a context) carries a FILE marker -- a dir load,
    or a bundle someone built with the documented separator. The ``files`` strategy is
    then the right default: file boundaries survive edits elsewhere, so the answer cache
    keeps hitting; ``lines`` boundaries shift on every edit above them and nothing does."""
    return _FILE_MARK.search(head) is not None


def chunk_text(text: str, strategy: str, *, chunk_lines: int, chunk_chars: int, overlap: int) -> list[Chunk]:
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose from {STRATEGIES}.")
    starts = _line_starts(text)

    if strategy == "lines":
        spans: list[tuple[int, int, str]] = []
        step = max(1, chunk_lines - max(0, overlap))
        for i in range(0, len(starts), step):
            s = starts[i]
            end_line_idx = min(i + chunk_lines, len(starts))
            e = starts[end_line_idx] if end_line_idx < len(starts) else len(text)
            spans.append((s, e, ""))
            if e >= len(text):
                break
        # Long lines (JSON-per-line logs) make chunk_lines alone a poor bound: 2000
        # lines of 4 KB is 8 MB, far past the sub-model. Cap on chunk_chars too.
        return _mk(text, starts, _cap_spans(spans, chunk_chars))

    # "semantic" is paragraph-boundary splitting with a larger target — NOT
    # embedding-based. Same code path, bigger chunks.
    if strategy in ("paragraphs", "semantic"):
        target = chunk_chars if strategy == "semantic" else min(chunk_chars, 40_000)
        paras = [m.start() for m in re.finditer(r"\n[ \t]*\n", text)]
        bounds = sorted(set([0, *[p + 1 for p in paras], len(text)]))
        spans, cur_start = [], 0
        for b in bounds[1:]:
            if b - cur_start >= target:
                spans.append((cur_start, b, ""))
                cur_start = b
        if cur_start < len(text):
            spans.append((cur_start, len(text), ""))
        spans = _split_points_to_spans(text, [s for s, _, _ in spans], chunk_chars)
        return _mk(text, starts, spans)

    # Both split on a marker regex; only the regex differs.
    marker = {"functions": _FUNC, "headings": _HEADING}.get(strategy)
    if marker is not None:
        pts = [m.start() for m in marker.finditer(text)]
        return _mk(text, starts, _split_points_to_spans(text, pts, chunk_chars))

    # strategy == "files"
    marks = list(_FILE_MARK.finditer(text))
    if not marks:
        return _mk(text, starts, _cap_spans([(0, len(text), "<single>")], chunk_chars))
    spans = []
    for i, m in enumerate(marks):
        s = m.start()
        e = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        spans.append((s, e, m.group(1)))
    # One file can exceed the sub-model on its own; keep its label on each part.
    return _mk(text, starts, _cap_spans(spans, chunk_chars))

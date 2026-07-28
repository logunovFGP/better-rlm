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


def _split_points_to_spans(text: str, points: list[int], max_chars: int) -> list[tuple[int, int, str]]:
    """Given sorted split offsets, build spans, splitting any span exceeding max_chars."""
    points = sorted(set([0, *points, len(text)]))
    spans: list[tuple[int, int, str]] = []
    for s, e in zip(points, points[1:]):
        while e - s > max_chars:
            spans.append((s, s + max_chars, ""))
            s += max_chars
        if e > s:
            spans.append((s, e, ""))
    return spans


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
        return _mk(text, starts, spans)

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

    if strategy == "functions":
        pts = [m.start() for m in _FUNC.finditer(text)]
        return _mk(text, starts, _split_points_to_spans(text, pts, chunk_chars))

    if strategy == "headings":
        pts = [m.start() for m in _HEADING.finditer(text)]
        return _mk(text, starts, _split_points_to_spans(text, pts, chunk_chars))

    # strategy == "files"
    marks = list(_FILE_MARK.finditer(text))
    if not marks:
        return _mk(text, starts, [(0, len(text), "<single>")])
    spans = []
    for i, m in enumerate(marks):
        s = m.start()
        e = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        spans.append((s, e, m.group(1)))
    return _mk(text, starts, spans)

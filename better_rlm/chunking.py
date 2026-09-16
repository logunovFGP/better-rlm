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
    #: sha256 of this chunk's own text. "" = unknown; set by ContextStore.set_chunks for
    #: the same reason as the offsets above, and readers must fall back the same way. It
    #: is the identity the answer cache keys on, so storing it here is what lets the cache
    #: scan skip re-reading every chunk purely to hash it.
    sha256: str = ""

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
    if max_chars <= 0:
        # A 0 from config.yaml used to spin forever below, appending empty spans. Read
        # it as "no char bound" rather than clamping to 1: clamping terminates but then
        # emits one chunk PER CHARACTER, which is a worse failure than the hang because
        # it looks like it worked. chunk_lines still bounds the walk.
        return [(s, e, label) for s, e, label in spans if e > s]
    out: list[tuple[int, int, str]] = []
    for s, e, label in spans:
        n = e - s
        if n <= 0:
            continue
        if n <= max_chars:
            out.append((s, e, label))
            continue
        # Split EVENLY, not max_chars-then-remainder. The default 2000-line window is
        # ~124 KB on a typical log, so capping at 120 KB used to emit 120 KB plus a
        # 4 KB sliver -- for *every* window. That doubled the chunk count (401 chunks
        # on a 400k-line log where 209 is right) and with it the sub_query_batch
        # fan-out, for chunks holding a few dozen lines. Ceil twice: k pieces of
        # ceil(n/k) chars, each guaranteed <= max_chars.
        k = -(-n // max_chars)
        size = -(-n // k)
        for i in range(k):
            a, b = s + i * size, min(s + (i + 1) * size, e)
            if b > a:
                out.append((a, b, label))
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
        # Bound each chunk by lines AND chars while walking, whichever binds first.
        # Long lines (JSON-per-line logs) make chunk_lines alone a poor bound: 2000
        # lines of 4 KB is 8 MB, far past the sub-model. But applying chunk_chars only
        # afterwards, via _cap_spans, is worse than it looks: _cap_spans cannot see line
        # boundaries, so a 2000-line window of ~124 KB against a 120 KB cap came back as
        # two half-full chunks. That doubled the chunk count -- 401 chunks on a 400k-line
        # log where 209 hold the same content -- and rlm_sub_query_batch pays per chunk.
        # Choosing the boundary here keeps every chunk as full as both limits allow.
        spans: list[tuple[int, int, str]] = []
        n = len(starts)
        i = 0
        while i < n:
            s = starts[i]
            by_lines = min(i + chunk_lines, n)
            # Last line start still within chunk_chars of s, so the span cannot exceed
            # it. chunk_chars <= 0 means "no char bound" (see _cap_spans) -- let lines
            # decide rather than collapsing every chunk to a single line.
            by_chars = (bisect.bisect_right(starts, s + chunk_chars) - 1
                        if chunk_chars > 0 else by_lines)
            # max(i + 1, ...) guarantees progress: a single line longer than chunk_chars
            # yields by_chars == i, and that lone long line falls to _cap_spans below.
            j = max(i + 1, min(by_lines, by_chars))
            e = starts[j] if j < n else len(text)
            spans.append((s, e, ""))
            if e >= len(text):
                break
            i = max(i + 1, j - max(0, overlap))
        # Still needed for the one case the walk cannot fix: a single line wider than
        # chunk_chars, which has no line boundary to break on.
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

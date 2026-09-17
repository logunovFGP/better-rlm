"""Port of cline-2's ``apps/cli/src/tui/components/searchable-list.tsx``.

Pure data: scoring, section rows, and the visible window. No terminal, no rich, so
the behaviour is testable without a TTY -- which is how cline tests it too
(``searchable-list.test.ts`` exercises exactly these functions).

Kept faithful rather than tidied. The scores (100/90/70/30), the three-pass window
loop and the "do not hide a lone section header" correction are upstream's; a
cleaner-looking rewrite would drift from the thing it is meant to mirror.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

MAX_VISIBLE = 10

_NORMALIZE_RE = re.compile(r"[^a-z0-9.]")


@dataclass(frozen=True)
class SearchableItem:
    """One selectable row. Mirrors upstream's ``SearchableItem`` interface."""

    key: str
    label: str
    section: str = ""
    detail: str = ""
    tag: str = ""
    right_label: str = ""
    search_text: str = ""
    payload: object = None       # what the caller gets back; not upstream, not rendered


@dataclass(frozen=True)
class Row:
    """A rendered line: a section header, or an item."""

    kind: Literal["header", "item"]
    key: str
    label: str = ""
    item: SearchableItem | None = None
    item_index: int = -1


@dataclass(frozen=True)
class Window:
    visible_rows: list[Row] = field(default_factory=list)
    above_count: int = 0
    below_count: int = 0
    show_above: bool = False
    show_below: bool = False


def normalize(s: str) -> str:
    return _NORMALIZE_RE.sub("", s)


def fuzzy_match(text: str, query: str) -> bool:
    """Subsequence match: every query char appears in order, gaps allowed."""
    qi = 0
    for ch in text:
        if qi >= len(query):
            break
        if ch == query[qi]:
            qi += 1
    return qi == len(query)


def score_item(item: SearchableItem, query: str) -> int:
    """Upstream's ladder: exact 100, prefix 90, substring 70, fuzzy 30, else 0."""
    targets = (item.label.lower(), item.key.lower(), (item.search_text or "").lower())
    n_query = normalize(query)
    best = 0
    for raw in targets:
        t = normalize(raw)
        if not t:
            continue
        if t == n_query:
            return 100
        if t.startswith(n_query):
            best = max(best, 90)
        elif n_query in t:
            best = max(best, 70)
        elif fuzzy_match(t, n_query):
            best = max(best, 30)
    return best


def _section_order(items: list[SearchableItem]) -> dict[str, int]:
    order: dict[str, int] = {}
    for it in items:
        if it.section and it.section not in order:
            order[it.section] = len(order)
    return order


def filter_items(items: list[SearchableItem], search: str) -> list[SearchableItem]:
    """Score, drop zeros, then sort by section order first and score second.

    Section rank beating score is upstream's choice and it matters: it keeps a
    catalogue's grouping stable while typing, instead of letting a high-scoring row
    jump out of its section.
    """
    if not search:
        return list(items)
    q = search.lower()
    order = _section_order(items)
    scored = [(it, score_item(it, q)) for it in items]
    scored = [(it, sc) for it, sc in scored if sc > 0]
    big = len(order) + 1

    def rank(pair: tuple[SearchableItem, int]) -> tuple[int, int]:
        it, sc = pair
        sect = order.get(it.section, big) if order else big
        return (sect, -sc)

    scored.sort(key=rank)
    return [it for it, _ in scored]


def build_rows(items: list[SearchableItem]) -> list[Row]:
    """Insert a header row whenever the section changes."""
    rows: list[Row] = []
    previous = None
    for idx, item in enumerate(items):
        if item.section and item.section != previous:
            rows.append(Row(kind="header", key=f"section-{item.section}-{idx}",
                            label=item.section))
        rows.append(Row(kind="item", key=item.key, item=item, item_index=idx))
        previous = item.section
    return rows


def _count_item_rows(rows: list[Row]) -> int:
    return sum(1 for r in rows if r.kind == "item")


def rows_window(items: list[SearchableItem], selected: int,
                max_visible: int = MAX_VISIBLE) -> Window:
    """The visible slice, with counts of what is hidden above and below.

    The three-pass loop is upstream's: each pass recomputes the limit knowing whether
    the "N above" / "N below" indicator lines are themselves taking a row, which
    changes how many items fit, which can change whether an indicator is needed.
    """
    rows = build_rows(items)
    if not rows:
        return Window()

    selected_row = next((i for i, r in enumerate(rows)
                         if r.kind == "item" and r.item_index == selected), 0)
    selected_row = max(0, selected_row)
    visible_limit = max_visible
    start, end = 0, len(rows)

    for _ in range(3):
        show_above = start > 0
        show_below = end < len(rows)
        visible_limit = max(1, max_visible - (1 if show_above else 0) - (1 if show_below else 0))
        start = max(0, selected_row - visible_limit // 2)
        if start + visible_limit > len(rows):
            start = max(0, len(rows) - visible_limit)
        end = min(len(rows), start + visible_limit)

    # A window starting on a section header with no items above it would hide that
    # header for nothing, so reset to the top instead.
    if start > 0 and _count_item_rows(rows[:start]) == 0:
        start = 0
        show_below = len(rows) > max_visible
        visible_limit = max(1, max_visible - (1 if show_below else 0))
        end = min(len(rows), visible_limit)

    above = _count_item_rows(rows[:start])
    below = _count_item_rows(rows[end:])
    return Window(visible_rows=rows[start:end], above_count=above, below_count=below,
                  show_above=above > 0, show_below=below > 0)


def move_up(selected: int, count: int) -> int:
    """Wraps to the last item, as upstream's moveUp does."""
    if count == 0:
        return 0
    return count - 1 if selected <= 0 else selected - 1


def move_down(selected: int, count: int) -> int:
    """Wraps to the first item, as upstream's moveDown does."""
    if count == 0:
        return 0
    return 0 if selected >= count - 1 else selected + 1

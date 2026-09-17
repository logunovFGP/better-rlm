"""The cline-2 searchable-list port, and the key reader that drives it.

The three section/window cases are ported from upstream's
``apps/cli/src/tui/components/searchable-list.test.ts`` so a divergence in the
behaviour we copied shows up as a failure here rather than as a picker that feels
subtly wrong.
"""

from __future__ import annotations

import os
import pytest

from better_rlm.searchable_list import (
    MAX_VISIBLE,
    SearchableItem,
    build_rows,
    filter_items,
    fuzzy_match,
    move_down,
    move_up,
    normalize,
    rows_window,
    score_item,
)


def _items(*specs: tuple[str, str]) -> list[SearchableItem]:
    return [SearchableItem(key=k, label=k, section=sec) for k, sec in specs]


# --- ported from cline's searchable-list.test.ts -------------------------------


def test_adds_section_headers_as_separate_rows():
    rows = build_rows(_items(("a", "one"), ("b", "one"), ("c", "two")))
    assert [r.kind for r in rows] == ["header", "item", "item", "header", "item"]
    assert [r.label for r in rows if r.kind == "header"] == ["one", "two"]


def test_keeps_window_counts_based_on_selectable_items():
    """Header rows take space but are not items, so the 'N more' counts must count
    items only -- otherwise the indicator lies about how much is hidden."""
    items = [SearchableItem(key=str(i), label=str(i), section=f"s{i // 3}")
             for i in range(20)]
    win = rows_window(items, 15, MAX_VISIBLE)
    assert win.above_count + len([r for r in win.visible_rows if r.kind == "item"]) \
        + win.below_count == len(items)
    assert win.show_above and win.show_below


def test_does_not_hide_only_a_section_header_before_showing_the_above_indicator():
    """Upstream's correction: a window that starts on a header with no items above
    it would spend a row hiding the header for nothing."""
    items = [SearchableItem(key=str(i), label=str(i), section="only") for i in range(12)]
    win = rows_window(items, 0, MAX_VISIBLE)
    assert not win.show_above
    assert win.visible_rows[0].kind == "header"


# --- scoring ------------------------------------------------------------------


@pytest.mark.parametrize("query,expected", [
    ("minimax", 100),        # exact
    ("mini", 90),            # prefix
    ("nima", 70),            # substring
    ("mnmx", 30),            # fuzzy subsequence
    ("zzz", 0),              # no match
])
def test_score_ladder_matches_upstream(query, expected):
    assert score_item(SearchableItem(key="m", label="MiniMax"), query) == expected


def test_normalize_strips_everything_but_alnum_and_dot():
    assert normalize("claude-sonnet-5") == "claudesonnet5"
    assert normalize("MiniMax-M2.7".lower()) == "minimaxm2.7"


def test_fuzzy_is_a_subsequence_not_a_substring():
    assert fuzzy_match("anthropic", "atc")
    assert not fuzzy_match("anthropic", "cta")   # order matters


def test_section_rank_beats_score():
    """Upstream sorts by section first. A high-scoring row must not jump out of its
    group while the operator types, or the list reshuffles under them."""
    items = [SearchableItem(key="a", label="zzz-exact", section="first"),
             SearchableItem(key="b", label="exact", section="second")]
    got = [i.key for i in filter_items(items, "exact")]
    assert got == ["a", "b"], "the 'second' section jumped ahead of 'first'"


def test_empty_search_returns_everything_unreordered():
    items = _items(("a", ""), ("b", ""), ("c", ""))
    assert [i.key for i in filter_items(items, "")] == ["a", "b", "c"]


# --- movement wraps, as upstream's moveUp/moveDown do -------------------------


@pytest.mark.parametrize("fn,start,count,expected", [
    (move_up, 0, 3, 2), (move_up, 2, 3, 1), (move_up, 0, 0, 0),
    (move_down, 2, 3, 0), (move_down, 0, 3, 1), (move_down, 0, 0, 0),
])
def test_movement_wraps(fn, start, count, expected):
    assert fn(start, count) == expected


# --- the key reader -----------------------------------------------------------


@pytest.mark.skipif(os.name != "posix", reason="pty is POSIX-only")
# pty.fork() warns when the parent is multi-threaded, because a child that takes a
# lock another thread held at fork time deadlocks. This child takes none: it sets
# sys.stdin, reads one key, writes to fd 1 and _exit()s without touching the
# allocator-heavy paths that make the warning matter.
@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")
def test_an_arrow_key_is_not_read_as_escape():
    """Regression, and it made the picker unusable rather than merely wrong.

    read_key used sys.stdin.read(1). Python's buffered text stream pulls the whole
    three-byte escape sequence in one syscall, returns the first character, and
    leaves the rest in a buffer the OS-level select() cannot see -- so every arrow
    key looked like a bare Esc and cancelled the dialog. Reading the fd directly is
    the fix; this drives a real pty because nothing smaller reproduces it.
    """
    import pty
    import select
    import time

    pid, fd = pty.fork()
    if pid == 0:                                   # pragma: no cover - child
        # os.write to fd 1, not print: under pytest sys.stdout is the capture
        # object created before the fork, so print() never reaches the pty.
        try:
            import sys as _sys

            # pytest replaces sys.stdin with a pseudofile that has no fileno();
            # the child needs the real one, which after pty.fork() is the slave.
            _sys.stdin = os.fdopen(0, "r")
            from better_rlm.picker import read_key

            os.write(1, b"READY\n")
            os.write(1, f"GOT:{read_key()}\n".encode())
        except BaseException as exc:               # surface it instead of hanging
            os.write(1, f"ERR:{type(exc).__name__}:{exc}\n".encode())
        os._exit(0)

    def drain(marker: bytes, timeout: float = 5.0) -> bytes:
        out = b""
        end = time.time() + timeout
        while time.time() < end and marker not in out:
            if select.select([fd], [], [], 0.1)[0]:
                try:
                    out += os.read(fd, 1024)
                except OSError:
                    break
        return out

    try:
        assert b"READY" in drain(b"READY"), "child never started"
        time.sleep(0.2)                            # let read_key reach raw mode
        os.write(fd, b"\x1b[B")
        out = drain(b"GOT:")
    finally:
        os.close(fd)
        try:
            os.waitpid(pid, 0)
        except ChildProcessError:
            pass
    assert b"GOT:down" in out, out[-200:]


# --- masked secret echo -------------------------------------------------------


@pytest.mark.parametrize("value,expected", [
    ("", ""),
    ("abc", "•••"),                       # too short to reveal anything
    ("sk-12345", "••••••••"),             # 8 chars: 2+2 would leak half
    ("sk-123456", "sk•••••56"),           # 9 chars: first two, last two
    ("sk-ant-api03-abcdefghijkl-xy", "sk••••••••••••••••••••••••xy"),
])
def test_mask_shows_first_and_last_two(value, expected):
    """cline echoes API keys in the clear; hiding them entirely was worse than both.

    A silent paste gives no way to tell whether the clipboard held the key, held
    nothing, or held the wrong thing -- and the answer only arrives at the first
    model call. Two characters each end is enough to recognise a key you just
    copied, and not enough to reconstruct one.
    """
    from better_rlm.picker import mask_secret

    assert mask_secret(value) == expected


def test_mask_never_reveals_more_than_four_characters():
    from better_rlm.picker import mask_secret

    for n in range(0, 60):
        secret = "x" * n
        assert len(mask_secret(secret).replace("•", "")) <= 4, n


def test_mask_preserves_length_so_a_truncated_paste_is_visible():
    from better_rlm.picker import mask_secret

    assert len(mask_secret("sk-abcdefghij")) == len("sk-abcdefghij")

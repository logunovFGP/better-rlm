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


# --- paste ---------------------------------------------------------------------


@pytest.mark.parametrize("raw,expected", [
    ("sk-abc123", "sk-abc123"),
    ("sk-abc123\n", "sk-abc123"),          # copied a line out of a file
    ("  sk-abc123  ", "sk-abc123"),        # copied with surrounding space
    ("sk-abc\r\n", "sk-abc"),              # CRLF from a Windows clipboard
    ("\n", ""),                            # nothing usable
    ("", ""),
])
def test_clean_paste_keeps_the_key_and_drops_the_wrapping(raw, expected):
    """A pasted key arrives with whatever the clipboard had around it. Rejecting the
    burst because isprintable() is False on the whole string threw the key away and
    left the prompt empty."""
    from better_rlm.picker import clean_paste

    assert clean_paste(raw) == expected


@pytest.mark.skipif(os.name != "posix", reason="pty is POSIX-only")
@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")
@pytest.mark.parametrize("payload", [
    b"sk-test-abc123xyz",                          # plain burst, older terminal
    b"\x1b[200~sk-test-abc123xyz\x1b[201~",        # bracketed paste
    b"\x1b[200~sk-test-abc123xyz\n\x1b[201~",      # with the trailing newline
])
def test_a_pasted_key_arrives_whole(payload):
    """Regression, and it made the prompt unusable rather than merely wrong.

    read_key set raw mode per keypress and restored it in a finally, so between keys
    the terminal was canonical. A paste is one burst: the first byte was read raw and
    the remaining bytes landed in the line discipline, which buffered them until a
    newline and then swallowed them as a line. Pasting a 40-character key produced
    exactly one character, which is what Cmd+V looked like from the operator's side.
    """
    import pty
    import select
    import time

    pid, fd = pty.fork()
    if pid == 0:                                   # pragma: no cover - child
        try:
            import sys as _sys

            # BOTH streams: picker.interactive() tests stdout too, and pytest's
            # capture object is not a tty, so read_secret would take the headless
            # path and never exercise raw mode at all.
            _sys.stdin = os.fdopen(0, "r")
            _sys.stdout = os.fdopen(1, "w")
            from rich.console import Console
            from better_rlm.picker import read_secret

            os.write(1, b"READY\n")
            got = read_secret(Console(file=_sys.stdout), "KEY")
            os.write(1, f"GOT:{got}\n".encode())
        except BaseException as exc:
            os.write(1, f"ERR:{type(exc).__name__}:{exc}\n".encode())
        os._exit(0)

    def drain(marker: bytes, timeout: float = 5.0) -> bytes:
        out = b""
        end = time.time() + timeout
        while time.time() < end and marker not in out:
            if select.select([fd], [], [], 0.1)[0]:
                try:
                    out += os.read(fd, 4096)
                except OSError:
                    break
        return out

    try:
        # Wait for the bracketed-paste enable sequence, not a sleep: it is the
        # definitive "raw mode is on and the prompt is listening" signal. A sleep
        # raced the child rich import and the paste landed in canonical mode.
        assert b"\x1b[?2004h" in drain(b"\x1b[?2004h"), "never entered raw mode"
        os.write(fd, payload)
        time.sleep(0.3)
        os.write(fd, b"\r")
        out = drain(b"GOT:")
    finally:
        os.close(fd)
        try:
            os.waitpid(pid, 0)
        except ChildProcessError:
            pass
    assert b"GOT:sk-test-abc123xyz" in out, out[-200:]


# --- key names are not text ----------------------------------------------------


@pytest.mark.parametrize("name", ["up", "down", "left", "right", "tab",
                                  "enter", "escape", "backspace"])
def test_a_named_key_carries_no_text(name):
    """read_key overloads one str for key names and literal text, which was safe
    only while text was always one character. When a paste started arriving whole,
    the `len(key) == 1` guards went with it -- and pressing Left while entering a
    credential appended the word "left" to it, invisibly, because the echo is
    masked."""
    from better_rlm.picker import key_text

    assert key_text(name) == ""


def test_real_text_still_reaches_the_buffer():
    from better_rlm.picker import key_text

    assert key_text("s") == "s"
    assert key_text("sk-ant-abc\n") == "sk-ant-abc"


def test_an_escape_sequence_riding_a_paste_is_stripped():
    """An unbracketed paste is drained with whatever is queued behind it, so an
    arrow pressed right after one arrives in the same burst. Filtering only
    non-printables left the "[D" tail in the value."""
    from better_rlm.picker import clean_paste

    assert clean_paste("sk-test-abc123xyz\x1b[D") == "sk-test-abc123xyz"
    assert clean_paste("sk-[abc]-123") == "sk-[abc]-123"      # real brackets survive


def test_backspace_removes_one_character_of_a_pasted_key():
    """buf held chunks, so a paste was one element and one Backspace deleted the
    whole credential."""
    buf: list[str] = []
    buf.extend("sk-ant-api03-abcdefghijklmnop")
    assert len(buf) == 29
    buf.pop()
    assert len(buf) == 28


def test_paste_ceilings_are_set():
    """A tty never reports EOF, so an end marker that never arrives must be bounded
    by size and time or read_key blocks forever."""
    from better_rlm.picker import _PASTE_MAX, _PASTE_TIMEOUT_S

    assert 0 < _PASTE_MAX <= 1024 * 1024
    assert 0 < _PASTE_TIMEOUT_S <= 10


# --- the Windows console path ------------------------------------------------
#
# picker.read_key has two implementations and only the POSIX one was covered; the
# Windows branch carried `# pragma: no cover` and three defects, each measured on a
# real Windows 11 console before these tests were written.

WINDOWS_ONLY = pytest.mark.skipif(os.name != "nt", reason="msvcrt is Windows-only")


@pytest.fixture
def fake_console(monkeypatch):
    """Drive the Windows key reader from a scripted buffer, not a real console.

    msvcrt is patched rather than stubbed as a module so the test exercises exactly
    the calls _read_key_windows makes: getwch to read, kbhit to ask whether more is
    already queued, ungetwch to hand a keypress back.
    """
    import msvcrt

    def install(chars):
        buf = list(chars)
        monkeypatch.setattr(msvcrt, "getwch", lambda: buf.pop(0))
        monkeypatch.setattr(msvcrt, "kbhit", lambda: bool(buf))
        monkeypatch.setattr(msvcrt, "ungetwch", lambda c: buf.insert(0, c))
        return buf

    return install


@WINDOWS_ONLY
def test_windows_ctrl_c_raises_like_posix(fake_console):
    """Ctrl+C must reach callers as KeyboardInterrupt on both platforms.

    POSIX gets that free: cbreak leaves ISIG on, so the kernel signals. msvcrt hands
    back the byte instead, so the old branch returned '\x03' as literal text -- the
    operator typed a control character into the search field with no way to abort.
    """
    from better_rlm.picker import read_key

    fake_console(["\x03"])
    with pytest.raises(KeyboardInterrupt):
        read_key()


@WINDOWS_ONLY
def test_windows_paste_arrives_whole_and_does_not_bleed(fake_console):
    """A paste is one unit, and a CR inside it stays a separate keypress.

    The old branch returned one character per call, so a pasted key submitted itself
    at the first CR and the remainder queued up to type itself into the NEXT prompt.
    """
    from better_rlm.picker import read_key

    rest = fake_console(list("sk-ant-SECRET") + ["\r"] + list("next-field"))
    assert read_key() == "sk-ant-SECRET"
    assert read_key() == "enter"
    assert "".join(rest) == "next-field"


@WINDOWS_ONLY
def test_windows_paste_is_bounded(fake_console):
    """Bounded like the POSIX paste path, so a stuck console cannot grow it forever."""
    from better_rlm.picker import _PASTE_MAX, read_key

    fake_console(list("x" * (_PASTE_MAX + 500)))
    assert len(read_key()) <= _PASTE_MAX


@WINDOWS_ONLY
@pytest.mark.parametrize("seq,want", [
    (["\xe0", "H"], "up"), (["\xe0", "P"], "down"),
    (["\xe0", "K"], "left"), (["\xe0", "M"], "right"),
    (["\r"], "enter"), (["\n"], "enter"), (["\x1b"], "escape"),
    (["\x08"], "backspace"), (["\t"], "tab"), (["a"], "a"),
])
def test_windows_named_keys(fake_console, seq, want):
    """The names the POSIX path returns, from the console's scan codes."""
    from better_rlm.picker import read_key

    fake_console(seq)
    assert read_key() == want


def test_every_name_the_windows_map_returns_is_a_registered_key_name():
    """A name outside KEY_NAMES is typed into the field as literal text.

    key_text() treats anything not in KEY_NAMES as text, so naming a key the set does
    not carry makes pressing it type its own name -- into the credential prompt too,
    where the masked echo hides what was typed. Caught while adding Home/End/Delete
    to the Windows map: key_text("home") returned "home". The Windows vocabulary is
    now exactly the POSIX one, and this keeps the two from drifting apart again.

    Runs on every platform: the maps are module-level data, so Linux CI guards the
    Windows branch here even though it cannot execute it.
    """
    from better_rlm.picker import _ARROWS, KEY_NAMES, _WIN_ARROWS, _WIN_NAMED

    assert set(_WIN_ARROWS.values()) <= KEY_NAMES, "unregistered arrow name"
    assert set(_WIN_NAMED.values()) <= KEY_NAMES, "unregistered key name"
    assert set(_ARROWS.values()) == set(_WIN_ARROWS.values()), (
        "the two platforms no longer name the same arrow set"
    )



def _throwaway_console():
    """A Console that renders into memory.

    StringIO, not open(os.devnull, "w"): that opens with the platform's default
    encoding, which is cp1252 on the Windows runners, and the pickers draw "\u25b8"
    and "\u2192". Both Windows jobs died with UnicodeEncodeError while the same
    tests passed on Linux and macOS. It also leaked the file handle.
    """
    import io

    from rich.console import Console

    return Console(file=io.StringIO(), width=100)


# --- Ctrl+C leaves; Esc goes back ---------------------------------------------
#
# These two keys were the same key. Every picker caught KeyboardInterrupt beside
# EOFError and returned CANCEL, which is what Esc returns, so its caller popped one
# level and drew the next screen: Ctrl+C read as "go back", and from a nested screen
# nothing left the program. Reported as "ctrl+c in the main menu opened another
# menu" -- the menu WAS the response.


@pytest.mark.parametrize("call", [
    lambda c, items: __import__("better_rlm.picker", fromlist=["x"]).choose(c, "t", items),
    lambda c, items: __import__("better_rlm.picker", fromlist=["x"]).choose_cards(c, "t", "s", items),
    lambda c, items: __import__("better_rlm.picker", fromlist=["x"]).read_secret(c, "key"),
])
def test_no_picker_swallows_ctrl_c(monkeypatch, call):
    """It must reach the caller. A picker that answers CANCEL cannot be told apart
    from Esc, and the caller's loop then redraws instead of exiting."""
    from rich.console import Console

    from better_rlm import picker

    monkeypatch.setattr(picker, "interactive", lambda: True)
    monkeypatch.setattr(picker, "raw_mode", lambda *a, **k: __import__("contextlib").nullcontext())
    monkeypatch.setattr(picker, "read_key", lambda: (_ for _ in ()).throw(KeyboardInterrupt))

    items = [SearchableItem(key="a", label="A"), SearchableItem(key="b", label="B")]
    with pytest.raises(KeyboardInterrupt):
        call(_throwaway_console(), items)


@pytest.mark.parametrize("fn", ["choose", "choose_cards"])
def test_eof_is_still_a_cancel(monkeypatch, fn):
    """The other half of the split. Headless stdin closing is not an interrupt --
    CI and --one-shot feed a pipe, and that path must keep answering CANCEL."""
    from rich.console import Console

    from better_rlm import picker

    monkeypatch.setattr(picker, "interactive", lambda: True)
    monkeypatch.setattr(picker, "raw_mode", lambda *a, **k: __import__("contextlib").nullcontext())
    monkeypatch.setattr(picker, "read_key", lambda: (_ for _ in ()).throw(EOFError))

    items = [SearchableItem(key="a", label="A")]
    args = ("t", "s", items) if fn == "choose_cards" else ("t", items)
    res = getattr(picker, fn)(_throwaway_console(), *args)
    assert res.key == picker.CANCEL


@pytest.mark.skipif(os.name != "posix", reason="pty is POSIX-only")
@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")
@pytest.mark.parametrize("descend", [0, 1], ids=["first-screen", "one-level-in"])
def test_ctrl_c_exits_the_program_not_just_the_screen(tmp_path, descend):
    """End to end, because the defect lived in the seam between the picker and its
    caller and no unit could see it: the picker returned, the caller looped, and the
    process stayed up drawing menus.

    Asserts the exit STATUS, not the screen. A screen assertion passes while the
    program is merely on its way out; 130 is only reachable by actually leaving.
    """
    import pty
    import select
    import signal
    import sys
    import time

    cfg = tmp_path / "config.yaml"
    cfg.write_text("mode: api\nprovider: minimax\n", encoding="utf-8")

    pid, fd = pty.fork()
    if pid == 0:                                   # pragma: no cover - child
        env = {**os.environ, "TERM": "xterm-256color", "COLUMNS": "100", "LINES": "40"}
        env.pop("NO_COLOR", None)
        os.execve(sys.executable,
                  [sys.executable, "-m", "better_rlm.cli", "--config", str(cfg)], env)

    def settle(seconds: float) -> None:
        end = time.time() + seconds
        while time.time() < end:
            if select.select([fd], [], [], 0.1)[0]:
                try:
                    if not os.read(fd, 65536):
                        return
                except OSError:
                    return

    try:
        settle(3.0)                                # first screen drawn
        for _ in range(descend):
            os.write(fd, b"\r")
            settle(1.5)
        os.write(fd, b"\x03")

        end, status = time.time() + 15.0, None
        while time.time() < end:
            settle(0.2)
            done, raw = os.waitpid(pid, os.WNOHANG)
            if done:
                status = os.waitstatus_to_exitcode(raw)
                break
        assert status is not None, (
            "Ctrl+C did not end the program -- it was swallowed as a cancel and the "
            "caller drew the next screen"
        )
        assert status == 130, f"expected 130 (SIGINT), got {status}"
    finally:
        try:
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
        except (ProcessLookupError, ChildProcessError):
            pass
        os.close(fd)

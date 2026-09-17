"""Interactive picker: cline-2's dialog interaction, on a terminal Python can drive.

cline renders its dialogs with opentui/React and reads keys through
``useDialogKeyboard`` (``components/dialogs/*.tsx``). The interaction is what
matters and is what this reproduces: a highlighted row you move with the arrow
keys, typing narrows the list, Enter resolves, Esc dismisses. Not a numbered list
you type a digit into.

Raw terminal input is stdlib (``termios``/``tty``, ``msvcrt`` on Windows) rather
than a new dependency -- rich is already shipped and does the drawing, and a TUI
framework would be a large dependency for one key loop.

**Headless is a first-class path, not a fallback bolted on.** The suite, CI and
``--one-shot`` all feed stdin from a pipe, where raw mode does not exist. Callers
get the numbered prompt there. cline draws the same line by testing the list model
(``searchable-list.test.ts``) rather than the terminal.
"""

from __future__ import annotations

import os
import re
import sys
from contextlib import contextmanager
from dataclasses import dataclass

from rich.align import Align
from rich.console import Console, Group
from rich.panel import Panel
from rich.text import Text

from .searchable_list import (
    MAX_VISIBLE,
    SearchableItem,
    filter_items,
    move_down,
    move_up,
    rows_window,
)

CANCEL = "__cancel__"
CUSTOM = "__custom__"

_CSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z~]")

_ARROWS = {"A": "up", "B": "down", "C": "right", "D": "left"}

#: Every value read_key can return that names a key rather than being literal text.
#: read_key overloads one str for both, which was safe only while text was always
#: exactly one character -- the `len(key) == 1` guards that used to exclude these
#: went away when a paste started arriving whole. Without this set, pressing Left
#: types "left" into whatever is being edited, including a credential, where the
#: masked echo hides it.
KEY_NAMES = frozenset({"up", "down", "left", "right", "enter", "escape",
                       "backspace", "tab"})


def key_text(key: str) -> str:
    """The literal text a key event carries: "" for a named key or an empty read."""
    return "" if key in KEY_NAMES else clean_paste(key)


def interactive() -> bool:
    """Whether a live picker can run at all: both ends must be a real terminal."""
    try:
        return sys.stdin.isatty() and sys.stdout.isatty()
    except (AttributeError, ValueError):
        return False


_RAW_HELD = False
_PASTE_START = "200~"
_PASTE_END = "\x1b[201~"
#: Ceilings for a bracketed paste. Generous for any credential or model id, and
#: small enough that a marker that never arrives fails fast instead of hanging.
_PASTE_MAX = 64 * 1024
_PASTE_TIMEOUT_S = 2.0


@contextmanager
def raw_mode(stream=None):
    """Hold the terminal in raw mode for a whole input loop, and turn on bracketed
    paste while we do.

    Setting raw per keypress and restoring it afterwards -- which is what this used
    to do -- leaves the terminal in canonical mode between keys. A paste arrives as
    one burst: the first byte is read raw, and the rest land in the line discipline,
    which buffers them until a newline and then swallows them as a line. Pasting a
    40-character key produced exactly one character.

    Bracketed paste (``ESC[?2004h``) makes the terminal wrap pasted content in
    markers, so a key containing a newline or an escape byte can never be mistaken
    for the operator pressing Enter or Esc.

    ``stream`` is where those escapes go. Callers pass the console they draw
    through, so a Console pointed at something other than stdout cannot leave
    bracketed paste toggled on a terminal it never wrote to.
    """
    global _RAW_HELD
    if _RAW_HELD or sys.platform == "win32":
        yield
        return
    import termios
    import tty

    out = stream or sys.stdout
    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    _RAW_HELD = True
    try:
        # setcbreak, NOT setraw. Both give unbuffered, unechoed input; setraw also
        # clears OPOST/ONLCR, which stops the terminal translating the "\n" rich
        # writes into CR-LF. The cursor then drops a row without returning to
        # column 0, rich's cursor-up arithmetic lands in the wrong place, and every
        # redraw appends a fresh copy of the frame instead of overwriting it.
        #
        # This only became visible when raw mode started being held across the whole
        # Live loop: before that it was set around a single os.read and restored, so
        # rich always rendered with post-processing on.
        #
        # cbreak also leaves ISIG on, so Ctrl+C arrives as SIGINT rather than as a
        # \x03 byte -- the KeyboardInterrupt the callers already catch, raised by the
        # kernel instead of by us.
        tty.setcbreak(fd)
        out.write("\x1b[?2004h")
        out.flush()
        yield
    finally:
        _RAW_HELD = False
        out.write("\x1b[?2004l")
        out.flush()
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)


def _read_pending(fd: int, limit: int = 65536) -> str:
    """Whatever is already queued, without blocking. A paste is a burst."""
    import select

    out = ""
    while select.select([fd], [], [], 0)[0]:
        chunk = os.read(fd, limit)
        if not chunk:
            break
        out += chunk.decode("utf-8", "replace")
    return out


#: Windows has no termios, so the console is read through msvcrt instead. These two
#: maps are the console's answer to the escape sequences _ARROWS covers on POSIX: a
#: special key arrives as a '\x00' or '\xe0' prefix followed by a scan code.
_WIN_ARROWS = {"H": "up", "P": "down", "K": "left", "M": "right"}
_WIN_NAMED = {"\r": "enter", "\n": "enter", "\x1b": "escape",
              "\x08": "backspace", "\t": "tab"}


def _read_key_windows() -> str:
    """The same contract as the POSIX path, on a console that has no termios.

    Three behaviours the previous branch did not have. Each was measured against the
    real msvcrt on Windows 11 before this was written:

      * **Ctrl+C came back as the byte '\\x03'.** On POSIX, cbreak leaves ISIG on, so
        the kernel raises KeyboardInterrupt and every caller already catches it --
        raw_mode's docstring says as much. msvcrt.getwch swallows the signal and
        returns the byte, so the operator typed a control character into the search
        field and had no way to abort. Raise it here instead, so both platforms hand
        callers the same exception.
      * **A paste arrived one character per call**, where the POSIX path returns the
        burst whole. Any CR inside the paste mapped to 'enter' and submitted the
        field mid-paste -- and the remainder stayed queued, typing itself into the
        NEXT prompt. msvcrt.kbhit() is this console's select(): drain what is already
        queued and return it as one unit, bounded by _PASTE_MAX like the POSIX path.
    The vocabulary is deliberately the same four arrows POSIX names and no more.
    Naming Home/End/Delete here looked like an easy win and is a trap: key_text()
    treats any name outside KEY_NAMES as literal text, so pressing Home would type
    "home" into whatever is being edited -- including the credential prompt, where
    the masked echo hides it. A new name has to be registered there first.

    There is no bracketed paste to enable: the console delivers pasted text as
    ordinary key events, which is exactly why kbhit is the right seam.
    """
    import msvcrt

    ch = msvcrt.getwch()
    if ch == "\x03":
        raise KeyboardInterrupt
    if ch in ("\x00", "\xe0"):
        return _WIN_ARROWS.get(msvcrt.getwch(), "")
    if ch in _WIN_NAMED:
        return _WIN_NAMED[ch]

    text = ch
    while msvcrt.kbhit() and len(text) < _PASTE_MAX:
        nxt = msvcrt.getwch()
        if nxt in ("\x00", "\xe0") or nxt in _WIN_NAMED or nxt == "\x03":
            # A keypress, not paste body. Put it back so the next read_key names it
            # -- a paste ending in CR then submits, which is what was intended.
            msvcrt.ungetwch(nxt)
            break
        text += nxt
    return text


def read_key() -> str:
    """One keypress, as a name -- 'up' 'down' 'left' 'right' 'enter' 'escape'
    'backspace' 'tab' -- or the literal text typed.

    Text may be longer than one character: a paste is a burst of bytes and is
    returned whole rather than one character per call, so callers append it as a
    unit. Bracketed-paste markers are stripped and their content is never parsed as
    keys.
    """
    if sys.platform == "win32":
        return _read_key_windows()

    import select

    with raw_mode():
        fd = sys.stdin.fileno()
        # os.read on the FD, never sys.stdin.read: Python's buffered text stream
        # pulls the whole escape sequence in one syscall, hands back the first
        # character, and leaves the rest in a buffer the OS-level select() cannot
        # see -- so every arrow key was read as a bare Esc.
        ch = os.read(fd, 1).decode("utf-8", "replace")

        if ch == "\x1b":
            # A lone Esc and the start of a sequence are the same byte. Nothing
            # queued behind it means the key itself.
            if not select.select([fd], [], [], 0.05)[0]:
                return "escape"
            rest = os.read(fd, 1).decode("utf-8", "replace")
            if rest != "[":
                return "escape"
            seq = ""
            while len(seq) < 8:
                # Time-bounded: the 0.05s guard above covers only the byte after
                # ESC. A partial CSI -- a disconnect mid-sequence, a resize racing
                # input -- otherwise blocks here forever with the terminal in raw
                # mode, where Ctrl+C is not delivered as a signal either.
                if not select.select([fd], [], [], 0.05)[0]:
                    return ""
                nxt = os.read(fd, 1).decode("utf-8", "replace")
                seq += nxt
                if nxt.isalpha() or nxt == "~":
                    break
            if seq == _PASTE_START:
                body = ""
                # Bounded both ways. A tty never reports EOF, so `if not more`
                # cannot end this loop: a paste cancelled mid-transfer, or a
                # terminal that sends the start marker but not the end, would
                # otherwise block forever while body grew without limit.
                while _PASTE_END not in body and len(body) < _PASTE_MAX:
                    if not select.select([fd], [], [], _PASTE_TIMEOUT_S)[0]:
                        break
                    more = os.read(fd, 4096).decode("utf-8", "replace")
                    if not more:
                        break
                    body += more
                return body.split(_PASTE_END)[0]
            return _ARROWS.get(seq[:1], "") if len(seq) == 1 else ""

        if ch in ("\r", "\n"):
            return "enter"
        if ch in ("\x7f", "\x08"):
            return "backspace"
        if ch == "\t":
            return "tab"
        if ch == "\x03":
            raise KeyboardInterrupt
        if ch == "\x04":
            raise EOFError
        if not ch.isprintable():
            return ""
        # A printable byte may be the first of a paste that is not bracketed (an
        # older terminal, or ssh). Take whatever else is already queued with it.
        return ch + _read_pending(fd)


@dataclass
class PickerResult:
    key: str
    item: SearchableItem | None = None
    search: str = ""


def _render(title: str, items: list[SearchableItem], selected: int, search: str,
            current_key: str, custom_hint: str) -> Panel:
    win = rows_window(items, selected, MAX_VISIBLE)
    body: list[Text] = []
    if win.show_above:
        body.append(Text(f"  ⌃ {win.above_count} more above", style="grey50"))
    for row in win.visible_rows:
        if row.kind == "header":
            body.append(Text(f"  {row.label.upper()}", style="bold grey50"))
            continue
        it = row.item
        chosen = row.item_index == selected
        line = Text()
        line.append("▸ " if chosen else "  ", style="cyan" if chosen else "")
        line.append(it.label, style="bold cyan" if chosen else "white")
        if it.key == current_key:
            line.append("  (current)", style="green")
        if it.tag:
            line.append(f"  [{it.tag}]", style="yellow")
        if it.detail:
            line.append(f"   {it.detail}", style="grey50" if not chosen else "grey70")
        body.append(line)
    if win.show_below:
        body.append(Text(f"  ⌄ {win.below_count} more below", style="grey50"))
    if not items:
        body.append(Text("  no match", style="yellow"))

    footer = Text()
    footer.append("\n")
    if search:
        footer.append("search: ", style="grey50")
        footer.append(search, style="bold")
        footer.append("   ")
    footer.append("↑/↓", style="cyan")
    footer.append(" move  ", style="grey50")
    footer.append("Enter", style="cyan")
    footer.append(" select  ", style="grey50")
    footer.append("type", style="cyan")
    footer.append(" to filter  ", style="grey50")
    if custom_hint:
        footer.append("Tab", style="cyan")
        footer.append(f" {custom_hint}  ", style="grey50")
    footer.append("Esc", style="cyan")
    footer.append(" cancel", style="grey50")
    body.append(footer)
    return Panel(Group(*body), title=f"[bold]{title}[/bold]", border_style="cyan")


def choose(console: Console, title: str, items: list[SearchableItem],
           current_key: str = "", custom_hint: str = "") -> PickerResult:
    """Run the live picker. Returns the chosen item, CANCEL, or CUSTOM.

    Mirrors upstream's key handling: arrows move with wrap, printable characters
    build a search that resets the selection to the top, Backspace edits it, Enter
    resolves, Esc dismisses. Tab is the "create custom" row cline calls
    ``CreateCustomModelRow``.
    """
    from rich.live import Live

    search = ""
    selected = 0
    shown = filter_items(items, search)
    # Open on the current value, the way a picker should: upstream marks it and the
    # operator expects Enter alone to be a no-op.
    for i, it in enumerate(shown):
        if it.key == current_key:
            selected = i
            break

    # auto_refresh=False on purpose: the picker is blocked on a keypress almost all
    # the time, and a background 30fps thread redraws a frame that cannot have
    # changed. It also fights the raw-mode key reader for the terminal, and floods a
    # pty fast enough to stall a writer. Redraw when the state changes, not on a clock.
    with raw_mode(console.file), Live(
            # screen=True draws on the terminal's ALTERNATE buffer, the way a
            # full-screen TUI does (cline's opentui the same). Every frame lands on
            # a clean screen, so there is no cursor-up arithmetic to get wrong --
            # which is what left the top of each previous frame on screen, once the
            # content grew past what rich counted. The original screen is restored
            # on exit, so the shell scrollback is untouched.
            _render(title, shown, selected, search, current_key, custom_hint),
            console=console, auto_refresh=False, screen=True) as live:
        while True:
            try:
                key = read_key()
            except (KeyboardInterrupt, EOFError):
                return PickerResult(CANCEL)
            if key == "escape":
                return PickerResult(CANCEL)
            if key == "enter":
                if not shown:
                    continue
                it = shown[min(selected, len(shown) - 1)]
                return PickerResult(it.key, it, search)
            if key == "tab" and custom_hint:
                return PickerResult(CUSTOM, None, search)
            if key == "up":
                selected = move_up(selected, len(shown))
            elif key == "down":
                selected = move_down(selected, len(shown))
            elif key == "backspace":
                search = search[:-1]
                shown = filter_items(items, search)
                selected = 0
            elif (text := key_text(key)):
                search += text
                shown = filter_items(items, search)
                selected = 0            # upstream's setSearch resets selection
            live.update(_render(title, shown, selected, search, current_key, custom_hint),
                        refresh=True)


def clean_paste(text: str) -> str:
    """What is usable from pasted text: printable characters, surrounds trimmed.

    Copying a key out of a file or a dashboard brings a trailing newline with it,
    and sometimes a leading space. Rejecting the burst because `isprintable()` is
    False on the whole string threw the key away and left the prompt empty. A
    newline is stripped rather than treated as Enter, so a two-line paste cannot
    silently submit half a credential.
    """
    # Strip CSI sequences first. An unbracketed paste is drained with whatever else
    # is queued behind it, so an arrow pressed right after one arrives in the same
    # burst -- and filtering only non-printables leaves its "[D" tail behind as
    # literal text in the value.
    return "".join(c for c in _CSI.sub("", text) if c.isprintable()).strip()


def mask_secret(value: str, keep: int = 2) -> str:
    """Render a secret as ``ab••••••yz`` -- enough to tell a good paste from a bad one.

    cline shows API keys in the clear while typing (``<input value={value}>`` in
    onboarding/screens.tsx). Hiding it entirely is what we had, and it was worse
    than either: a silent paste gives no way to tell whether the clipboard held the
    key, held nothing, or held the wrong one, and the failure only shows up at the
    first model call.

    Short values are fully masked rather than mostly revealed: showing 2+2 of an
    8-character string leaks half of it. Real keys are 40+ characters, so the guard
    only ever catches a mistyped fragment.
    """
    if not value:
        return ""
    if len(value) <= keep * 2 + 4:
        return "•" * len(value)
    return f"{value[:keep]}{'•' * (len(value) - keep * 2)}{value[-keep:]}"


def read_secret(console: Console, label: str, keep: int = 2) -> str:
    """Prompt for a secret, echoing it masked as it is typed.

    Returns "" when cancelled. The value is returned to the caller and never
    printed, logged, or put in a prompt string; only ``mask_secret`` of it reaches
    the screen.
    """
    if not interactive():
        # No raw mode on a pipe. getpass already degrades honestly there, and the
        # caller says so, so do not pretend to mask what the terminal will echo.
        from rich.prompt import Prompt

        return Prompt.ask(label, console=console).strip()

    console.print(f"[grey50]{label}[/grey50] ", end="")
    with raw_mode(console.file):
        return _read_secret_loop(console, label, keep)


def _read_secret_loop(console: Console, label: str, keep: int) -> str:
    # Characters, never chunks: a pasted burst extended one element at a time so
    # Backspace removes one character rather than the whole credential.
    buf: list[str] = []
    while True:
        try:
            key = read_key()
        except (KeyboardInterrupt, EOFError):
            console.print()
            return ""
        if key == "enter":
            console.print()
            return "".join(buf)
        if key == "escape":
            console.print("\n[grey50]cancelled[/grey50]")
            return ""
        if key == "backspace":
            if buf:
                buf.pop()
        else:
            # May be a whole pasted key rather than one character, and a paste out
            # of a file carries a trailing newline. A named key carries no text.
            text = key_text(key)
            if not text:
                continue
            buf.extend(text)
        shown = mask_secret("".join(buf), keep)
        # \r and a pad wide enough to erase the previous, possibly longer, render.
        # Sized from the character count: len(buf) was the element count, which a
        # paste made 1, leaving the old mask on screen after a Backspace.
        console.file.write("\r" + " " * (len(label) + len("".join(buf)) + 24) + "\r")
        console.file.write(f"{label} {shown}")
        console.file.flush()


def choose_cards(console: Console, title: str, subtitle: str,
                 items: list[SearchableItem], footer: str = "") -> PickerResult:
    """cline's first-run screen: a heading, a subtitle, and one bordered card per
    option with an arrow on the selected one.

    ``OnboardingMainMenuScreen`` in views/onboarding/screens.tsx renders each option
    as its own rounded box -- icon, label, detail, and a right-aligned arrow when
    selected -- rather than as rows of a table. On a first run that framing IS the
    question being asked; a dense list reads as a reference card and leaves the
    operator wondering what they are being asked to do.
    """
    from rich.live import Live

    selected = 0

    # cline sizes its onboarding the same way (contentWidth = min(width - 4,
    # HOME_VIEW_MAX_WIDTH)). A panel exactly as wide as the console wraps its last
    # border character onto the next row, which makes the frame one line taller than
    # rich accounted for -- so its cursor-up lands short and the previous frame's
    # top lines are left on screen instead of being overwritten.
    width = max(20, min(console.width - 4, 76))

    def frame() -> Group:
        parts: list[object] = [
            Text(title, style="bold", justify="center"),
            Text(subtitle, style="grey50", justify="center"),
            Text(""),
        ]
        for i, it in enumerate(items):
            chosen = i == selected
            body = Text()
            body.append(f"{it.tag or ' '}  ", style="cyan" if chosen else "grey50")
            body.append(it.label, style="bold" if chosen else "grey62")
            if chosen:
                body.append("   →", style="cyan")
            if it.detail:
                body.append("\n   ")
                body.append(it.detail, style="grey50")
            parts.append(Panel(body, border_style="cyan" if chosen else "grey30",
                               padding=(0, 1), width=width))
        parts.append(Text(footer or "↑/↓ navigate, Enter to select, Esc to exit",
                          style="grey50", justify="center"))
        # ONE Align around the whole group, never one per part: Align.center with an
        # explicit width pads each element vertically, so the frame occupied three
        # more rows than rich counted, its cursor-up cleared too few, and the top of
        # the previous frame survived every redraw.
        return Align.center(Group(*parts), width=width)

    with raw_mode(console.file), Live(frame(), console=console, auto_refresh=False,
                                      screen=True) as live:
        while True:
            try:
                key = read_key()
            except (KeyboardInterrupt, EOFError):
                return PickerResult(CANCEL)
            if key == "escape":
                return PickerResult(CANCEL)
            if key == "enter":
                return PickerResult(items[selected].key, items[selected])
            if key == "up":
                selected = move_up(selected, len(items))
            elif key == "down":
                selected = move_down(selected, len(items))
            else:
                continue
            live.update(frame(), refresh=True)

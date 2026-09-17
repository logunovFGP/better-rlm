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
import sys
from dataclasses import dataclass

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

_ARROWS = {"A": "up", "B": "down", "C": "right", "D": "left"}


def interactive() -> bool:
    """Whether a live picker can run at all: both ends must be a real terminal."""
    try:
        return sys.stdin.isatty() and sys.stdout.isatty()
    except (AttributeError, ValueError):
        return False


def read_key() -> str:
    """One keypress, as a name. 'up' 'down' 'left' 'right' 'enter' 'escape'
    'backspace' 'tab', or the literal character typed.

    Escape is ambiguous in a terminal: a bare Esc and the start of an arrow
    sequence are the same byte. Upstream gets this resolved by its runtime; here
    the distinction is made by asking whether more bytes are already waiting,
    which is what separates "the user pressed Esc" from "an escape sequence is
    arriving".
    """
    if sys.platform == "win32":                             # pragma: no cover
        import msvcrt

        ch = msvcrt.getwch()
        if ch in ("\x00", "\xe0"):
            return {"H": "up", "P": "down", "K": "left", "M": "right"}.get(
                msvcrt.getwch(), "")
        return {"\r": "enter", "\x1b": "escape", "\x08": "backspace",
                "\t": "tab"}.get(ch, ch)

    import select
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        # os.read on the FD, never sys.stdin.read: Python's buffered text stream
        # pulls the whole escape sequence in one syscall, hands back the first
        # character, and leaves the rest in a buffer the OS-level select() cannot
        # see -- so every arrow key was read as a bare Esc and cancelled the picker.
        ch = os.read(fd, 1).decode("utf-8", "replace")
        if ch == "\x1b":
            # A lone Esc and the start of an arrow sequence are the same byte. Ask
            # whether more is already queued; nothing waiting means the key itself.
            if not select.select([fd], [], [], 0.05)[0]:
                return "escape"
            rest = os.read(fd, 2).decode("utf-8", "replace")
            if not rest.startswith("["):
                return "escape"
            return _ARROWS.get(rest[1:2], "")
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
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


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
    with Live(_render(title, shown, selected, search, current_key, custom_hint),
              console=console, auto_refresh=False, transient=True) as live:
        live.refresh()
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
            elif len(key) == 1 and key.isprintable():
                search += key
                shown = filter_items(items, search)
                selected = 0            # upstream's setSearch resets selection
            live.update(_render(title, shown, selected, search, current_key, custom_hint),
                        refresh=True)


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

    buf: list[str] = []
    console.print(f"[grey50]{label}[/grey50] ", end="")
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
        elif len(key) == 1 and key.isprintable():
            buf.append(key)
        else:
            continue
        shown = mask_secret("".join(buf), keep)
        # \r and a pad wide enough to erase the previous, possibly longer, render.
        console.file.write("\r" + " " * (len(label) + len(buf) + 24) + "\r")
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
                               padding=(0, 1)))
        parts.append(Text(footer or "↑/↓ navigate, Enter to select, Esc to exit",
                          style="grey50", justify="center"))
        return Group(*parts)

    with Live(frame(), console=console, auto_refresh=False, transient=True) as live:
        live.refresh()
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

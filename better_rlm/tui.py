"""Operator-facing TUI for configuring the better-rlm transport stack.

Mirrors cline-2's mode/model picker (``apps/cli/src/tui/
components/model-selector/`` + ``hooks/use-model-selector.tsx``) on top of
rich.prompt -- so the same "compare two modes, pick a model" flow is
available here without pulling in a JS bundle. The CLI is
optional: the MCP server itself runs unchanged on ``python -m better_rlm.server``.

What this TUI does:

  * shows the current ``mode`` / ``provider`` / ``root_model`` /
    ``sub_model`` (the values that decide where model calls land);
  * lets the operator change the mode and the three models via rich.prompt
    pickers whose prose matches ``describe.py`` (cline-2's describeMode);
  * runs ``uv run --extra dev pytest -q`` via the ``/test`` slash command
    so the TUI is the same surface operators use to verify a config
    change before they restart the server.

``provider`` is shown but NOT offered as a picker: auth.require_anthropic
refuses every value but ``anthropic`` at every door that leads to a model
call, so /status flags a bad one instead of helping you write one.

What it deliberately does NOT do:

  * does not spawn the MCP server or the rlm engine. The one model call it
    can make is ``/auth-probe``, explicitly, on request;
  * does not write to ``.env`` (that's still the installer's job -- see
    install.sh --auth). It only touches ``config.yaml``.
  * does not require a TTY. The slash commands are read from stdin and
    output is plain text, so a one-shot ``/status`` works in CI / scripts.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from . import config_writer
from .config import (
    MODEL_OPUS,
    MODEL_SONNET,
    MODEL_SONNET_5,
    MODEL_HAIKU,
    PKG_ROOT,
    config_file,
    env_file,
)
from . import envfile
from .describe import (
    AUTH_API_KEY,
    AUTH_CLI,
    MODE_API,
    MODE_AUTO,
    MODE_CLI,
    PROVIDER_CUSTOM,
    PROVIDERS,
    VALID_MODES,
    all_modes,
    describe_mode,
    describe_provider,
    provider_for_config,
    providers_for_mode,
)

# Sentinels that mean "the user wants to back out / re-enter the picker".
# The cline-2 picker uses the same idiom (``CHANGE_PROVIDER_ACTION``,
# ``BROWSE_ALL_ACTION`` in model-selector.tsx): a non-model string the
# caller pattern-matches on to redirect flow without raising.
ACTION_CANCEL = "__cancel__"


@dataclass(frozen=True)
class Status:
    """One snapshot of the operator-visible state.

    Frozen so the ``/status`` command and the pickers share one source of
    truth: a picker picks against this, then writes its result back and
    the next ``/status`` reflects the new disk state.
    """

    mode: str
    provider: str
    root_model: str
    root_model_override: str
    sub_model: str
    cli_path: str
    cli_available: bool
    cli_logged_in: bool | None  # None = unknown (probe failed)
    env_mode: str | None        # set when RLM_MODE override is in effect
    env_provider: str | None    # set when RLM_PROVIDER override is in effect
    has_api_key: bool
    base_url: str = ""          # "" = Anthropic's own endpoint (the SDK default)

    def mode_is_pinned(self) -> bool:
        """True when an RLM_MODE env var overrides config.yaml.

        Tells the user the picker write may not take effect on the next
        server start -- the env var still wins.
        """
        return self.env_mode is not None and self.env_mode != self.mode


def load_status(config_path: Path | None = None) -> Status:
    """Snapshot the current configuration and runtime state from disk + env.

    The values shown are what ``config.load_config`` would resolve to on
    the NEXT server start, NOT the in-process state of any already-running
    MCP server. CLAUDE.md spells out why that distinction matters: a
    running server holds ``better_rlm/`` from startup, so a TUI write is not
    live until the server reconnects.
    """
    cfg_path = config_path or config_file()

    def _read(key: str, default: str) -> str:
        on_disk = config_writer.read_scalar(cfg_path, key)
        return default if on_disk is None else on_disk

    mode = _read("mode", MODE_AUTO)
    provider = _read("provider", "anthropic")
    root_model = _read("root_model", MODEL_SONNET_5)
    root_model_override = _read("root_model_override", MODEL_OPUS)
    sub_model = _read("sub_model", MODEL_HAIKU)
    cli_path = _read("cli_path", "claude")
    # RLM_BASE_URL wins on disk the same way RLM_MODE does, so what /status shows is
    # what the next server start resolves -- not what config.yaml alone says.
    base_url = (os.getenv("RLM_BASE_URL") or _read("base_url", "")).strip()

    env_mode = os.getenv("RLM_MODE")
    env_provider = os.getenv("RLM_PROVIDER")
    cli_available = shutil.which(cli_path) is not None

    # CLI login probe is non-fatal -- a probe failure is reported as "unknown", not as
    # an exception, so /status always renders. transport.cli_auth_status already asks
    # `claude auth status --json` and parses it; it only reads cfg.cli_path, so the TUI
    # hands it that one field rather than building a whole Config. Do NOT re-roll this:
    # without --json there is no "loggedIn" key to find, and sniffing the human output
    # reports NOT LOGGED IN for a CLI that is signed in.
    cli_logged_in: bool | None = None
    if cli_available:
        from .transport import cli_auth_status   # lazy: pulls the anthropic SDK
        raw = cli_auth_status(SimpleNamespace(cli_path=cli_path))
        flag = raw.get("loggedIn") if raw is not None else None
        # Only a real bool is an answer. bool() on a missing key would render a
        # signed-in CLI as NOT LOGGED IN, and on the string "false" would render a
        # signed-out one as logged in -- the confident-wrong-answer shape this
        # function was rewritten to remove. Anything else falls to "unknown".
        cli_logged_in = flag if isinstance(flag, bool) else None

    has_api_key = bool(os.getenv("ANTHROPIC_API_KEY"))

    return Status(
        mode=mode,
        provider=provider,
        root_model=root_model,
        root_model_override=root_model_override,
        sub_model=sub_model,
        cli_path=cli_path,
        cli_available=cli_available,
        cli_logged_in=cli_logged_in,
        env_mode=env_mode,
        env_provider=env_provider,
        has_api_key=has_api_key,
        base_url=base_url,
    )


def _provider_line(st: Status) -> str:
    """Which provider the next server start will use, named rather than shown as a URL.

    Flags the one combination that silently does nothing: a base_url set while mode
    still spawns the `claude` CLI, which reaches Anthropic whatever this says.
    """
    pid = provider_for_config(st.base_url, st.mode)
    d = describe_provider(pid)
    where = st.base_url or ("the `claude` CLI" if d.auth == AUTH_CLI else "api.anthropic.com")
    if st.base_url and st.mode == MODE_CLI:
        return f"{d.label} — {where}   IGNORED: mode={st.mode} uses the claude CLI"
    return f"{d.label} — {where}"


def render_status(st: Status) -> str:
    """Plain-text rendering of one Status -- used by /status and pickers.

    Kept string-returning (not Console.print) so the same function serves
    the interactive TUI and any future log/capture path.
    """
    cli_line = (
        "available" if st.cli_available else f"NOT on PATH (looked for {st.cli_path!r})"
    )
    if st.cli_available:
        if st.cli_logged_in is True:
            cli_line += ", logged in"
        elif st.cli_logged_in is False:
            cli_line += ", NOT LOGGED IN"
        else:
            cli_line += ", login status unknown"

    pin = ""
    if st.env_mode:
        pin += f"\n  RLM_MODE={st.env_mode} env var pins mode at registration (wins over config.yaml)"
    if st.env_provider:
        pin += f"\n  RLM_PROVIDER={st.env_provider} env var pins provider at registration"

    api_line = "ANTHROPIC_API_KEY=set" if st.has_api_key else "ANTHROPIC_API_KEY=MISSING"

    # auth.require_anthropic raises on every other provider at every door that leads
    # to a model call, so say so here rather than letting the next rlm_query be the
    # one to find out. The TUI cannot set this key -- only a hand edit or RLM_PROVIDER.
    bad_provider = (
        "" if st.provider.strip().lower() == "anthropic"
        else "   UNSUPPORTED — only provider=anthropic can make model calls"
    )

    return (
        f"mode:           {st.mode}\n"
        f"provider:       {_provider_line(st)}\n"
        f"protocol:       {st.provider}{bad_provider}\n"
        f"root_model:     {st.root_model}\n"
        f"override_model: {st.root_model_override}\n"
        f"sub_model:      {st.sub_model}\n"
        f"claude cli:     {cli_line}\n"
        f"api key:        {api_line}"
        f"{pin}"
    )


# -- Pickers ---------------------------------------------------------------

# Constants used as prompt answer values. Kept as module-level so test
# code can drive the pickers headlessly without spinning up a fake stdin.
PICKER_KEEP = "__keep__"
PICKER_CUSTOM = "__custom__"


def _prompt_choice(
    console: Console,
    title: str,
    options: list[tuple[str, str]],
    current: str,
    allow_custom: bool = False,
    custom_hint: str = "type a value",
) -> str:
    """Render a numbered picker and return the chosen value.

    Mirrors cline-2's ``ModelSelectorContent``: numbered list with the
    current value marked, plus an optional free-text "custom" entry that
    the cline-2 picker calls ``CreateCustomModelRow``. We use a plain
    numbered prompt (rich.prompt.IntPrompt doesn't expose ``choices=``,
    so we show the table, then ask for the index).

    Returns:
      * the option value (string)
      * ``PICKER_KEEP`` if the user just presses Enter on the current one
      * ``PICKER_CUSTOM`` if a custom value was entered and ``allow_custom``
      * ``ACTION_CANCEL`` if the user types ``q`` or blank
    """
    table = Table(title=title, show_header=False, header_style="bold magenta")
    table.add_column("#", justify="right", style="cyan", no_wrap=True)
    table.add_column("value", style="white")
    table.add_column("description", style="grey50")

    # Sentinel option: keep the current value (Enter on it). Always
    # offered first so the user can abort a picker without a back-out
    # key. Same role as cline-2's "Cancel" entry on the picker.
    table.add_row(
        "0",
        f"[bold cyan]keep[/bold cyan] {current}",
        "[grey50]no change[/grey50]",
    )
    for i, (value, description) in enumerate(options, start=1):
        marker = "  [bold green](current)[/bold green]" if value == current else ""
        table.add_row(str(i), f"{value}{marker}", description)

    if allow_custom:
        table.add_row("c", "[bold yellow]custom[/bold yellow] …", f"[grey50]{custom_hint}[/grey50]")

    table.add_row("q", "[bold red]cancel[/bold red]", "[grey50]leave picker without changes[/grey50]")

    console.print(table)
    while True:
        raw = Prompt.ask(
            "[bold]Pick one[/bold]",
            default="0",
            console=console,
            show_default=False,
        ).strip()
        if not raw or raw.lower() in ("q", "cancel"):
            return ACTION_CANCEL
        if raw == "0":
            return PICKER_KEEP
        if allow_custom and raw.lower() in ("c", "custom"):
            return PICKER_CUSTOM
        if raw.isdigit():
            n = int(raw)
            if 1 <= n <= len(options):
                return options[n - 1][0]
        console.print(f"[red]enter a number 0–{len(options) + (1 if allow_custom else 0)}, or q[/red]")


def pick_mode(console: Console, current: str) -> str:
    """Two-column compare picker that returns the new mode or a sentinel.

    Mirrors ``ModePickerContent`` in cline-2 (host vs proxy), with an
    ``auto`` row added on top so the operator sees the zero-setup option
    first. ``PICKER_KEEP`` means "leave config.yaml alone".
    """
    options: list[tuple[str, str]] = []
    for m in all_modes():
        d = describe_mode(m)
        options.append((m, d.summary))
    console.print()
    console.print(
        Panel(
            render_mode_compare(),
            title="[bold]Mode comparison[/bold]",
            border_style="cyan",
            expand=False,
        )
    )
    return _prompt_choice(
        console,
        title=f"Mode (current: {current})",
        options=options,
        current=current,
    )


def pick_provider(console: Console, mode: str, current: str) -> str:
    """Provider picker, narrowed to the providers the chosen mode can reach.

    Ported from cline-2's ``ProviderPickerContent``, which filters on
    ``p.mode === modeFilter`` for the same reason: the caller has just run the mode
    step, so offering a provider that mode cannot use would present a choice that
    silently fails later. ``auto`` offers everything, because it resolves at launch.

    Returns a provider id, ``PICKER_CUSTOM``, or a sentinel.
    """
    options: list[tuple[str, str]] = []
    for pid in providers_for_mode(mode):
        if pid == PROVIDER_CUSTOM:
            continue          # reached through the custom entry, not as a row
        d = PROVIDERS[pid]
        auth = "claude CLI login" if d.auth == AUTH_CLI else f"{d.key_env} in .env"
        options.append((pid, f"{d.summary}  [{auth}]"))
    return _prompt_choice(
        console,
        title=f"Provider (current: {current})",
        options=options,
        current=current,
        allow_custom=mode != MODE_CLI,
        custom_hint="an Anthropic-compatible base URL",
    )


def auth_step(console: Console, provider_id: str, st: Status) -> bool:
    """The credential step, branching on how the provider authenticates.

    cline-2 splits this three ways in ``runProviderChange``: an OAuth login screen, a
    local-CLI status screen, and a config-fields form. better-rlm has two of those --
    the `claude` CLI holds its own credential (cline's LocalCliStatusContent), and
    everything else takes an API key (cline's ProviderConfigInputContent).

    Returns True when the provider is usable afterwards.
    """
    d = describe_provider(provider_id)

    if d.auth == AUTH_CLI:
        # The CLI owns the credential; this screen reports whether it has one, the
        # way cline's local-CLI step reports on its binary. Nothing to type here.
        if not st.cli_available:
            console.print(
                f"[red]`{st.cli_path}` is not on PATH.[/red] Install Claude Code "
                "(https://claude.com/download), or choose an API provider instead."
            )
            return False
        if st.cli_logged_in is True:
            console.print("[green]`claude` CLI is logged in.[/green] Nothing to supply.")
            console.print(
                "[grey50]For a server you leave running, prefer a long-lived token: "
                "`./install.sh --auth`. An interactive login expires and cannot "
                "self-refresh.[/grey50]"
            )
            return True
        state = "not logged in" if st.cli_logged_in is False else "login state unknown"
        console.print(f"[yellow]`claude` CLI is {state}.[/yellow]")
        console.print("Fix it with either:")
        console.print("  [bold]./install.sh --auth[/bold]   long-lived token, best for a running server")
        console.print("  [bold]claude auth login[/bold]     interactive; expires, cannot self-refresh")
        return False

    # API key. Read hidden, written straight to .env, never echoed and never returned
    # to the caller -- only its fingerprint is reported. Same rule install.sh follows.
    path = env_file()
    if envfile.has_var(path, d.key_env):
        console.print(f"[green]{d.key_env} is already set[/green] in {path}.")
        keep = Prompt.ask(
            "Keep it?", console=console, choices=["y", "n"], default="y"
        ).strip().lower()
        if keep != "n":
            return True
    # password=True routes through getpass, which warns "Password input may be echoed"
    # and falls back to plain input when stdin is not a terminal -- a pipe, CI, a test.
    # Claiming hidden input there would be a lie, so say which one this is.
    hidden = sys.stdin.isatty()
    console.print(
        f"[grey50]Paste your key for {d.label}. "
        + ("Input is hidden.[/grey50]" if hidden
           else "[yellow]stdin is not a terminal: input will NOT be hidden.[/yellow][/grey50]")
    )
    try:
        key = Prompt.ask(d.key_env, console=console, password=hidden).strip()
    except (EOFError, KeyboardInterrupt):
        console.print("\n[grey50]cancelled[/grey50]")
        return False
    if not key:
        console.print("[grey50]empty, nothing written[/grey50]")
        return False
    fp = envfile.set_var(path, d.key_env, key)
    del key
    console.print(f"[green]wrote {d.key_env}[/green] to {path} ({fp}, mode 0600)")
    return True


def run_setup(console: Console, config_path: Path) -> bool:
    """Guided configuration: mode, then provider, then credential, then models.

    The order is cline-2's onboarding machine (``views/onboarding/model.ts``:
    ``mode_picker -> byo_provider -> {byo_apikey | local_cli_setup} -> model_picker``),
    because each step narrows the next: the mode decides which providers can be
    reached, and the provider decides what credential is even asked for.

    Returns True if anything was written. Cancelling any step leaves config.yaml as
    it was -- each step writes as it completes, so an abort keeps what came before
    rather than rolling back a mode the operator did choose.
    """
    console.print(Panel(render_status(load_status(config_path)),
                        title="[bold]Current configuration[/bold]", border_style="grey50"))

    st = load_status(config_path)
    wrote = False

    # 1. mode -- host vs proxy, the two-column compare
    mode = pick_mode(console, st.mode)
    if mode in (ACTION_CANCEL,):
        console.print("[grey50]setup cancelled[/grey50]")
        return wrote
    if mode not in (PICKER_KEEP,) and mode in VALID_MODES:
        wrote |= _save(config_path, {"mode": mode}, console)
    else:
        mode = st.mode

    # 2. provider, narrowed to that mode
    st = load_status(config_path)
    current_provider = provider_for_config(st.base_url, mode)
    choice = pick_provider(console, mode, current_provider)
    if choice == ACTION_CANCEL:
        console.print("[grey50]setup cancelled; mode kept[/grey50]")
        return wrote
    provider_id = current_provider
    if choice == PICKER_CUSTOM:
        url = Prompt.ask("[bold]base URL[/bold]", console=console).strip()
        if url:
            wrote |= _save(config_path, {"base_url": url}, console)
            provider_id = PROVIDER_CUSTOM
    elif choice != PICKER_KEEP:
        provider_id = choice
        wrote |= _save(config_path, {"base_url": describe_provider(choice).base_url}, console)

    # 3. credential for that provider
    st = load_status(config_path)
    ok = auth_step(console, provider_id, st)

    # 4. models
    st = load_status(config_path)
    for label, key, current in (
        ("root", "root_model", st.root_model),
        ("sub", "sub_model", st.sub_model),
    ):
        pick = pick_model(console, current, kind=label)
        if pick == ACTION_CANCEL:
            break
        if pick == PICKER_CUSTOM:
            custom = Prompt.ask(f"[bold]{label} model id[/bold]", console=console).strip()
            if custom:
                wrote |= _save(config_path, {key: custom}, console)
        elif pick != PICKER_KEEP and pick:
            wrote |= _save(config_path, {key: pick}, console)

    console.print()
    console.print(Panel(render_status(load_status(config_path)),
                        title="[bold]Configured[/bold]",
                        border_style="green" if ok else "yellow"))
    if not ok:
        console.print("[yellow]The credential step did not complete — model calls will "
                      "fail until it does.[/yellow]")
    if wrote:
        console.print("[grey50]A running server keeps its own copy: "
                      "`claude mcp restart rlm` to pick this up.[/grey50]")
    return wrote


def render_mode_compare() -> str:
    """Two-column compare block: ``render_mode_compare``-equivalent.

    Mirrors the cline-2 ``ModePickerContent`` two-pane render (label +
    summary at top, pros/cons underneath) but laid out top-to-bottom so
    it survives a 24-line terminal. Each column is a labelled block.
    """
    parts: list[str] = []
    for m in all_modes():
        d = describe_mode(m)
        parts.append(f"[bold]{d.label}[/bold]   ({m})")
        parts.append(f"  {d.summary}")
        for pro in d.pros:
            parts.append(f"    [green]+ {pro}[/green]")
        for con in d.cons:
            parts.append(f"    [red]- {con}[/red]")
        parts.append("")
    return "\n".join(parts).rstrip()


def pick_model(console: Console, current: str, kind: str = "root") -> str:
    """Model picker -- curated list of Anthropic models + custom entry.

    Curated list tracks the constants in ``better_rlm/config.py`` so the
    picker's defaults stay in sync with what the engine accepts. Custom
    entry covers a model id newer than this checkout's constants.
    """
    curated: list[tuple[str, str]] = [
        (MODEL_SONNET_5, "current default root (1M ctx)"),
        (MODEL_SONNET, "prior root (1M ctx)"),
        (MODEL_OPUS, "override for the hardest tasks (1M ctx)"),
        (MODEL_HAIKU, "cheap sub-LLM (200K ctx)"),
    ]
    return _prompt_choice(
        console,
        title=f"{kind} model (current: {current})",
        options=curated,
        current=current,
        allow_custom=True,
        custom_hint="type a model id (e.g. claude-opus-4-8) and press Enter",
    )


# -- Persistence -----------------------------------------------------------

def _save(
    config_path: Path,
    updates: dict[str, str],
    console: Console,
) -> bool:
    """Write ``updates`` to ``config.yaml`` and report what changed.

    Returns True if anything was written, so the caller can decide whether
    to show the "restart the server" reminder.
    """
    if not updates:
        return False
    changed = config_writer.write_scalars(config_path, updates)
    if changed:
        console.print(
            f"[green]wrote[/green] {', '.join(f'{k}={v!r}' for k, v in updates.items())} "
            f"to {config_path}"
        )
        console.print(
            "[yellow]note[/yellow]: a running MCP server keeps the old config until reconnect; "
            "restart it (or run [bold]claude mcp restart rlm[/bold]) to pick up the change."
        )
    else:
        console.print("[grey50]no change[/grey50]")
    return changed


# -- Slash command loop ---------------------------------------------------

SLASH_COMMANDS: list[tuple[str, str]] = [
    ("/help",            "show this list"),
    ("/status",          "show current mode / provider / model / cli login"),
    ("/mode-help",       "compare the three transport modes side-by-side"),
    ("/setup",           "guided setup: mode -> provider -> credential -> models"),
    ("/mode",            "open the mode picker (writes config.yaml)"),
    ("/provider",        "pick the provider for the current mode (writes config.yaml)"),
    ("/model",           "open the root-model picker (writes config.yaml)"),
    ("/override",        "open the override-model picker (writes config.yaml)"),
    ("/sub",             "open the sub-model picker (writes config.yaml)"),
    ("/test",            "run `uv run --extra dev pytest -q` (the verify gate)"),
    ("/test-config",     "run a focused pytest on config/auth/transport modules"),
    ("/auth-probe",      "send one tiny sub-model call to verify the auth path"),
    ("/quit",            "exit the TUI (also /exit)"),
]


HELP_TEXT = "\n".join(f"  [bold cyan]{cmd:14}[/bold cyan] {desc}" for cmd, desc in SLASH_COMMANDS)


def _run_pytest(console: Console, args: list[str]) -> int:
    """Invoke the project's pytest command and stream its output.

    Used by /test and /test-config. Honours the same `uv run --extra dev`
    invocation as the pre-push hook (CLAUDE.md) so a TUI run produces
    the same green/red as `git push`.
    """
    cmd = ["uv", "run", "--extra", "dev", "pytest", "-q", *args]
    console.print(f"[grey50]$ {' '.join(cmd)}[/grey50]")
    try:
        return subprocess.call(cmd, cwd=str(PKG_ROOT))
    except FileNotFoundError:
        console.print(
            "[red]`uv` not found on PATH[/red] -- pytest cannot run. "
            "Install uv (https://docs.astral.sh/uv/) or run pytest directly "
            "from .venv_sh."
        )
        return 127


def _run_auth_probe(console: Console) -> None:
    """Send one tiny sub-model call to verify the auth path.

    Mirrors ``server._auth_probe_line`` -- same payload, same threshold,
    so /auth-probe and the MCP startup probe are interchangeable. Calls
    directly into ``subquery.sub_query`` so it doesn't have to start the
    MCP server.

    Reads the checkout's own config.yaml: ``load_config`` takes no path, so a TUI
    started with ``--config`` elsewhere probes against THIS checkout's config, not
    that file. Spends one sub-model call.
    """
    from . import models
    from .config import load_config
    from .subquery import sub_query

    try:
        # Inside the guard: models.select resolves the auth mode, which runs
        # auth.require_anthropic -- that raises NotImplementedError for any provider
        # but anthropic, the exact value /status flags as UNSUPPORTED.
        cfg = load_config()
        target = models.select(cfg, models.Role.SUB)
        res = sub_query(cfg, "Reply with exactly: ok", target, max_tokens=16)
    except Exception as exc:                       # noqa: BLE001
        console.print(f"[red]auth probe FAILED[/red]: {type(exc).__name__}: {exc}")
        return
    if res.error:
        console.print(f"[red]auth probe FAILED[/red]: {res.error}")
        return
    console.print(f"[green]auth probe ok[/green]: sub-model replied {res.answer.strip()[:20]!r}")


#: Exit status of the last dispatched command, for ``--one-shot``. The dispatcher's
#: own return value is the quit flag (False = leave the REPL), so it cannot also carry a
#: status -- and README advertises ``--one-shot`` as CI-friendly, where a /test that
#: exits 0 on a red suite is worse than no gate at all.
LAST_EXIT_CODE = 0


def _dispatch(
    line: str,
    console: Console,
    config_path: Path,
) -> bool:
    """Run one slash command.

    Returns False when the user asked to quit, so the REPL can break out. The command's
    exit status goes to the module-level LAST_EXIT_CODE, which ``main`` returns for
    ``--one-shot``.
    """
    global LAST_EXIT_CODE
    LAST_EXIT_CODE = 0
    cmd = line.strip()
    if not cmd:
        return True
    if cmd in ("/quit", "/exit"):
        return False

    if cmd == "/help":
        console.print(Panel(HELP_TEXT, title="[bold]better-rlm TUI commands[/bold]", border_style="cyan"))
        return True

    if cmd == "/status":
        console.print(Panel(render_status(load_status(config_path)), title="[bold]current configuration[/bold]", border_style="cyan"))
        return True

    if cmd == "/mode-help":
        console.print(Panel(render_mode_compare(), title="[bold]mode comparison[/bold]", border_style="cyan"))
        return True

    if cmd == "/mode":
        st = load_status(config_path)
        choice = pick_mode(console, st.mode)
        if choice == PICKER_KEEP or choice == ACTION_CANCEL:
            console.print("[grey50]no change[/grey50]")
        elif choice in VALID_MODES:
            _save(config_path, {"mode": choice}, console)
        return True

    if cmd == "/setup":
        run_setup(console, config_path)
        return True

    if cmd == "/provider":
        st = load_status(config_path)
        current = provider_for_config(st.base_url, st.mode)
        choice = pick_provider(console, st.mode, current)
        if choice in (PICKER_KEEP, ACTION_CANCEL):
            console.print("[grey50]no change[/grey50]")
            return True
        if choice == PICKER_CUSTOM:
            url = Prompt.ask("[bold]base URL[/bold]", console=console).strip()
            if not url:
                console.print("[grey50]empty value, no change[/grey50]")
                return True
            provider_id = PROVIDER_CUSTOM
        else:
            provider_id = choice
            url = describe_provider(choice).base_url
        _save(config_path, {"base_url": url}, console)
        auth_step(console, provider_id, load_status(config_path))
        return True

    if cmd in ("/model", "/override", "/sub"):
        st = load_status(config_path)
        if cmd == "/model":
            current, key = st.root_model, "root_model"
            kind = "root"
        elif cmd == "/override":
            current, key = st.root_model_override, "root_model_override"
            kind = "override"
        else:
            current, key = st.sub_model, "sub_model"
            kind = "sub"
        choice = pick_model(console, current, kind=kind)
        if choice == PICKER_KEEP or choice == ACTION_CANCEL:
            console.print("[grey50]no change[/grey50]")
        elif choice == PICKER_CUSTOM:
            custom = Prompt.ask(f"[bold]{kind} model id[/bold]", console=console).strip()
            if custom:
                _save(config_path, {key: custom}, console)
            else:
                console.print("[grey50]empty value, no change[/grey50]")
        elif choice:
            _save(config_path, {key: choice}, console)
        return True

    if cmd == "/test":
        rc = _run_pytest(console, [])
        LAST_EXIT_CODE = rc
        console.print(f"[grey50]pytest exited with rc={rc}[/grey50]")
        return True

    if cmd == "/test-config":
        rc = _run_pytest(console, ["tests/test_config.py", "tests/test_auth.py", "tests/test_transport.py"])
        LAST_EXIT_CODE = rc
        console.print(f"[grey50]pytest exited with rc={rc}[/grey50]")
        return True

    if cmd == "/auth-probe":
        _run_auth_probe(console)
        return True

    LAST_EXIT_CODE = 2      # a typo'd --one-shot must not look like success either
    console.print(f"[red]unknown command[/red]: {cmd!r}. Type [bold]/help[/bold] for the list.")
    return True


def _dispatch_guarded(line: str, console: Console, config_path: Path) -> bool:
    """Run one command, reporting a handler exception instead of ending the session.

    Every handler reaches the operator through here, so this is the one place that has
    to catch: without it an OSError from a config write, or require_anthropic refusing
    a provider, exits the REPL with a traceback mid-session. Returns True (keep going)
    on failure and leaves LAST_EXIT_CODE nonzero so --one-shot still reports it.
    """
    global LAST_EXIT_CODE
    try:
        return _dispatch(line, console, config_path)
    except Exception as exc:                       # noqa: BLE001
        LAST_EXIT_CODE = 1
        console.print(f"[red]command FAILED[/red]: {type(exc).__name__}: {exc}")
        return True


def run_repl(
    config_path: Path | None = None,
    console: Console | None = None,
    setup: bool = False,
) -> int:
    """Main REPL -- reads slash commands from stdin until /quit.

    Headless: every line is read with input(); non-interactive shells
    (CI, scripts) feed the lines and the loop exits on EOF or /quit.
    Interactive: a prompt is drawn for each line via rich.prompt.Prompt.

    ``setup`` runs the guided flow first, which is what a bare ``better-rlm`` does:
    an operator who types the command with no arguments wants to configure the
    thing, not to be handed a prompt and a list of slash commands. Explicit rather
    than inferred from isatty(), so the behaviour is identical in a script.
    """
    cfg_path = config_path or config_file()
    console = console or Console()
    if setup:
        try:
            run_setup(console, cfg_path)
        except (EOFError, KeyboardInterrupt):
            console.print("\n[grey50]setup cancelled[/grey50]")
        console.print()
    console.print(
        Panel(
            "[bold]better-rlm TUI[/bold]\n"
            "Type [bold cyan]/setup[/bold cyan] to re-run guided configuration, "
            "[bold cyan]/help[/bold cyan] for all commands, [bold cyan]/quit[/bold cyan] to exit.",
            border_style="green",
        )
    )
    while True:
        try:
            line = Prompt.ask("[bold green]rlm[/bold green]", console=console, default="").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[grey50]bye[/grey50]")
            return 0
        if not _dispatch_guarded(line, console, cfg_path):
            console.print("[grey50]bye[/grey50]")
            return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry: ``python -m better_rlm.cli [--config PATH] [--one-shot CMD]``.

    ``--one-shot`` runs ONE slash command and exits with that command's status
    (pytest's rc for /test and /test-config, 2 for an unknown command, else 0) so CI
    can gate on it. Without it, run_repl takes over stdin.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    args_were_given = bool(args)
    config_path: Path | None = None
    one_shot: str | None = None
    while args:
        a = args.pop(0)
        if a in ("-h", "--help"):
            print(__doc__ or "")
            return 0
        if a == "--config" and args:
            config_path = Path(args.pop(0)).expanduser().resolve()
            continue
        if a == "--one-shot" and args:
            one_shot = args.pop(0)
            continue
        print(f"unknown flag: {a}", file=sys.stderr)
        return 2

    console = Console()
    cfg_path = config_path or config_file()
    if one_shot is not None:
        _dispatch_guarded(one_shot, console, cfg_path)
        return LAST_EXIT_CODE
    # A bare invocation opens in configuration mode. Passing --config or any other
    # flag means the caller is driving it deliberately, so they get the plain REPL.
    return run_repl(cfg_path, console=console, setup=not args_were_given)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

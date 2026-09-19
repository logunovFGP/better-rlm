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
import dataclasses
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
    MODEL_SONNET_5,
    MODEL_HAIKU,
    PKG_ROOT,
    config_file,
    env_file,
    load_config,
)
from . import envfile, picker
from .searchable_list import SearchableItem
from .describe import (
    AUTH_CLI,
    MODE_API,
    MODE_AUTO,
    MODE_CLI,
    PROVIDER_CUSTOM,
    PROVIDERS,
    VALID_MODES,
    all_modes,
    describe_mode,
    describe_model,
    describe_provider,
    models_for,
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
    key_env: str = ""           # the ACTIVE provider's key variable ("" = CLI login)
    config_written: bool = True  # False = no config.yaml on disk; every value is a default

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
    config_written = cfg_path.exists()

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

    # The ACTIVE provider's variable, not a fixed one: a MiniMax setup with no
    # ANTHROPIC_API_KEY is configured, and saying "MISSING" there would be wrong.
    key_var = describe_provider(provider_for_config(base_url, mode)).key_env
    has_api_key = bool(key_var and os.getenv(key_var))

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
        key_env=key_var,
        config_written=config_written,
    )


def cli_will_win(st: Status) -> bool:
    """True when the resolved transport will be the `claude` CLI, so base_url is dead.

    ``auto`` is the subtle one and the case this originally missed: it prefers the CLI
    on PATH PRESENCE ALONE -- ``auth.claude_cli_available`` only calls shutil.which,
    and login is left to surface as an auth error at call time. So a machine with the
    CLI installed silently ignores a configured endpoint under auto, which looks
    configured and is not.
    """
    if st.mode == MODE_CLI:
        return True
    if st.mode == MODE_AUTO:
        return st.cli_available
    return False


def _provider_line(st: Status) -> str:
    """Which provider the next server start will use, named rather than shown as a URL.

    Flags the combinations that silently do nothing: a base_url set while the resolved
    transport is the `claude` CLI, which reaches Anthropic whatever this says.
    """
    pid = provider_for_config(st.base_url, st.mode)
    d = describe_provider(pid)
    where = st.base_url or ("the `claude` CLI" if d.auth == AUTH_CLI else "api.anthropic.com")
    if st.base_url and cli_will_win(st):
        why = ("mode=claude-cli" if st.mode == MODE_CLI
               else f"mode=auto prefers the `{st.cli_path}` CLI, which is on PATH")
        return f"{d.label} — {where}   IGNORED: {why}"
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

    # Name the variable the active provider actually reads, so the operator knows
    # which one to set rather than being told a fixed name that may not apply.
    if not st.key_env:
        api_line = "not needed — the `claude` CLI holds the credential"
    else:
        api_line = f"{st.key_env}={'set' if st.has_api_key else 'MISSING'}"

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
#: Not a second literal that happens to match: _select maps picker.CUSTOM onto
#: this, and a synthetic "custom" row in an options list is keyed with it. Two
#: independent strings agreeing by coincidence is a bug waiting for an edit.
PICKER_CUSTOM = picker.CUSTOM


def _select(console: Console, title: str, options: list[tuple[str, str]],
            current: str, allow_custom: bool = False, custom_hint: str = "",
            sections: dict[str, str] | None = None,
            tags: dict[str, str] | None = None) -> str:
    """One selection, through the live picker on a terminal and the numbered prompt
    off one.

    Every picker goes through here so the two paths cannot drift: a terminal gets
    cline's interaction (arrow keys, a highlight, type-to-filter), and a pipe -- the
    suite, CI, ``--one-shot`` -- gets the prompt it can actually answer. Returns the
    same sentinels either way, so no caller has to know which ran.
    """
    if not picker.interactive():
        return _prompt_choice(console, title, options, current,
                              allow_custom=allow_custom, custom_hint=custom_hint)
    items = [
        SearchableItem(key=value, label=value, detail=detail,
                       section=(sections or {}).get(value, ""),
                       tag=(tags or {}).get(value, ""))
        for value, detail in options
    ]
    res = picker.choose(console, title, items, current_key=current,
                        custom_hint=custom_hint if allow_custom else "")
    if res.key == picker.CANCEL:
        return ACTION_CANCEL
    if res.key == picker.CUSTOM:
        return PICKER_CUSTOM
    # Upstream opens on the current value and Enter alone is a no-op; keep that,
    # so a picker opened by accident changes nothing.
    return PICKER_KEEP if res.key == current else res.key


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
        try:
            raw = Prompt.ask(
                "[bold]Pick one[/bold]",
                default="0",
                console=console,
                show_default=False,
            ).strip()
        except EOFError:
            # A pipe that has run out is a cancel, not a crash. This never fired
            # while the wizard was gated on a TTY; dropping that gate made every
            # screen reachable from a pipe, and the last screen then ended the run
            # with "command FAILED: EOFError" instead of leaving cleanly.
            console.print()
            return ACTION_CANCEL
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


def select_cards(console: Console, title: str, subtitle: str,
                 rows: list[tuple[str, str, str, str]], current: str = "") -> str:
    """One selection from cline's card screen, on a terminal or off one.

    ``rows`` is (key, label, detail, icon). The card screen had no headless twin,
    which is why the first-run flow used to be gated on picker.interactive() --
    in violation of the rule it cites. A pipe gets the numbered prompt instead, and
    both return the same sentinels, so no caller branches.
    """
    if not picker.interactive():
        choice = _prompt_choice(console, title,
                                [(key, f"{label} - {detail}") for key, label, detail, _ in rows],
                                current)
        # Enter on a screen with nothing chosen yet means the first card, which is
        # what the live picker does -- it opens on row one.
        if choice == PICKER_KEEP:
            return current or (rows[0][0] if rows else ACTION_CANCEL)
        return choice
    items = [SearchableItem(key=key, label=label, detail=detail, tag=icon)
             for key, label, detail, icon in rows]
    res = picker.choose_cards(console, title, subtitle, items)
    return ACTION_CANCEL if res.key == picker.CANCEL else res.key


def prospective_config(config_path: Path, mode: str, base_url: str):
    """The Config a set of about-to-be-written settings would produce.

    The probe has to test what the wizard is about to write, not what is on disk.
    _run_auth_probe used to call load_config() with no argument and so probed the
    DEFAULT config even under --config other.yaml -- reporting on a file nobody was
    editing. Unrelated tunables (timeouts, ledger paths) still come from the loaded
    config, because those are not what is being decided here.
    """
    st = load_status(config_path)
    return dataclasses.replace(load_config(), mode=mode, base_url=base_url,
                               cli_path=st.cli_path)


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
    return _select(console, f"Mode (current: {current})", options, current)


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
    sections = {pid: ("proxy — the `claude` CLI runs it" if PROVIDERS[pid].mode == MODE_CLI
                      else "host — better-rlm runs it")
                for pid in providers_for_mode(mode) if pid != PROVIDER_CUSTOM}
    tags = {pid: (PROVIDERS[pid].key_env or "no key needed")
            for pid in providers_for_mode(mode) if pid != PROVIDER_CUSTOM}
    return _select(console, f"Provider (current: {current})", options, current,
                   allow_custom=mode != MODE_CLI,
                   custom_hint="an Anthropic-compatible base URL",
                   sections=sections, tags=tags)


def env_for_config(config_path: Path) -> Path:
    """Where a credential belongs, for the config file being edited.

    The default config gets the default .env, which knows about the checkout vs
    wheel split. A `--config /tmp/other.yaml` run gets /tmp/.env instead: a config
    file and its credential are a pair, and writing the key to the repo's .env while
    editing someone else's config would silently overwrite a live credential.
    """
    try:
        if config_path.resolve() == config_file().resolve():
            return env_file()
    except OSError:
        pass
    return config_path.parent / ".env"


def load_sidecar_env(config_path: Path) -> Path | None:
    """Load the .env that belongs to the config being edited. Returns it, or None.

    config.py runs ``load_dotenv(env_file())`` at IMPORT, which resolves the DEFAULT
    .env and knows nothing about ``--config``. ``env_for_config`` writes beside the
    config named on the command line. The two disagreed, so under
    ``--config elsewhere`` the wizard wrote a key to a file nothing ever read back:
    setup reported Ready, and the very next launch reported the key MISSING and
    re-ran the whole wizard.

    ``override=True`` because a config file and its credential are a pair -- the key
    beside the file being edited has to beat whatever the default .env put into
    os.environ at import. The cost is that an exported shell variable also loses,
    which is the right trade for a flag whose whole purpose is "edit THAT config":
    testing config A while silently authenticating as config B is the confusion this
    pairing exists to prevent.
    """
    path = env_for_config(config_path)
    if path == env_file() or not path.is_file():
        return None
    from dotenv import load_dotenv

    load_dotenv(path, override=True)
    return path


def warn_unknown_model(console: Console, model_id: str) -> None:
    """Say what is NOT known about a hand-typed model id.

    A custom id is deliberately accepted without checking it against a list -- the
    endpoint is the authority, not us. But accepting it silently hides two
    consequences the operator cannot see: an id the engine has no window for is
    assumed to be the default (8x smaller than a 1M model, so it chunks far
    earlier), and an id with no published rate makes every cost line read unpriced.
    """
    from rlm.utils.token_utils import DEFAULT_CONTEXT_LIMIT, get_context_limit

    from .config import COST_PER_MTOK

    if describe_model(model_id) is not None:
        return
    notes = []
    if get_context_limit(model_id) == DEFAULT_CONTEXT_LIMIT:
        notes.append(f"its context window is unknown, so {DEFAULT_CONTEXT_LIMIT:,} "
                     "tokens is assumed")
    if model_id not in COST_PER_MTOK:
        notes.append("it has no published rate here, so cost reports as unpriced")
    if notes:
        console.print(f"[yellow]{model_id} is not in the catalogue: "
                      + "; ".join(notes) + ".[/yellow]")


def auth_step(console: Console, provider_id: str, st: Status,
              env_path: Path | None = None) -> bool:
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
    path = env_path or env_file()
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
    console.print(
        f"[grey50]Paste your key for {d.label}. "
        "Shown masked — first and last two characters, so you can tell a good paste "
        "from an empty clipboard.[/grey50]"
    )
    key = picker.read_secret(console, d.key_env)
    if not key:
        console.print("[grey50]empty, nothing written[/grey50]")
        return False
    fp = envfile.set_var(path, d.key_env, key)
    # config.py ran load_dotenv at import, so the write above is invisible to this
    # process: auth.api_key_for reads os.getenv, and without this line /status says
    # MISSING and the connectivity probe reports a good key as rejected.
    os.environ[d.key_env] = key
    del key
    console.print(f"[green]wrote {d.key_env}[/green] to {path} ({fp}, mode 0600)")
    return True


def _warn_if_endpoint_is_dead(console: Console, st: Status) -> None:
    """Say it at the moment of writing, not only on the next /status.

    An operator who picks MiniMax and is told nothing has every reason to think it
    took effect.
    """
    if st.base_url and cli_will_win(st):
        why = ("mode is claude-cli" if st.mode == MODE_CLI
               else f"mode is auto and the `{st.cli_path}` CLI is on PATH, which auto prefers")
        console.print(
            f"[yellow]This endpoint will be ignored: {why}, and the CLI talks to "
            f"Anthropic. Run [bold]/mode[/bold] and pick `api` to use it.[/yellow]"
        )


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


def pick_model(console: Console, current: str, kind: str = "root",
               provider: str = "") -> str:
    """Model picker -- the rows the CONFIGURED endpoint actually serves.

    This used to be one hardcoded list of four Anthropic ids whatever the endpoint
    was, so a MiniMax install was offered claude-sonnet-5 and, taking it, ran with a
    Claude id pointed at api.minimax.io. The list now comes from describe.MODELS via
    the provider, which is the reverse lookup of base_url + mode.

    A provider with no catalogue -- a custom endpoint, or an empty argument -- gets
    the custom entry alone. Offering it another vendor's ids would be the same bug
    in a different place.
    """
    rows = [(m.id, m.detail()) for m in models_for(provider)]
    return _select(console, f"{kind} model (current: {current})", rows, current,
                   allow_custom=True, custom_hint="type a model id")


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
            "start a new Claude Code session to pick up the change."
        )
    else:
        console.print("[grey50]no change[/grey50]")
    return changed


# -- Slash command loop ---------------------------------------------------

SLASH_COMMANDS: list[tuple[str, str]] = [
    ("/help",            "show this list"),
    ("/status",          "show current mode / provider / model / cli login"),
    ("/mode-help",       "compare the three transport modes side-by-side"),
    ("/setup",           "guided setup: provider -> transport -> credentials -> models"),
    ("/mode",            "open the mode picker (writes config.yaml)"),
    ("/provider",        "pick the provider for the current mode (writes config.yaml)"),
    ("/model",           "open the root-model picker (writes config.yaml)"),
    ("/override",        "open the override-model picker (writes config.yaml)"),
    ("/sub",             "open the sub-model picker (writes config.yaml)"),
    ("/test",            "run `uv run --extra dev pytest -q` (the verify gate)"),
    ("/test-config",     "run a focused pytest on config/auth/transport modules"),
    ("/auth-probe",      "check the configured endpoint can be reached"),
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


def _run_auth_probe(console: Console, config_path: Path | None = None) -> None:
    """Can this configuration reach its endpoint? The same check the wizard runs.

    Was a real sub-model call through subquery against ``load_config()`` -- which
    takes no path, so a TUI started with ``--config`` elsewhere probed THIS
    checkout's config rather than the file being edited. It now shares probe.py with
    the setup wizard, so the menu row and the wizard cannot disagree about whether
    an endpoint works, and the proxy path costs nothing at all.
    """
    from . import describe as _d
    from . import probe

    path = config_path or config_file()
    st = load_status(path)
    cfg = prospective_config(path, st.mode, st.base_url)
    pid = _d.provider_for_config(st.base_url, st.mode)
    res = probe.probe_endpoint(cfg, st.sub_model or _d.role_defaults(pid)[2])
    if res.ok:
        console.print(f"[green]auth probe ok[/green]: {res.detail} ({res.where})")
        return
    console.print(f"[red]auth probe FAILED[/red] ({res.code}): {res.detail}")
    if res.fix:
        console.print(f"[grey50]{res.fix}[/grey50]")
    return


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
        from . import onboard
        onboard.run(console, config_path)
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
        previous = st.base_url
        _save(config_path, {"base_url": url}, console)
        st = load_status(config_path)
        _warn_if_endpoint_is_dead(console, st)
        if not auth_step(console, provider_id, st, env_for_config(config_path)):
            # The endpoint moved but no credential exists for it, and api_key_for
            # deliberately will not fall back to another provider's key -- so leaving
            # this written turns a working install into one where every model call
            # fails. Put the old endpoint back and say so.
            _save(config_path, {"base_url": previous}, console)
            console.print("[yellow]No credential supplied, so the endpoint was left "
                          "as it was.[/yellow]")
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
        choice = pick_model(console, current, kind=kind,
                            provider=provider_for_config(st.base_url, st.mode))
        if choice == PICKER_KEEP or choice == ACTION_CANCEL:
            console.print("[grey50]no change[/grey50]")
        elif choice == PICKER_CUSTOM:
            custom = Prompt.ask(f"[bold]{kind} model id[/bold]", console=console).strip()
            if custom:
                warn_unknown_model(console, custom)
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
        _run_auth_probe(console, config_path)
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


@dataclass(frozen=True)
class MenuItem:
    """One row of the main menu. Mirrors cline-2's ``MenuOption``
    (``views/onboarding/model.ts``): a label, a detail line explaining what it does
    right now, and the slash command it stands for."""

    key: str
    label: str
    detail: str
    command: str
    section: str = "Actions"


def build_menu(st: Status) -> list[MenuItem]:
    """What to offer, given the state the configuration is actually in.

    cline-2's machine opens on ``menu``, not on the mode picker, and filters the rows
    by state (``getMainMenuOptions``). The same idea, with one addition that matters
    more here than there: anything currently BROKEN leads. An operator who lands on a
    config that cannot make a model call should be offered the fix first, not asked to
    know which slash command repairs it.
    """
    items: list[MenuItem] = []

    # Problems first, most-blocking first.
    needs_key = bool(st.key_env) and not st.has_api_key
    if st.mode != MODE_CLI and needs_key:
        provider = describe_provider(provider_for_config(st.base_url, st.mode)).label
        items.append(MenuItem(
            "", f"Supply your {provider} key",
            f"{st.key_env} is not set — every model-backed tool fails until it is",
            "/provider-key", "Needs attention",
        ))
    if st.base_url and cli_will_win(st):
        items.append(MenuItem(
            "", "Fix the mode that is ignoring your endpoint",
            f"mode={st.mode} routes to the `{st.cli_path}` CLI, so your endpoint is unused",
            "/mode", "Needs attention",
        ))
    if st.mode == MODE_CLI and st.cli_logged_in is False:
        items.append(MenuItem(
            "", "Sign the `claude` CLI in",
            "mode=claude-cli, but the CLI has no login — model calls will fail",
            "/provider-key", "Needs attention",
        ))

    items += [
        MenuItem("", "Run guided setup", "provider, transport, credentials, connection test, models", "/setup"),
        MenuItem("", "Change mode", f"currently {st.mode}", "/mode"),
        MenuItem("", "Change provider",
                 f"currently {describe_provider(provider_for_config(st.base_url, st.mode)).label}",
                 "/provider"),
        MenuItem("", "Change models", f"root {st.root_model}, sub {st.sub_model}", "/model"),
        MenuItem("", "Test the connection", "one tiny model call that proves auth works", "/auth-probe"),
        MenuItem("", "Run the verify gate", "the test suite", "/test"),
    ]
    return [dataclasses.replace(it, key=str(n)) for n, it in enumerate(items, 1)]


def menu_items_to_searchable(items: list[MenuItem]) -> list[SearchableItem]:
    """Menu rows as picker rows, with the problems grouped apart from the routine
    actions so a broken config reads as a short list of fixes rather than a wall."""
    out: list[SearchableItem] = []
    for it in items:
        out.append(SearchableItem(key=it.command + "#" + it.label, label=it.label,
                                  detail=it.detail, section=it.section, payload=it))
    return out


def render_menu(items: list[MenuItem]) -> Table:
    """The headless rendering. The live picker draws its own."""
    table = Table(title="What would you like to do?", show_header=False)
    table.add_column("#", justify="right", style="cyan", no_wrap=True)
    table.add_column("action", style="white")
    table.add_column("detail", style="grey50")
    for it in items:
        table.add_row(it.key, it.label, it.detail)
    table.add_row("s", "Slash commands", "type any /command directly — /help lists them")
    table.add_row("q", "Quit", "")
    return table


MODE_CARDS: dict[str, tuple[str, str, str]] = {
    # mode -> (icon, label, detail). The labels are describe.py's, which are
    # cline's host/proxy terminology, so the same words appear on this screen and
    # in /mode-help rather than two descriptions of one thing.
    MODE_CLI: ("✦", "OAuth — proxy mode",
               "Reuse your `claude` CLI login. No API key, no per-token cost, "
               "subject to subscription limits."),
    MODE_API: ("⚙", "API key — host mode",
               "Talk to the endpoint directly. Higher limits, per-token cost, "
               "needs a key in .env."),
}


def needs_onboarding(st: Status) -> bool:
    """Whether to run setup rather than the maintenance menu.

    Two reasons, and either one is enough.

    **Nothing has been configured yet.** This is cline's rule verbatim --
    ``if (!isProviderConfigured(props.config)) return "onboarding"``. Without it a
    brand-new install skipped setup entirely whenever the `claude` CLI happened to
    be logged in: no config.yaml means `mode` defaults to `auto`, auto accepts a
    signed-in CLI, so the gate answered "already fine" and went straight to the
    maintenance menu. Every value on that screen was a default the operator had
    never chosen. Having Claude Code installed is not the same as having configured
    this tool.

    **Or the configuration cannot make a model call.** This part is not cline's and
    is kept: a config.yaml naming a provider whose key is missing is not configured
    either, whatever the file says.
    """
    if not st.config_written:
        return True
    if st.mode == MODE_CLI:
        return st.cli_logged_in is not True
    if st.key_env and st.has_api_key:
        return False
    # auto can fall back to the CLI, so a signed-in CLI counts.
    return not (st.mode == MODE_AUTO and st.cli_available and st.cli_logged_in is True)


def run_menu(config_path: Path | None = None, console: Console | None = None,
             menu_only: bool = False) -> int:
    """The default surface: show the configuration, offer what to do about it, repeat.

    A bare ``better-rlm`` used to drop straight into the mode picker, which assumes
    the operator wants to walk all four steps. Most of the time they want one thing --
    usually the thing that is currently broken -- and the menu names it.
    """
    cfg_path = config_path or config_file()
    console = console or Console()
    while True:
        st = load_status(cfg_path)
        # No picker.interactive() gate any more: every screen in the wizard
        # has a headless path, so a pipe gets asked the same questions.
        if not menu_only and needs_onboarding(st):
            # Nothing here can make a model call yet, so ask rather than presenting a
            # maintenance menu to someone who has not configured anything.
            #
            # cline's root.tsx is the shape being copied, and it does NOT fall through:
            #
            #     onComplete -> setAppView("home")
            #     onExit     -> exitCline()
            #
            # Neither branch lands on a list. Falling through did, unconditionally,
            # which is why every first run ended on the maintenance menu no matter
            # what the operator chose -- including Esc, where the menu WAS the
            # response to "I want out". `better-rlm` is the setup surface; the menu
            # is `better-rlm config`, asked for by name.
            from . import onboard
            done = onboard.run(console, cfg_path)
            st = load_status(cfg_path)
            if not done:
                console.print("[grey50]setup cancelled[/grey50]")
                return 0
            console.print()
            console.print(Panel(render_status(st), title="[bold]configured[/bold]",
                                border_style="green"))
            console.print("[grey50]`better-rlm config` to change any of it[/grey50]")
            return 0
        console.print()
        console.print(Panel(render_status(st), title="[bold]better-rlm[/bold]",
                            border_style="green"))
        items = build_menu(st)

        if picker.interactive():
            rows = menu_items_to_searchable(items)
            rows.append(SearchableItem(key="__repl__", label="Slash commands",
                                       detail="type any /command directly",
                                       section="Other"))
            rows.append(SearchableItem(key="__quit__", label="Quit", section="Other"))
            res = picker.choose(console, "What would you like to do?", rows)
            if res.key in (picker.CANCEL, "__quit__"):
                console.print("[grey50]bye[/grey50]")
                return 0
            if res.key == "__repl__":
                return run_repl(cfg_path, console=console)
            chosen = res.item.payload if res.item else None
            if chosen is None:
                continue
            command = chosen.command
        else:
            console.print(render_menu(items))
            try:
                choice = Prompt.ask("[bold green]Pick one[/bold green]", console=console,
                                    default="q").strip()
            except EOFError:
                console.print("\n[grey50]bye[/grey50]")
                return 0
            if choice.lower() in ("q", "quit", "exit"):
                console.print("[grey50]bye[/grey50]")
                return 0
            if choice.lower() == "s":
                return run_repl(cfg_path, console=console)
            if choice.startswith("/"):
                if not _dispatch_guarded(choice, console, cfg_path):
                    return 0
                continue
            match = next((it for it in items if it.key == choice), None)
            if match is None:
                console.print(f"[yellow]not an option: {choice!r}[/yellow]")
                continue
            command = match.command

        if command == "/provider-key":
            # The credential alone, without re-walking the provider picker.
            pid = provider_for_config(st.base_url, st.mode)
            auth_step(console, pid, st, env_for_config(cfg_path))
            continue
        if not _dispatch_guarded(command, console, cfg_path):
            return 0


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
            from . import onboard
            onboard.run(console, cfg_path)
        except EOFError:
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
        except EOFError:
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
    config_path: Path | None = None
    one_shot: str | None = None
    menu_only = False
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
        if a == "--menu":
            # Set by `better-rlm config`. Bare `better-rlm` is the setup surface and
            # gates on needs_onboarding; naming `config` asks for the menu itself,
            # configured or not.
            menu_only = True
            continue
        print(f"unknown flag: {a}", file=sys.stderr)
        return 2

    console = Console()
    cfg_path = config_path or config_file()
    load_sidecar_env(cfg_path)
    try:
        if one_shot is not None:
            _dispatch_guarded(one_shot, console, cfg_path)
            return LAST_EXIT_CODE
        # The menu is the surface, whether or not --config pointed somewhere else:
        # that flag only says WHICH config to edit. Only --one-shot means "do this
        # one thing and exit", and it is handled above.
        return run_menu(cfg_path, console=console, menu_only=menu_only)
    except KeyboardInterrupt:
        # The single handler, and the reason no picker catches this. Every screen
        # used to turn Ctrl+C into the CANCEL that Esc returns, so its caller looped
        # and drew the next menu -- Ctrl+C read as "go back", and from a nested
        # screen no key left the program at all.
        #
        # Here rather than in cli.main because `python -m better_rlm.tui` reaches
        # this function without passing through cli. The newline is because raw mode
        # echoes nothing, so the cursor sits wherever the picker left it. 130 is the
        # conventional status for death by SIGINT.
        console.print()
        console.print("[grey50]bye[/grey50]")
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

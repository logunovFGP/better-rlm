"""The setup wizard: a port of cline-2's onboarding flow, two vendors only.

cline drives its onboarding from views/onboarding/ -- model.ts holds a flat union
of step names, controller.ts holds the state, keyboard.ts owns Escape and the back
transitions, view.tsx is an if-chain that picks one presentational screen per step.
This module is all four, because our screens draw through picker.py and tui._select
rather than owning any layout of their own.

The order is cline's, reduced to the two vendors this fork supports:

    vendor -> transport (host/proxy) -> credentials -> connectivity -> models

Steps cline has that cannot happen here, and why: oauth_pending and device_code
(no device flow -- the claude CLI owns its own login, which is our CLI_CHECK, its
local_cli_setup); byo_provider (with two vendors the vendor screen IS that list);
cline_model and cline_pass_subscription (cline-account specific); thinking_level
(no corresponding config key).

Two things here are NOT cline's, both deliberate:

  * the connectivity probe. cline removed its API-path pre-flight on purpose. Ours
    runs on both paths so a wrong URL or a mistyped key is caught on the screen
    that produced it. See probe.py.
  * three model screens instead of one. The engine needs a root, an override and a
    sub-model; each is a different job, so each is asked with its job explained
    rather than derived behind the operator's back.

Nothing reaches config.yaml until the last step. A cancel anywhere leaves the file
byte-identical, so there is no rollback to get wrong.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from . import describe, envfile, picker, probe
from .describe import (
    MODE_CLI,
    PROVIDER_CUSTOM,
    describe_provider,
    describe_vendor,
    modes_for_vendor,
    provider_for,
)

#: A pipe that answers the same way forever, plus back-navigation, is a loop. The
#: React original never needed this because a human eventually stops.
MAX_TRANSITIONS = 100


class Step(str, Enum):
    VENDOR = "vendor"
    MODE = "mode"
    CLI_CHECK = "cli_check"
    CREDENTIALS = "credentials"
    PROBE = "probe"
    MODELS = "models"
    DONE = "done"
    EXIT = "exit"


TERMINAL = frozenset({Step.DONE, Step.EXIT})

#: (config key, screen title, what this model is actually for). Three screens
#: rather than one derivation: someone choosing where their money goes should be
#: told what each choice does, in the order the engine uses them.
ROLES: tuple[tuple[str, str, str], ...] = (
    ("root_model", "Root model - the orchestrator",
     "Reads your question, decides how to split an oversized context, and writes "
     "the final answer. Most of the output quality comes from this one."),
    ("root_model_override", "Override model - for the hardest tasks",
     "Used only when a query explicitly asks for the top tier (rlm_query with "
     "model set to opus). Set it to the strongest model you are willing to pay "
     "for; nothing else reaches it."),
    ("sub_model", "Sub-model - the chunk worker",
     "Runs once per chunk, so it makes the large majority of the calls and most "
     "of the spend. Cheap and fast matters more than clever here."),
)


@dataclass(frozen=True)
class Wizard:
    """Everything the flow has decided so far.

    There is no api_key field, on purpose. A credential goes prompt -> .env ->
    os.environ and the local is dropped; a frozen dataclass holding one is a single
    uncaught exception away from printing it in a traceback.
    """

    step: Step = Step.VENDOR
    history: tuple[Step, ...] = ()
    vendor: str = ""
    mode: str = ""
    provider: str = ""
    base_url: str = ""
    key_env: str = ""
    role_index: int = 0
    models: tuple[tuple[str, str], ...] = ()
    probe_note: str = ""
    error: str = ""


def goto(w: Wizard, step: Step, **fields) -> Wizard:
    """Advance, remembering where we came from."""
    return replace(w, step=step, history=w.history + (w.step,), error="", **fields)


def back(w: Wizard, **fields) -> Wizard:
    """Escape: one screen back, or out.

    Pops a pushed history rather than consulting a static table, because two edges
    are path-dependent and a static table gets both wrong. MiniMax skips the
    transport screen going forward, so Escape must skip it coming back or it lands
    on a one-row question; and the models screen is reached from CREDENTIALS or from
    CLI_CHECK depending on a choice made three screens earlier.
    """
    if not w.history:
        return replace(w, step=Step.EXIT, error="", **fields)
    return replace(w, step=w.history[-1], history=w.history[:-1], error="", **fields)


# --------------------------------------------------------------------------- #
# Screens
# --------------------------------------------------------------------------- #
def vendor_screen(console: Console, config_path: Path, w: Wizard) -> Wizard:
    """cline's MAIN_MENU: whose account do you have?"""
    from .tui import ACTION_CANCEL, MODE_CARDS, select_cards

    rows = [(vid, describe_vendor(vid).label, describe_vendor(vid).summary,
             describe_vendor(vid).icon) for vid in describe.all_vendors()]
    choice = select_cards(console, "Welcome to better-rlm",
                          "Which provider do you have an account with?", rows,
                          current=w.vendor)
    if choice == ACTION_CANCEL:
        return back(w)

    modes = modes_for_vendor(choice)
    if len(modes) == 1:
        # cline shows ModePickerContent only when the transport is genuinely open;
        # a screen with one answer is not a question. Say why it was skipped, so a
        # skipped screen does not read as a missing one.
        console.print(f"[grey50]{describe_vendor(choice).label} is reached one way: "
                      f"{MODE_CARDS[modes[0]][1]}.[/grey50]")
        # replace, not goto(Step.MODE): a screen skipped going forward must be
        # skipped coming back. Pushing MODE onto the history here made Escape from
        # the credential form land on a transport question with exactly one answer.
        return _after_mode(replace(w, vendor=choice), modes[0])
    return goto(w, Step.MODE, vendor=choice)


def mode_screen(console: Console, config_path: Path, w: Wizard) -> Wizard:
    """Host or proxy -- cline's mode_picker, over the same two-column comparison."""
    from .tui import ACTION_CANCEL, MODE_CARDS, render_mode_compare, select_cards

    console.print(render_mode_compare())
    rows = [(m, MODE_CARDS[m][1], MODE_CARDS[m][2], MODE_CARDS[m][0])
            for m in modes_for_vendor(w.vendor)]
    choice = select_cards(console, describe_vendor(w.vendor).label,
                          "How should better-rlm reach it?", rows, current=w.mode)
    if choice == ACTION_CANCEL:
        return back(w)
    return _after_mode(w, choice)


def _after_mode(w: Wizard, mode: str) -> Wizard:
    """Resolve the provider and branch to the credential screen that fits it."""
    pid = provider_for(w.vendor, mode)
    d = describe_provider(pid)
    nxt = Step.CLI_CHECK if mode == MODE_CLI else Step.CREDENTIALS
    return goto(w, nxt, mode=mode, provider=pid, base_url=d.base_url, key_env=d.key_env)


def cli_check_screen(console: Console, config_path: Path, w: Wizard) -> Wizard:
    """Proxy path: cline's local_cli_setup, including its r-to-recheck.

    Enter is a no-op until the CLI is ready, which is cline's rule --
    saveLocalCliConfig returns early unless isLocalCliReady.
    """
    from .tui import load_status

    while True:
        st = load_status(config_path)
        ready = st.cli_available and st.cli_logged_in is True
        if ready:
            console.print(Panel(
                f"[green]The {st.cli_path} CLI is installed and logged in.[/green]\n"
                "[grey50]No API key needed -- calls reuse that login.[/grey50]",
                title="Claude Code", border_style="green"))
        else:
            why = ("not on PATH" if not st.cli_available
                   else "not logged in" if st.cli_logged_in is False
                   else "login state unknown")
            console.print(Panel(
                f"[yellow]The {st.cli_path} CLI is {why}.[/yellow]\n\n"
                "Fix it with either:\n"
                "  [bold]./install.sh --auth[/bold]   long-lived token, best for a "
                "server you leave running\n"
                "  [bold]claude auth login[/bold]     interactive; expires and cannot "
                "self-refresh",
                title="Claude Code", border_style="yellow"))
        if not picker.interactive():
            # r exists so you can fix the CLI in another window; a pipe has no other
            # window. Same decision on both paths, taken once.
            return goto(w, Step.PROBE) if ready else back(w)
        console.print("[grey37]Enter continue - r re-check - Esc back[/grey37]")
        with picker.raw_mode(console.file):
            try:
                key = picker.read_key()
            except EOFError:
                return back(w)
        if key == "escape":
            return back(w)
        if key == "enter" and ready:
            return goto(w, Step.PROBE)


def credentials_screen(console: Console, config_path: Path, w: Wizard) -> Wizard:
    """Host path: cline's byo_apikey form -- the endpoint and the key on one screen.

    cline runs no validation here at all, and neither do we: the probe on the next
    screen is what catches a bad value, and it catches more than a regex could.
    """
    from .tui import env_for_config

    d = describe_provider(w.provider)
    fields = []
    if d.base_url or w.provider == PROVIDER_CUSTOM:
        fields.append(picker.Field(
            "base_url", "Base URL", value=w.base_url,
            hint="The SDK appends /v1/messages itself, so do not end this in /v1."))
    fields.append(picker.Field(
        "api_key", w.key_env or "API key", secret=True,
        hint="Echoed masked -- first and last two characters, so you can tell a "
             "good paste from an empty clipboard."))

    env_path = env_for_config(config_path)
    res = picker.ask(console, f"{describe_vendor(w.vendor).label} credentials", fields,
                     subtitle=f"Written to {env_path}, readable only by you.",
                     error=w.error)
    if res.cancelled:
        return back(w)

    url = res.values.get("base_url", w.base_url).strip()
    # Resolve the key variable from the URL actually submitted, not from the vendor
    # picked two screens ago. An edited MiniMax URL reverse-maps to the custom
    # provider, whose variable is RLM_API_KEY -- writing the key to MINIMAX_API_KEY
    # would leave it unreadable, with every call failing "not set".
    pid = describe.provider_for_config(url, w.mode)
    key_env = describe_provider(pid).key_env or w.key_env

    key = res.values.get("api_key", "").strip()
    if not key:
        if envfile.has_var(env_path, key_env):
            console.print(f"[grey50]{key_env} left as it was.[/grey50]")
            return goto(w, Step.PROBE, base_url=url, provider=pid, key_env=key_env)
        return replace(w, base_url=url,
                       error=f"{key_env} is needed to reach {url or 'the endpoint'}.")

    fp = envfile.set_var(env_path, key_env, key)
    # config.py ran load_dotenv at import, so this process has not seen that write.
    # Without this line the probe below reports "key rejected" for a key that is fine.
    os.environ[key_env] = key
    del key
    console.print(f"[green]wrote {key_env}[/green] to {env_path} ({fp})")
    return goto(w, Step.PROBE, base_url=url, provider=pid, key_env=key_env)


def probe_screen(console: Console, config_path: Path, w: Wizard) -> Wizard:
    """The connectivity test. On failure: retry, go back and edit, or write anyway."""
    from .tui import ACTION_CANCEL, PICKER_KEEP, _select, prospective_config

    model = describe.role_defaults(w.provider)[2]
    where = w.base_url or "api.anthropic.com"
    console.print(f"[grey50]Checking {where} ...[/grey50]")
    res = probe.probe_endpoint(prospective_config(config_path, w.mode, w.base_url), model)

    if res.ok:
        console.print(Panel(f"[green]{res.detail}[/green]\n[grey50]{res.where}[/grey50]",
                            title="Connected", border_style="green"))
        return goto(w, Step.MODELS, probe_note="", role_index=0)

    body = f"[red]{res.detail}[/red]" + (f"\n\n[white]{res.fix}[/white]" if res.fix else "")
    console.print(Panel(body, title=f"Could not reach it ({res.code})",
                        border_style="red"))

    choice = _select(console, "What now?", [
        ("retry", "try the same settings again"),
        ("edit", "go back and change the endpoint or the key"),
        ("anyway", "write the configuration regardless -- fix it later"),
    ], current="retry")
    if choice in (ACTION_CANCEL, "edit"):
        return back(w)
    if choice in (PICKER_KEEP, "retry"):
        return w
    return goto(w, Step.MODELS, role_index=0,
                probe_note=f"{res.code}: {res.fix or res.detail}")


def models_screen(console: Console, config_path: Path, w: Wizard) -> Wizard:
    """One screen per role, each explaining what that model is actually for.

    The list is the provider's own -- describe.models_for -- which is the whole
    point of the catalogue: a MiniMax endpoint is never offered a Claude id again.
    """
    from .tui import ACTION_CANCEL, PICKER_CUSTOM, PICKER_KEEP, _select, warn_unknown_model

    key, title, purpose = ROLES[w.role_index]
    rows = [(m.id, m.detail()) for m in describe.models_for(w.provider)]
    # Re-opening a screen you came BACK to shows what you chose, not the catalogue
    # default. Escaping from the override screen used to drop the root pick and
    # re-open on the default, so the cursor silently disagreed with the choice you
    # had already made.
    default = (dict(w.models).get(key)
               or describe.role_defaults(w.provider)[w.role_index])

    console.print(Panel(purpose, title=f"[bold]{title}[/bold]", border_style="cyan",
                        title_align="left"))
    if not rows:
        console.print("[grey50]Nothing is known about this endpoint's models; "
                      "type an id.[/grey50]")

    choice = _select(console, f"{title}  (step {w.role_index + 1} of {len(ROLES)})",
                     rows, current=default, allow_custom=True,
                     custom_hint="type a model id this endpoint serves")
    if choice == ACTION_CANCEL:
        if w.role_index:
            # Keep the picks. They are replaced by index on the way forward, so
            # going back and forward again overwrites rather than appends.
            return replace(w, role_index=w.role_index - 1, error="")
        return back(w)
    if choice == PICKER_CUSTOM:
        res = picker.ask(console, title, [picker.Field(
            "model", "Model id",
            hint="Exactly as the endpoint names it; it is not checked against a list.")])
        picked = res.values.get("model", "").strip()
        if res.cancelled or not picked:
            return w                      # cline: empty input stays on the screen
        choice = picked
        warn_unknown_model(console, choice)
    elif choice == PICKER_KEEP:
        choice = default

    # Replace at this role's slot rather than appending, so walking back and
    # forward again cannot leave two entries for one config key.
    i = w.role_index
    models = w.models[:i] + ((key, choice),) + w.models[i + 1:]
    if i + 1 < len(ROLES):
        return replace(w, role_index=i + 1, models=models, error="")
    return goto(w, Step.DONE, models=models)


SCREENS = {
    Step.VENDOR: vendor_screen,
    Step.MODE: mode_screen,
    Step.CLI_CHECK: cli_check_screen,
    Step.CREDENTIALS: credentials_screen,
    Step.PROBE: probe_screen,
    Step.MODELS: models_screen,
}


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def commit(console: Console, config_path: Path, w: Wizard) -> bool:
    """The one and only write. Everything above this line merely decided things.

    provider is deliberately absent: it names the wire protocol, stays anthropic,
    and MiniMax is reached as a base_url. See CLAUDE.md, Endpoints vs providers.
    """
    from .tui import _save, load_status, render_status

    updates: dict[str, str] = {"mode": w.mode, "base_url": w.base_url}
    updates.update(dict(w.models))
    _save(config_path, updates, console)

    ok = not w.probe_note
    console.print()
    console.print(Panel(
        render_status(load_status(config_path)),
        title="[bold]Ready[/bold]" if ok else "[bold]Written, unverified[/bold]",
        border_style="green" if ok else "yellow"))
    if w.probe_note:
        console.print(f"[yellow]The connection test did not pass: {w.probe_note}[/yellow]")
    console.print("[grey50]A running server keeps its own copy: "
                  "claude mcp restart rlm to pick this up.[/grey50]")
    return ok


def run(console: Console, config_path: Path) -> bool:
    """Drive the flow. True when a configuration was written."""
    w = Wizard()
    for _ in range(MAX_TRANSITIONS):
        if w.step in TERMINAL:
            break
        w = SCREENS[w.step](console, config_path, w)
    else:
        console.print("[red]Setup did not settle; nothing written.[/red]")
        return False
    if w.step is Step.DONE:
        return commit(console, config_path, w)
    console.print("[grey50]setup cancelled - nothing written[/grey50]")
    return False

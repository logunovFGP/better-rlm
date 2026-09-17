"""The endpoint picker: pointing the Anthropic client somewhere other than Anthropic.

`provider` names the wire protocol, not the vendor. b7282f9 removed the old provider
picker because it offered four vendors whose selection wrote a config.yaml that
auth.require_anthropic then rejected at the first model call. This replaces it with
endpoints that all speak the Anthropic messages format, so the session-window ledger,
the 95% floor and ceiling-learning survive the switch -- the client is unchanged, only
its URL moves.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from unittest.mock import patch

import pytest

from better_rlm import config as cfgmod
from better_rlm.config import load_config
from better_rlm.describe import (
    AUTH_API_KEY,
    AUTH_CLI,
    MODE_API,
    MODE_AUTO,
    MODE_CLI,
    PROVIDER_ANTHROPIC,
    PROVIDER_CLAUDE_CLI,
    PROVIDER_CUSTOM,
    PROVIDER_MINIMAX,
    PROVIDERS,
    all_providers,
    describe_provider,
    provider_for_config,
    providers_for_mode,
)


# --- catalogue ----------------------------------------------------------------


def test_every_api_provider_speaks_the_anthropic_protocol():
    """The safety argument is about the CLIENT, not the variable name.

    This used to assert every provider's key_env == ANTHROPIC_API_KEY, which read like
    a protocol check and was really just "they all share one variable" -- the bug that
    let configuring MiniMax overwrite the Anthropic key. What must hold is that every
    provider is reached through the Anthropic client, so the ledger stays in the stack.
    """
    for pid in all_providers():
        d = PROVIDERS[pid]
        assert d.auth in (AUTH_CLI, AUTH_API_KEY), pid
        if d.auth == AUTH_API_KEY:
            assert d.key_env, f"{pid} has no key variable of its own"


def test_no_two_providers_share_a_key_variable():
    """The defect this replaced. One shared variable meant configuring a second
    provider destroyed the first one's key, and switching back sent whatever remained
    to whichever endpoint was configured."""
    seen = [PROVIDERS[p].key_env for p in all_providers() if PROVIDERS[p].key_env]
    assert len(seen) == len(set(seen)), f"shared key variable among {seen}"


def test_a_providers_key_is_never_read_for_another(monkeypatch):
    """Falling back would hand the key you issued to Anthropic to a third-party
    endpoint. A missing key must fail as missing."""
    import dataclasses
    from better_rlm.auth import api_key_for, key_env_for

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    cfg = dataclasses.replace(load_config(), mode=MODE_API,
                              base_url=PROVIDERS[PROVIDER_MINIMAX].base_url)
    assert key_env_for(cfg) == "MINIMAX_API_KEY"
    assert api_key_for(cfg) == "", "the Anthropic key leaked to a MiniMax endpoint"


def test_mode_filter_matches_clines_provider_picker():
    """cline-2 filters on `p.mode === modeFilter` so the operator is never offered a
    provider the mode they just chose cannot reach."""
    assert providers_for_mode(MODE_CLI) == (PROVIDER_CLAUDE_CLI,)
    assert PROVIDER_CLAUDE_CLI not in providers_for_mode(MODE_API)
    assert PROVIDER_MINIMAX in providers_for_mode(MODE_API)
    # auto resolves at launch, so it cannot honestly narrow the list.
    assert set(providers_for_mode(MODE_AUTO)) == set(all_providers())


def test_the_cli_provider_needs_no_key():
    d = PROVIDERS[PROVIDER_CLAUDE_CLI]
    assert d.auth == AUTH_CLI and d.key_env == ""


def test_anthropic_resolves_to_the_sdk_default():
    assert PROVIDERS[PROVIDER_ANTHROPIC].base_url == "", (
        "Anthropic must use the SDK default, not a hardcoded URL that could drift"
    )


@pytest.mark.parametrize(
    "url,mode,expected",
    [
        ("", MODE_API, PROVIDER_ANTHROPIC),
        ("   ", MODE_API, PROVIDER_ANTHROPIC),
        ("", MODE_CLI, PROVIDER_CLAUDE_CLI),
        ("https://api.minimax.io/anthropic", MODE_API, PROVIDER_MINIMAX),
        ("https://api.minimax.io/anthropic/", MODE_API, PROVIDER_MINIMAX),  # trailing slash
        ("https://gateway.example/anthropic", MODE_API, PROVIDER_CUSTOM),
    ],
)
def test_reverse_lookup_names_the_configured_provider(url, mode, expected):
    """/status shows a name, not a bare URL, so an operator can tell at a glance
    whether they are pointed where they think."""
    assert provider_for_config(url, mode) == expected


def test_unknown_provider_id_is_safe():
    d = describe_provider("wat")
    assert d.base_url == ""
    assert "pick one" in d.summary


# --- config -------------------------------------------------------------------


def test_base_url_defaults_to_empty(monkeypatch, tmp_path):
    """Hermetic on purpose. Reading the real config.yaml made this assert against
    whatever the developer happens to have configured -- the ambient-state trap that
    kept a transport test red on CI for weeks."""
    monkeypatch.setattr(cfgmod, "config_file", lambda: tmp_path / "absent.yaml")
    monkeypatch.delenv("RLM_BASE_URL", raising=False)
    assert load_config().base_url == ""


def test_rlm_base_url_env_wins_like_rlm_mode(monkeypatch):
    # Same precedence as RLM_MODE/RLM_PROVIDER, so a pod can pin it at registration.
    monkeypatch.setenv("RLM_BASE_URL", "https://api.minimax.io/anthropic")
    assert load_config().base_url == "https://api.minimax.io/anthropic"


def test_anthropic_base_url_is_not_read_by_config(monkeypatch, tmp_path):
    """The SDK honours ANTHROPIC_BASE_URL itself. Reading it here too would give one
    setting two owners that can disagree, and `better-rlm where` could not say which
    won -- so make_client passes the configured value explicitly instead."""
    monkeypatch.setattr(cfgmod, "config_file", lambda: tmp_path / "absent.yaml")
    monkeypatch.delenv("RLM_BASE_URL", raising=False)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://sneaky.example")
    assert load_config().base_url == ""


# --- the client actually gets it ----------------------------------------------


def test_make_client_omits_base_url_when_unset(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    from better_rlm import auth
    c = auth.make_client(base_url="")
    assert "api.anthropic.com" in str(c.base_url)


@pytest.mark.parametrize("url", ["https://api.minimax.io/anthropic", "  https://x.test/v  "])
def test_make_client_honours_the_configured_endpoint(monkeypatch, url):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    from better_rlm import auth
    c = auth.make_client(base_url=url)
    assert str(c.base_url).rstrip("/") == url.strip().rstrip("/")


def test_api_transport_reads_cfg_not_a_private_alias(monkeypatch):
    """Regression. The first wiring used `self._cfg.base_url`; ApiTransport stores
    `self.cfg`, and the client is built LAZILY -- so this raised AttributeError only on
    the first real model call, which no test makes. Build the client to catch it."""
    monkeypatch.setenv("MINIMAX_API_KEY", "sk-test")   # per-provider, not Anthropic's
    from better_rlm.transport import ApiTransport
    cfg = dataclasses.replace(load_config(), mode=MODE_API,
                              base_url="https://api.minimax.io/anthropic")
    t = ApiTransport(cfg)
    assert "minimax" in str(t._sync_client().base_url)
    assert "minimax" in str(t._async_client().base_url)


# --- the picker ---------------------------------------------------------------


def _cfg_file(tmp_path: Path) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text("mode: api\nprovider: anthropic\nbase_url: \"\"\n")
    return p


def test_picker_writes_the_selected_endpoint(tmp_path, quiet_console=None):
    from rich.console import Console
    import io
    from better_rlm.tui import _dispatch
    p = _cfg_file(tmp_path)
    console = Console(file=io.StringIO(), quiet=True)
    with patch("better_rlm.tui.pick_provider", return_value=PROVIDER_MINIMAX), \
            patch("better_rlm.tui.auth_step", return_value=True):
        _dispatch("/provider", console, p)
    from better_rlm.config_writer import read_scalar
    assert read_scalar(p, "base_url") == "https://api.minimax.io/anthropic"


def test_picker_writes_through_the_provider_command(tmp_path):
    """/provider writes base_url and then runs the credential step, mirroring
    cline's runProviderChange."""
    import io
    from rich.console import Console
    from better_rlm.tui import _dispatch
    p = tmp_path / "config.yaml"
    p.write_text("mode: auto\nprovider: anthropic\nbase_url: \"\"\n")
    buf = io.StringIO()
    console = Console(file=buf, width=200)
    with patch("better_rlm.tui.pick_provider", return_value=PROVIDER_MINIMAX), \
            patch("better_rlm.tui.auth_step", return_value=True):
        _dispatch("/provider", console, p)
    from better_rlm.config_writer import read_scalar
    assert read_scalar(p, "base_url") == "https://api.minimax.io/anthropic"


def test_status_flags_a_provider_that_mode_ignores():
    from better_rlm.tui import Status, _provider_line
    ignored = Status(
        mode="auto", provider="anthropic", root_model="m", root_model_override="m",
        sub_model="m", cli_path="claude", cli_available=True, cli_logged_in=True,
        env_mode=None, env_provider=None, has_api_key=False,
        base_url="https://api.minimax.io/anthropic",
    )
    ignored = dataclasses.replace(ignored, mode="claude-cli")
    assert "IGNORED" in _provider_line(ignored)
    assert "IGNORED" not in _provider_line(dataclasses.replace(ignored, mode="api"))
    assert "IGNORED" not in _provider_line(dataclasses.replace(ignored, base_url=""))


# --- the suite must never touch the developer's own config --------------------


def test_no_test_writes_the_repo_config_or_env(request):
    """A test that defaults to config_file() edits the machine it runs on.

    This session wrote `mode: api` and a MiniMax base_url into the repo's own
    config.yaml while exercising the picker, which would have pointed a live MCP
    server at an endpoint it had no key for. Nothing detected it; it was noticed by
    eye in `git status`. This fixture-free check makes the class visible: it records
    both files' digests at session start and compares at teardown.
    """
    import hashlib
    from better_rlm.config import PKG_ROOT

    watched = [PKG_ROOT / "config.yaml", PKG_ROOT / ".env"]

    def digest(p):
        return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else "absent"

    before = {p: digest(p) for p in watched}

    def check():
        for p in watched:
            assert digest(p) == before[p], (
                f"a test modified {p.name} in the checkout. Tests must pass an explicit "
                f"path (tmp_path) or monkeypatch config_file()/env_file()."
            )

    request.addfinalizer(check)


# --- the guided flow ----------------------------------------------------------


def test_setup_runs_clines_step_order(tmp_path, monkeypatch):
    """mode -> provider -> credential -> models, the order cline-2's onboarding
    machine uses (views/onboarding/model.ts). Each step narrows the next: the mode
    decides which providers are reachable, the provider decides which credential is
    even asked for, so running them out of order asks questions that cannot be
    answered yet."""
    import io
    from rich.console import Console
    from better_rlm import tui

    calls: list[str] = []
    p = tmp_path / "config.yaml"
    p.write_text("mode: auto\nprovider: anthropic\n")

    monkeypatch.setattr(tui, "pick_mode", lambda *a, **k: (calls.append("mode"), "api")[1])
    monkeypatch.setattr(tui, "pick_provider",
                        lambda *a, **k: (calls.append("provider"), PROVIDER_MINIMAX)[1])
    monkeypatch.setattr(tui, "auth_step", lambda *a, **k: (calls.append("auth"), True)[1])
    monkeypatch.setattr(tui, "pick_model", lambda *a, **k: (calls.append("model"), tui.PICKER_KEEP)[1])

    tui.run_setup(Console(file=io.StringIO(), quiet=True), p)
    assert calls[:3] == ["mode", "provider", "auth"], calls
    assert "model" in calls

    from better_rlm.config_writer import read_scalar
    assert read_scalar(p, "mode") == "api"
    assert read_scalar(p, "base_url") == "https://api.minimax.io/anthropic"


def test_setup_passes_the_chosen_mode_to_the_provider_picker(tmp_path, monkeypatch):
    """cline passes modeFilter so the user is not offered a provider the mode they
    just picked cannot reach. Without this the wizard would offer claude-cli after
    the operator chose api, and the selection would quietly do nothing."""
    import io
    from rich.console import Console
    from better_rlm import tui

    seen = {}
    p = tmp_path / "config.yaml"
    p.write_text("mode: auto\n")
    monkeypatch.setattr(tui, "pick_mode", lambda *a, **k: MODE_CLI)
    monkeypatch.setattr(tui, "pick_provider",
                        lambda console, mode, current: seen.setdefault("mode", mode) and None or tui.ACTION_CANCEL)
    tui.run_setup(Console(file=io.StringIO(), quiet=True), p)
    assert seen["mode"] == MODE_CLI


def test_cli_provider_asks_for_no_key(tmp_path, monkeypatch):
    """The claude CLI holds its own credential. Prompting for a key there would be
    asking for something that is never read."""
    import io
    from rich.console import Console
    from better_rlm import tui
    from better_rlm.describe import PROVIDER_CLAUDE_CLI

    st = tui.Status(
        mode=MODE_CLI, provider="anthropic", root_model="m", root_model_override="m",
        sub_model="m", cli_path="claude", cli_available=True, cli_logged_in=True,
        env_mode=None, env_provider=None, has_api_key=False, base_url="",
    )
    with patch.object(tui.Prompt, "ask", side_effect=AssertionError("prompted for a key")):
        assert tui.auth_step(Console(file=io.StringIO(), quiet=True), PROVIDER_CLAUDE_CLI, st) is True


def test_cli_provider_reports_a_missing_login_instead_of_claiming_success(tmp_path):
    import io
    from rich.console import Console
    from better_rlm import tui
    from better_rlm.describe import PROVIDER_CLAUDE_CLI

    buf = io.StringIO()
    st = tui.Status(
        mode=MODE_CLI, provider="anthropic", root_model="m", root_model_override="m",
        sub_model="m", cli_path="claude", cli_available=True, cli_logged_in=False,
        env_mode=None, env_provider=None, has_api_key=False, base_url="",
    )
    assert tui.auth_step(Console(file=buf, width=200), PROVIDER_CLAUDE_CLI, st) is False
    assert "not logged in" in buf.getvalue()


# --- two defects found by driving the real binary -----------------------------


def test_the_credential_follows_the_config_being_edited(tmp_path, monkeypatch):
    """`better-rlm --config /tmp/other.yaml`, then entering a key, used to write it to
    the REPO's .env -- overwriting a live credential while editing an unrelated config.
    A config file and its credential are a pair."""
    from better_rlm.tui import env_for_config
    from better_rlm.config import config_file, env_file

    assert env_for_config(config_file()) == env_file()          # default is unchanged
    scratch = tmp_path / "other.yaml"
    assert env_for_config(scratch) == tmp_path / ".env"         # follows the config


def test_auto_mode_shadowing_an_endpoint_is_flagged():
    """The trap /status originally missed. auth.claude_cli_available only calls
    shutil.which, so `auto` prefers the CLI on PATH PRESENCE ALONE -- a machine with
    the CLI installed silently ignores a configured endpoint, which looks configured
    and is not. Login is deliberately not consulted, because auto does not consult it.
    """
    import dataclasses
    from better_rlm.tui import Status, cli_will_win, _provider_line

    base = Status(
        mode="auto", provider="anthropic", root_model="m", root_model_override="m",
        sub_model="m", cli_path="claude", cli_available=True, cli_logged_in=True,
        env_mode=None, env_provider=None, has_api_key=False,
        base_url="https://api.minimax.io/anthropic",
    )
    assert cli_will_win(base) is True
    assert "IGNORED" in _provider_line(base)

    # A logged-OUT CLI still wins under auto: presence is the only test.
    assert cli_will_win(dataclasses.replace(base, cli_logged_in=False)) is True
    # No CLI on PATH: auto falls through to the API key, so the endpoint is live.
    assert cli_will_win(dataclasses.replace(base, cli_available=False)) is False
    assert "IGNORED" not in _provider_line(dataclasses.replace(base, cli_available=False))
    # api pins the SDK path regardless of any CLI.
    assert cli_will_win(dataclasses.replace(base, mode="api")) is False
    # claude-cli is the obvious case.
    assert cli_will_win(dataclasses.replace(base, mode="claude-cli")) is True
    # No endpoint configured: nothing to shadow, nothing to warn about.
    assert "IGNORED" not in _provider_line(dataclasses.replace(base, base_url=""))


def test_the_key_written_by_the_wizard_reaches_the_client(tmp_path, monkeypatch):
    """End to end minus the network: wizard -> .env -> load -> ApiTransport client."""
    from better_rlm import envfile, tui
    from better_rlm.transport import ApiTransport
    import dataclasses, io
    from rich.console import Console
    from better_rlm.describe import PROVIDER_MINIMAX

    env_path = tmp_path / ".env"
    st = tui.Status(
        mode="api", provider="anthropic", root_model="m", root_model_override="m",
        sub_model="m", cli_path="claude", cli_available=False, cli_logged_in=None,
        env_mode=None, env_provider=None, has_api_key=False, base_url="",
    )
    with patch.object(tui.Prompt, "ask", return_value="sk-fake-roundtrip"):
        assert tui.auth_step(Console(file=io.StringIO(), quiet=True),
                             PROVIDER_MINIMAX, st, env_path) is True
    assert envfile.has_var(env_path, "MINIMAX_API_KEY")
    assert oct(env_path.stat().st_mode & 0o777) == "0o600"
    assert "sk-fake-roundtrip" not in envfile.fingerprint("sk-fake-roundtrip")

    monkeypatch.setenv("MINIMAX_API_KEY", "sk-fake-roundtrip")
    cfg = dataclasses.replace(load_config(), mode=MODE_API,
                              base_url=PROVIDERS[PROVIDER_MINIMAX].base_url)
    client = ApiTransport(cfg)._sync_client()
    assert "minimax" in str(client.base_url)
    assert client.api_key == "sk-fake-roundtrip"


# --- the main menu ------------------------------------------------------------


def _st(**over):
    from better_rlm.tui import Status
    base = dict(
        mode="auto", provider="anthropic", root_model="claude-sonnet-5",
        root_model_override="o", sub_model="claude-haiku-4-5", cli_path="claude",
        cli_available=True, cli_logged_in=True, env_mode=None, env_provider=None,
        has_api_key=False, base_url="", key_env="",
    )
    base.update(over)
    return Status(**base)


def test_a_healthy_config_offers_no_problem_rows():
    """The menu is state-aware like cline's getMainMenuOptions. A working setup should
    not be nagged about problems it does not have."""
    from better_rlm.tui import build_menu

    labels = [i.label for i in build_menu(_st(mode="claude-cli", cli_logged_in=True))]
    assert not any("Supply" in l or "Fix the mode" in l or "Sign the" in l for l in labels)
    assert "Run guided setup" in labels


def test_a_missing_key_leads_the_menu():
    """An operator landing on a config that cannot make a model call is offered the
    fix first, rather than having to know which slash command repairs it."""
    from better_rlm.tui import build_menu

    items = build_menu(_st(mode=MODE_API, key_env="MINIMAX_API_KEY", has_api_key=False,
                           base_url="https://api.minimax.io/anthropic"))
    assert items[0].label.startswith("Supply your MiniMax key")
    assert "MINIMAX_API_KEY" in items[0].detail
    assert items[0].key == "1"


def test_an_endpoint_the_mode_ignores_leads_the_menu():
    from better_rlm.tui import build_menu

    items = build_menu(_st(mode="auto", cli_available=True, key_env="MINIMAX_API_KEY",
                           has_api_key=True, base_url="https://api.minimax.io/anthropic"))
    assert "Fix the mode" in items[0].label


def test_a_signed_out_cli_leads_the_menu():
    from better_rlm.tui import build_menu

    items = build_menu(_st(mode=MODE_CLI, cli_logged_in=False))
    assert "Sign the" in items[0].label


def test_a_cli_config_is_never_asked_for_an_api_key():
    """mode=claude-cli reads no key variable, so 'MINIMAX_API_KEY is not set' would be
    nagging about something that is never consulted."""
    from better_rlm.tui import build_menu

    labels = [i.label for i in build_menu(_st(mode=MODE_CLI, key_env="", has_api_key=False))]
    assert not any("Supply your" in l for l in labels)


def test_every_menu_row_maps_to_something_dispatchable():
    """A row whose command does not exist is a dead end the operator cannot escape."""
    from better_rlm.tui import build_menu, SLASH_COMMANDS

    known = set(dict(SLASH_COMMANDS)) | {"/provider-key"}   # the key-only internal action
    for mode in ("auto", MODE_CLI, MODE_API):
        for item in build_menu(_st(mode=mode, key_env="ANTHROPIC_API_KEY")):
            assert item.command in known, item


def test_menu_keys_are_unique_and_sequential():
    from better_rlm.tui import build_menu

    for st in (_st(), _st(mode=MODE_API, key_env="MINIMAX_API_KEY")):
        keys = [i.key for i in build_menu(st)]
        assert keys == [str(n) for n in range(1, len(keys) + 1)], keys


# --- first run ----------------------------------------------------------------


def test_onboarding_runs_when_no_call_can_be_made():
    """cline runs onboarding when no provider is configured. The equivalent question
    is whether a credential exists: a config that cannot call a model is not
    configured, whatever config.yaml says."""
    from better_rlm.tui import needs_onboarding

    assert needs_onboarding(_st(mode=MODE_API, key_env="MINIMAX_API_KEY", has_api_key=False))
    assert needs_onboarding(_st(mode=MODE_CLI, cli_logged_in=False))
    assert needs_onboarding(_st(mode=MODE_AUTO, cli_available=False, has_api_key=False))


def test_onboarding_is_skipped_once_it_can():
    from better_rlm.tui import needs_onboarding

    assert not needs_onboarding(_st(mode=MODE_API, key_env="MINIMAX_API_KEY", has_api_key=True))
    assert not needs_onboarding(_st(mode=MODE_CLI, cli_logged_in=True))
    # auto falls back to the CLI, so a signed-in CLI is a working config.
    assert not needs_onboarding(_st(mode=MODE_AUTO, cli_available=True, cli_logged_in=True))


def test_every_onboarding_choice_sets_a_coherent_mode():
    """One question sets both, cline-style. A row pairing a CLI provider with api --
    or an endpoint provider with claude-cli -- would write a config that ignores the
    answer the operator just gave."""
    from better_rlm.tui import ONBOARDING_CHOICES
    from better_rlm.describe import AUTH_CLI, PROVIDERS

    for pid, icon, label, detail, mode in ONBOARDING_CHOICES:
        assert pid in PROVIDERS, pid
        assert icon and label and detail
        expected = MODE_CLI if PROVIDERS[pid].auth == AUTH_CLI else MODE_API
        assert mode == expected, f"{pid} would be configured unreachable"

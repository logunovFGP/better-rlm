"""Tests for ``better_rlm.tui`` -- the operator-facing TUI.

Most TUI functions either prompt for input (interactive, can't be tested
without a fake stdin) or invoke subprocesses (``/test``, ``/auth-probe``).
The tests here pin the parts that are pure data: the dispatcher's
recognition of each slash command, the pickers' option lists, and the
``Status`` snapshot construction.
"""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import patch

import pytest

from better_rlm import tui
from better_rlm.describe import MODE_API, MODE_AUTO, MODE_CLI
from better_rlm.tui import (
    ACTION_CANCEL,
    HELP_TEXT,
    PICKER_KEEP,
    SLASH_COMMANDS,
    Status,
    _dispatch,
    load_status,
    pick_mode,
    render_mode_compare,
    render_status,
)


@pytest.fixture
def quiet_console():
    """A rich Console that writes nowhere -- so pickers can run in CI without
    polluting test output.

    StringIO, not open("/dev/null"): that path does not exist on Windows, where
    release.yml also runs this suite, and it leaked an unclosed file handle.
    """
    from rich.console import Console

    return Console(file=io.StringIO(), quiet=True)


def test_status_dataclass_is_frozen() -> None:
    """Status is a snapshot, not a mutable config handle -- frozen so it
    can't be accidentally mutated by a caller that holds a reference."""
    st = Status(
        mode=MODE_AUTO,
        provider="anthropic",
        root_model="claude-sonnet-5",
        root_model_override="claude-opus-4-8",
        sub_model="claude-haiku-4-5",
        cli_path="claude",
        cli_available=True,
        cli_logged_in=True,
        env_mode=None,
        env_provider=None,
        has_api_key=False,
    )
    with pytest.raises((AttributeError, TypeError)):
        st.mode = "api"  # type: ignore[misc]


def test_status_mode_is_pinned_only_when_env_differs() -> None:
    """``mode_is_pinned`` is True when an env var forces a different value
    than config.yaml -- the picker must warn the user that the write may
    not take effect."""
    pinned = Status(
        mode=MODE_AUTO, provider="anthropic",
        root_model="x", root_model_override="x", sub_model="x",
        cli_path="claude", cli_available=True, cli_logged_in=True,
        env_mode=MODE_API, env_provider=None, has_api_key=False,
    )
    not_pinned = Status(
        mode=MODE_AUTO, provider="anthropic",
        root_model="x", root_model_override="x", sub_model="x",
        cli_path="claude", cli_available=True, cli_logged_in=True,
        env_mode=None, env_provider=None, has_api_key=False,
    )
    same_env = Status(
        mode=MODE_AUTO, provider="anthropic",
        root_model="x", root_model_override="x", sub_model="x",
        cli_path="claude", cli_available=True, cli_logged_in=True,
        env_mode=MODE_AUTO, env_provider=None, has_api_key=False,
    )
    assert pinned.mode_is_pinned() is True
    assert not_pinned.mode_is_pinned() is False
    assert same_env.mode_is_pinned() is False


def test_load_status_reads_config_yaml(tmp_path: Path) -> None:
    """``load_status`` looks at config.yaml on disk (not the in-process
    Config), so it shows what the next server start would resolve to."""
    cfg = tmp_path / "config.yaml"
    cfg.write_text("mode: api\nprovider: openai\nroot_model: gpt-4o\n"
                   "root_model_override: gpt-4o-mini\nsub_model: gpt-4o-mini\n")
    # sub_path: sub-MODEL-related constants come from config.py; we just need ANY
    # value to come back. The fixture values are placeholders -- the test is about
    # the keys the TUI reads. (No PKG_ROOT patch: load_status takes the path
    # explicitly, and the TUI resolves config through config_file() now, so patching
    # tui.PKG_ROOT changed nothing and implied coverage that was not there.)
    st = load_status(cfg)
    assert st.mode == "api"
    assert st.provider == "openai"
    assert st.root_model == "gpt-4o"
    assert st.sub_model == "gpt-4o-mini"


def test_render_status_lists_every_key() -> None:
    """/status shows every key the TUI can change -- so a glance at the
    panel answers "what's currently set" without scrolling."""
    st = Status(
        mode=MODE_AUTO, provider="anthropic",
        root_model="x", root_model_override="y", sub_model="z",
        cli_path="claude", cli_available=True, cli_logged_in=True,
        env_mode=None, env_provider=None, has_api_key=False,
    )
    out = render_status(st)
    assert "mode:" in out
    assert "provider:" in out
    assert "root_model:" in out
    assert "override_model:" in out
    assert "sub_model:" in out
    assert "claude cli:" in out
    assert "api key:" in out


def test_render_status_warns_when_pinned() -> None:
    """When RLM_MODE pins a different mode than config.yaml, /status must
    say so -- otherwise the user thinks the picker is broken when their
    write doesn't take effect."""
    st = Status(
        mode=MODE_AUTO, provider="anthropic",
        root_model="x", root_model_override="y", sub_model="z",
        cli_path="claude", cli_available=True, cli_logged_in=True,
        env_mode=MODE_API, env_provider=None, has_api_key=False,
    )
    out = render_status(st)
    assert "RLM_MODE=" in out
    assert "env var pins mode" in out


def test_pick_mode_uses_describe_prose(quiet_console, monkeypatch) -> None:
    """The picker shows the same prose as ``describe_mode`` -- so the
    comparison never lies. Patch Prompt.ask to return PICKER_KEEP and
    verify the function returns it cleanly."""
    monkeypatch.setattr("better_rlm.tui.Prompt.ask", lambda *a, **kw: "0")
    out = pick_mode(quiet_console, MODE_AUTO)
    assert out == PICKER_KEEP


def test_pick_mode_returns_chosen_mode(quiet_console, monkeypatch) -> None:
    """Choosing option '2' (claude-cli) returns the value, not the index."""
    monkeypatch.setattr("better_rlm.tui.Prompt.ask", lambda *a, **kw: "2")
    out = pick_mode(quiet_console, MODE_AUTO)
    assert out == MODE_CLI


def test_pick_mode_cancel_via_q(quiet_console, monkeypatch) -> None:
    """Typing ``q`` returns ACTION_CANCEL -- the dispatcher treats it as
    'no change' just like PICKER_KEEP."""
    monkeypatch.setattr("better_rlm.tui.Prompt.ask", lambda *a, **kw: "q")
    out = pick_mode(quiet_console, MODE_AUTO)
    assert out == ACTION_CANCEL


def test_picker_never_offers_a_provider_the_server_would_refuse(quiet_console, tmp_path) -> None:
    """The picker is back, and this is the property that lets it be.

    It used to assert `/provider` must not exist. b7282f9 removed the original picker
    because it offered four vendors whose selection wrote a config.yaml
    auth.require_anthropic rejects at the first model call -- the write succeeds and
    the next rlm_query is what fails. The ban was aimed at the picker rather than at
    the trap. Every provider offered now is provider=anthropic, differing by endpoint
    and credential, so any selection produces a config the server accepts.
    """
    from better_rlm import tui as tui_mod
    from better_rlm.describe import MODE_API, MODE_AUTO, MODE_CLI, PROVIDERS, providers_for_mode

    assert hasattr(tui_mod, "pick_provider")
    assert "/provider" in dict(SLASH_COMMANDS)

    for mode in (MODE_AUTO, MODE_CLI, MODE_API):
        offered = providers_for_mode(mode)
        assert offered, f"{mode} offers nothing"
        for pid in offered:
            # A row reached through a different CLIENT is the trap coming back; the
            # key variable is per provider and says nothing about the protocol.
            assert PROVIDERS[pid].auth in ("cli", "api_key"), (mode, pid)

def test_render_mode_compare_includes_every_mode() -> None:
    """The compare view must mention every mode the picker offers -- so
    /mode-help is a substitute for /mode when stdin is broken."""
    out = render_mode_compare()
    for mode in (MODE_AUTO, MODE_CLI, MODE_API):
        assert mode in out


# -- Dispatcher -------------------------------------------------------------


@pytest.mark.parametrize(
    "cmd, expected_keep",
    [
        ("/quit", False),
        ("/exit", False),
    ],
)
def test_dispatch_quit_returns_false(quiet_console, tmp_path, cmd, expected_keep) -> None:
    """``/quit`` and ``/exit`` are the only commands that return False from
    ``_dispatch`` -- the REPL turns False into 'break out of the loop'."""
    assert _dispatch(cmd, quiet_console, tmp_path / "config.yaml") is expected_keep


def test_dispatch_unknown_command(quiet_console, tmp_path) -> None:
    """An unknown slash command does NOT crash; it prints an error and
    returns True so the REPL keeps running."""
    p = tmp_path / "config.yaml"
    p.write_text("")
    assert _dispatch("/foo", quiet_console, p) is True


def test_dispatch_help_mentions_every_command(quiet_console, tmp_path) -> None:
    """The /help panel must list every registered slash command -- the
    REPL shows HELP_TEXT directly, so a missing entry is a missing
    discoverable feature."""
    p = tmp_path / "config.yaml"
    _dispatch("/help", quiet_console, p)
    for cmd, _desc in SLASH_COMMANDS:
        assert cmd in HELP_TEXT


def test_dispatch_status_runs_against_disk_config(quiet_console, tmp_path) -> None:
    """``/status`` reports the on-disk config, not the in-process Config --
    the user can edit config.yaml from outside the TUI and /status picks
    up the change."""
    p = tmp_path / "config.yaml"
    p.write_text("mode: api\nprovider: openai\nroot_model: m1\n"
                 "root_model_override: m2\nsub_model: m3\n")
    # No PKG_ROOT patch -- _dispatch takes the config path explicitly.
    _dispatch("/status", quiet_console, p)


def test_dispatch_mode_pick_keeps_current(quiet_console, tmp_path, monkeypatch) -> None:
    """Choosing 'keep' in /mode writes nothing -- so accidentally pressing
    Enter on the picker does not silently change the config."""
    p = tmp_path / "config.yaml"
    p.write_text("mode: auto\n")
    monkeypatch.setattr("better_rlm.tui.Prompt.ask", lambda *a, **kw: "0")
    _dispatch("/mode", quiet_console, p)
    from better_rlm.config_writer import read_scalar
    assert read_scalar(p, "mode") == "auto"


def test_dispatch_mode_pick_writes_choice(quiet_console, tmp_path, monkeypatch) -> None:
    """Choosing option '3' (api) writes mode=api to disk -- the picker's
    writeback is what makes /mode different from /mode-help."""
    p = tmp_path / "config.yaml"
    p.write_text("mode: auto\n")
    monkeypatch.setattr("better_rlm.tui.Prompt.ask", lambda *a, **kw: "3")
    _dispatch("/mode", quiet_console, p)
    from better_rlm.config_writer import read_scalar
    assert read_scalar(p, "mode") == "api"


def test_render_status_flags_a_provider_the_server_will_refuse() -> None:
    """A non-anthropic provider is surfaced by /status, not silently rendered.

    The TUI cannot write this key any more, but a hand edit or RLM_PROVIDER still can,
    and auth.require_anthropic only raises once a model call is attempted. Saying so on
    the status line is cheaper than discovering it mid-query.
    """
    def _st(provider: str) -> Status:
        return Status(
            mode=MODE_AUTO, provider=provider,
            root_model="x", root_model_override="y", sub_model="z",
            cli_path="claude", cli_available=True, cli_logged_in=True,
            env_mode=None, env_provider=None, has_api_key=False,
        )

    assert "UNSUPPORTED" in render_status(_st("openai"))
    assert "UNSUPPORTED" not in render_status(_st("anthropic"))
    assert "UNSUPPORTED" not in render_status(_st("  Anthropic  "))


def test_dispatch_model_pick_writes_choice(quiet_console, tmp_path, monkeypatch) -> None:
    """``/model`` writes ``root_model`` -- the picker mirrors /mode and
    /provider in shape.

    Answer "3" is a ROW POSITION, so this asserts against the catalogue rather than
    a hardcoded id: the list used to be four constants and is now the provider own
    table, and a test naming the id at position 3 silently changes meaning whenever
    a row is inserted above it.
    """
    from better_rlm.describe import models_for

    p = tmp_path / "config.yaml"
    p.write_text("root_model: claude-sonnet-5\n")
    monkeypatch.setattr("better_rlm.tui.Prompt.ask", lambda *a, **kw: "3")
    _dispatch("/model", quiet_console, p)
    from better_rlm.config_writer import read_scalar
    assert read_scalar(p, "root_model") == models_for("anthropic")[2].id


def test_dispatch_model_custom_entry(quiet_console, tmp_path, monkeypatch) -> None:
    """``c`` enters custom-id mode and the next Prompt.ask is the model id."""
    p = tmp_path / "config.yaml"
    p.write_text("root_model: claude-sonnet-5\n")
    answers = iter(["c", "claude-opus-4-9"])
    monkeypatch.setattr("better_rlm.tui.Prompt.ask", lambda *a, **kw: next(answers))
    _dispatch("/model", quiet_console, p)
    from better_rlm.config_writer import read_scalar
    assert read_scalar(p, "root_model") == "claude-opus-4-9"


def test_dispatch_test_runs_pytest(quiet_console, tmp_path, monkeypatch) -> None:
    """``/test`` invokes ``uv run --extra dev pytest -q`` -- the same gate
    the pre-push hook runs (CLAUDE.md)."""
    p = tmp_path / "config.yaml"
    called: list[list[str]] = []
    monkeypatch.setattr("subprocess.call", lambda cmd, **kw: called.append(cmd) or 0)
    _dispatch("/test", quiet_console, p)
    assert called, "pytest was not invoked"
    cmd = called[0]
    assert cmd[0] == "uv"
    assert "--extra" in cmd
    assert "dev" in cmd
    assert "pytest" in cmd


def test_dispatch_test_config_focused_subset(quiet_console, tmp_path, monkeypatch) -> None:
    """``/test-config`` runs only the auth/config/transport tests so a
    picker-only change doesn't have to run the whole suite to verify."""
    p = tmp_path / "config.yaml"
    called: list[list[str]] = []
    monkeypatch.setattr("subprocess.call", lambda cmd, **kw: called.append(cmd) or 0)
    _dispatch("/test-config", quiet_console, p)
    cmd = called[0]
    assert "tests/test_config.py" in cmd
    assert "tests/test_auth.py" in cmd
    assert "tests/test_transport.py" in cmd


def test_dispatch_unknown_provider_writes_nothing(quiet_console, tmp_path, monkeypatch) -> None:
    """Defensive: if ``_prompt_choice`` ever returns something not in the
    catalogue, the dispatcher must NOT write it. Guards against a future
    picker bug that lets an unvalidated string through."""
    p = tmp_path / "config.yaml"
    p.write_text("mode: auto\n")

    # Bypass the picker entirely -- drive the dispatcher with a synthetic
    # "would-be" return value that bypasses the catalogue. The picker
    # loops on invalid input; testing that loop here would be testing the
    # picker, not the dispatcher's defence-in-depth.
    from better_rlm import tui as tui_mod
    monkeypatch.setattr(tui_mod, "pick_mode", lambda *a, **kw: "this_is_not_a_mode")
    _dispatch("/mode", quiet_console, p)
    from better_rlm.config_writer import read_scalar
    assert read_scalar(p, "mode") == "auto"


def test_login_state_comes_from_cli_auth_status(tmp_path: Path, monkeypatch) -> None:
    """/status reads the login flag from `claude auth status --json`, via transport.

    Regression: load_status used to run `claude auth status` WITHOUT --json and test
    `"loggedIn" in stdout` -- a JSON key that never appears in the human-readable
    output. A signed-in CLI therefore rendered as "NOT LOGGED IN" on the one surface
    whose job is diagnosing auth. transport.cli_auth_status already does this properly
    (parses the JSON, passes CliTransport._subprocess_env()), so the TUI calls it.
    """
    import better_rlm.transport as transport

    cfg = tmp_path / "config.yaml"
    cfg.write_text("mode: auto\ncli_path: claude\n", encoding="utf-8")
    monkeypatch.setattr("shutil.which", lambda _p: "/usr/local/bin/claude")

    seen: list[str] = []

    def fake(cfg_like):
        seen.append(cfg_like.cli_path)      # only cli_path is read off the config
        return {"loggedIn": True}

    monkeypatch.setattr(transport, "cli_auth_status", fake)
    assert load_status(cfg).cli_logged_in is True
    assert seen == ["claude"]

    monkeypatch.setattr(transport, "cli_auth_status", lambda _c: {"loggedIn": False})
    assert load_status(cfg).cli_logged_in is False

    # Probe failure stays "unknown" rather than a false negative or an exception.
    monkeypatch.setattr(transport, "cli_auth_status", lambda _c: None)
    assert load_status(cfg).cli_logged_in is None


def test_one_shot_exits_with_the_command_status(tmp_path: Path, monkeypatch) -> None:
    """`--one-shot /test` must exit non-zero on a red suite.

    Regression: main() used to `return 0 if _dispatch(...) else 0` -- both branches
    zero -- while README advertises --one-shot as CI-friendly and /test runs the verify
    gate. A gate that is always green is worse than no gate.
    """
    cfg = str(tmp_path / "config.yaml")

    monkeypatch.setattr("subprocess.call", lambda cmd, **kw: 1)
    assert tui.main(["--config", cfg, "--one-shot", "/test"]) == 1

    monkeypatch.setattr("subprocess.call", lambda cmd, **kw: 0)
    assert tui.main(["--config", cfg, "--one-shot", "/test"]) == 0

    # A command with nothing to report is still a success.
    assert tui.main(["--config", cfg, "--one-shot", "/help"]) == 0


def test_one_shot_typo_is_not_reported_as_success(tmp_path: Path) -> None:
    """A misspelled --one-shot command exits 2, so a scripted caller notices."""
    assert tui.main(["--config", str(tmp_path / "c.yaml"), "--one-shot", "/stauts"]) == 2



def test_readme_documents_exactly_the_shipped_slash_commands() -> None:
    """The README command table is hand-maintained; nothing tied it to SLASH_COMMANDS.

    The /provider row was removed by hand when the picker went, and the next command
    added or removed can desync the published docs from the shipped surface with a
    green suite. test_dispatch_help_mentions_every_command pins /help, not the README.
    """
    import re

    readme = (Path(__file__).resolve().parent.parent / "README.md").read_text(
        encoding="utf-8")
    table = readme.split("Slash commands exposed by the TUI:")[1].split("\n\n")[1]

    for cmd, _desc in SLASH_COMMANDS:
        assert cmd in table, f"{cmd} is shipped but missing from the README table"

    # /exit is a documented alias of /quit, handled in _dispatch but not listed in
    # SLASH_COMMANDS. Anything else in the table must be a real command.
    shipped = {cmd for cmd, _desc in SLASH_COMMANDS} | {"/exit"}
    # Backtick-delimited only: the prose in these rows says things like
    # "host/proxy terminology", which is not a command.
    for documented in set(re.findall(r"`(/[a-z][a-z-]*)`", table)):
        assert documented in shipped, f"README documents {documented}, which does not exist"


def _status(**over):
    """A Status with every field pinned, so no test reads this machine.

    The first version of the test below called load_status and monkeypatched a
    name that does not exist (`_cli_available`, with raising=False, so the patch
    silently did nothing). It passed here only because this machine has a
    logged-in `claude` CLI, and failed on all four CI jobs, which have none:
    cli_logged_in came back None and mode `claude-cli` then reported
    "not configured".
    """
    base = dict(
        mode=tui.MODE_AUTO, provider="anthropic", root_model="claude-sonnet-5",
        root_model_override="claude-opus-4-8", sub_model="claude-haiku-4-5",
        cli_path="claude", cli_available=True, cli_logged_in=True,
        env_mode=None, env_provider=None, has_api_key=False,
        base_url="", key_env="", config_written=True,
    )
    base.update(over)
    return tui.Status(**base)


def test_a_brand_new_install_runs_setup_even_when_the_claude_cli_is_logged_in(tmp_path):
    """The gate cline states as `if (!isProviderConfigured(config))`.

    needs_onboarding asked only whether a model call could succeed. With no
    config.yaml, `mode` defaults to `auto`, and auto accepts a signed-in `claude`
    CLI -- so on any machine with Claude Code installed the answer was "already
    configured" and setup never ran. The operator landed on the maintenance menu
    showing defaults they had never chosen, which is what was reported.

    Having Claude Code installed is not the same as having configured this tool.
    """
    # Absent file: real load_status, and the machine cannot influence it because
    # config_written short-circuits before any CLI state is consulted.
    st = tui.load_status(tmp_path / "config.yaml")
    assert not st.config_written, "no file on disk, so nothing has been configured"
    assert tui.needs_onboarding(st), (
        "a machine with a logged-in `claude` CLI skipped setup entirely"
    )

    # Written and reachable: the menu is right. This is the case that must not
    # regress into always onboarding.
    assert not tui.needs_onboarding(
        _status(mode=tui.MODE_CLI, cli_available=True, cli_logged_in=True))

    # Written but unreachable stays onboarding -- the half this repo adds to cline's.
    assert tui.needs_onboarding(
        _status(mode=tui.MODE_API, key_env="ANTHROPIC_API_KEY", has_api_key=False))

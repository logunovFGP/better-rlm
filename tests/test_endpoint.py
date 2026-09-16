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
    ENDPOINT_ANTHROPIC,
    ENDPOINT_CUSTOM,
    ENDPOINT_MINIMAX,
    ENDPOINTS,
    all_endpoints,
    describe_endpoint,
    endpoint_for_base_url,
)


# --- catalogue ----------------------------------------------------------------


def test_every_offered_endpoint_speaks_the_anthropic_protocol():
    """The whole safety argument. An entry needing a different client would lose the
    ledger, which is exactly what require_anthropic exists to prevent."""
    for eid in all_endpoints():
        assert ENDPOINTS[eid].key_env == "ANTHROPIC_API_KEY", eid


def test_anthropic_is_first_and_keyless():
    assert all_endpoints()[0] == ENDPOINT_ANTHROPIC
    assert ENDPOINTS[ENDPOINT_ANTHROPIC].base_url == "", (
        "Anthropic must resolve to the SDK default, not a hardcoded URL that could "
        "drift from whatever the SDK ships"
    )


@pytest.mark.parametrize(
    "url,expected",
    [
        ("", ENDPOINT_ANTHROPIC),
        ("   ", ENDPOINT_ANTHROPIC),
        ("https://api.minimax.io/anthropic", ENDPOINT_MINIMAX),
        ("https://api.minimax.io/anthropic/", ENDPOINT_MINIMAX),  # trailing slash
        ("https://gateway.example/anthropic", ENDPOINT_CUSTOM),
    ],
)
def test_reverse_lookup_names_the_configured_url(url, expected):
    """/status shows a name, not a bare URL, so an operator can tell at a glance
    whether they are pointed where they think."""
    assert endpoint_for_base_url(url) == expected


def test_unknown_endpoint_id_is_safe():
    d = describe_endpoint("wat")
    assert d.base_url == ""
    assert "pick one" in d.note


# --- config -------------------------------------------------------------------


def test_base_url_defaults_to_empty():
    assert load_config().base_url == ""


def test_rlm_base_url_env_wins_like_rlm_mode(monkeypatch):
    # Same precedence as RLM_MODE/RLM_PROVIDER, so a pod can pin it at registration.
    monkeypatch.setenv("RLM_BASE_URL", "https://api.minimax.io/anthropic")
    assert load_config().base_url == "https://api.minimax.io/anthropic"


def test_anthropic_base_url_is_not_read_by_config(monkeypatch):
    """The SDK honours ANTHROPIC_BASE_URL itself. Reading it here too would give one
    setting two owners that can disagree, and `better-rlm where` could not say which
    won -- so make_client passes the configured value explicitly instead."""
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
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    from better_rlm.transport import ApiTransport
    cfg = dataclasses.replace(load_config(), base_url="https://api.minimax.io/anthropic")
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
    with patch("better_rlm.tui.pick_endpoint", return_value=ENDPOINT_MINIMAX):
        _dispatch("/endpoint", console, p)
    from better_rlm.config_writer import read_scalar
    assert read_scalar(p, "base_url") == "https://api.minimax.io/anthropic"


def test_picker_warns_when_mode_would_ignore_the_endpoint(tmp_path):
    """The one way to leave the TUI with a config that silently does nothing: a
    base_url set while mode still spawns the claude CLI, which talks to Anthropic."""
    import io
    from rich.console import Console
    from better_rlm.tui import _dispatch
    p = tmp_path / "config.yaml"
    p.write_text("mode: auto\nprovider: anthropic\nbase_url: \"\"\n")
    buf = io.StringIO()
    console = Console(file=buf, width=200)
    with patch("better_rlm.tui.pick_endpoint", return_value=ENDPOINT_MINIMAX):
        _dispatch("/endpoint", console, p)
    assert "ignores this endpoint" in buf.getvalue()


def test_status_flags_an_endpoint_that_mode_ignores():
    from better_rlm.tui import Status, _endpoint_line
    ignored = Status(
        mode="auto", provider="anthropic", root_model="m", root_model_override="m",
        sub_model="m", cli_path="claude", cli_available=True, cli_logged_in=True,
        env_mode=None, env_provider=None, has_api_key=False,
        base_url="https://api.minimax.io/anthropic",
    )
    assert "IGNORED" in _endpoint_line(ignored)
    assert "IGNORED" not in _endpoint_line(dataclasses.replace(ignored, mode="api"))
    assert "IGNORED" not in _endpoint_line(dataclasses.replace(ignored, base_url=""))

import dataclasses

import pytest

import src.auth as auth
from src.config import load_config


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("RLM_MODE", raising=False)


def _cfg(**over):
    return dataclasses.replace(load_config(), **over)


def test_clean_secret_ignores_placeholders_and_whitespace():
    assert auth._clean_secret("<paste>") is None
    assert auth._clean_secret("   ") is None
    assert auth._clean_secret("has space") is None
    assert auth._clean_secret("sk-ant-oat01-real") == "sk-ant-oat01-real"


def test_auth_status_reports_present_credentials(monkeypatch):
    assert auth.auth_status() == "none"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api-x")
    assert auth.auth_status() == "apikey"
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat01-x")
    assert auth.auth_status() == "oauth"  # token wins for the "present credential" report


def test_claude_cli_available(monkeypatch):
    cfg = _cfg(cli_path="claude")
    monkeypatch.setattr(auth.shutil, "which", lambda p: "/usr/local/bin/claude")
    assert auth.claude_cli_available(cfg) is True
    monkeypatch.setattr(auth.shutil, "which", lambda p: None)
    assert auth.claude_cli_available(cfg) is False


def test_auto_prefers_cli_with_no_token_or_key(monkeypatch):
    # The headline fix: with the CLI present, auto mode selects it — NO token/key.
    monkeypatch.setattr(auth.shutil, "which", lambda p: "/bin/claude")
    assert auth.resolve_auth_mode(_cfg(mode="auto")) == "oauth"


def test_auto_falls_back_to_apikey_without_cli(monkeypatch):
    monkeypatch.setattr(auth.shutil, "which", lambda p: None)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api-x")
    assert auth.resolve_auth_mode(_cfg(mode="auto")) == "apikey"


def test_auto_raises_when_nothing_available(monkeypatch):
    monkeypatch.setattr(auth.shutil, "which", lambda p: None)
    with pytest.raises(RuntimeError):
        auth.resolve_auth_mode(_cfg(mode="auto"))


def test_explicit_claude_cli_requires_cli(monkeypatch):
    monkeypatch.setattr(auth.shutil, "which", lambda p: None)
    with pytest.raises(RuntimeError):
        auth.resolve_auth_mode(_cfg(mode="claude-cli"))
    monkeypatch.setattr(auth.shutil, "which", lambda p: "/bin/claude")
    assert auth.resolve_auth_mode(_cfg(mode="claude-cli")) == "oauth"


def test_explicit_api_requires_key(monkeypatch):
    monkeypatch.setattr(auth.shutil, "which", lambda p: "/bin/claude")  # present but mode forces api
    with pytest.raises(RuntimeError):
        auth.resolve_auth_mode(_cfg(mode="api"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api-x")
    assert auth.resolve_auth_mode(_cfg(mode="api")) == "apikey"


def test_non_anthropic_provider_needs_its_own_key(monkeypatch):
    g = _cfg(provider="gemini")
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        auth.resolve_auth_mode(g)
    monkeypatch.setenv("GEMINI_API_KEY", "dummy-not-a-real-key")
    assert auth.resolve_auth_mode(g) == "apikey"
    assert auth.provider_key(g) == "dummy-not-a-real-key"


def test_patch_engine_only_throttles_for_non_anthropic(monkeypatch):
    """Anthropic gets its client REPLACED (routed to our transport). Any other provider
    keeps the engine's native client and only gains throttle/retry — the vendor the
    engine already speaks must not be hijacked."""
    import rlm.clients.anthropic as ant_mod
    import rlm.core.rlm as core

    monkeypatch.setenv("GEMINI_API_KEY", "dummy-not-a-real-key")
    monkeypatch.setattr("src.config.load_config", lambda: _cfg(provider="gemini"))
    monkeypatch.setattr(core, "_rlmmcp_throttled", False, raising=False)
    monkeypatch.setattr(ant_mod, "_rlmmcp_patched", False, raising=False)

    before_client, before_get = ant_mod.AnthropicClient, core.get_client
    monkeypatch.setattr(core, "get_client", before_get)  # restored by monkeypatch
    auth.patch_engine()
    assert ant_mod.AnthropicClient is before_client   # NOT hijacked
    assert core.get_client is not before_get          # wrapped for throttle/retry
    wrapped = core.get_client
    auth.patch_engine()                               # idempotent
    assert core.get_client is wrapped


def test_make_client_requires_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        auth.make_client()


def test_make_client_builds_from_api_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api-x")
    assert auth.make_client() is not None
    assert auth.make_client(async_=True) is not None

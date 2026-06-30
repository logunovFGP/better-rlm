import pytest

import src.auth as auth


@pytest.fixture(autouse=True)
def _clear_creds(monkeypatch):
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


def test_clean_secret_ignores_placeholders_and_whitespace():
    assert auth._clean_secret("<paste>") is None
    assert auth._clean_secret("   ") is None
    assert auth._clean_secret("has space") is None
    assert auth._clean_secret("sk-ant-oat01-real") == "sk-ant-oat01-real"


def test_resolve_auth_prefers_oauth(monkeypatch):
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat01-x")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api-x")
    info = auth.resolve_auth()
    assert info.mode == "oauth" and info.secret == "sk-ant-oat01-x"


def test_resolve_auth_apikey_fallback(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api-x")
    assert auth.resolve_auth().mode == "apikey"


def test_resolve_auth_none_raises():
    with pytest.raises(RuntimeError):
        auth.resolve_auth()


def test_auth_status_transitions(monkeypatch):
    assert auth.auth_status() == "none"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api-x")
    assert auth.auth_status() == "apikey"
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat01-x")
    assert auth.auth_status() == "oauth"  # oauth wins


def test_make_client_rejects_oauth(monkeypatch):
    # OAuth must NOT build an SDK client — it drives the claude CLI transport.
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat01-x")
    with pytest.raises(RuntimeError, match="API-key auth only"):
        auth.make_client()


def test_make_client_builds_apikey_client(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api-x")
    client = auth.make_client()
    assert client is not None
    aclient = auth.make_client(async_=True)
    assert aclient is not None

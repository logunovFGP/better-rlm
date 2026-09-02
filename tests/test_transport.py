import dataclasses
import json
import types

import pytest

import src.transport as tp
from src.transport import (
    ApiTransport,
    CliCompletionError,
    CliRateLimitError,
    CliTransport,
    flatten_messages,
    get_transport,
    split_prompt,
)


# --------------------------- message helpers --------------------------- #
def test_split_prompt_string():
    msgs, system = split_prompt("hello")
    assert msgs == [{"role": "user", "content": "hello"}]
    assert system is None


def test_split_prompt_list_extracts_system():
    msgs, system = split_prompt([
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "U"},
        {"role": "assistant", "content": "A"},
    ])
    assert system == "SYS"
    assert msgs == [{"role": "user", "content": "U"}, {"role": "assistant", "content": "A"}]


def test_split_prompt_invalid_type():
    with pytest.raises(ValueError):
        split_prompt(42)


def test_flatten_single_user_is_verbatim():
    assert flatten_messages([{"role": "user", "content": "just this"}]) == "just this"


def test_flatten_multiturn_builds_transcript():
    out = flatten_messages([
        {"role": "user", "content": "U1"},
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "REPL OUT"},
    ])
    assert "## USER" in out and "## ASSISTANT" in out
    assert out.rstrip().endswith("## ASSISTANT")  # cue for the next assistant turn
    assert "U1" in out and "A1" in out and "REPL OUT" in out


def test_content_blocks_join_to_text():
    msgs, _ = split_prompt([{"role": "user", "content": [
        {"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}])
    assert flatten_messages(msgs) == "ab"


# --------------------------- strategy selection --------------------------- #
def inner_of(transport):
    """Look through the ledger wrapper get_transport applies to every backend."""
    return getattr(transport, "inner", transport)


def test_get_transport_selects_by_mode(cfg):
    tp._CACHE.clear()
    assert isinstance(inner_of(get_transport("oauth", cfg)), CliTransport)
    assert isinstance(inner_of(get_transport("apikey", cfg)), ApiTransport)
    tp._CACHE.clear()


def test_get_transport_selects_by_provider(cfg, monkeypatch):
    """Anthropic keeps its two dedicated transports (the local path); every other
    provider goes through the engine's own client."""
    import dataclasses

    tp._CACHE.clear()
    monkeypatch.setenv("GEMINI_API_KEY", "dummy-not-a-real-key")
    assert cfg.provider == "anthropic"                      # default = the local run
    assert isinstance(inner_of(get_transport("oauth", cfg)), CliTransport)

    for provider in ("gemini", "openai", "azure_openai", "portkey"):
        other = dataclasses.replace(cfg, provider=provider)
        t = get_transport("apikey", other)
        assert isinstance(inner_of(t), tp.EngineClientTransport), provider
        # The INNER transport is what gets cached (it owns the client and the neutral
        # cwd). The ledger wrapper is rebuilt per call ON PURPOSE, because it binds a
        # cfg -- caching it would hand the first caller's config, and its spend
        # ledger, to every later caller.
        assert inner_of(get_transport("apikey", other)) is inner_of(t)
    # the anthropic entry must not have been clobbered by the shared "apikey" mode
    assert isinstance(inner_of(get_transport("oauth", cfg)), CliTransport)
    tp._CACHE.clear()


# --------------------------- CLI argv --------------------------- #
def test_cli_argv_append_mode(cfg):
    c = dataclasses.replace(cfg, cli_system_prompt_mode="append")
    argv = CliTransport(c)._argv("claude-sonnet-4-6", "SYS")
    assert argv[:5] == [c.cli_path, "-p", "--output-format", "json", "--model"]
    assert "claude-sonnet-4-6" in argv
    assert "--safe-mode" in argv
    assert "--append-system-prompt" in argv and "SYS" in argv
    assert "--system-prompt" not in argv
    assert argv[argv.index("--tools") + 1] == ""   # --tools "" disables tools
    assert "--no-session-persistence" in argv


def test_cli_argv_replace_mode(cfg):
    c = dataclasses.replace(cfg, cli_system_prompt_mode="replace")
    argv = CliTransport(c)._argv("claude-haiku-4-5", "SYS")
    assert "--system-prompt" in argv
    assert "--append-system-prompt" not in argv


def test_cli_argv_no_system_omits_flag(cfg):
    argv = CliTransport(cfg)._argv("claude-haiku-4-5", None)
    assert "--append-system-prompt" not in argv and "--system-prompt" not in argv


# --------------------------- output parsing --------------------------- #
def test_parse_success():
    out = json.dumps({"subtype": "success", "is_error": False, "result": "ANSWER",
                      "usage": {"input_tokens": 11, "output_tokens": 7},
                      "total_cost_usd": 0.001})
    res = tp._parse_cli_output(0, out, "", "claude-sonnet-4-6")
    assert res.text == "ANSWER"
    assert res.input_tokens == 11 and res.output_tokens == 7
    assert res.cost_usd == 0.001 and res.model == "claude-sonnet-4-6"


def test_parse_rate_limit_error_raises_retryable():
    out = json.dumps({"subtype": "error", "is_error": True,
                      "result": "Rate limit exceeded, please retry"})
    with pytest.raises(CliRateLimitError):
        tp._parse_cli_output(1, out, "", "m")


def test_parse_other_error_raises_nonretryable():
    out = json.dumps({"subtype": "error_during_execution", "is_error": True, "result": "boom"})
    with pytest.raises(CliCompletionError):
        tp._parse_cli_output(1, out, "", "m")


def test_parse_non_json_raises():
    with pytest.raises(CliCompletionError):
        tp._parse_cli_output(1, "not json", "stderr detail", "m")


def test_parse_non_json_rate_limit_detected_from_stderr():
    with pytest.raises(CliRateLimitError):
        tp._parse_cli_output(1, "", "429 Too Many Requests", "m")


# --------------------------- complete() with mocked subprocess --------------------------- #
def test_cli_complete_with_mocked_subprocess(cfg, monkeypatch):
    captured = {}

    def fake_run(argv, **kw):
        captured["argv"] = argv
        captured["input"] = kw.get("input")
        return types.SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"subtype": "success", "is_error": False, "result": "OK",
                               "usage": {"input_tokens": 3, "output_tokens": 2}}),
            stderr="",
        )

    monkeypatch.setattr(tp.subprocess, "run", fake_run)
    res = CliTransport(cfg).complete(
        [{"role": "user", "content": "hi"}], "SYS", "claude-haiku-4-5", 100)
    assert res.text == "OK" and res.input_tokens == 3 and res.output_tokens == 2
    assert captured["input"] == "hi"          # single user msg flattened verbatim
    assert "--model" in captured["argv"]


def test_cli_rate_limit_is_recognized_by_ratelimit():
    import src.ratelimit as rl
    assert rl._is_rate_limit(CliRateLimitError("rate limited"))
    assert not rl._is_rate_limit(CliCompletionError("some other failure"))


class _CfgStub:
    cli_path = "claude"


_CFG_STUB = _CfgStub()

# --- CLI login diagnosis (free: no model call) --------------------------------
def test_auth_status_parses_the_cli_json(monkeypatch, tmp_path):
    import types as _t
    monkeypatch.setattr(tp.shutil, "which", lambda p: "/usr/local/bin/claude")
    monkeypatch.setattr(tp.subprocess, "run", lambda *a, **k: _t.SimpleNamespace(
        stdout='{"loggedIn": true, "authMethod": "oauth", "apiProvider": "firstParty"}',
        stderr="", returncode=0))
    st = tp.cli_auth_status(_CFG_STUB)
    assert st["loggedIn"] is True and st["authMethod"] == "oauth"


def test_auth_status_is_none_when_the_cli_cannot_be_asked(monkeypatch):
    monkeypatch.setattr(tp.shutil, "which", lambda p: None)
    assert tp.cli_auth_status(_CFG_STUB) is None

    monkeypatch.setattr(tp.shutil, "which", lambda p: "/usr/local/bin/claude")
    monkeypatch.setattr(tp.subprocess, "run",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("boom")))
    assert tp.cli_auth_status(_CFG_STUB) is None


def test_auth_failure_carries_the_remediation():
    """The error a user actually sees must say what to do about it. Being signed in
    to Claude Code does not sign in the CLI, and nothing else surfaces that."""
    payload = json.dumps({"subtype": "success", "is_error": True,
                          "result": "Failed to authenticate: OAuth session expired"})
    with pytest.raises(tp.CliAuthError) as e:
        tp._parse_cli_output(0, payload, "", "m")
    msg = str(e.value)
    assert "claude auth login" in msg
    assert "claude setup-token" in msg
    assert "does NOT sign in the CLI" in msg



# --------------------------- the floor sits under EVERY completion ------------------ #
def test_the_transport_refuses_a_call_that_would_cross_the_line_without_spending(cfg):
    """rlm_query's recursive fan-out had no stop at all; the batch's Gate only reached
    the batch. Placing the check here, where the ledger already is, makes it structural:
    the refused call must never reach the backend and must not be ledgered."""
    import dataclasses
    import src.budget as budget

    cfg = dataclasses.replace(cfg, session_budget_tokens=100_000, budget_stop_fraction=0.95)
    budget.record(cfg, "m", 94_000, 0)
    calls = []

    class _Backend:
        def complete(self, messages, system, model, max_tokens):
            calls.append(1)
            return tp.CompletionResult(text="ok", input_tokens=1, output_tokens=1, model="m")

        async def acomplete(self, *a):
            raise AssertionError("not used")

    w = tp._LedgeredTransport(_Backend(), cfg)
    with pytest.raises(budget.BudgetStopError):
        w.complete([{"role": "user", "content": "x" * 400}], None, "m", 2048)   # ~100 + 2048

    assert calls == [], "the refused call reached the backend"
    assert budget.spent(cfg).tokens == 94_000, "a refused call was ledgered"

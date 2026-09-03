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


def test_get_transport_refuses_every_provider_it_cannot_ledger(cfg, monkeypatch):
    """Anthropic keeps its two dedicated transports (the local path). Every other provider
    is refused HERE rather than handed the engine's own client, which resolved no
    transport of ours and so recorded no spend, passed no floor and learned no ceiling."""
    import dataclasses

    tp._CACHE.clear()
    monkeypatch.setenv("GEMINI_API_KEY", "dummy-not-a-real-key")
    assert cfg.provider == "anthropic"                      # default = the local run
    assert isinstance(inner_of(get_transport("oauth", cfg)), CliTransport)

    for provider in ("gemini", "openai", "azure_openai", "portkey"):
        other = dataclasses.replace(cfg, provider=provider)
        with pytest.raises(NotImplementedError, match=provider):
            get_transport("apikey", other)
        assert not any(k.startswith(provider) for k in tp._CACHE), (
            f"{provider} left a cached transport behind; a refusal must build nothing")
    # the anthropic entry must not have been disturbed by the refusals
    assert isinstance(inner_of(get_transport("oauth", cfg)), CliTransport)
    tp._CACHE.clear()


def test_the_inner_transport_is_cached_but_the_ledger_wrapper_is_not(cfg):
    """The INNER transport is cached (it owns the client and the neutral cwd). The ledger
    wrapper is rebuilt per call ON PURPOSE, because it binds a cfg -- caching it would
    hand the first caller's config, and its spend ledger, to every later caller."""
    tp._CACHE.clear()
    first = get_transport("oauth", cfg)
    second = get_transport("oauth", cfg)
    assert inner_of(first) is inner_of(second)
    assert first is not second
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


def test_the_floor_reserves_what_a_call_emits_not_the_cap_the_cli_discards(cfg):
    """CliTransport is handed max_tokens and has nowhere to put it — `claude` takes no
    output flag — so on the OAuth path a call can emit several times the cap. Reserving the
    cap under-reserves, and a floor whose whole job is to stop short of the wall lets one
    call step over it. Ceiling picked so the cap fits and the measured figure does not."""
    import dataclasses
    import src.budget as budget

    cfg = dataclasses.replace(cfg, session_budget_tokens=90_000, budget_stop_fraction=0.95)
    for _ in range(8):                       # measured mean output: 10,000 per call
        budget.record(cfg, "m", 0, 10_000)
    assert budget.spent(cfg).tokens == 80_000
    calls = []

    class _Backend:
        def complete(self, messages, system, model, max_tokens):
            calls.append(1)
            return tp.CompletionResult(text="ok", input_tokens=1, output_tokens=1, model="m")

        async def acomplete(self, *a):
            raise AssertionError("not used")

    w = tp._LedgeredTransport(_Backend(), cfg)
    # est_in ~100. Reserving the cap gives 82,148 — under the 85,500 line, so it would be
    # admitted. Reserving what the path actually emits gives 90,100, which is not.
    with pytest.raises(budget.BudgetStopError):
        w.complete([{"role": "user", "content": "x" * 400}], None, "m", 2048)

    assert calls == [], "reserved the unenforceable cap, so the floor admitted the call"


# --- input accounting: the cap defect with the operands swapped ------------------------

def test_total_input_counts_the_cache_not_just_the_uncached_remainder():
    """`input_tokens` alone is what is left AFTER prompt caching, and on this path that is
    almost nothing. Measured on a real 30k review chunk: input_tokens 10 against 38,470
    cache_creation and 25,679 cache_read -- so reading that one field understated the
    call's input by more than 6,000x (docs/07 §12)."""
    usage = {"input_tokens": 10, "cache_creation_input_tokens": 38_470,
             "cache_read_input_tokens": 25_679, "output_tokens": 15_768}
    assert tp._total_input(usage) == 64_159

    class _SdkUsage:            # the SDK hands an object, not a mapping
        input_tokens = 10
        cache_creation_input_tokens = 38_470
        cache_read_input_tokens = 25_679

    assert tp._total_input(_SdkUsage()) == 64_159
    # A transport that reports no cache fields at all must still work, unchanged.
    assert tp._total_input({"input_tokens": 700}) == 700


def test_the_ledger_records_the_transports_total_when_it_beats_our_estimate(cfg):
    """docs/07 §4 wrote the transport's input figure off as unusable, on the strength of a
    field that reported 1027 for ~3M tokens. It is usable once the cache fields are added
    in -- and max() picks whichever is closer to the truth with no knowledge of which
    transport is in play, exactly as expected_output does for the output side."""
    import json
    import src.budget as budget

    class _Backend:
        def complete(self, messages, system, model, max_tokens):
            # est_in for this message is ~4 tokens; the transport reports far more.
            return tp.CompletionResult(text="ok", input_tokens=64_159,
                                       output_tokens=15_768, model="m")

        async def acomplete(self, *a):
            raise AssertionError("not used")

    tp._LedgeredTransport(_Backend(), cfg).complete(
        [{"role": "user", "content": "hi"}], None, "m", 2048)

    rec = json.loads(cfg.budget_ledger.read_text(encoding="utf-8").splitlines()[-1])
    assert rec["itok"] == 64_159, "ledgered our estimate over the transport's real total"
    assert rec["est"] < 100, "the local estimate must ride along for input_overhead"
    assert rec["otok"] == 15_768


def test_the_floor_reserves_the_measured_input_overhead(cfg):
    """The mirror of the output defect. Our estimate covers what we hand over, not the
    harness context that rides along with it -- ~29k per call on the CLI path, measured.
    Unreserved, that is the one number a floor exists to keep honest.

    Ceiling picked so the bare estimate fits and the estimate-plus-overhead does not.
    """
    import dataclasses
    import src.budget as budget

    # Teaching the overhead necessarily spends: 8 records of itok 29,001 against est 1.
    # The ceiling is then picked so the leftover headroom sits BETWEEN the two reserves,
    # which is the only arrangement where the overhead is what decides the refusal.
    cfg = dataclasses.replace(cfg, session_budget_tokens=250_000, budget_stop_fraction=0.95)
    for _ in range(8):
        budget.record(cfg, "m", 29_001, 0, est=1)
    assert budget.input_overhead(cfg) == 29_000
    headroom = int(250_000 * 0.95) - budget.spent(cfg).tokens
    assert 0 < headroom < 29_000, f"ceiling mis-picked: headroom {headroom:,}"

    msg = [{"role": "user", "content": "x" * 40}]
    # Asserted in BOTH directions, or this passes on a budget that was merely exhausted --
    # which is exactly how the first draft of it passed with the overhead term deleted.
    bare = tp._LedgeredTransport._est_in(msg, None) + budget.expected_output(cfg, 16)
    budget.check_or_raise(cfg, bare)      # fits comfortably without the overhead

    class _Backend:
        def complete(self, messages, system, model, max_tokens):
            raise AssertionError("the floor should have refused before dispatch")

        async def acomplete(self, *a):
            raise AssertionError("not used")

    w = tp._LedgeredTransport(_Backend(), cfg)
    with pytest.raises(budget.BudgetStopError):
        w.complete(msg, None, "m", 16)    # ...and does not fit with it


def test_both_parsers_route_input_through_total_input():
    """_total_input can be correct and unreached. Nothing else in the suite proves either
    parser calls it, so deleting the call from one -- or from both -- was invisible: the
    same gap a tested-but-uncalled sweep had. One assertion per parser, on the wiring.
    """
    import json
    usage = {"input_tokens": 10, "cache_creation_input_tokens": 38_470,
             "cache_read_input_tokens": 25_679, "output_tokens": 15_768}

    cli = tp._parse_cli_output(
        0, json.dumps({"type": "result", "subtype": "success", "result": "ok",
                       "usage": usage, "total_cost_usd": 0.1943}), "", "m")
    assert cli.input_tokens == 64_159, "the CLI parser reported the uncached remainder"
    assert cli.output_tokens == 15_768

    resp = types.SimpleNamespace(
        content=[types.SimpleNamespace(type="text", text="ok")],
        usage=types.SimpleNamespace(**usage))

    sdk = tp._result_from_sdk_response(resp, "m")
    assert sdk.input_tokens == 64_159, "the SDK parser reported the uncached remainder"

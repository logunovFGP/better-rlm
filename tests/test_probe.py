"""The connectivity probe: what it tells the operator, and what it must never tell them.

cline-2 deliberately has no pre-flight on its API path -- bad credentials surface at
the first real call. Running one anyway is this fork's divergence, so it has to earn
it: a probe that cannot tell "wrong URL" from "wrong key" is worse than no probe,
because it sends people to fix the thing that was already right.

Every test here stubs ``probe._probe_call``. conftest's spend guard stubs it too, so
a test that forgets fails loudly instead of quietly making a billed call -- the exact
failure mode that guard was written for.
"""

from __future__ import annotations

import dataclasses

import anthropic
import httpx
import pytest

from better_rlm import probe
from better_rlm.describe import MODE_API, MODE_CLI

MINIMAX_URL = "https://api.minimax.io/anthropic"
FAKE_KEY = "sk-fake-probe-key-0123456789abcdef"


def _response(status: int, text: str = "") -> httpx.Response:
    return httpx.Response(status, request=httpx.Request("POST", "https://x/v1/messages"),
                          text=text)


def _read_timeout(client) -> float:
    """The SDK stores timeout as a float or an httpx.Timeout."""
    return float(getattr(client.timeout, "read", client.timeout))


def _cfg(tmp_path, mode: str = MODE_API, base_url: str = MINIMAX_URL):
    """A config pointed at an endpoint, with nothing else touched."""
    from better_rlm.config import load_config

    return dataclasses.replace(load_config(), mode=mode, base_url=base_url,
                               cli_path="claude")


@pytest.fixture(autouse=True)
def _a_key_is_present(monkeypatch):
    """The host path needs a key to get as far as the call it is being tested on."""
    monkeypatch.setenv("MINIMAX_API_KEY", FAKE_KEY)
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_KEY)
    # A base_url the catalogue does not recognise -- an edited one, or cline
    # /v1 string -- reverse-maps to the custom provider, which reads its own
    # variable. Without this the /v1 test fails as "no credential" and never
    # reaches the 404 it is actually about.
    monkeypatch.setenv("RLM_API_KEY", FAKE_KEY)


# --- the taxonomy ---------------------------------------------------------------


@pytest.mark.parametrize("exc, expected", [
    (anthropic.AuthenticationError("bad key", response=_response(401), body=None),
     probe.PROBE_KEY_REJECTED),
    (anthropic.PermissionDeniedError("no access", response=_response(403), body=None),
     probe.PROBE_KEY_REJECTED),
    (anthropic.NotFoundError("not found", response=_response(404), body=None),
     probe.PROBE_URL_WRONG),
    (anthropic.NotFoundError("model MiniMax-M2.7 does not exist",
                             response=_response(404), body=None),
     probe.PROBE_MODEL_UNKNOWN),
    (anthropic.BadRequestError("malformed", response=_response(400), body=None),
     probe.PROBE_BAD_REQUEST),
    (anthropic.BadRequestError("unknown model MiniMax-M2.7", response=_response(400),
                               body=None),
     probe.PROBE_MODEL_UNKNOWN),
    (anthropic.RateLimitError("slow down", response=_response(429), body=None),
     probe.PROBE_RATE_LIMITED),
    (anthropic.InternalServerError("boom", response=_response(500), body=None),
     probe.PROBE_SERVER_ERROR),
    (anthropic.APIConnectionError(request=httpx.Request("POST", "https://x")),
     probe.PROBE_NETWORK_DOWN),
    (anthropic.APITimeoutError(request=httpx.Request("POST", "https://x")),
     probe.PROBE_NETWORK_DOWN),
    (ValueError("something else entirely"), probe.PROBE_UNEXPECTED),
])
def test_the_probe_maps_every_failure_to_its_own_code(tmp_path, monkeypatch, exc, expected):
    """The three the operator most needs told apart -- URL, key, network -- come from
    three disjoint exception branches, not from sniffing one message."""
    monkeypatch.setattr(probe, "_probe_call",
                        lambda *_a, **_k: (_ for _ in ()).throw(exc))
    res = probe.probe_endpoint(_cfg(tmp_path), "MiniMax-M2.7")
    assert res.code == expected


def test_a_rate_limit_counts_as_reachable():
    """429 means the endpoint accepted the credential in order to throttle it.

    Treating it as a failure would send someone to re-paste a key that is fine, and
    on a busy endpoint would make setup impossible to finish.
    """
    assert probe.ProbeResult(probe.PROBE_RATE_LIMITED, "d", "w").ok is True
    assert probe.ProbeResult(probe.PROBE_KEY_REJECTED, "d", "w").ok is False
    assert probe.ProbeResult(probe.PROBE_URL_WRONG, "d", "w").ok is False


def test_a_404_names_the_v1_footgun(tmp_path, monkeypatch):
    """The single likeliest way to misconfigure MiniMax is pasting cline's own URL,
    which ends in /v1 -- the SDK then asks for /v1/v1/messages. Say so at the 404."""
    monkeypatch.setattr(probe, "_probe_call", lambda *_a, **_k: (_ for _ in ()).throw(
        anthropic.NotFoundError("nope", response=_response(404), body=None)))
    res = probe.probe_endpoint(_cfg(tmp_path, base_url=MINIMAX_URL + "/v1"), "MiniMax-M3")
    assert res.code == probe.PROBE_URL_WRONG
    assert "/v1" in res.fix


def test_a_reachable_endpoint_reports_ok(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, "_probe_call", lambda *_a, **_k: None)
    res = probe.probe_endpoint(_cfg(tmp_path), "MiniMax-M3")
    assert res.ok and res.code == probe.PROBE_OK
    assert res.where == MINIMAX_URL


# --- what it must never say -----------------------------------------------------


def test_the_probe_never_puts_the_key_in_its_result(tmp_path, monkeypatch):
    """An endpoint that echoes the credential back in an error body must not get it
    reprinted onto the operator's terminal, or into whatever captures that."""
    leaky = anthropic.AuthenticationError(
        f"invalid x-api-key: {FAKE_KEY}", response=_response(401), body=None)
    monkeypatch.setattr(probe, "_probe_call",
                        lambda *_a, **_k: (_ for _ in ()).throw(leaky))
    res = probe.probe_endpoint(_cfg(tmp_path), "MiniMax-M3")
    assert res.code == probe.PROBE_KEY_REJECTED
    for part in (res.detail, res.where, res.fix):
        assert FAKE_KEY not in part, f"the key leaked into {part!r}"
    assert "***" in res.detail


def test_a_scrubbed_detail_stays_one_short_line():
    """It is printed into a panel. A multi-kilobyte HTML error page would bury the
    screen, and a newline would break the frame."""
    long = ("x" * 5000) + chr(10) + chr(10) + ("y" * 5000)
    out = probe._scrub(long, "")
    assert len(out) <= 200 and chr(10) not in out


# --- which config, and at whose expense -----------------------------------------


def test_the_probe_tests_the_config_being_edited(tmp_path, monkeypatch):
    """_run_auth_probe called load_config() with no path, so a TUI started with
    --config elsewhere probed THIS checkout's config -- reporting on a file nobody
    was editing. The probe takes the prospective config as an argument instead."""
    seen: dict[str, str] = {}

    def capture(client, model):
        seen["url"] = str(client.base_url)
        seen["model"] = model

    monkeypatch.setattr(probe, "_probe_call", capture)
    probe.probe_endpoint(_cfg(tmp_path, base_url=MINIMAX_URL), "MiniMax-M2.7")
    assert MINIMAX_URL in seen["url"]
    assert seen["model"] == "MiniMax-M2.7"


def test_the_probe_builds_a_client_that_gives_up_in_seconds(tmp_path, monkeypatch):
    """The completion timeout is 600s. A setup wizard that hangs ten minutes on an
    unreachable host is indistinguishable from one that has crashed."""
    seen: dict[str, float] = {}
    monkeypatch.setattr(probe, "_probe_call",
                        lambda client, model: seen.update(timeout=_read_timeout(client)))
    probe.probe_endpoint(_cfg(tmp_path), "MiniMax-M3", timeout_s=20.0)
    assert seen["timeout"] == 20.0


def test_the_probe_does_not_go_through_the_budget_ledger(tmp_path, monkeypatch):
    """The ledger refuses calls past the session stop line. Routing the probe through
    it would report an exhausted budget as an auth failure -- telling someone their
    key is broken when it is not."""
    called: list[str] = []
    monkeypatch.setattr(probe, "_probe_call", lambda *_a, **_k: None)

    import better_rlm.transport as t
    monkeypatch.setattr(t, "get_transport",
                        lambda *a, **k: called.append("transport"))

    assert probe.probe_endpoint(_cfg(tmp_path), "MiniMax-M3").ok
    assert called == [], "the probe went through the ledgered transport"


# --- the proxy half -------------------------------------------------------------


def test_the_proxy_probe_spends_nothing(tmp_path, monkeypatch):
    """claude auth status answers the question directly. A model call to infer the
    same thing from a success would cost tokens for no extra information."""
    spent: list[str] = []
    monkeypatch.setattr(probe, "_probe_call", lambda *_a, **_k: spent.append("call"))
    monkeypatch.setattr(probe, "cli_auth_status",
                        lambda cfg: {"loggedIn": True, "authMethod": "oauth"})
    monkeypatch.setattr(probe.shutil, "which", lambda _n: "/usr/bin/claude")

    res = probe.probe_endpoint(_cfg(tmp_path, mode=MODE_CLI, base_url=""), "ignored")
    assert res.ok and spent == []


@pytest.mark.parametrize("which, status, expected", [
    (None, None, probe.PROBE_CLI_MISSING),
    ("/usr/bin/claude", None, probe.PROBE_CLI_UNKNOWN),
    ("/usr/bin/claude", {"loggedIn": False}, probe.PROBE_CLI_LOGGED_OUT),
])
def test_the_proxy_probe_separates_absent_from_silent_from_logged_out(
        tmp_path, monkeypatch, which, status, expected):
    """Three different fixes. "It did not work" would name none of them."""
    monkeypatch.setattr(probe.shutil, "which", lambda _n: which)
    monkeypatch.setattr(probe, "cli_auth_status", lambda cfg: status)
    res = probe.probe_endpoint(_cfg(tmp_path, mode=MODE_CLI, base_url=""), "ignored")
    assert res.code == expected
    assert res.fix, "a failure with no suggested fix leaves the operator stuck"


def test_a_missing_credential_is_not_reported_as_a_rejected_one(tmp_path, monkeypatch):
    """Nothing to send and something sent-and-refused are different problems."""
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(probe.shutil, "which", lambda _n: None)
    res = probe.probe_endpoint(_cfg(tmp_path), "MiniMax-M3")
    assert res.code == probe.PROBE_NO_CREDENTIAL
    assert not res.ok

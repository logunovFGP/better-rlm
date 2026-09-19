"""Connectivity probe -- can this configuration actually make a model call?

Split from transport.py rather than appended to it: that module was already 640
lines and the repo's ceiling is 800. This is one cohesive job (ask the endpoint a
question, classify the answer), so it gets its own file instead of pushing a file
that does a different job over the limit.

Imports transport for the proxy half -- `cli_auth_status` already answers "is the
CLI logged in?" directly and for free, so there is one connectivity concept here,
not two. transport does not import this module, so there is no cycle.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass

from . import failures
from .transport import AUTH_REMEDIATION, cli_auth_status

# --------------------------------------------------------------------------- #
# Connectivity probe
# --------------------------------------------------------------------------- #
#
# cline-2 deliberately has NO pre-flight on its API path -- controller.ts carries
# the comment "No required-field validation. If credentials are missing or wrong,
# the provider's own auth response is the authoritative error." It DOES probe the
# local-CLI path (checkLocalCli spawns the agent binary and asks its auth status).
#
# The wizard running it on BOTH paths is a deliberate divergence: a wrong base_url
# or a mistyped key is caught while the operator is still looking at the screen
# that produced it, not at the first real query an hour later. It earns the
# deviation twice, because the credential form has no validation (faithful to
# cline) and something has to catch a bad key before the model screens.

# The shared vocabulary, re-exported under the names the wizard and the TUI already
# import. One spelling of "key_rejected" for the probe panel, the retry decision and the
# fan-out abort -- see failures.py for what they used to disagree about.
PROBE_OK = failures.CODE_OK
PROBE_RATE_LIMITED = failures.CODE_RATE_LIMITED
PROBE_KEY_REJECTED = failures.CODE_KEY_REJECTED
PROBE_URL_WRONG = failures.CODE_URL_WRONG
PROBE_MODEL_UNKNOWN = failures.CODE_MODEL_UNKNOWN
PROBE_BAD_REQUEST = failures.CODE_BAD_REQUEST
PROBE_NETWORK_DOWN = failures.CODE_NETWORK_DOWN
PROBE_SERVER_ERROR = failures.CODE_SERVER_ERROR
PROBE_NO_CREDENTIAL = failures.CODE_NO_CREDENTIAL
PROBE_CLI_MISSING = "cli_missing"
PROBE_CLI_LOGGED_OUT = "cli_logged_out"
PROBE_CLI_UNKNOWN = "cli_unknown"
PROBE_UNEXPECTED = failures.CODE_UNEXPECTED

#: Codes meaning the credential and the endpoint are both fine. A rate limit is a
#: pass: the endpoint had to accept the credential in order to throttle it.
_PROBE_GOOD = frozenset({PROBE_OK, PROBE_RATE_LIMITED})

_DETAIL_MAX = 200


@dataclass(frozen=True)
class ProbeResult:
    """What the connectivity test learned. Never carries a credential.

    ``detail`` is the scrubbed one-line reason, safe to print. ``where`` names what
    was contacted. ``fix`` is the single next action, or "" when there is none.
    """

    code: str
    detail: str
    where: str
    fix: str = ""

    @property
    def ok(self) -> bool:
        return self.code in _PROBE_GOOD


def _scrub(text: str, secret: str) -> str:
    """One printable line with the live key removed.

    The SDK's exception text carries the response body and status, never the request
    headers, so this is belt-and-braces at a trust boundary -- which is exactly where
    being thorough costs less than being sorry.
    """
    one = " ".join((text or "").split())
    if secret and len(secret) >= 8:
        one = one.replace(secret, "***")
    return one[:_DETAIL_MAX] or "no detail"


def _probe_call(client, model: str) -> None:
    """Issue the smallest real call the endpoint will accept.

    A separate function so a test can replace exactly this and nothing else --
    conftest's spend guard hooks it the way it hooks ``subquery._call``.
    """
    client.messages.create(
        model=model,
        max_tokens=16,
        messages=[{"role": "user", "content": "ping"}],
    )


def _endpoint_label(cfg) -> str:
    return (cfg.base_url or "").strip() or "api.anthropic.com"


def _probe_cli(cfg) -> ProbeResult:
    """Proxy path: ask the CLI whether it is logged in. Free, ~215 ms, no tokens."""
    where = f"the {cfg.cli_path} CLI"
    if not shutil.which(cfg.cli_path):
        return ProbeResult(PROBE_CLI_MISSING, f"{cfg.cli_path} is not on PATH", where,
                           "Install Claude Code, or choose the API-key path instead.")
    st = cli_auth_status(cfg)
    if st is None:
        return ProbeResult(PROBE_CLI_UNKNOWN, "the CLI did not answer auth status --json",
                           where, "Run `claude auth status` by hand to see what it says.")
    if not st.get("loggedIn"):
        return ProbeResult(PROBE_CLI_LOGGED_OUT, "the CLI reports it is not logged in",
                           where, AUTH_REMEDIATION)
    return ProbeResult(PROBE_OK, f"logged in ({st.get('authMethod', '?')})", where)


def _classify(exc: Exception, model: str, where: str, key: str) -> ProbeResult:
    """Turn one SDK exception into a ProbeResult: shared code, local prose.

    The code comes from failures.classify -- disjoint exception branches, not string
    sniffing, except in the one place the protocol needs it: a 404 means either
    "nothing here speaks this protocol" or "this endpoint does not serve that model",
    and only the body tells them apart, which is why ``model`` is passed through.

    What stays here is the fix text. A taxonomy answers "what happened"; this module
    exists to answer "what do I do now", on a screen an operator is looking at.
    """
    import anthropic

    detail = _scrub(str(exc), key)
    code = failures.classify(exc, model=model)

    # What to DO about it -- the half that is genuinely this module's. The classification
    # is shared with the transports (failures.py), because the 403 the wizard calls "key
    # rejected" has to be the same 403 that aborts a fan-out instead of being retried
    # once per chunk. It was not, until these were one function.
    fix = {
        PROBE_KEY_REJECTED:
            f"{where} refused the credential. Re-paste the key, or check that the "
            "account has access to this endpoint.",
        PROBE_RATE_LIMITED:
            "The credential was accepted; the endpoint is busy. Nothing to fix.",
        PROBE_MODEL_UNKNOWN:
            f"{where} does not serve {model}. Pick another model.",
        PROBE_URL_WRONG:
            f"Nothing speaking the Anthropic messages format at {where}. Note the SDK "
            "appends /v1/messages itself, so a base URL ending in /v1 asks for "
            "/v1/v1/messages.",
        PROBE_BAD_REQUEST: f"{where} rejected the request shape.",
        PROBE_SERVER_ERROR: f"{where} is reachable but failing. Vendor-side; retry.",
        PROBE_NETWORK_DOWN:
            f"Could not reach {where} -- DNS, TLS, proxy or firewall. "
            "The credential was never sent.",
    }.get(code, "")

    # Two details the taxonomy cannot carry, because they are presentation: a timeout's
    # exception text says nothing useful, and an unexpected error is worth naming.
    if isinstance(exc, anthropic.APITimeoutError):
        detail = f"timed out contacting {where}"
    elif code == PROBE_UNEXPECTED:
        detail = f"{type(exc).__name__}: {detail}"
    return ProbeResult(code, detail, where, fix)


def probe_endpoint(cfg, model: str, *, timeout_s: float = 20.0) -> ProbeResult:
    """Can this configuration actually make a model call? Never raises.

    Pass a PROSPECTIVE config -- ``dataclasses.replace(load_config(), mode=...,
    base_url=...)`` -- so the wizard tests the settings it is about to write rather
    than whatever is already on disk. ``_run_auth_probe`` used to call
    ``load_config()`` with no path and so probed the DEFAULT config even under
    ``--config other.yaml``.

    Deliberately does NOT go through ``get_transport``: the ledger wrapper refuses
    calls past the session stop line, so a probe on an exhausted budget would report
    a budget stop as an auth failure. Nothing is recorded either -- 16 tokens is
    noise, and on a first run the ledger may not exist yet.
    """
    from .auth import MODE_CLI, api_key_for, make_client, resolve_auth_mode

    # An explicitly configured proxy mode is answered by the proxy check, before
    # resolve_auth_mode gets a chance to raise its generic "no transport available".
    # Otherwise a missing CLI reported as "no credential" would send the operator
    # looking for an API key on the path that deliberately has none.
    if cfg.mode == MODE_CLI:
        return _probe_cli(cfg)

    try:
        auth_mode = resolve_auth_mode(cfg)
    except Exception as exc:                   # noqa: BLE001 - report, never raise
        return ProbeResult(PROBE_NO_CREDENTIAL, _scrub(str(exc), ""),
                           _endpoint_label(cfg),
                           "Neither the claude CLI login nor an API key is usable yet.")

    if auth_mode == "oauth":
        return _probe_cli(cfg)

    where = _endpoint_label(cfg)
    key = api_key_for(cfg)
    try:
        client = make_client(base_url=cfg.base_url, cfg=cfg, timeout=timeout_s)
    except Exception as exc:                   # noqa: BLE001
        return ProbeResult(PROBE_NO_CREDENTIAL, _scrub(str(exc), key), where,
                           "Set the API key for this endpoint and try again.")
    try:
        _probe_call(client, model)
    except Exception as exc:                   # noqa: BLE001 - classified, never raised
        return _classify(exc, model, where, key)
    return ProbeResult(PROBE_OK, f"{model} answered", where)

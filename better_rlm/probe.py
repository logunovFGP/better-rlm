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

PROBE_OK = "ok"
PROBE_RATE_LIMITED = "rate_limited"
PROBE_KEY_REJECTED = "key_rejected"
PROBE_URL_WRONG = "url_wrong"
PROBE_MODEL_UNKNOWN = "model_unknown"
PROBE_BAD_REQUEST = "bad_request"
PROBE_NETWORK_DOWN = "network_down"
PROBE_SERVER_ERROR = "server_error"
PROBE_NO_CREDENTIAL = "no_credential"
PROBE_CLI_MISSING = "cli_missing"
PROBE_CLI_LOGGED_OUT = "cli_logged_out"
PROBE_CLI_UNKNOWN = "cli_unknown"
PROBE_UNEXPECTED = "unexpected"

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
    """Map one SDK exception onto the taxonomy.

    Disjoint exception branches, not string sniffing -- except in the one place the
    protocol genuinely needs it: a 404 means either "nothing here speaks this
    protocol" or "this endpoint does not serve that model", and only the body tells
    them apart.
    """
    import anthropic

    detail = _scrub(str(exc), key)
    names_model = bool(model) and model.lower() in detail.lower()

    if isinstance(exc, anthropic.APITimeoutError):
        return ProbeResult(PROBE_NETWORK_DOWN, f"timed out contacting {where}", where,
                           "Check the URL, the network and any proxy. "
                           "The credential was never sent.")
    if isinstance(exc, anthropic.APIConnectionError):
        return ProbeResult(PROBE_NETWORK_DOWN, detail, where,
                           f"Could not reach {where} -- DNS, TLS, proxy or firewall. "
                           "The credential was never sent.")
    if isinstance(exc, anthropic.APIResponseValidationError):
        return ProbeResult(PROBE_URL_WRONG, detail, where,
                           f"{where} answered, but not in the Anthropic messages "
                           "format. Check the base URL.")
    if isinstance(exc, (anthropic.AuthenticationError, anthropic.PermissionDeniedError)):
        return ProbeResult(PROBE_KEY_REJECTED, detail, where,
                           f"{where} refused the credential. Re-paste the key, or check "
                           "that the account has access to this endpoint.")
    if isinstance(exc, anthropic.RateLimitError):
        return ProbeResult(PROBE_RATE_LIMITED, detail, where,
                           "The credential was accepted; the endpoint is busy. "
                           "Nothing to fix.")
    if isinstance(exc, anthropic.NotFoundError):
        if names_model:
            return ProbeResult(PROBE_MODEL_UNKNOWN, detail, where,
                               f"{where} does not serve {model}. Pick another model.")
        return ProbeResult(PROBE_URL_WRONG, detail, where,
                           f"Nothing speaking the Anthropic messages format at {where}. "
                           "Note the SDK appends /v1/messages itself, so a base URL "
                           "ending in /v1 asks for /v1/v1/messages.")
    if isinstance(exc, (anthropic.BadRequestError, anthropic.UnprocessableEntityError)):
        if names_model:
            return ProbeResult(PROBE_MODEL_UNKNOWN, detail, where,
                               f"{where} rejected the model id {model}. Pick another.")
        return ProbeResult(PROBE_BAD_REQUEST, detail, where,
                           f"{where} rejected the request shape.")
    if isinstance(exc, anthropic.InternalServerError):
        return ProbeResult(PROBE_SERVER_ERROR, detail, where,
                           f"{where} is reachable but failing. Vendor-side; retry.")
    if isinstance(exc, anthropic.APIStatusError):
        status = getattr(exc, "status_code", 0) or 0
        if status >= 500:
            return ProbeResult(PROBE_SERVER_ERROR, detail, where,
                               f"{where} returned {status}. Vendor-side; retry.")
        return ProbeResult(PROBE_UNEXPECTED, detail, where, "")
    return ProbeResult(PROBE_UNEXPECTED, f"{type(exc).__name__}: {detail}", where, "")


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

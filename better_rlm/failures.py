"""What kind of failure is this? Asked once, for both transports.

The strategy split is real and stays: one route spawns the `claude` CLI, the other calls
the Anthropic SDK. What must NOT be per-route is the judgement downstream of it -- and it
was, in four places:

  * ``transport``  string markers over the CLI's own message;
  * ``ratelimit``  SDK exception types, for the retry decision;
  * ``probe``      a fuller taxonomy, again from SDK types, for the setup wizard;
  * ``transport._note_if_limit``  a third spelling of "is this a 429", inline.

Each was extended on the path whose bug prompted it, so they disagreed. Measured before
this module existed:

  * **403 fanned out.** ``PermissionDeniedError`` is not an ``AuthenticationError``
    subclass, so the API path retried nothing and aborted nothing: one doomed call per
    chunk, the exact waste the 401 fast-fail exists to prevent. ``probe`` had already
    been calling it ``key_rejected``.
  * **529 retried on OAuth only.** ``"overloaded"`` is a CLI marker; the SDK side had no
    5xx branch at all.
  * **A bare ``"429"`` substring** decided whether a rate limit was recorded as evidence
    about the account's ceiling -- so a message carrying ``5,429,000`` raised a floor
    nobody had hit.

RESOLUTION ORDER IS THE DESIGN. Duck-typed flags first, so an error the CLI raised
classifies without the SDK being consulted at all; then SDK exception types; then string
markers, which are the CLI's only signal and the weakest evidence here. Both routes reach
one answer from the evidence each actually has.

The vocabulary is what the setup wizard already shows an operator (``onboard`` prints the
code in a panel title, ``tui`` in the probe line), so these names are not new -- they are
the existing public ones, moved where every caller can reach them.
"""

from __future__ import annotations

CODE_OK = "ok"
CODE_RATE_LIMITED = "rate_limited"
CODE_KEY_REJECTED = "key_rejected"
CODE_URL_WRONG = "url_wrong"
CODE_MODEL_UNKNOWN = "model_unknown"
CODE_BAD_REQUEST = "bad_request"
CODE_NETWORK_DOWN = "network_down"
CODE_SERVER_ERROR = "server_error"
CODE_NO_CREDENTIAL = "no_credential"
CODE_UNEXPECTED = "unexpected"

#: Substrings in the CLI's own message. It reports no status code and raises no typed
#: error, so this is the only signal that path gives. Best-effort by design: if a marker
#: stops matching we degrade to "unexpected", which is the previous behaviour, never to
#: something worse.
_RATE_LIMIT_MARKERS = (
    "rate limit", "rate_limit", "overloaded", "usage limit",
    "too many requests", "please run /upgrade", "quota",
)

#: ``"429"`` is deliberately NOT a marker. As a bare substring it matches any
#: comma-formatted number containing it -- ``5,429,000 tokens`` -- and this
#: classification feeds ``budget.note_limit_hit``, which raises the learned ceiling. The
#: status code is read from ``status_code`` instead, where it is unambiguous.
_AUTH_FAIL_MARKERS = (
    "failed to authenticate",
    "oauth session expired",
    "oauth token has expired",
    "invalid api key",
    "authentication_error",
    "please run /login",
)


def _status(exc: object) -> int:
    try:
        return int(getattr(exc, "status_code", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _names_model(exc: BaseException, model: str) -> bool:
    return bool(model) and model.lower() in str(exc).lower()


def classify(exc: BaseException, *, model: str = "") -> str:
    """One code for one failure, from whatever evidence this exception carries.

    ``model`` splits the one case the protocol cannot: a 404 means either "nothing here
    speaks this protocol" or "this endpoint does not serve that model", and only the body
    names which. A caller that does not know the model gets ``url_wrong``, the safer
    reading -- a wrong base URL is by far the more common setup mistake.
    """
    # 1. Duck-typed flags. The CLI's own errors carry these, and honouring them first is
    #    what lets this module classify that path without ever seeing the SDK.
    if getattr(exc, "is_rate_limit", False):
        return CODE_RATE_LIMITED
    # `is_fatal_subcall` carries two meanings today: "the login is dead" (CliAuthError)
    # and "the session budget is exhausted" (budget.BudgetStopError). Only the first is
    # an auth failure. The second is scheduled work, and calling it key_rejected would
    # tell an operator their credential was refused when it was not. The engine's own
    # STOPS_RUN_ATTR is what separates them, so use it rather than invent a flag.
    if getattr(exc, "is_fatal_subcall", False) and not getattr(exc, "is_session_budget_stop", False):
        return CODE_KEY_REJECTED

    # 2. SDK exception types. Imported lazily: ratelimit and transport sit on the import
    #    path of every tool call, and this keeps them off the anthropic package until a
    #    failure actually needs classifying.
    import anthropic

    if isinstance(exc, (anthropic.APITimeoutError, anthropic.APIConnectionError)):
        return CODE_NETWORK_DOWN
    if isinstance(exc, anthropic.APIResponseValidationError):
        return CODE_URL_WRONG
    if isinstance(exc, (anthropic.AuthenticationError, anthropic.PermissionDeniedError)):
        # 403 sits with 401 on purpose. An org without access to this endpoint is as dead
        # for the next chunk as a bad key is, and treating it as a per-call failure is
        # what produced one doomed call per chunk on the API path.
        return CODE_KEY_REJECTED
    if isinstance(exc, anthropic.RateLimitError):
        return CODE_RATE_LIMITED
    if isinstance(exc, anthropic.NotFoundError):
        return CODE_MODEL_UNKNOWN if _names_model(exc, model) else CODE_URL_WRONG
    if isinstance(exc, (anthropic.BadRequestError, anthropic.UnprocessableEntityError)):
        return CODE_MODEL_UNKNOWN if _names_model(exc, model) else CODE_BAD_REQUEST
    if isinstance(exc, anthropic.InternalServerError) or _status(exc) >= 500:
        # 529 "overloaded" included. The CLI path always retried it, because "overloaded"
        # is one of its markers; the SDK path did not, because nothing here looked past
        # 429. Same condition, same answer now.
        return CODE_SERVER_ERROR
    if _status(exc) == 429:
        return CODE_RATE_LIMITED

    # 3. The CLI's raw message: last, and the weakest evidence.
    low = str(exc).lower()
    if any(m in low for m in _RATE_LIMIT_MARKERS):
        return CODE_RATE_LIMITED
    if any(m in low for m in _AUTH_FAIL_MARKERS):
        return CODE_KEY_REJECTED
    return CODE_UNEXPECTED


def is_rate_limit(exc: BaseException) -> bool:
    """Retryable: the credential was accepted, and the endpoint is busy or overloaded."""
    return classify(exc) in (CODE_RATE_LIMITED, CODE_SERVER_ERROR)


def is_auth_dead(exc: BaseException) -> bool:
    """Fatal to a fan-out: a property of the login, not of this one call.

    Retrying cannot help and neither can the next chunk, so batch callers abort rather
    than reproducing one identical failure per chunk.
    """
    return classify(exc) == CODE_KEY_REJECTED

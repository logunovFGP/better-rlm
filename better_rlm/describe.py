"""Mode descriptions for the TUI selector.

Cline-2 ships ``describeMode()`` (``sdk/packages/llms/src/providers/
session-mode.ts``) — a small, data-only function that returns label +
summary + pros + cons for a mode, so every picker / status line / doc
shares the same prose and the same comparison.

This module is the Python twin for the better-rlm TUI. It maps better-rlm's
three transport modes (``auto`` | ``claude-cli`` | ``api``) onto the
host / proxy terminology the cline-2 picker speaks, because the comparison
itself is the same shape:

  * host mode = talk to a model endpoint directly (API key)
  * proxy mode = spawn an installed agent CLI (reuses the existing login)

For better-rlm, "spawn the agent CLI" is ``claude-cli`` mode (drives the
official ``claude`` binary, reusing the Claude Code keychain), and "talk
to the endpoint" is ``api`` mode (Anthropic SDK + ``ANTHROPIC_API_KEY``).
``auto`` is the same as ``host`` but with keyless fallback — described
under the auto entry below so the picker's two-column compare stays a
strict binary.
"""

from __future__ import annotations

from dataclasses import dataclass

# -- Mode catalogue ----------------------------------------------------------

# These strings match ``auth.MODE_AUTO/CLI/API`` exactly. Re-stated as
# module-level constants so describe.py has no import dependency on
# auth.py (which would import rlm via config.py and slow CLI startup).
MODE_AUTO = "auto"
MODE_CLI = "claude-cli"
MODE_API = "api"

VALID_MODES: tuple[str, ...] = (MODE_AUTO, MODE_CLI, MODE_API)

# Cline-2 uses host / proxy terminology. We mirror it in the picker so the
# comparison reads identically to anyone familiar with that UI, but the
# persistence keys remain the better-rlm-native names (so existing
# config.yaml files keep working).
PROXY_MODE_LABEL = "Binary — proxy mode"
HOST_MODE_LABEL = "API — host mode"


@dataclass(frozen=True)
class ModeDescription:
    """Two-column compare entry for the mode picker."""

    label: str
    summary: str
    pros: tuple[str, ...]
    cons: tuple[str, ...]


# Mirrors MODE_DESCRIPTIONS in cline-2's session-mode.ts, with the
# better-rlm-specific facts woven in. ``claude-cli`` is the proxy path
# (it spawns the `claude` binary); ``api`` is the host path (it talks to
# the Anthropic endpoint directly with ANTHROPIC_API_KEY).
_MODE_DESCRIPTIONS: dict[str, ModeDescription] = {
    MODE_CLI: ModeDescription(
        label=PROXY_MODE_LABEL,
        summary="Spawn the installed `claude` CLI. The CLI runs the session.",
        pros=(
            "Reuses your existing Claude Code login (keychain or setup-token)",
            "No ANTHROPIC_API_KEY needed — uses the subscription",
            "No premium-model gating to work around",
            "Nothing extra to configure once `claude` is on PATH and signed in",
        ),
        cons=(
            "Requires the `claude` CLI installed and signed in",
            "Subject to subscription rate limits (retried with backoff)",
            "No direct HTTP path, so harder to profile than `api`",
        ),
    ),
    MODE_API: ModeDescription(
        label=HOST_MODE_LABEL,
        summary="Talk to the model endpoint directly. better-rlm runs the session.",
        pros=(
            "No dependency on the `claude` CLI being installed or signed in",
            "Higher limits than the subscription path",
            "Direct HTTP path — easier to debug and profile",
        ),
        cons=(
            "Needs ANTHROPIC_API_KEY in .env",
            "Per-token cost on paid endpoints",
            "No reuse of an existing Claude Code login",
        ),
    ),
}

# `auto` shares the api-mode cons (it can fall back to api) but keeps the
# zero-setup property of cli. Rendered as its own row so the user sees
# "auto" before being asked to pick a concrete mode.
AUTO_DESCRIPTION = ModeDescription(
    label="Auto — pick the best path",
    summary=(
        "Prefer the `claude` CLI (proxy); fall back to ANTHROPIC_API_KEY (host) "
        "if the CLI isn't installed. Anthropic-only."
    ),
    pros=(
        "Zero setup — reuses the Claude Code login when present",
        "Falls back to ANTHROPIC_API_KEY so a fresh machine still works",
        "Anthropic-only path that picks the right transport per launch",
    ),
    cons=(
        "Less predictable than pinning `claude-cli` or `api` directly",
    ),
)


def describe_mode(mode: str) -> ModeDescription:
    """Return label/summary/pros/cons for a transport mode.

    Mirrors cline-2's ``describeMode()`` so the picker compares pros/cons
    the same way. Unknown modes return an empty description rather than
    raising — callers (the picker, the status line) render an empty entry
    as a validation hint instead of crashing on a config typo.
    """
    m = (mode or "").strip().lower()
    if m == MODE_AUTO:
        return AUTO_DESCRIPTION
    return _MODE_DESCRIPTIONS.get(
        m,
        ModeDescription(
            label=m or "(unknown mode)",
            summary=f"Unrecognised mode: {mode!r}. Expected one of {', '.join(VALID_MODES)}.",
            pros=(),
            cons=(
                "Pick a mode from the picker so config.yaml is rewritten with a valid value.",
            ),
        ),
    )


def all_modes() -> tuple[str, ...]:
    """Modes the picker offers, in the order they should be shown."""
    return VALID_MODES


# -- Endpoint catalogue --------------------------------------------------------
#
# What the picker offers is ENDPOINTS, not vendors, and the distinction is the whole
# reason this came back after b7282f9 removed the old provider picker.
#
# ``provider`` names the wire protocol. ``auth.require_anthropic`` refuses anything but
# ``anthropic`` because the session-window ledger, the 95% floor and ceiling-learning
# live in ``transport._LedgeredTransport``, which is only in the stack when the
# Anthropic client is -- a Gemini or OpenAI client would spend past a budget that
# refused nothing. That guard is correct and stays.
#
# But it was over-read as "Anthropic the company is the only option", and the old
# picker made the opposite error: it offered four vendors whose selection produced a
# config.yaml the server then rejected at the first model call. Both miss that plenty
# of endpoints speak the Anthropic messages format. Pointing the Anthropic client at
# one keeps every safety property, because the client -- and therefore the ledger -- is
# unchanged. Only the URL moves.
#
# So: every entry here is provider=anthropic. The picker writes base_url.

ENDPOINT_ANTHROPIC = "anthropic"
ENDPOINT_MINIMAX = "minimax"
ENDPOINT_CUSTOM = "custom"


@dataclass(frozen=True)
class EndpointDescription:
    """One row of the endpoint picker."""

    label: str
    base_url: str          # "" means the SDK default, api.anthropic.com
    key_env: str
    note: str


ENDPOINTS: dict[str, EndpointDescription] = {
    ENDPOINT_ANTHROPIC: EndpointDescription(
        label="Anthropic",
        base_url="",
        key_env="ANTHROPIC_API_KEY",
        note="The default. Works keyless in claude-cli/auto mode via your Claude Code login.",
    ),
    ENDPOINT_MINIMAX: EndpointDescription(
        label="MiniMax",
        base_url="https://api.minimax.io/anthropic",
        key_env="ANTHROPIC_API_KEY",
        note="Anthropic-compatible. Needs mode=api and a MiniMax key; set models to "
             "MiniMax ids (MiniMax-M2.7, MiniMax-M3, ...).",
    ),
    ENDPOINT_CUSTOM: EndpointDescription(
        label="Custom endpoint",
        base_url="",
        key_env="ANTHROPIC_API_KEY",
        note="Any other endpoint speaking the Anthropic messages format: a gateway, a "
             "proxy, a self-hosted model server.",
    ),
}


def describe_endpoint(endpoint: str) -> EndpointDescription:
    """Row for one endpoint id; a safe placeholder for an unknown one."""
    e = (endpoint or "").strip().lower()
    info = ENDPOINTS.get(e)
    if info is None:
        return EndpointDescription(
            label=endpoint or "(unknown)",
            base_url="",
            key_env="ANTHROPIC_API_KEY",
            note="Not a known endpoint id; pick one from the list.",
        )
    return info


def all_endpoints() -> tuple[str, ...]:
    """Endpoint ids the picker offers, in declaration order."""
    return tuple(ENDPOINTS.keys())


def endpoint_for_base_url(base_url: str) -> str:
    """Which catalogue entry a configured base_url corresponds to.

    Lets /status and the picker show the current choice by name instead of making
    the operator recognise a URL. A trailing slash is not a different endpoint.
    """
    u = (base_url or "").strip().rstrip("/")
    if not u:
        return ENDPOINT_ANTHROPIC
    for eid, info in ENDPOINTS.items():
        if info.base_url and info.base_url.rstrip("/") == u:
            return eid
    return ENDPOINT_CUSTOM

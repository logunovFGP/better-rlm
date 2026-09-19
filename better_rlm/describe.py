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


# -- Provider catalogue --------------------------------------------------------
#
# Ported from cline-2's provider model (``apps/cli/src/tui/components/dialogs/
# provider-picker.tsx``, filtered by ``p.mode === modeFilter``). Each provider
# declares which SESSION MODE offers it, so the picker never shows a provider the
# mode cannot reach -- the same reason cline passes ``modeFilter`` after its mode
# step: you are not re-offered a choice you just made.
#
# Each provider stores its key under its OWN env var, the way cline-2 keeps a separate
# settings entry per provider (``saveLocalProviderSettings(manager, {providerId,
# apiKey})``). One shared variable meant configuring MiniMax destroyed the Anthropic
# key that was already there, and switching back silently sent whatever remained to
# whichever endpoint was configured -- so a key given to one vendor could be handed to
# another. Keys are NEVER read across providers, for that reason.
#
# Every provider here is ``provider: anthropic`` in config.yaml, because that names
# the WIRE PROTOCOL. ``auth.require_anthropic`` refuses anything else: the
# session-window ledger, the 95% floor and ceiling-learning live in
# ``transport._LedgeredTransport``, which is in the stack only because the Anthropic
# client is. Providers differ by endpoint and by how you authenticate, not by client.

AUTH_CLI = "cli"          # the `claude` CLI holds the credential (cline: LocalCliStatus)
AUTH_API_KEY = "api_key"  # a key in .env                       (cline: ProviderConfigInput)

PROVIDER_CLAUDE_CLI = "claude-cli"
PROVIDER_ANTHROPIC = "anthropic"
PROVIDER_MINIMAX = "minimax"
PROVIDER_CUSTOM = "custom"


@dataclass(frozen=True)
class ProviderDescription:
    """One row of the provider picker."""

    label: str
    mode: str              # the session mode that offers it
    auth: str              # AUTH_CLI | AUTH_API_KEY
    base_url: str          # "" = the SDK default, api.anthropic.com
    key_env: str
    summary: str


PROVIDERS: dict[str, ProviderDescription] = {
    PROVIDER_CLAUDE_CLI: ProviderDescription(
        label="Claude Code subscription",
        mode=MODE_CLI,
        auth=AUTH_CLI,
        base_url="",
        key_env="",
        summary="Reuse your `claude` CLI login. No API key, no per-token cost.",
    ),
    PROVIDER_ANTHROPIC: ProviderDescription(
        label="Anthropic",
        mode=MODE_API,
        auth=AUTH_API_KEY,
        base_url="",
        key_env="ANTHROPIC_API_KEY",
        summary="api.anthropic.com directly, with an API key.",
    ),
    PROVIDER_MINIMAX: ProviderDescription(
        label="MiniMax",
        mode=MODE_API,
        auth=AUTH_API_KEY,
        base_url="https://api.minimax.io/anthropic",
        key_env="MINIMAX_API_KEY",
        summary="Anthropic-compatible. Use MiniMax model ids (MiniMax-M2.7, MiniMax-M3).",
    ),
    PROVIDER_CUSTOM: ProviderDescription(
        label="Custom endpoint",
        mode=MODE_API,
        auth=AUTH_API_KEY,
        base_url="",
        key_env="RLM_API_KEY",
        summary="Any other endpoint speaking the Anthropic messages format.",
    ),
}


def describe_provider(provider: str) -> ProviderDescription:
    """Row for one provider id; a safe placeholder for an unknown one."""
    info = PROVIDERS.get((provider or "").strip().lower())
    if info is None:
        return ProviderDescription(
            label=provider or "(unknown)",
            mode=MODE_API,
            auth=AUTH_API_KEY,
            base_url="",
            key_env="RLM_API_KEY",
            summary="Not a known provider id; pick one from the list.",
        )
    return info


def providers_for_mode(mode: str) -> tuple[str, ...]:
    """Provider ids offered by one session mode, in declaration order.

    ``auto`` resolves to whichever transport is available at launch, so it offers
    everything rather than pretending to know which one will win.
    """
    m = (mode or "").strip().lower()
    if m == MODE_AUTO:
        return tuple(PROVIDERS)
    return tuple(pid for pid, d in PROVIDERS.items() if d.mode == m)


def all_providers() -> tuple[str, ...]:
    """Every provider id the picker knows, in declaration order."""
    return tuple(PROVIDERS)


def provider_for_config(base_url: str, mode: str) -> str:
    """Which catalogue row a saved (base_url, mode) pair corresponds to.

    Lets the wizard and /status show the current choice by name instead of making
    the operator recognise a URL. A trailing slash is not a different endpoint.
    """
    m = (mode or "").strip().lower()
    if m == MODE_CLI:
        return PROVIDER_CLAUDE_CLI
    u = (base_url or "").strip().rstrip("/")
    if not u:
        return PROVIDER_CLAUDE_CLI if m == MODE_CLI else PROVIDER_ANTHROPIC
    for pid, d in PROVIDERS.items():
        if d.base_url and d.base_url.rstrip("/") == u:
            return pid
    return PROVIDER_CUSTOM


# -- Vendors: who you have an account with -------------------------------------
#
# A provider id here pairs a vendor with a transport -- `claude-cli` and `anthropic`
# are the same vendor reached two ways. That flattening is fine for the maintenance
# picker, where the operator is changing one setting, and wrong for a first run,
# where the question is "whose account do you have?" and the transport follows.
#
# cline splits it the same way: MAIN_MENU offers vendors ("Sign in with Claude
# Code", "Bring your own provider") and the transport is decided by which row was
# picked, with ModePickerContent only appearing when the answer is genuinely open.

VENDOR_CLAUDE = "claude"
VENDOR_MINIMAX = "minimax"


@dataclass(frozen=True)
class VendorDescription:
    """One row of the first-run provider screen."""

    label: str
    icon: str
    summary: str
    modes: dict[str, str]        # session mode -> provider id it resolves to


VENDORS: dict[str, VendorDescription] = {
    VENDOR_CLAUDE: VendorDescription(
        label="Claude",
        icon="✦",
        summary="Anthropic's models, by subscription or API key.",
        modes={MODE_CLI: PROVIDER_CLAUDE_CLI, MODE_API: PROVIDER_ANTHROPIC},
    ),
    VENDOR_MINIMAX: VendorDescription(
        label="MiniMax",
        icon="⚙",
        summary="Anthropic-compatible endpoint. API key only.",
        modes={MODE_API: PROVIDER_MINIMAX},
    ),
}


def all_vendors() -> tuple[str, ...]:
    return tuple(VENDORS)


def describe_vendor(vendor: str) -> VendorDescription:
    v = (vendor or "").strip().lower()
    info = VENDORS.get(v)
    if info is None:
        return VendorDescription(label=vendor or "(unknown)", icon=" ",
                                 summary="Not a known vendor; pick one from the list.",
                                 modes={})
    return info


def modes_for_vendor(vendor: str) -> tuple[str, ...]:
    """The transports this vendor can actually be reached by.

    One entry means there is no question to ask, and asking anyway is the shape
    cline avoids by branching in runProviderChange rather than always showing the
    mode picker.
    """
    return tuple(describe_vendor(vendor).modes)


def provider_for(vendor: str, mode: str) -> str:
    """Which provider id a (vendor, mode) pair resolves to.

    Raises on a pair that cannot be reached. Defaulting to Anthropic meant
    ``provider_for("minimax", MODE_CLI)`` quietly configured a different vendor than
    the operator picked, and the only symptom would be calls landing somewhere
    unexpected. Callers pass pairs drawn from ``VENDORS``; anything else is a bug.
    """
    modes = describe_vendor(vendor).modes
    if mode not in modes:
        raise ValueError(
            f"{vendor!r} cannot be reached by mode {mode!r} "
            f"(it offers {sorted(modes) or 'nothing'})"
        )
    return modes[mode]


def vendor_of(provider: str) -> str:
    """Reverse: which vendor a provider id belongs to, or "" when none does.

    PROVIDER_CUSTOM is deliberately absent from VENDORS, so it has no vendor. This
    returned VENDOR_CLAUDE for it -- and for any unknown id -- which would label a
    custom endpoint "Claude" on any screen that named the vendor.
    """
    for vid, d in VENDORS.items():
        if provider in d.modes.values():
            return vid
    return ""


# -- Model catalogue ---------------------------------------------------------
#
# cline-2 ships a bundled per-provider model table (its generated
# `catalog.generated.ts`, reached through `getProviderConfig(id).knownModels`)
# and the model picker renders exactly the rows for the provider in hand. It does
# NOT ask the endpoint what it serves: for anthropic and minimax the network
# refresh is a no-op, because neither declares a `modelsSourceUrl`. A static
# table is the faithful port, not a shortcut.
#
# This table is the reason the picker can stop offering Claude ids to a MiniMax
# endpoint -- the defect that made a MiniMax install run with
# `root_model: claude-sonnet-5` pointed at api.minimax.io.
#
# Keyed by VENDOR, not provider: `claude-cli` and `anthropic` are one vendor
# reached two ways, and two copies of one list is how they drift.


@dataclass(frozen=True)
class ModelDescription:
    """One row of a model picker: what it is, and what it can hold.

    No price columns. A rate is a published claim about a vendor's billing that
    this table cannot verify and cannot keep current, and every row that lacked
    one still had to render as something -- so the picker quoted numbers it had
    no way to check against the invoice.
    """

    id: str
    label: str
    context: int                 # input window, tokens
    max_tokens: int              # ceiling on output per call
    note: str = ""

    def detail(self) -> str:
        """The picker's right-hand column: window, output ceiling, what it is for."""
        window = (f"{self.context // 1_000_000}M ctx" if self.context >= 1_000_000
                  else f"{self.context // 1000}K ctx")
        out = f"{self.max_tokens // 1000}K out"
        return " - ".join(p for p in (window, out, self.note) if p)


MODELS: dict[str, tuple[ModelDescription, ...]] = {
    VENDOR_CLAUDE: (
        ModelDescription("claude-opus-5", "Claude Opus 5", 1_000_000, 128_000, note="deepest reasoning"),
        ModelDescription("claude-sonnet-5", "Claude Sonnet 5", 1_000_000, 128_000, note="best all-round; the default root"),
        ModelDescription("claude-fable-5", "Claude Fable 5", 1_000_000, 128_000,
                         note="remapped to Opus 4.8 on the CLI path"),
        ModelDescription("claude-opus-4-8", "Claude Opus 4.8", 1_000_000, 128_000, note="the default hardest-task override"),
        ModelDescription("claude-opus-4-7", "Claude Opus 4.7", 1_000_000, 128_000),
        ModelDescription("claude-sonnet-4-6", "Claude Sonnet 4.6", 1_000_000, 128_000, note="prior root, still selectable"),
        ModelDescription("claude-opus-4-6", "Claude Opus 4.6", 1_000_000, 128_000),
        ModelDescription("claude-opus-4-5", "Claude Opus 4.5 (latest)", 200_000, 64_000),
        ModelDescription("claude-opus-4-5-20251101", "Claude Opus 4.5", 200_000, 64_000, note="pinned build"),
        ModelDescription("claude-haiku-4-5", "Claude Haiku 4.5 (latest)", 200_000, 64_000, note="fastest; the default sub-model"),
        ModelDescription("claude-haiku-4-5-20251001", "Claude Haiku 4.5", 200_000, 64_000, note="pinned build"),
        ModelDescription("claude-sonnet-4-5", "Claude Sonnet 4.5 (latest)", 1_000_000, 64_000),
        ModelDescription("claude-sonnet-4-5-20250929", "Claude Sonnet 4.5", 1_000_000, 64_000, note="pinned build"),
    ),
    VENDOR_MINIMAX: (
        ModelDescription("MiniMax-M3", "MiniMax M3", 1_048_576, 512_000, note="1M window; the default root"),
        ModelDescription("MiniMax-M2.7", "MiniMax M2.7", 204_800, 131_072, note="the default sub-model"),
        ModelDescription("MiniMax-M2.7-highspeed", "MiniMax M2.7 highspeed", 204_800, 131_072, note="same model, lower latency"),
        ModelDescription("MiniMax-M2.5", "MiniMax M2.5", 204_800, 131_072),
        ModelDescription("MiniMax-M2.5-highspeed", "MiniMax M2.5 highspeed", 204_800, 131_072, note="same model, lower latency"),
        ModelDescription("MiniMax-M2.1", "MiniMax M2.1", 204_800, 131_072),
        ModelDescription("MiniMax-M2", "MiniMax M2", 204_800, 131_072),
    ),
}

#: Where each model screen puts its cursor: (root, override, sub) per vendor.
#: The operator still picks -- these only decide what Enter-alone selects.
#: Claude's row is identical to config._DEFAULTS, so an existing install that
#: re-runs setup and presses Enter three times keeps exactly what it had.
#: MiniMax has no stronger tier than M3, so override = root is the honest answer
#: rather than a fake upgrade.
ROLE_DEFAULTS: dict[str, tuple[str, str, str]] = {
    VENDOR_CLAUDE: ("claude-sonnet-5", "claude-opus-4-8", "claude-haiku-4-5"),
    VENDOR_MINIMAX: ("MiniMax-M3", "MiniMax-M3", "MiniMax-M2.7"),
}


def models_for(provider: str) -> tuple[ModelDescription, ...]:
    """The models this provider serves, in picker order.

    Empty for a custom endpoint: we know nothing about what it serves, so the
    picker offers free-text entry alone rather than a list from another vendor.
    """
    return MODELS.get(vendor_of(provider), ())


def describe_model(model_id: str) -> ModelDescription | None:
    """The catalogue row for an id, or None when nothing known serves it."""
    for rows in MODELS.values():
        for m in rows:
            if m.id == model_id:
                return m
    return None


def role_defaults(provider: str) -> tuple[str, str, str]:
    """(root, override, sub) cursor positions for a provider.

    Falls back to Claude's row for a provider with no vendor, which is the only
    safe guess: a custom endpoint is reached over the Anthropic protocol, and the
    operator overrides all three on the screens anyway.
    """
    return ROLE_DEFAULTS.get(vendor_of(provider), ROLE_DEFAULTS[VENDOR_CLAUDE])

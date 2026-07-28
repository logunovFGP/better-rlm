"""Model-selection policy.

Centralizes role -> model selection so the logic lives in ONE place instead of
being hardcoded across engine/server/subquery. Selection depends on auth mode:

* API key / first-party  -> use the configured model per role verbatim.
* Claude Code OAuth       -> map each role's model to the closest sibling the
                             subscription actually serves (grounded in a live
                             probe: current 4.x IDs work; Claude Fable 5 returns
                             404 "use Opus 4.8"; deprecated dated models 404).

To change selection behavior, edit here — not every call site.
"""

from __future__ import annotations

from enum import Enum

from .config import Config


class Role(str, Enum):
    ROOT = "root"          # orchestrator (default)
    OVERRIDE = "override"  # hardest tasks
    SUB = "sub"            # cheap chunk-level sub-LLM


# Closest-sibling map applied under Claude Code OAuth for models the
# subscription does not serve under their first-party IDs.
OAUTH_SIBLING: dict[str, str] = {
    "claude-fable-5": "claude-opus-4-8",   # API: "Claude Fable 5 is not available. Please use Opus 4.8."
    "claude-mythos-5": "claude-opus-4-8",
    "claude-3-5-haiku-20241022": "claude-haiku-4-5",
    "claude-3-5-sonnet-20241022": "claude-sonnet-4-6",
    "claude-3-5-sonnet-20240620": "claude-sonnet-4-6",
}


def configured(cfg: Config, role: Role) -> str:
    """The model configured for a role, before any auth-mode mapping."""
    return {
        Role.ROOT: cfg.root_model,
        Role.OVERRIDE: cfg.root_model_override,
        Role.SUB: cfg.sub_model,
    }[role]


def map_for_mode(auth_mode: str, model_id: str) -> str:
    """Apply an auth mode's policy to one model id: OAuth remaps to the closest
    subscription-supported sibling, the API-key path passes through."""
    return OAUTH_SIBLING.get(model_id, model_id) if auth_mode == "oauth" else model_id


def policy_name(auth_mode: str) -> str:
    """Name of the active policy, for status display."""
    return "OAuthSibling" if auth_mode == "oauth" else "Direct"


def _mode(cfg: Config) -> str:
    from .auth import resolve_auth_mode  # lazy to avoid an import cycle
    return resolve_auth_mode(cfg)


def select(cfg: Config, role: Role) -> str:
    """Resolve the model for a role under the active auth mode."""
    return map_for_mode(_mode(cfg), configured(cfg, role))


def map_model(cfg: Config, model_id: str) -> str:
    """Resolve an explicit user-supplied model id under the active auth mode."""
    return map_for_mode(_mode(cfg), model_id)

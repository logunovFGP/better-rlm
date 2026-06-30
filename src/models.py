"""Model-selection strategy.

Centralizes role -> model selection so the logic lives in ONE place instead of
being hardcoded across engine/server/subquery. The strategy is chosen by auth
mode:

* API key / first-party  -> use the configured model per role verbatim.
* Claude Code OAuth       -> map each role's model to the closest sibling the
                             subscription actually serves (grounded in a live
                             probe: current 4.x IDs work; Claude Fable 5 returns
                             404 "use Opus 4.8"; deprecated dated models 404).

To change selection behavior, edit here — not every call site.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
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


class ModelStrategy(ABC):
    """Maps a Role to a concrete model id."""

    def __init__(self, cfg: Config):
        self._by_role = {
            Role.ROOT: cfg.root_model,
            Role.OVERRIDE: cfg.root_model_override,
            Role.SUB: cfg.sub_model,
        }

    @abstractmethod
    def model_for(self, role: Role) -> str: ...

    @abstractmethod
    def map_explicit(self, model_id: str) -> str:
        """Map a user-supplied explicit model id through the same policy."""


class DirectStrategy(ModelStrategy):
    """API key / first-party: configured model per role, verbatim."""

    def model_for(self, role: Role) -> str:
        return self._by_role[role]

    def map_explicit(self, model_id: str) -> str:
        return model_id


class OAuthSiblingStrategy(ModelStrategy):
    """Claude Code OAuth: each role -> closest subscription-supported sibling."""

    def model_for(self, role: Role) -> str:
        return OAUTH_SIBLING.get(self._by_role[role], self._by_role[role])

    def map_explicit(self, model_id: str) -> str:
        return OAUTH_SIBLING.get(model_id, model_id)


def get_strategy(cfg: Config, auth_mode: str) -> ModelStrategy:
    return OAuthSiblingStrategy(cfg) if auth_mode == "oauth" else DirectStrategy(cfg)


def _current_strategy(cfg: Config) -> ModelStrategy:
    from .auth import resolve_auth_mode  # lazy to avoid import cycle
    return get_strategy(cfg, resolve_auth_mode(cfg))


def select(cfg: Config, role: Role) -> str:
    """Resolve the model for a role under the active auth mode."""
    return _current_strategy(cfg).model_for(role)


def map_model(cfg: Config, model_id: str) -> str:
    """Resolve an explicit user-supplied model id under the active auth mode."""
    return _current_strategy(cfg).map_explicit(model_id)

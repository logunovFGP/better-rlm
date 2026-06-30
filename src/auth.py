"""Authentication for Claude calls — reuse Claude Code's auth, no API key required.

Two auth modes, resolved from the environment / ``.env``:

  * ``oauth``  — a Claude subscription login. Run ``claude setup-token`` once and
    put the token in rlm-mcp/.env as ``CLAUDE_CODE_OAUTH_TOKEN`` (or rely on the
    Claude Code keychain session). In this mode the server does **not** call the
    HTTP API: it drives the official ``claude`` CLI (see transport.CliTransport),
    which is itself the authorized Claude Code tool, so the subscription "just
    works" — no token plumbing into requests, no premium-model gating to hack.
  * ``apikey`` — ``ANTHROPIC_API_KEY`` is set. Calls go out over the Anthropic SDK
    (see transport.ApiTransport). Opt-in fallback only.

The engine's AnthropicClient hardcodes api_key and does its own retries, so
patch_engine() rebinds it to a subclass that routes every completion through the
selected transport (Strategy) and the shared throttle + auth-aware 429 retry
(ratelimit.py). SDK-side retries are disabled so retry policy lives in one place.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import anthropic
from dotenv import load_dotenv

from .config import PKG_ROOT

load_dotenv(PKG_ROOT / ".env")

_CLIENT_TIMEOUT = 600.0
_SDK_MAX_RETRIES = 0  # retry/backoff is owned by ratelimit.retry_and_queue_retries


def _clean_secret(value: str | None) -> str | None:
    if not value:
        return None
    v = value.strip()
    if not v or v.startswith("<") or v.endswith(">") or any(c.isspace() for c in v):
        return None
    return v


@dataclass(frozen=True)
class AuthInfo:
    mode: str   # "oauth" | "apikey"
    secret: str


def resolve_auth() -> AuthInfo:
    tok = _clean_secret(os.getenv("CLAUDE_CODE_OAUTH_TOKEN"))
    if tok:
        return AuthInfo("oauth", tok)
    key = _clean_secret(os.getenv("ANTHROPIC_API_KEY"))
    if key:
        return AuthInfo("apikey", key)
    raise RuntimeError(
        "No Claude credentials found (a literal placeholder like `<paste>` is ignored). "
        "Reuse your Claude Code login: run `claude setup-token`, copy the token it prints, "
        "and write it to rlm-mcp/.env as CLAUDE_CODE_OAUTH_TOKEN — no API key needed. "
        "Alternatively set ANTHROPIC_API_KEY."
    )


def auth_status() -> str:
    if _clean_secret(os.getenv("CLAUDE_CODE_OAUTH_TOKEN")):
        return "oauth"
    if _clean_secret(os.getenv("ANTHROPIC_API_KEY")):
        return "apikey"
    return "none"


def make_client(async_: bool = False):
    """Build an Anthropic SDK client for **API-key** auth (used by
    transport.ApiTransport). OAuth does NOT use this — it drives the ``claude`` CLI
    via transport.CliTransport. SDK retries are disabled (ratelimit.py owns retry)."""
    info = resolve_auth()
    if info.mode != "apikey":
        raise RuntimeError(
            "make_client() is for API-key auth only; OAuth uses the claude CLI transport "
            "(see transport.CliTransport)."
        )
    cls = anthropic.AsyncAnthropic if async_ else anthropic.Anthropic
    return cls(api_key=info.secret, timeout=_CLIENT_TIMEOUT, max_retries=_SDK_MAX_RETRIES)


def patch_engine() -> None:
    """Rebind ``rlm.clients.anthropic.AnthropicClient`` so the engine routes every
    completion through the selected transport (OAuth → claude CLI, API key → SDK)
    plus the shared throttle + auth-aware 429 retry. Idempotent."""
    import rlm.clients.anthropic as ant_mod

    if getattr(ant_mod, "_rlmmcp_patched", False):
        return
    from .config import load_config
    from .ratelimit import aretry_and_queue_retries, retry_and_queue_retries
    from .transport import get_transport, split_prompt

    base = ant_mod.AnthropicClient
    cfg = load_config()

    class _ClaudeCodeAnthropicClient(base):  # type: ignore[misc, valid-type]
        def __init__(self, api_key=None, model_name=None, max_tokens=32768, **kwargs):
            # The placeholder key satisfies the engine's required field; the real
            # auth lives in the transport (CLI session or SDK api key).
            super().__init__(api_key="rlm-mcp-oauth", model_name=model_name,
                             max_tokens=max_tokens, **kwargs)
            # Strategy: pick transport once, by the active auth mode.
            self._transport = get_transport(auth_status(), cfg)

        def _resolve(self, prompt, model):
            messages, system = split_prompt(prompt)
            model = model or self.model_name
            if not model:
                raise ValueError("Model name is required for the Anthropic client.")
            return messages, system, model

        def _record(self, model, input_tokens, output_tokens):
            # Mirror the engine's _track_cost so get_last_usage()/get_usage_summary()
            # report correct per-model token counts (this proves routing).
            self.model_call_counts[model] += 1
            self.model_input_tokens[model] += input_tokens
            self.model_output_tokens[model] += output_tokens
            self.model_total_tokens[model] += input_tokens + output_tokens
            self.last_prompt_tokens = input_tokens
            self.last_completion_tokens = output_tokens

        @retry_and_queue_retries
        def completion(self, prompt, model=None):
            messages, system, model = self._resolve(prompt, model)
            res = self._transport.complete(messages, system, model, self.max_tokens)
            self._record(res.model, res.input_tokens, res.output_tokens)
            return res.text

        @aretry_and_queue_retries
        async def acompletion(self, prompt, model=None):
            messages, system, model = self._resolve(prompt, model)
            res = await self._transport.acomplete(messages, system, model, self.max_tokens)
            self._record(res.model, res.input_tokens, res.output_tokens)
            return res.text

    ant_mod.AnthropicClient = _ClaudeCodeAnthropicClient
    ant_mod._rlmmcp_patched = True

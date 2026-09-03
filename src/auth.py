"""Authentication & transport-mode selection — reuse Claude Code's login, no API key.

A transport **Strategy** (src/transport.py) decides how each model call is made;
this module decides *which* transport, from ``cfg.mode`` (``auto`` | ``claude-cli``
| ``api``, overridable via the ``RLM_MODE`` env var):

  * ``claude-cli`` → drive the official ``claude`` CLI (``claude -p``). The CLI is
    itself the authorized Claude Code tool, so it authenticates from your existing
    login (keychain, or a ``claude setup-token`` token if present). **No API key and
    no token plumbing of our own** — and no premium-model gating to work around.
  * ``api`` → the Anthropic SDK, using ``ANTHROPIC_API_KEY``.
  * ``auto`` (default) → prefer the ``claude`` CLI when it's installed (zero setup —
    reuse the Claude Code login); otherwise fall back to ``ANTHROPIC_API_KEY``.

``resolve_auth_mode(cfg)`` returns the internal selector ``"oauth"`` (claude CLI) or
``"apikey"`` (SDK) consumed by the transport, model-selection strategy, and retry
backoff. The engine's AnthropicClient hardcodes api_key + its own retries, so
``patch_engine()`` rebinds it to a subclass that routes every completion through the
chosen transport + the shared throttle/retry (ratelimit.py).
"""

from __future__ import annotations

import os
import shutil

import anthropic

# Importing .config runs its module body, which loads .env — no second load here.
from .config import Config

MODE_AUTO = "auto"
MODE_CLI = "claude-cli"
MODE_API = "api"

_CLIENT_TIMEOUT = 600.0
_SDK_MAX_RETRIES = 0  # retry/backoff is owned by ratelimit.retry_and_queue_retries


def _clean_secret(value: str | None) -> str | None:
    if not value:
        return None
    v = value.strip()
    if not v or v.startswith("<") or v.endswith(">") or any(c.isspace() for c in v):
        return None
    return v


def require_anthropic(cfg: Config) -> None:
    """Refuse any provider but Anthropic, at every door that leads to a model call.

    Not a policy preference — a correctness guard. The session-window ledger, the 95%
    floor and the ceiling-learning all live in ``transport._LedgeredTransport``, which is
    only ever in the stack when the Anthropic client has been rebound onto
    ``get_transport``. The other providers kept the engine's own client, which resolves no
    transport, so a Gemini/OpenAI run recorded no spend, passed no gate and learned no
    ceiling: the budget existed on paper and refused nothing. Half-gated spending is worse
    than none, because the estimate then reports headroom that was already consumed.

    Raised here rather than in ``load_config`` on purpose: the read-only tools (grep,
    exec, chunk, inspect) never call a model and must keep working whatever the config
    says. Only the paths that are about to spend refuse.
    """
    provider = (cfg.provider or "anthropic").strip().lower()
    if provider != "anthropic":
        raise NotImplementedError(
            f"provider={provider!r} is not supported: only provider=anthropic routes "
            "through the session-window budget (ledger, 95% floor, resumable stop). "
            "Set provider: anthropic in config.yaml."
        )


def claude_cli_available(cfg: Config) -> bool:
    """True if the ``claude`` CLI binary is on PATH. Login is the CLI's own concern —
    a present-but-not-logged-in CLI surfaces a clear auth error at call time."""
    return shutil.which(cfg.cli_path) is not None


def auth_status() -> str:
    """Which explicit credential is present in the environment (for status display):
    'oauth' (a CLAUDE_CODE_OAUTH_TOKEN), 'apikey' (ANTHROPIC_API_KEY), or 'none'.
    NOTE: claude-cli mode needs NEITHER — the CLI uses its own keychain login."""
    if _clean_secret(os.getenv("CLAUDE_CODE_OAUTH_TOKEN")):
        return "oauth"
    if _clean_secret(os.getenv("ANTHROPIC_API_KEY")):
        return "apikey"
    return "none"


def resolve_auth_mode(cfg: Config) -> str:
    """Authoritative transport selector → 'oauth' (claude CLI) | 'apikey' (SDK).

    Honors ``cfg.mode`` — the three values config.yaml documents. ``auto`` prefers
    the ``claude`` CLI — reusing the existing Claude Code login, so NO token or API
    key is needed — and falls back to ``ANTHROPIC_API_KEY``.
    """
    # Anthropic is the only supported provider, and the only one with a keyless path
    # (the `claude` CLI login). Everything below chooses between ITS two transports.
    require_anthropic(cfg)
    m = (cfg.mode or MODE_AUTO).strip().lower()
    if m == MODE_CLI:
        if not claude_cli_available(cfg):
            raise RuntimeError(
                f"mode=claude-cli but the `claude` CLI was not found (cli_path={cfg.cli_path!r}). "
                "Install Claude Code and log in, or set mode=api with ANTHROPIC_API_KEY."
            )
        return "oauth"
    if m == MODE_API:
        if not _clean_secret(os.getenv("ANTHROPIC_API_KEY")):
            raise RuntimeError("mode=api but ANTHROPIC_API_KEY is not set.")
        return "apikey"
    # auto: prefer the CLI (reuse the Claude Code login), else the API key.
    if claude_cli_available(cfg):
        return "oauth"
    if _clean_secret(os.getenv("ANTHROPIC_API_KEY")):
        return "apikey"
    raise RuntimeError(
        "No transport available. Either install + log into the `claude` CLI "
        "(recommended — reuses your Claude Code login, no key needed), or set "
        "ANTHROPIC_API_KEY and mode=api."
    )


def make_client(async_: bool = False):
    """Build an Anthropic SDK client for the **api** transport, from ANTHROPIC_API_KEY
    (used by transport.ApiTransport). The claude-cli transport does NOT use this — the
    CLI authenticates itself. SDK retries are disabled (ratelimit.py owns retry)."""
    key = _clean_secret(os.getenv("ANTHROPIC_API_KEY"))
    if not key:
        raise RuntimeError(
            "The API transport requires ANTHROPIC_API_KEY. Use mode=claude-cli "
            "(or the default 'auto') to reuse your Claude Code login instead."
        )
    cls = anthropic.AsyncAnthropic if async_ else anthropic.Anthropic
    return cls(api_key=key, timeout=_CLIENT_TIMEOUT, max_retries=_SDK_MAX_RETRIES)


def patch_engine() -> None:
    """Make the engine's model calls go through our transport + throttle/retry.

    Rebind ``AnthropicClient`` so every completion routes through the selected transport —
    claude CLI on 'oauth', Anthropic SDK on 'apikey' — and therefore through the ledger,
    the floor and the retry. Idempotent.

    Any other provider raises: see ``require_anthropic``. It used to fall back to the
    engine's native client with only a throttle bolted on, which quietly bypassed the
    whole budget.
    """
    from .config import load_config

    cfg = load_config()
    require_anthropic(cfg)

    import rlm.clients.anthropic as ant_mod

    if getattr(ant_mod, "_rlmmcp_patched", False):
        return
    from .ratelimit import aretry_and_queue_retries, retry_and_queue_retries
    from .transport import get_transport, split_prompt

    base = ant_mod.AnthropicClient

    class _ClaudeCodeAnthropicClient(base):  # type: ignore[misc, valid-type]
        def __init__(self, api_key=None, model_name=None, max_tokens=32768, **kwargs):
            # The placeholder key satisfies the engine's required field; the real
            # auth lives in the transport (CLI session or SDK api key).
            super().__init__(api_key="rlm-mcp-oauth", model_name=model_name,
                             max_tokens=max_tokens, **kwargs)
            # Strategy: pick transport once, by the resolved mode.
            self._transport = get_transport(resolve_auth_mode(cfg), cfg)

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

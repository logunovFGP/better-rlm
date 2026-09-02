"""Completion transport — the Strategy that decides HOW a model call is made.

Two transports behind one interface (``CompletionTransport.complete`` / ``acomplete``):

  * ``ApiTransport`` — API-key auth: the Anthropic SDK over HTTPS.
  * ``CliTransport`` — OAuth auth: spawn the official ``claude`` CLI (``claude -p``)
    and read its output. The CLI is itself the authorized Claude Code tool, so a
    Claude subscription "just works" with no token plumbing and no premium-model
    gating — that is the whole reason this path exists. The CLI uses the existing
    Claude Code login (keychain / ``claude setup-token`` token); we never feed it
    an API key.

``get_transport(auth_mode, cfg)`` returns the right one. The function interface is
identical; only the transport varies. Both are wrapped by the shared retry/throttle
decorators at the call sites (auth.patch_engine's client + subquery), because
either transport can hit limits and fail — a CLI failure surfaces as
``CliRateLimitError`` (retryable) or ``CliCompletionError`` (not).

The engine hands completion prompts as a message list that includes a
``{"role": "system"}`` entry; ``split_prompt`` separates that out. ``claude -p``
takes a single prompt, so for multi-turn histories ``flatten_messages`` renders a
role-delimited transcript fed over stdin (argv has length limits).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

from . import budget
from .config import Config, estimate_tokens
from .logsetup import log_event

_LOG = logging.getLogger("rlm-mcp")

_RATE_LIMIT_MARKERS = (
    "rate limit", "rate_limit", "429", "overloaded", "usage limit",
    "too many requests", "please run /upgrade", "quota",
)


@dataclass(frozen=True)
class CompletionResult:
    """What every transport returns — text plus the token usage we record."""

    text: str
    input_tokens: int
    output_tokens: int
    model: str
    cost_usd: float | None = None


class CliCompletionError(RuntimeError):
    """Non-retryable failure from the `claude` CLI (bad invocation, crash,
    unparseable output)."""


class CliRateLimitError(RuntimeError):
    """CLI hit a rate/usage limit. Flagged so ratelimit._is_rate_limit retries it."""

    is_rate_limit = True


class CliAuthError(CliCompletionError):
    """CLI could not authenticate — the login is dead, so every other call fails
    identically.

    ``is_fatal_subcall`` is the engine's duck-typed contract
    (``rlm.environments.base_env.FATAL_SUBCALL_ATTR``): any batched fan-out, ours
    in subquery.py or the engine's own, aborts on it instead of repeating one
    global failure once per prompt. One attribute, both layers, no import across
    the boundary.
    """

    is_fatal_subcall = True


# --------------------------------------------------------------------------- #
# Message helpers
# --------------------------------------------------------------------------- #
def _content_to_text(content) -> str:
    """Flatten message content (a string or a list of content blocks) to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text", "") or "")
            else:
                parts.append(str(block))
        return "".join(parts)
    if content is None:
        return ""
    return str(content)


def split_prompt(prompt) -> tuple[list[dict], object | None]:
    """Split an engine prompt into (messages, system).

    Mirrors the engine's AnthropicClient._prepare_messages: a plain string becomes
    one user message; a message list has its ``system`` entry pulled out. ``system``
    is returned raw (str or content blocks) — each transport normalizes it.
    """
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}], None
    if isinstance(prompt, list):
        messages: list[dict] = []
        system = None
        for msg in prompt:
            if isinstance(msg, dict) and msg.get("role") == "system":
                system = msg.get("content")
            else:
                messages.append(msg)
        return messages, system
    raise ValueError(f"Invalid prompt type: {type(prompt)!r}")


def flatten_messages(messages: list[dict]) -> str:
    """Render the user/assistant turns into a single prompt for ``claude -p``.

    A lone user message passes through verbatim (the common first-turn case).
    Multi-turn histories become a role-delimited transcript ending on an
    ``## ASSISTANT`` cue so the model continues with exactly its next turn.
    """
    if len(messages) == 1 and messages[0].get("role") == "user":
        return _content_to_text(messages[0].get("content"))

    parts = ["[Conversation so far. Continue ONLY with the assistant's next turn.]", ""]
    for msg in messages:
        role = str(msg.get("role", "user")).upper()
        parts.append(f"## {role}")
        parts.append(_content_to_text(msg.get("content")))
        parts.append("")
    parts.append("## ASSISTANT")
    return "\n".join(parts)


def _looks_rate_limited(text: str | None) -> bool:
    low = (text or "").lower()
    return any(marker in low for marker in _RATE_LIMIT_MARKERS)


# ponytail: substring match on the CLI's own message — the only signal it gives.
# If these ever stop matching we degrade to the previous behaviour (one doomed
# call per chunk), never to something worse, so this stays best-effort by design.
_AUTH_FAIL_MARKERS = (
    "failed to authenticate",
    "oauth session expired",
    "oauth token has expired",
    "invalid api key",
    "authentication_error",
    "please run /login",
)


def _looks_auth_failed(text: str | None) -> bool:
    low = (text or "").lower()
    return any(marker in low for marker in _AUTH_FAIL_MARKERS)


#: What to actually DO about a dead CLI login. Static text, so building it costs
#: nothing and it is attached to every auth failure. The live check is
#: cli_auth_status() below, which rlm_status calls.
#:
#: The trap this exists for: being signed in to Claude Code (or the desktop app)
#: does NOT sign in the `claude` CLI. The host session holds its own credential
#: and a nested `claude -p` cannot borrow it — measured with the delegation env
#: both stripped AND left intact, identical failure either way. Meanwhile
#: `claude auth status` reports loggedIn: false, which is the ground truth.
AUTH_REMEDIATION = (
    "The `claude` CLI has no usable login of its own. Being signed in to Claude Code "
    "or the desktop app does NOT sign in the CLI — the host session's credential "
    "cannot be borrowed by a nested `claude -p`. Check with `claude auth status`, "
    "then pick ONE:\n"
    "  1. `claude auth login`            — interactive; refreshes the CLI login\n"
    "  2. `claude setup-token`           — long-lived token (needs a Claude "
    "subscription); put it in CLAUDE_CODE_OAUTH_TOKEN. The durable choice for a "
    "server left running, since a headless refresh cannot complete interactive OAuth\n"
    "  3. ANTHROPIC_API_KEY + `mode: api` — the SDK path, no CLI involved"
)


_AUTH_LABEL: str | None = None


def auth_label(cfg) -> str:
    """How the calls are actually authenticated, e.g. ``oauth (oauth_token)``.

    Cached for the process: it costs a ~215 ms subprocess and cannot change
    without a restart, because config.yaml and .env are read once at import.
    """
    global _AUTH_LABEL
    if _AUTH_LABEL is not None:
        return _AUTH_LABEL
    from .auth import resolve_auth_mode          # lazy: auth imports this module
    try:
        mode = resolve_auth_mode(cfg)
    except Exception as exc:                     # noqa: BLE001 - report, never raise
        _AUTH_LABEL = f"UNRESOLVED ({type(exc).__name__})"
        return _AUTH_LABEL
    if mode != "oauth":
        _AUTH_LABEL = "apikey (anthropic SDK)"
        return _AUTH_LABEL
    st = cli_auth_status(cfg)
    if st is None:
        _AUTH_LABEL = "oauth (claude CLI, status unknown)"
    elif st.get("loggedIn"):
        _AUTH_LABEL = f"oauth (claude CLI, {st.get('authMethod', '?')})"
    else:
        _AUTH_LABEL = "oauth (claude CLI, NOT LOGGED IN)"
    return _AUTH_LABEL


def cli_auth_status(cfg) -> dict | None:
    """`claude auth status --json` — is the CLI actually logged in? FREE: no model
    call, ~215 ms measured. Returns the parsed dict, or None if the CLI could not be
    asked (absent, timed out, unexpected output).

    Worth preferring over a probe completion: it answers the question directly
    instead of inferring it from a failure, and it costs no tokens.
    """
    claude = shutil.which(cfg.cli_path)
    if not claude:
        return None
    try:
        proc = subprocess.run(
            [claude, "auth", "status", "--json"],
            capture_output=True, text=True, timeout=30,
            cwd=None, env=CliTransport._subprocess_env(),
        )
        return json.loads(proc.stdout)
    except Exception:      # absent, timeout, or a shape we do not recognise
        return None


def _spawn_err(stderr: str | None, stdout: str | None) -> str:
    """One-line failure reason for a non-zero CLI spawn (stderr preferred, stdout
    fallback). Whitespace-collapsed; log_event clamps the length. Never empty."""
    return " ".join((stderr or stdout or "").split()) or "no output"


# --------------------------------------------------------------------------- #
# Transports
# --------------------------------------------------------------------------- #
class CompletionTransport(ABC):
    """How a single completion is produced. Same interface, swappable backend."""

    @abstractmethod
    def complete(self, messages: list[dict], system, model: str,
                 max_tokens: int) -> CompletionResult: ...

    @abstractmethod
    async def acomplete(self, messages: list[dict], system, model: str,
                        max_tokens: int) -> CompletionResult: ...


def _result_from_sdk_response(resp, model: str) -> CompletionResult:
    text = "".join(
        getattr(b, "text", "") for b in resp.content
        if getattr(b, "type", None) == "text"
    )
    usage = resp.usage
    return CompletionResult(
        text=text,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        model=model,
    )


class ApiTransport(CompletionTransport):
    """API-key auth: call the Anthropic SDK directly. Clients are built lazily so
    constructing this transport never requires credentials."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._client = None
        self._aclient = None

    def _sync_client(self):
        if self._client is None:
            from .auth import make_client
            self._client = make_client(async_=False)
        return self._client

    def _async_client(self):
        if self._aclient is None:
            from .auth import make_client
            self._aclient = make_client(async_=True)
        return self._aclient

    @staticmethod
    def _kwargs(messages, system, model, max_tokens) -> dict:
        kwargs = {"model": model, "max_tokens": max_tokens, "messages": messages}
        if system:
            kwargs["system"] = system
        return kwargs

    def complete(self, messages, system, model, max_tokens) -> CompletionResult:
        resp = self._sync_client().messages.create(
            **self._kwargs(messages, system, model, max_tokens))
        return _result_from_sdk_response(resp, model)

    async def acomplete(self, messages, system, model, max_tokens) -> CompletionResult:
        resp = await self._async_client().messages.create(
            **self._kwargs(messages, system, model, max_tokens))
        return _result_from_sdk_response(resp, model)


class CliTransport(CompletionTransport):
    """OAuth auth: drive the official ``claude`` CLI. No API key, no HTTP — the CLI
    authenticates with the existing Claude Code subscription session."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._cwd: str | None = None

    def _neutral_cwd(self) -> str:
        # Run in an empty dir so no project CLAUDE.md / settings are auto-discovered
        # (belt-and-suspenders alongside --safe-mode).
        if self._cwd is None:
            d = self.cfg.store_dir / ".cli-cwd"
            d.mkdir(parents=True, exist_ok=True)
            self._cwd = str(d)
        return self._cwd

    @staticmethod
    def _subprocess_env() -> dict:
        """Build the environment for the nested ``claude`` CLI.

        Two jobs:
          1. Force the subscription/OAuth path (drop ``ANTHROPIC_API_KEY``).
          2. Strip the Claude Code *session* markers this server inherits when it is
             itself launched **by** Claude Code (as an MCP server). Left in place,
             vars like ``CLAUDECODE`` / ``CLAUDE_CODE_CHILD_SESSION`` /
             ``CLAUDE_CODE_SESSION_ID`` / ``CLAUDE_CODE_SDK_HAS_*_REFRESH`` /
             ``CLAUDE_CODE_OAUTH_SCOPES`` make the nested CLI take the *delegated
             child-session* auth path (which fails with a misleading "organization
             has disabled Claude subscription access") instead of authenticating
             standalone from the Claude Code login (macOS keychain / Windows
             credential store / ``~/.claude/.credentials.json``). Stripping them makes
             the spawned ``claude -p`` behave exactly like one run from a clean shell,
             which is the login that actually works. A *valid* explicit
             ``CLAUDE_CODE_OAUTH_TOKEN`` (the headless-box case) is preserved.
        """
        from .auth import _clean_secret
        tok = _clean_secret(os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"))
        env = {}
        for k, v in os.environ.items():
            if k in ("ANTHROPIC_API_KEY", "CLAUDECODE", "CLAUDE_AGENT_SDK_VERSION"):
                continue
            # Drop inherited Claude Code session/SDK markers; keep only a valid
            # explicit OAuth token (re-added below).
            if k.startswith("CLAUDE_CODE_") and k != "CLAUDE_CODE_OAUTH_TOKEN":
                continue
            env[k] = v
        if tok:
            env["CLAUDE_CODE_OAUTH_TOKEN"] = tok
        else:
            env.pop("CLAUDE_CODE_OAUTH_TOKEN", None)
        return env

    def _argv(self, model: str, system_text: str | None) -> list[str]:
        cfg = self.cfg
        argv = [cfg.cli_path, "-p", "--output-format", "json", "--model", model]
        if cfg.cli_safe_mode:
            argv.append("--safe-mode")  # no hooks/CLAUDE.md/skills/MCP → no recursion
        if system_text:
            flag = "--system-prompt" if cfg.cli_system_prompt_mode == "replace" \
                else "--append-system-prompt"
            argv += [flag, system_text]
        if cfg.cli_disable_tools:
            argv += ["--tools", ""]  # RLM runs its OWN sandbox; CLI must only emit text
        if cfg.cli_no_session_persistence:
            argv.append("--no-session-persistence")
        if cfg.cli_fallback_model:
            argv += ["--fallback-model", cfg.cli_fallback_model]
        argv += list(cfg.cli_extra_args)
        return argv

    def _prepare(self, messages, system, model) -> tuple[list[str], str]:
        system_text = _content_to_text(system) if system else None
        return self._argv(model, system_text), flatten_messages(messages)

    def complete(self, messages, system, model, max_tokens) -> CompletionResult:
        argv, prompt = self._prepare(messages, system, model)
        start = time.monotonic()
        try:
            proc = subprocess.run(
                argv, input=prompt, capture_output=True, text=True,
                cwd=self._neutral_cwd(), env=self._subprocess_env(),
                timeout=self.cfg.cli_timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            log_event(_LOG, "cli_spawn", model=model,
                      dur_ms=round((time.monotonic() - start) * 1000),
                      outcome="timeout", limit_s=self.cfg.cli_timeout_s)
            raise CliCompletionError(
                f"claude CLI timed out after {self.cfg.cli_timeout_s}s") from exc
        rc = proc.returncode
        log_event(_LOG, "cli_spawn", model=model,
                  dur_ms=round((time.monotonic() - start) * 1000), exit=rc,
                  err=_spawn_err(proc.stderr, proc.stdout) if rc != 0 else None)
        return _parse_cli_output(rc, proc.stdout, proc.stderr, model)

    async def acomplete(self, messages, system, model, max_tokens) -> CompletionResult:
        argv, prompt = self._prepare(messages, system, model)
        start = time.monotonic()
        proc = await asyncio.create_subprocess_exec(
            *argv, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE, cwd=self._neutral_cwd(),
            env=self._subprocess_env(),
        )
        try:
            out, err = await asyncio.wait_for(
                proc.communicate(prompt.encode()), timeout=self.cfg.cli_timeout_s)
        except asyncio.TimeoutError as exc:
            proc.kill()
            log_event(_LOG, "cli_spawn", model=model,
                      dur_ms=round((time.monotonic() - start) * 1000),
                      outcome="timeout", limit_s=self.cfg.cli_timeout_s)
            raise CliCompletionError(
                f"claude CLI timed out after {self.cfg.cli_timeout_s}s") from exc
        rc = proc.returncode
        out_s = out.decode(errors="replace")
        err_s = err.decode(errors="replace")
        log_event(_LOG, "cli_spawn", model=model,
                  dur_ms=round((time.monotonic() - start) * 1000), exit=rc,
                  err=_spawn_err(err_s, out_s) if rc != 0 else None)
        return _parse_cli_output(rc, out_s, err_s, model)


def _parse_cli_output(returncode: int, stdout: str, stderr: str,
                      model: str) -> CompletionResult:
    """Parse `claude -p --output-format json` output into a CompletionResult.

    Raises CliRateLimitError on rate/usage limits (so the retry decorator backs off
    and retries) and CliCompletionError on any other failure.
    """
    body = (stdout or "").strip()
    data = None
    if body:
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            data = None

    if not isinstance(data, dict):
        msg = (stderr or stdout or "no output").strip()[:500]
        if _looks_rate_limited(msg):
            raise CliRateLimitError(f"claude CLI rate limited: {msg}")
        if _looks_auth_failed(msg):
            raise CliAuthError(f"claude CLI auth failed: {msg}\n{AUTH_REMEDIATION}")
        raise CliCompletionError(f"claude CLI failed (exit {returncode}): {msg}")

    is_error = bool(data.get("is_error")) or data.get("subtype") not in (None, "success")
    if is_error:
        msg = str(data.get("result") or data.get("error") or stderr or "error").strip()[:500]
        if _looks_rate_limited(msg) or _looks_rate_limited(str(data.get("subtype"))):
            raise CliRateLimitError(f"claude CLI rate limited: {msg}")
        if _looks_auth_failed(msg):
            raise CliAuthError(f"claude CLI auth failed: {msg}\n{AUTH_REMEDIATION}")
        raise CliCompletionError(
            f"claude CLI error (subtype={data.get('subtype')}): {msg}")

    usage = data.get("usage") or {}
    cost = data.get("total_cost_usd")
    return CompletionResult(
        text=data.get("result") or "",
        input_tokens=int(usage.get("input_tokens") or 0),
        output_tokens=int(usage.get("output_tokens") or 0),
        model=model,
        cost_usd=float(cost) if isinstance(cost, (int, float)) else None,
    )


_CACHE: dict[str, CompletionTransport] = {}


class _LedgeredTransport(CompletionTransport):
    """Wraps a transport so every completion lands in the session-window spend ledger.

    Placed HERE, at the one point all callers resolve a transport through, because the
    alternative was recording per call site and the call sites are not equivalent: our
    own map-reduce goes through subquery.py, but rlm_query's recursive fan-out goes
    through the engine's client, which src/auth.py rebinds onto this same factory. With
    the recording in subquery.py only, an rlm_query run — the most expensive tool here —
    spent its entire window budget invisibly, and rlm_estimate would then report
    headroom that had already been consumed.

    Input tokens are the LOCAL estimate of the prompt, never the transport's report:
    the CLI path reported 1,027 input tokens for a batch whose real input was ~3M.
    """

    def __init__(self, inner: CompletionTransport, cfg: Config):
        self._inner = inner
        self._cfg = cfg

    @property
    def inner(self) -> CompletionTransport:
        """The wrapped transport. Public because WHICH backend got selected is a real
        behaviour with its own tests, and wrapping must not hide it behind isinstance."""
        return self._inner

    def __getattr__(self, name):
        # auth_label and friends are read off the concrete transport; forward anything
        # this wrapper does not define so wrapping stays invisible to callers.
        return getattr(self._inner, name)

    @staticmethod
    def _est_in(messages: list[dict], system) -> int:
        n = sum(len(str(m.get("content", ""))) for m in messages) + len(str(system or ""))
        return estimate_tokens(n)

    def complete(self, messages, system, model, max_tokens) -> CompletionResult:
        est_in = self._est_in(messages, system)
        # The hard floor, for EVERY caller. The batch's Gate stops politely one layer
        # up; this is what stops rlm_query's recursive fan-out, which had nothing. It
        # raises BEFORE the call, so a refused call costs zero tokens.
        budget.check_or_raise(self._cfg, est_in + max_tokens)
        try:
            res = self._inner.complete(messages, system, model, max_tokens)
        except Exception as exc:
            self._note_if_limit(exc)
            raise
        budget.record(self._cfg, res.model or model, est_in, res.output_tokens)
        return res

    async def acomplete(self, messages, system, model, max_tokens) -> CompletionResult:
        est_in = self._est_in(messages, system)
        budget.check_or_raise(self._cfg, est_in + max_tokens)
        try:
            res = await self._inner.acomplete(messages, system, model, max_tokens)
        except Exception as exc:
            self._note_if_limit(exc)
            raise
        budget.record(self._cfg, res.model or model, est_in, res.output_tokens)
        return res

    def _note_if_limit(self, exc: BaseException) -> None:
        """Record a rate/usage limit as EVIDENCE about the account's real ceiling.

        A lower bound on it, not a cap: at the wall the account has spent its ceiling C,
        of which this ledger saw S while other Claude sessions spent the invisible rest, so
        C = S + other >= S. ``budget.ceiling`` therefore reports the observation rather
        than gating on it — an earlier version gated on it, which put the stop line at
        0.95*S, below the spend that produced it, and refused everything after.

        That also makes recording here safe. This wrapper is inside the retry, so it sees
        every attempt including the transient 429s the retry then handles; since the number
        only ever advises, an over-eager observation costs nothing. Recorded HERE rather
        than in ratelimit's retry decision because that decorator wraps engine methods
        whose first argument is ``self`` and has no cfg to write with -- it had to read a
        module global, which is exactly what wrote test spend into the operator's real
        budget state. This wrapper already holds the cfg whose ledger is being measured.
        """
        if _looks_rate_limited(str(exc)) or getattr(exc, "is_rate_limit", False)                 or getattr(exc, "status_code", None) == 429:
            budget.note_limit_hit(self._cfg)


def get_transport(auth_mode: str, cfg: Config) -> CompletionTransport:
    """Strategy selector: OAuth → claude CLI, apikey → Anthropic SDK. Cached per mode so
    clients and the neutral cwd are reused across calls. Every transport is wrapped so its
    spend reaches the ledger — which is why a provider with no transport of ours is
    refused here rather than silently routed around the budget (see auth.require_anthropic)."""
    from .auth import require_anthropic          # lazy: auth imports this module

    require_anthropic(cfg)
    ckey = f"anthropic:{auth_mode}"
    inner = _CACHE.get(ckey)
    if inner is None:
        inner = CliTransport(cfg) if auth_mode == "oauth" else ApiTransport(cfg)
        _CACHE[ckey] = inner
    # The INNER transport is cached (it owns the client and the neutral cwd); the wrapper
    # is rebuilt per call because it binds a cfg. Caching the wrapper would hand the first
    # caller's config to every later one -- with a test and the server holding different
    # configs, that writes one's spend into the other's ledger.
    return _LedgeredTransport(inner, cfg)

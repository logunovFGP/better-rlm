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
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

from .config import Config
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
        raise CliCompletionError(f"claude CLI failed (exit {returncode}): {msg}")

    is_error = bool(data.get("is_error")) or data.get("subtype") not in (None, "success")
    if is_error:
        msg = str(data.get("result") or data.get("error") or stderr or "error").strip()[:500]
        if _looks_rate_limited(msg) or _looks_rate_limited(str(data.get("subtype"))):
            raise CliRateLimitError(f"claude CLI rate limited: {msg}")
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


def get_transport(auth_mode: str, cfg: Config) -> CompletionTransport:
    """Strategy selector: OAuth → CLI, anything else (apikey) → SDK. Cached per
    auth mode so SDK clients / neutral cwd are reused across calls."""
    transport = _CACHE.get(auth_mode)
    if transport is None:
        transport = CliTransport(cfg) if auth_mode == "oauth" else ApiTransport(cfg)
        _CACHE[auth_mode] = transport
    return transport

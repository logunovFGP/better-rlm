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
import contextlib
import json
import logging
import os
import shutil
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

from . import budget, failures
from .config import Config, estimate_tokens
from .logsetup import log_event

_LOG = logging.getLogger("rlm-mcp")


@dataclass(frozen=True)
class CompletionResult:
    """What every transport returns — text plus the token usage we record."""

    text: str
    input_tokens: int
    output_tokens: int
    model: str
    #: The model stopped because it hit ``max_tokens``, so ``text`` is a fragment --
    #: or, for a reasoning model, EMPTY. Measured on MiniMax-M2.7 over one 12.9k-token
    #: log chunk: 13,015 output tokens, of which 99% was the thinking block and 286
    #: characters were the answer. At the old 4096 cap the budget ran out mid-thought,
    #: no text block was ever emitted, and the sub-query returned a blank answer under
    #: a cheerful header with a token receipt attached. A paid call that produced
    #: nothing must not look like one that had nothing to say.
    truncated: bool = False


def truncation_note(text: str, cap: int) -> str:
    """What to append to a truncated answer, or "" when it is whole.

    Here rather than in a caller because BOTH consumers of a CompletionResult need it and
    only one had it: ``batch.one`` reported truncation while ``auth.patch_engine``'s
    engine shim returned ``res.text`` and dropped the flag, so an ``rlm_query`` over a
    MiniMax endpoint got silently empty chunk answers long after the sub-query stopped.
    """
    if not text.strip():
        return (f"[TRUNCATED at max_tokens ({cap:,}) with no answer emitted -- the model "
                "spent the whole output budget on reasoning. Use a smaller chunk.]")
    return f"\n\n[TRUNCATED at max_tokens ({cap:,}) -- the answer above is incomplete.]"


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


#: The three fields that together are one call's real input. Prompt caching splits it,
#: and ``input_tokens`` alone is only the UNCACHED remainder.
_INPUT_FIELDS = ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")


def _total_input(usage) -> int:
    """Every input token a call consumed, cache included. Accepts a dict (CLI JSON) or an
    SDK usage object.

    ``input_tokens`` on its own is badly misleading wherever prompt caching is in play. A
    real 30k-token review chunk reported **10** while the account was billed **64,149** —
    the rest sat in ``cache_creation_input_tokens`` and ``cache_read_input_tokens``. A
    22-character prompt reported 10 against 29,268 of cache creation, which is the CLI's
    own baseline context arriving before any of ours. docs/07 §4 wrote the transport's
    input figure off as unusable on the strength of that one field; it is usable, just not
    from that field alone.

    Summed unweighted. Cache creation and reads are priced differently from fresh input
    (1.25x and 0.1x on the API), so this is tokens PROCESSED rather than a cost-weighted
    number — the right quantity for a context-window budget, and the honest one to report
    while the weighting for a subscription window is undocumented.
    """
    get = usage.get if isinstance(usage, dict) else lambda k, d=0: getattr(usage, k, d)
    return sum(int(get(k, 0) or 0) for k in _INPUT_FIELDS)


def _result_from_sdk_response(resp, model: str) -> CompletionResult:
    # Only `text` blocks. A reasoning model also returns `thinking` blocks, which are
    # its scratchpad and not an answer -- but they are billed as output and they are
    # emitted FIRST, so a cap that runs out during them yields no text block at all.
    # Hence stop_reason: without it, that case is indistinguishable from a model that
    # chose to say nothing.
    text = "".join(
        getattr(b, "text", "") for b in resp.content
        if getattr(b, "type", None) == "text"
    )
    usage = resp.usage
    return CompletionResult(
        text=text,
        input_tokens=_total_input(usage),
        output_tokens=usage.output_tokens,
        model=model,
        truncated=getattr(resp, "stop_reason", None) == "max_tokens",
    )


@contextlib.contextmanager
def _tag_fatal_auth():
    """Mark an SDK auth failure with the attribute the engine's fan-out honours.

    ``rlm.utils.exceptions.aborts_batch`` is duck-typed on ``is_fatal_subcall`` alone --
    correct layering, since the engine must not import a backend to know a call is
    hopeless. But the SDK's ``AuthenticationError`` carries no such attribute, so the
    contract only ever worked for the CLI path, whose ``CliAuthError`` sets it.

    Measured before this: ``aborts_batch(anthropic.AuthenticationError) is False`` while
    ``aborts_batch(CliAuthError) is True``. A dead API key therefore stopped OUR batch
    (subquery checks the exception type too) and not the engine's, so an ``rlm_query``
    fan-out issued one doomed call per prompt -- exactly what the attribute exists to
    prevent, on the one path that never got it.

    Set on the instance rather than the class: it says something about this failure
    reaching this caller, and mutating the SDK's class would leak into every other user
    of the library in the process.
    """
    try:
        yield
    except Exception as exc:
        if failures.is_auth_dead(exc):
            exc.is_fatal_subcall = True
        raise


class ApiTransport(CompletionTransport):
    """API-key auth: call the Anthropic SDK directly. Clients are built lazily so
    constructing this transport never requires credentials."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._clients: dict[bool, object | None] = {False: None, True: None}

    def _client(self, async_: bool):
        """The SDK client for one of the two shapes, built on first use.

        Lazily, so constructing this transport never requires a credential -- and once,
        because an Anthropic client owns a connection pool worth reusing.
        """
        if self._clients.get(async_) is None:
            from .auth import make_client
            self._clients[async_] = make_client(
                async_=async_, base_url=self.cfg.base_url, cfg=self.cfg)
        return self._clients[async_]

    # Kept as named methods: tests and the two call sites read better for it, and both
    # are one line over the dict above rather than a second copy of the lazy init.
    def _sync_client(self):
        return self._client(False)

    def _async_client(self):
        return self._client(True)

    @staticmethod
    def _kwargs(messages, system, model, max_tokens) -> dict:
        kwargs = {"model": model, "max_tokens": max_tokens, "messages": messages}
        if system:
            kwargs["system"] = system
        return kwargs

    def complete(self, messages, system, model, max_tokens) -> CompletionResult:
        with _tag_fatal_auth():
            resp = self._sync_client().messages.create(
                **self._kwargs(messages, system, model, max_tokens))
        return _result_from_sdk_response(resp, model)

    async def acomplete(self, messages, system, model, max_tokens) -> CompletionResult:
        with _tag_fatal_auth():
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

    def _timeout(self, model: str, start: float, exc: BaseException) -> CliCompletionError:
        """Both halves time out the same way; only the exception they catch differs."""
        log_event(_LOG, "cli_spawn", model=model,
                  dur_ms=round((time.monotonic() - start) * 1000),
                  outcome="timeout", limit_s=self.cfg.cli_timeout_s)
        return CliCompletionError(
            f"claude CLI timed out after {self.cfg.cli_timeout_s}s")

    def _finish(self, model: str, start: float, rc: int, out: str,
                err: str) -> CompletionResult:
        """Record the spawn and parse it. Identical on both halves, so it lives once."""
        log_event(_LOG, "cli_spawn", model=model,
                  dur_ms=round((time.monotonic() - start) * 1000), exit=rc,
                  err=_spawn_err(err, out) if rc != 0 else None)
        return _parse_cli_output(rc, out, err, model)

    def complete(self, messages, system, model, max_tokens) -> CompletionResult:
        argv, prompt = self._prepare(messages, system, model)
        start = time.monotonic()
        try:
            # subprocess.run bundles spawn, stdin, wait, timeout and decode into one
            # call, and kills+reaps the child itself on timeout.
            proc = subprocess.run(
                argv, input=prompt, capture_output=True, text=True,
                cwd=self._neutral_cwd(), env=self._subprocess_env(),
                timeout=self.cfg.cli_timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            raise self._timeout(model, start, exc) from exc
        return self._finish(model, start, proc.returncode, proc.stdout, proc.stderr)

    async def acomplete(self, messages, system, model, max_tokens) -> CompletionResult:
        argv, prompt = self._prepare(messages, system, model)
        start = time.monotonic()
        # Four things genuinely differ from the sync half: the pipes must be named
        # (asyncio inherits them otherwise), the wait is separate from the spawn, the
        # child survives a cancelled wait so it needs killing, and there is no text=
        # so the streams are decoded by hand.
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
            raise self._timeout(model, start, exc) from exc
        return self._finish(model, start, proc.returncode,
                            out.decode(errors="replace"), err.decode(errors="replace"))


def _raise_cli_failure(msg: str, fallback: str, *, reported: str = "") -> None:
    """Turn the CLI's own message into the right exception type.

    The classification is ``failures.classify`` -- the same call the SDK path makes,
    reaching the same answer from the only evidence this path has, the CLI's text.

    ``reported`` is what the operator sees when it should differ from what was
    classified: the error envelope carries the limit in ``subtype`` often enough to be
    worth reading, but quoting that back adds noise to the message.

    Rate limit is still tested before auth, unchanged: a message matching both markers
    stays retryable. Reversing it would abort a whole fan-out on a transient limit that
    merely mentioned logging in, which is the worse failure of the two.
    """
    shown = reported or msg
    code = failures.classify(RuntimeError(msg))
    if code in (failures.CODE_RATE_LIMITED, failures.CODE_SERVER_ERROR):
        raise CliRateLimitError(f"claude CLI rate limited: {shown}")
    if code == failures.CODE_KEY_REJECTED:
        raise CliAuthError(f"claude CLI auth failed: {shown}\n{AUTH_REMEDIATION}")
    raise CliCompletionError(fallback)


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
        _raise_cli_failure(msg, f"claude CLI failed (exit {returncode}): {msg}")

    is_error = bool(data.get("is_error")) or data.get("subtype") not in (None, "success")
    if is_error:
        msg = str(data.get("result") or data.get("error") or stderr or "error").strip()[:500]
        # The subtype rides along for classification only: a limit sometimes arrives
        # there with a generic `result` beside it. The reported message stays `msg`.
        _raise_cli_failure(f"{msg} {data.get('subtype') or ''}".strip(),
                           f"claude CLI error (subtype={data.get('subtype')}): {msg}",
                           reported=msg)

    usage = data.get("usage") or {}
    # The CLI also reports total_cost_usd. Deliberately dropped: on a subscription it
    # is a notional price for a call nobody was separately billed for, and printing it
    # beside real token counts makes a quote look like a measurement.
    return CompletionResult(
        text=data.get("result") or "",
        input_tokens=_total_input(usage),
        output_tokens=int(usage.get("output_tokens") or 0),
        model=model,
    )


_CACHE: dict[str, CompletionTransport] = {}


class _LedgeredTransport(CompletionTransport):
    """Wraps a transport so every completion lands in the session-window spend ledger.

    Placed HERE, at the one point all callers resolve a transport through, because the
    alternative was recording per call site and the call sites are not equivalent: our
    own map-reduce goes through subquery.py, but rlm_query's recursive fan-out goes
    through the engine's client, which better_rlm/auth.py rebinds onto this same factory. With
    the recording in subquery.py only, an rlm_query run — the most expensive tool here —
    spent its entire window budget invisibly, and the next run's pre-flight would
    then admit calls against headroom that had already been consumed.

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

    def _reserve(self, messages, system, max_tokens) -> int:
        """Tokens to hold against the floor for the call about to be made.

        Output is ``budget.expected_output``, not ``max_tokens``: the CLI takes no output
        flag, so on the OAuth path the cap is never sent and a call can emit several times
        it. Reserving the cap there under-reserves, which is how a floor whose whole job is
        to stop short of the wall lets a single call step over it.

        Input is the local estimate PLUS ``budget.input_overhead``, for the mirror-image
        reason: the estimate covers what we hand over and not the harness context that
        rides along with it, measured at roughly 29k per call on the CLI path. Both terms
        are learned from the ledger and both return the un-learned value on a cold one, so
        a fresh install reserves exactly what it used to.
        """
        return (self._est_in(messages, system)
                + budget.input_overhead(self._cfg)
                + budget.expected_output(self._cfg, max_tokens))

    def _before(self, messages, system, max_tokens) -> tuple[int, float]:
        """The floor, for EVERY caller. Raises BEFORE the call, so a refused one costs
        nothing. The batch's Gate stops politely one layer up; this is what stops
        rlm_query's recursive fan-out, which had nothing."""
        budget.check_or_raise(self._cfg, self._reserve(messages, system, max_tokens))
        return self._est_in(messages, system), time.monotonic()

    def _after(self, res, model, est_in, start, mode) -> CompletionResult:
        """Ledger the spend, then record the call. Both transports, one shape.

        The transport's own input total when it has one (cache included -- see
        _total_input), the estimate otherwise. max() needs no knowledge of which
        transport is in play and can only move the recorded figure UP, toward the truth;
        est_in rides along so input_overhead can learn the gap between them.

        THE LOG RECORD IS THE POINT OF PUTTING THIS HERE. Every log_event in this module
        used to sit inside CliTransport, so the SDK path produced no transport-level
        record at all -- no call, no duration, no outcome. That is why a MiniMax
        truncation that billed 4,096 output tokens for an empty answer could only be
        found by driving the tool by hand. Emitted from the wrapper both transports pass
        through, it is written once and cannot drift between them.
        """
        budget.record(self._cfg, res.model or model,
                      max(est_in, res.input_tokens), res.output_tokens, est=est_in)
        log_event(_LOG, "model_call", model=res.model or model, mode=mode,
                  dur_ms=round((time.monotonic() - start) * 1000),
                  in_tok=res.input_tokens, out_tok=res.output_tokens,
                  truncated=res.truncated or None)
        return res

    def _failed(self, exc, model, start, mode) -> None:
        self._note_if_limit(exc)
        log_event(_LOG, "model_call", model=model, mode=mode,
                  dur_ms=round((time.monotonic() - start) * 1000),
                  outcome=failures.classify(exc))

    def complete(self, messages, system, model, max_tokens) -> CompletionResult:
        est_in, start = self._before(messages, system, max_tokens)
        try:
            res = self._inner.complete(messages, system, model, max_tokens)
        except Exception as exc:
            self._failed(exc, model, start, "sync")
            raise
        return self._after(res, model, est_in, start, "sync")

    async def acomplete(self, messages, system, model, max_tokens) -> CompletionResult:
        est_in, start = self._before(messages, system, max_tokens)
        try:
            res = await self._inner.acomplete(messages, system, model, max_tokens)
        except Exception as exc:
            self._failed(exc, model, start, "async")
            raise
        return self._after(res, model, est_in, start, "async")

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
        if failures.is_rate_limit(exc):
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

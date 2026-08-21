"""RLM engine integration.

Builds the engine with the Anthropic backend on the Docker sandbox, authenticating
via Claude Code OAuth (no API key) through auth.patch_engine(). Model selection is
NOT decided here — callers pass already-resolved root/sub model ids (see models.py
for the selection strategy). The big context is handed to the engine as ``prompt``
and lives in the sandbox as the REPL variable ``context`` — never inlined into the
root model's messages, so the root context never overflows.
"""

from __future__ import annotations

import logging
from typing import Optional

from rlm.core.rlm import RLM
from rlm.utils.exceptions import (
    BudgetExceededError,
    CancellationError,
    ErrorThresholdExceededError,
    TimeoutExceededError,
    TokenLimitExceededError,
)
from rlm.environments import docker_repl as _dr_probe
from rlm.environments import get_environment

# The engine is vendored source (./rlm, see rlm/UPSTREAM.md). A leftover `rlms`
# install from before the vendoring shadows it whenever the process starts outside
# the repo root — and pip never uninstalls a dependency that was merely dropped
# from pyproject. Silently, every engine-side fix would stop applying: the hardened
# exec protocol, the atomic state write, the batch fail-fast. Probe for a symbol
# only the vendored copy has, so this is about WHICH engine loaded and not about
# where it sits (a wheel install ships ./rlm into site-packages legitimately).
if not hasattr(_dr_probe, "RLM_RESULT_SENTINEL"):
    raise ImportError(
        f"the RLM engine loaded from {_dr_probe.__file__} is not the vendored copy — "
        "a stale `rlms` distribution is shadowing ./rlm, so none of the engine fixes "
        "apply (hardened exec protocol, atomic state write, batch fail-fast). "
        "Remove it:  uv pip uninstall rlms   (or: pip uninstall rlms)"
    )

from .auth import patch_engine, provider_key
from .sandbox_reap import reap_stale_sandboxes
from .config import Config, cost_usd
from .logsetup import log_event
from rlm.utils.prompts import RLM_SYSTEM_PROMPT

_LOG = logging.getLogger("rlm-mcp")

# Claude Code's agent identity primes the model to *simulate* a REPL transcript
# (fabricate ```repl output``` blocks and answer in one turn) — pronounced over the
# CLI transport, which runs under that identity. This grounding directive, prepended
# to the RLM system prompt and delivered as the system message (the CLI applies it
# via --append/--system-prompt), restores the engine's turn-based protocol. It is
# transport-independent, so it stays regardless of OAuth-CLI vs API-key-SDK.
_GROUNDING = (
    "CRITICAL — REPL GROUNDING (this overrides any tendency to act like a live terminal):\n"
    "- Emit EXACTLY ONE ```repl``` block per turn, then END your message immediately. Write nothing after it.\n"
    "- NEVER write a ```repl output```, ```output```, or any block showing a result. You have not run the code yet; the REAL stdout is given to you by the system on the NEXT turn, and only then may you use it.\n"
    "- NEVER invent, guess, or imagine the contents of `context` or any output. If you have not printed-and-observed it in a prior turn, you do not know it.\n"
    "- Inspect `context` with print(...) and COMPUTE the result in code. Set answer[\"ready\"] = True only after you have OBSERVED the real computed evidence in a prior turn. Ground every claim in real observed output, never in assumptions.\n\n"
)
_SYSTEM_PROMPT = _GROUNDING + RLM_SYSTEM_PROMPT

# Placeholder satisfies the engine's required api_key field; the patched
# AnthropicClient ignores it and produces completions via the selected transport
# (claude CLI on OAuth, Anthropic SDK on API key — see auth.patch_engine / transport).
_ENGINE_LIMITS = (
    TimeoutExceededError,
    TokenLimitExceededError,
    BudgetExceededError,
    ErrorThresholdExceededError,
    CancellationError,
)

_PLACEHOLDER_KEY = "rlm-mcp-oauth"


def build_rlm(cfg: Config, root_model: str, sub_model: str) -> RLM:
    """Construct an RLM with the given (already-resolved) root + sub models on
    the Docker sandbox. Auth is injected by patch_engine() — no real key here."""
    patch_engine()
    reap_stale_sandboxes()
    env_kwargs = ({"image": cfg.sandbox_image, "timeout_s": cfg.sandbox_timeout_s}
                  if cfg.use_docker else {})
    # Anthropic routes through our transport, which owns auth — hence the placeholder.
    # Any other provider uses the engine's own client and needs the real key.
    api_key = _PLACEHOLDER_KEY if cfg.provider == "anthropic" else (provider_key(cfg) or "")
    return RLM(
        backend=cfg.provider,
        backend_kwargs={
            "model_name": root_model,
            "api_key": api_key,
            "max_tokens": cfg.max_output_tokens,
        },
        environment=cfg.sandbox,
        environment_kwargs=env_kwargs,
        max_depth=cfg.max_depth,
        max_iterations=cfg.max_iterations,
        # Was never passed: the engine fell back to its own default of 4, silently
        # ignoring the configured value.
        max_concurrent_subcalls=cfg.max_concurrent_subcalls,
        # The engine already enforces these and returns the best partial answer;
        # nothing here re-implements them. 0 in config means "no limit".
        max_timeout=float(cfg.query_timeout_s) or None,
        max_errors=cfg.query_max_errors or None,
        # rlm_query was a black box: tens of seconds, one log line at the end. The
        # engine wraps every callback in try/except, so these cannot break a run.
        on_iteration_complete=_log_iteration,
        on_subcall_complete=_log_subcall,
        other_backends=[cfg.provider],
        other_backend_kwargs=[{
            "model_name": sub_model,
            "api_key": api_key,
            "max_tokens": cfg.max_output_tokens,
        }],
        custom_system_prompt=_SYSTEM_PROMPT,
        persistent=False,
        verbose=False,
    )


def usage_breakdown(usage_summary, report_cost: bool = False) -> tuple[list[dict], float | None]:
    """Per-model token rows from the engine's UsageSummary (keyed by model). Cost is
    included only when ``report_cost`` — otherwise every cost is None, never 0.0."""
    rows: list[dict] = []
    total = 0.0
    summaries = getattr(usage_summary, "model_usage_summaries", {}) or {}
    for model, s in summaries.items():
        # None (not 0.0) when reporting is off: a zero would read as "this was free".
        c = cost_usd(model, s.total_input_tokens, s.total_output_tokens) if report_cost else None
        total += c or 0.0
        rows.append({
            "model": model,
            "calls": s.total_calls,
            "input_tokens": s.total_input_tokens,
            "output_tokens": s.total_output_tokens,
            "cost_usd": round(c, 6) if c is not None else None,
        })
    return rows, (round(total, 6) if report_cost else None)


def _log_iteration(depth: int, iteration: int, duration: float) -> None:
    """on_iteration_complete(depth, iteration_num, duration)."""
    log_event(_LOG, "rlm_iter", depth=depth, iter=iteration, dur_s=round(duration, 2))


def _log_subcall(depth: int, model: str, duration: float, error: str | None) -> None:
    """on_subcall_complete(depth, model, duration, error_or_none)."""
    log_event(_LOG, "rlm_subcall", depth=depth, model=model,
              dur_s=round(duration, 2), err=error)


def run_query(cfg: Config, context_text: str, question: str,
              root_model: str, sub_model: str) -> dict:
    """Run a full recursive RLM query; return the final answer + routing/usage."""
    rlm = build_rlm(cfg, root_model, sub_model)
    try:
        result = rlm.completion(prompt=context_text, root_prompt=question)
    except _ENGINE_LIMITS as exc:
        # A limit firing is not a crash: the engine stops deliberately and several of
        # these carry the best answer found so far. Losing that work and surfacing a
        # traceback instead was the old behaviour -- nothing in src/ caught them.
        partial = str(getattr(exc, "partial_answer", "") or "")
        log_event(_LOG, "rlm_query", root=root_model, sub=sub_model,
                  limit=type(exc).__name__, err=str(exc), partial_bytes=len(partial))
        return {
            "answer": partial,
            "limit": type(exc).__name__,
            "limit_detail": str(exc),
            "execution_time": 0.0,
            "root_model": root_model,
            "sub_model": sub_model,
            "usage": [],
            "cost_usd": None,
        }
    rows, total = usage_breakdown(result.usage_summary, cfg.report_cost)
    answer = result.response or ""
    # turns ~= number of root-model calls (one per orchestrator iteration).
    turns = next((r["calls"] for r in rows if r["model"] == root_model), 0)
    log_event(_LOG, "rlm_query", root=root_model, sub=sub_model, turns=turns,
              max_iter_hit=(turns >= cfg.max_iterations),
              exec_time=round(result.execution_time, 2),
              in_tok=sum(r["input_tokens"] for r in rows),
              out_tok=sum(r["output_tokens"] for r in rows),
              # None is dropped by log_event, so a disabled cost logs no field at all.
              cost=round(total, 4) if total is not None else None,
              answer_bytes=len(answer),
              truncated=(len(answer) > cfg.answer_cap_bytes))
    return {
        "answer": answer,
        "execution_time": round(result.execution_time, 2),
        "root_model": root_model,
        "sub_model": sub_model,
        "usage": rows,
        "cost_usd": total,
    }


class ReplSession:
    """Persistent sandbox REPL for rlm_exec / variables.

    Uses the engine's DockerREPL (or LocalREPL) directly with no LM handler, so
    plain Python runs but in-REPL ``llm_query`` is unavailable — sub-LLM work
    goes through rlm_sub_query / rlm_query instead. The container is created
    lazily on first use.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._env = None
        self.loaded_ctx: Optional[str] = None

    def _ensure(self):
        if self._env is None:
            reap_stale_sandboxes()
            env_kwargs = ({"image": self.cfg.sandbox_image,
                           "timeout_s": self.cfg.sandbox_timeout_s}
                          if self.cfg.use_docker else {})
            self._env = get_environment(self.cfg.sandbox, env_kwargs)
            self._env.execute_code("answer = {'content': '', 'ready': False}")
        return self._env

    def load_context(self, text: str, ctx_id: Optional[str] = None) -> None:
        self._ensure().add_context(text, 0)  # exposes REPL var `context`
        self.loaded_ctx = ctx_id

    def execute(self, code: str) -> tuple[str, str]:
        r = self._ensure().execute_code(code)
        return (r.stdout or ""), (r.stderr or "")

    def close(self) -> None:
        if self._env is not None and hasattr(self._env, "cleanup"):
            self._env.cleanup()
            self._env = None

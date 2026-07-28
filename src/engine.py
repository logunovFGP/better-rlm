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
from rlm.environments import get_environment

from .auth import patch_engine
from .sandbox_patch import patch_sandbox
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
_PLACEHOLDER_KEY = "rlm-mcp-oauth"


def build_rlm(cfg: Config, root_model: str, sub_model: str) -> RLM:
    """Construct an RLM with the given (already-resolved) root + sub models on
    the Docker sandbox. Auth is injected by patch_engine() — no real key here."""
    patch_engine()
    patch_sandbox()
    env_kwargs = {"image": cfg.sandbox_image} if cfg.use_docker else {}
    return RLM(
        backend="anthropic",
        backend_kwargs={
            "model_name": root_model,
            "api_key": _PLACEHOLDER_KEY,
            "max_tokens": cfg.max_output_tokens,
        },
        environment=cfg.sandbox,
        environment_kwargs=env_kwargs,
        max_depth=cfg.max_depth,
        max_iterations=cfg.max_iterations,
        # Was never passed: the engine fell back to its own default of 4, silently
        # ignoring the configured value.
        max_concurrent_subcalls=cfg.max_concurrent_subcalls,
        other_backends=["anthropic"],
        other_backend_kwargs=[{
            "model_name": sub_model,
            "api_key": _PLACEHOLDER_KEY,
            "max_tokens": cfg.max_output_tokens,
        }],
        custom_system_prompt=_SYSTEM_PROMPT,
        persistent=False,
        verbose=False,
    )


def usage_breakdown(usage_summary) -> tuple[list[dict], float]:
    """Per-model token + cost rows from the engine's UsageSummary (keyed by model)."""
    rows: list[dict] = []
    total = 0.0
    summaries = getattr(usage_summary, "model_usage_summaries", {}) or {}
    for model, s in summaries.items():
        c = cost_usd(model, s.total_input_tokens, s.total_output_tokens)
        total += c
        rows.append({
            "model": model,
            "calls": s.total_calls,
            "input_tokens": s.total_input_tokens,
            "output_tokens": s.total_output_tokens,
            "cost_usd": round(c, 6),
        })
    return rows, round(total, 6)


def run_query(cfg: Config, context_text: str, question: str,
              root_model: str, sub_model: str) -> dict:
    """Run a full recursive RLM query; return the final answer + routing/usage."""
    rlm = build_rlm(cfg, root_model, sub_model)
    result = rlm.completion(prompt=context_text, root_prompt=question)
    rows, total = usage_breakdown(result.usage_summary)
    answer = result.response or ""
    # turns ~= number of root-model calls (one per orchestrator iteration).
    turns = next((r["calls"] for r in rows if r["model"] == root_model), 0)
    log_event(_LOG, "rlm_query", root=root_model, sub=sub_model, turns=turns,
              max_iter_hit=(turns >= cfg.max_iterations),
              exec_time=round(result.execution_time, 2),
              in_tok=sum(r["input_tokens"] for r in rows),
              out_tok=sum(r["output_tokens"] for r in rows),
              cost=round(total, 4), answer_bytes=len(answer),
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
            patch_sandbox()
            env_kwargs = {"image": self.cfg.sandbox_image} if self.cfg.use_docker else {}
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

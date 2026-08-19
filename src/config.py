"""Configuration for the RLM MCP server.

Secrets come from ``.env``; tunables come from ``config.yaml`` layered over the
baked-in defaults below. Real environment variables always win over ``.env``.
Model IDs and cost rates were verified against the authoritative ``claude-api``
reference, not memory.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PKG_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PKG_ROOT / ".env")  # no-op if the file is absent

# --- Verified Anthropic model IDs (claude-api reference) ---
MODEL_SONNET_5 = "claude-sonnet-5"  # 1M ctx — current default root/orchestrator
MODEL_SONNET = "claude-sonnet-4-6"  # 1M ctx — prior root (still selectable)
MODEL_OPUS = "claude-opus-4-8"      # 1M ctx — root override for the hardest tasks
MODEL_HAIKU = "claude-haiku-4-5"    # 200K ctx — cheap sub-LLM for chunk work

COST_PER_MTOK: dict[str, tuple[float, float]] = {
    # Cost is informational only on the OAuth/CLI path (it draws on the subscription).
    # Sonnet 5 rates cloned from Sonnet 4.6 pending published pricing.
    MODEL_SONNET_5: (3.0, 15.0),
    MODEL_SONNET: (3.0, 15.0),
    MODEL_OPUS: (5.0, 25.0),
    MODEL_HAIKU: (1.0, 5.0),
}

HAIKU_CONTEXT_TOKENS = 200_000

# Provider → env var holding its API key. The engine (rlm.clients.get_client) already
# routes these backends; this map is only how we locate the credential. Anthropic is
# listed but special: it is the ONE provider that can authenticate with no key at all,
# through the local `claude` CLI login — see auth.resolve_auth_mode.
PROVIDER_KEY_ENV: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
    "azure_openai": "AZURE_OPENAI_API_KEY",
    "portkey": "PORTKEY_API_KEY",
}

_DEFAULTS: dict[str, Any] = {
    "root_model": MODEL_SONNET_5,
    "root_model_override": MODEL_OPUS,
    "sub_model": MODEL_HAIKU,
    "max_depth": 2,
    "max_iterations": 20,
    "max_output_tokens": 32768,
    "sandbox": "docker",
    "sandbox_image": "rlm-sandbox",
    "sandbox_timeout_s": 300,
    "max_concurrent_subcalls": 2,
    "subquery_concurrency": 3,
    "output_cap_bytes": 4096,      # raw-content tools (load/inspect/chunk/grep/read/list/exec/status)
    "answer_cap_bytes": 131072,    # synthesis tools (rlm_query/sub_query[_batch]) — the answer IS the deliverable
    "chunk_strategy": "lines",
    "chunk_lines": 2000,
    "chunk_chars": 120_000,
    "chunk_overlap": 0,
    # Rate-limit handling (see ratelimit.py): global throttle + auth-aware retry.
    "throttle_max_concurrency": 3,     # at most N API calls in flight at once
    "throttle_min_interval_s": 1.0,    # >= this many seconds between dispatches
    "oauth_retry_waits": [5, 10, 15],  # 429 backoff on OAuth (tight subscription limits)
    "apikey_retry_waits": [1, 2, 4],   # 429 backoff on API key (higher limits)
    # Transport mode: auto | claude-cli | api. "auto" prefers the `claude` CLI
    # (reuse the Claude Code login — no token/key needed) and falls back to the
    # API key. Overridable per-launch with the RLM_MODE env var (e.g. in the
    # `claude mcp add -e RLM_MODE=claude-cli` registration).
    "mode": "auto",
    # Cost reporting is OFF by default: the rate table below only covers Anthropic
    # models, and on the claude-CLI path the reported input-token count under-counts
    # the piped prompt — so a printed "$" figure would be confidently wrong. Turn it on
    # only when the configured models are in COST_PER_MTOK and you trust the counts.
    "report_cost": False,
    # Context ceiling of the SUB model, used to refuse an oversized single sub-query and
    # to skip a too-large reduce pass. 200K is Haiku's; other providers differ wildly
    # (Gemini is 1M+), so it MUST track the configured sub_model, not a vendor constant.
    "sub_context_tokens": HAIKU_CONTEXT_TOKENS,
    # Which vendor answers. "anthropic" is the ONLY one that works with no API key,
    # via the local `claude` CLI login — so it stays the default and the local path is
    # untouched. Remote deploys (no keychain in a pod) set this + the provider's key.
    "provider": "anthropic",
    # Claude Code CLI transport (see transport.py). Under claude-cli mode the server
    # drives the official `claude` CLI instead of the HTTP API, so it "just works" via
    # the existing Claude Code login — no token plumbing, no premium-model gating.
    "cli_path": "claude",
    "cli_system_prompt_mode": "replace",  # replace (clean RLM prompt; verified to keep premium access) | append
    "cli_timeout_s": 600,                # per-call CLI budget (cold start + generation)
    "cli_disable_tools": True,           # --tools "" : RLM runs its OWN sandbox, CLI emits text only
    "cli_safe_mode": True,               # --safe-mode : no hooks/CLAUDE.md/skills/MCP (prevents recursion)
    "cli_no_session_persistence": True,  # --no-session-persistence : stateless calls
    "cli_fallback_model": "",            # optional --fallback-model on overload
    "cli_extra_args": [],                # escape hatch: extra argv tokens
    # Structured file logging + bounded retention (see logsetup.py). Per-PID rotated
    # files in log_dir; a startup sweep caps total files/bytes/age across ALL processes
    # so many churning server processes can never fill the disk.
    "log_level": "INFO",                 # file handler level; stderr stays WARNING-only
    "log_to_file": True,                 # master switch for file logging + sweep
    "log_max_bytes": 2_097_152,          # 2 MB per file before rotation
    "log_backup_count": 3,               # per-PID rotated backups (one family <= 8 MB)
    "log_retention_files": 20,           # sweep: keep newest N rlm-mcp-*.log*
    "log_retention_total_bytes": 52_428_800,  # sweep: total across all files <= 50 MB
    "log_retention_days": 7,             # sweep: delete files older than this
    "log_sweep_cooldown_s": 60,          # skip the sweep if it ran within this window
    "log_dir": "~/.rlm/logs",
    "store_dir": "~/.rlm/contexts",
    # Registry of named external sources (see sources.py) — commands an operator declares
    # that the server may run and load. Outside the repo on purpose: a site's clusters,
    # endpoints and tooling stay out of the checkout and out of every diff. Absent file =
    # no sources; this server ships none.
    "sources_file": "~/.rlm/sources.yaml",
}


def _sandbox(value: str) -> str:
    """Validate the sandbox backend. ``use_docker`` is ``sandbox == "docker"``, so an
    unrecognised value would silently mean *host execution of model-written Python*.
    Reject it loudly instead of degrading into the unsafe mode."""
    v = value.strip().lower()
    if v not in ("docker", "local"):
        raise ValueError(
            f"sandbox must be 'docker' or 'local', got {value!r} "
            "(check config.yaml or the RLM_SANDBOX env var)."
        )
    return v


def _load_yaml() -> dict[str, Any]:
    path = PKG_ROOT / "config.yaml"
    if not path.exists():
        return {}
    with open(path) as fh:
        data = yaml.safe_load(fh) or {}
    return data if isinstance(data, dict) else {}


@dataclass(frozen=True)
class Config:
    """Immutable resolved configuration (defaults <- config.yaml)."""

    root_model: str
    root_model_override: str
    sub_model: str
    max_depth: int
    max_iterations: int
    max_output_tokens: int
    sandbox: str
    sandbox_image: str
    sandbox_timeout_s: int
    max_concurrent_subcalls: int
    subquery_concurrency: int
    output_cap_bytes: int
    answer_cap_bytes: int
    chunk_strategy: str
    chunk_lines: int
    chunk_chars: int
    chunk_overlap: int
    throttle_max_concurrency: int
    throttle_min_interval_s: float
    oauth_retry_waits: tuple[float, ...]
    apikey_retry_waits: tuple[float, ...]
    report_cost: bool
    sub_context_tokens: int
    mode: str
    provider: str
    cli_path: str
    cli_system_prompt_mode: str
    cli_timeout_s: int
    cli_disable_tools: bool
    cli_safe_mode: bool
    cli_no_session_persistence: bool
    cli_fallback_model: str
    cli_extra_args: tuple[str, ...]
    log_level: str
    log_to_file: bool
    log_max_bytes: int
    log_backup_count: int
    log_retention_files: int
    log_retention_total_bytes: int
    log_retention_days: int
    log_sweep_cooldown_s: int
    log_dir: Path
    store_dir: Path
    sources_file: Path

    @property
    def use_docker(self) -> bool:
        return self.sandbox == "docker"


def load_config() -> Config:
    """Build the resolved Config from defaults overlaid with config.yaml."""
    m = {**_DEFAULTS, **_load_yaml()}
    return Config(
        root_model=str(m["root_model"]),
        root_model_override=str(m["root_model_override"]),
        sub_model=str(m["sub_model"]),
        max_depth=int(m["max_depth"]),
        max_iterations=int(m["max_iterations"]),
        max_output_tokens=int(m["max_output_tokens"]),
        # RLM_SANDBOX env wins over config.yaml, same as mode/provider — so a host with
        # no Docker daemon selects `local` at registration instead of committing an
        # unsafe default. Validated: a typo must not silently become host exec.
        sandbox=_sandbox(str(os.getenv("RLM_SANDBOX") or m["sandbox"])),
        sandbox_image=str(m["sandbox_image"]),
        sandbox_timeout_s=int(m["sandbox_timeout_s"]),
        max_concurrent_subcalls=int(m["max_concurrent_subcalls"]),
        subquery_concurrency=int(m["subquery_concurrency"]),
        output_cap_bytes=int(m["output_cap_bytes"]),
        answer_cap_bytes=int(m["answer_cap_bytes"]),
        chunk_strategy=str(m["chunk_strategy"]),
        chunk_lines=int(m["chunk_lines"]),
        chunk_chars=int(m["chunk_chars"]),
        chunk_overlap=int(m["chunk_overlap"]),
        throttle_max_concurrency=int(m["throttle_max_concurrency"]),
        throttle_min_interval_s=float(m["throttle_min_interval_s"]),
        oauth_retry_waits=tuple(float(x) for x in m["oauth_retry_waits"]),
        apikey_retry_waits=tuple(float(x) for x in m["apikey_retry_waits"]),
        # RLM_MODE env wins over config.yaml so the mode can be set at registration.
        report_cost=bool(m["report_cost"]),
        sub_context_tokens=int(m["sub_context_tokens"]),
        mode=str(os.getenv("RLM_MODE") or m["mode"]).strip().lower(),
        # RLM_PROVIDER env wins, same as mode, so a pod sets it at registration.
        provider=str(os.getenv("RLM_PROVIDER") or m["provider"]).strip().lower(),
        cli_path=str(m["cli_path"]),
        cli_system_prompt_mode=str(m["cli_system_prompt_mode"]),
        cli_timeout_s=int(m["cli_timeout_s"]),
        cli_disable_tools=bool(m["cli_disable_tools"]),
        cli_safe_mode=bool(m["cli_safe_mode"]),
        cli_no_session_persistence=bool(m["cli_no_session_persistence"]),
        cli_fallback_model=str(m["cli_fallback_model"]),
        cli_extra_args=tuple(str(x) for x in m["cli_extra_args"]),
        log_level=str(m["log_level"]).upper(),
        log_to_file=bool(m["log_to_file"]),
        log_max_bytes=int(m["log_max_bytes"]),
        log_backup_count=int(m["log_backup_count"]),
        log_retention_files=int(m["log_retention_files"]),
        log_retention_total_bytes=int(m["log_retention_total_bytes"]),
        log_retention_days=int(m["log_retention_days"]),
        log_sweep_cooldown_s=int(m["log_sweep_cooldown_s"]),
        log_dir=Path(os.path.expanduser(str(m["log_dir"]))),
        store_dir=Path(os.path.expanduser(str(m["store_dir"]))),
        sources_file=Path(os.path.expanduser(str(m["sources_file"]))),
    )


def estimate_tokens(text_or_len: str | int) -> int:
    """Rough token estimate (~4 chars/token). Use count_tokens for precision."""
    n = len(text_or_len) if isinstance(text_or_len, str) else int(text_or_len)
    return max(1, n // 4)


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """USD cost for a model call from token counts and known rates."""
    rate = COST_PER_MTOK.get(model)
    if not rate:
        return 0.0
    return (input_tokens / 1_000_000) * rate[0] + (output_tokens / 1_000_000) * rate[1]

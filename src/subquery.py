"""Haiku map-reduce sub-queries via the active completion transport (CLI or SDK).

The actual model call is wrapped by retry_and_queue_retries, so sub-queries share
the global throttle (3 concurrent, 1s spacing) and the auth-aware backoff (429 on
the SDK path, rate/usage limits on the CLI path) with every other call in the
process. Model is the configured sub-model (Haiku). The transport is chosen by
auth mode (OAuth -> claude CLI, API key -> Anthropic SDK).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from .auth import resolve_auth_mode
from .config import load_config
from .logsetup import bind_rid, current_rid
from .ratelimit import is_fatal_auth, retry_and_queue_retries
from .transport import get_transport

_CFG = load_config()


@dataclass
class SubResult:
    index: int
    answer: str
    input_tokens: int
    output_tokens: int
    error: str | None = None


@retry_and_queue_retries
def _call(model: str, prompt: str, max_tokens: int,
          system: str | None) -> tuple[str, int, int]:
    transport = get_transport(resolve_auth_mode(_CFG), _CFG)
    res = transport.complete(
        [{"role": "user", "content": prompt}], system, model, max_tokens)
    return res.text, res.input_tokens, res.output_tokens


def sub_query(prompt: str, model: str, *, max_tokens: int = 4096,
              system: str | None = None) -> SubResult:
    try:
        text, itok, otok = _call(model, prompt, max_tokens, system)
        return SubResult(0, text, itok, otok)
    except Exception as exc:  # surfaced to caller, not swallowed
        return SubResult(0, "", 0, 0, error=str(exc))


def sub_query_batch(prompts: list[str], model: str, *, concurrency: int,
                    max_tokens: int = 2048, system: str | None = None) -> list[SubResult]:
    # Pool workers start with a fresh contextvars context, so capture the caller's
    # correlation id here and re-bind it inside each worker — otherwise the nested
    # cli_spawn/retry events lose the originating tool call's rid.
    parent_rid = current_rid()
    # A dead login is global, not per-chunk: without this, every remaining prompt
    # spawns its own doomed call (measured: 20 chunks, 20 identical "OAuth session
    # expired"). Only auth aborts the batch — a chunk-specific failure must not
    # discard the chunks that would have succeeded.
    # ponytail: plain list as the flag. The benign race lets the calls already in
    # flight finish; a threading.Event only matters if that ever costs something.
    fatal: list[str] = []

    def work(item: tuple[int, str]) -> SubResult:
        idx, prompt = item
        if fatal:
            return SubResult(idx, "", 0, 0, error=f"skipped — {fatal[0]}")
        with bind_rid(parent_rid):
            try:
                text, itok, otok = _call(model, prompt, max_tokens, system)
                return SubResult(idx, text, itok, otok)
            except Exception as exc:
                if is_fatal_auth(exc):
                    fatal.append(str(exc))
                return SubResult(idx, "", 0, 0, error=str(exc))

    # pool.map yields in submission order, so results already line up with `prompts`.
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        return list(pool.map(work, enumerate(prompts)))

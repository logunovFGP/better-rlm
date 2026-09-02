"""Haiku map-reduce sub-queries via the active completion transport (CLI or SDK).

The actual model call is wrapped by retry_and_queue_retries, so sub-queries share
the global throttle (3 concurrent, 1s spacing) and the auth-aware backoff (429 on
the SDK path, rate/usage limits on the CLI path) with every other call in the
process. Model is the configured sub-model (Haiku). The transport is chosen by
auth mode (OAuth -> claude CLI, API key -> Anthropic SDK).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from . import budget
from .auth import resolve_auth_mode
from .config import estimate_tokens, load_config
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
    #: Model that actually produced this answer, as reported by the transport --
    #: not the id we asked for. On OAuth these differ: models.select maps a
    #: configured id to its closest subscription-supported sibling, so the only
    #: way to know what ran is to read it back.
    model: str = ""


@retry_and_queue_retries
def _call(model: str, prompt: str, max_tokens: int,
          system: str | None) -> tuple[str, int, int, str]:
    transport = get_transport(resolve_auth_mode(_CFG), _CFG)
    res = transport.complete(
        [{"role": "user", "content": prompt}], system, model, max_tokens)
    return res.text, res.input_tokens, res.output_tokens, res.model


def sub_query(prompt: str, model: str, *, max_tokens: int = 4096,
              system: str | None = None) -> SubResult:
    try:
        text, itok, otok, used = _call(model, prompt, max_tokens, system)
        # Ledger gets our LOCAL estimate of what we sent, not `itok`: the CLI transport
        # reported 1027 input tokens for a batch whose real input was ~3M, so a budget
        # built on the reported number would promise headroom that does not exist.
        budget.record(_CFG, used or model, estimate_tokens(prompt), otok)
        return SubResult(0, text, itok, otok, model=used)
    except Exception as exc:  # surfaced to caller, not swallowed
        return SubResult(0, "", 0, 0, error=str(exc))


def sub_query_batch(prompts: Sequence[str | Callable[[], str]], model: str, *, concurrency: int,
                    max_tokens: int = 2048, system: str | None = None,
                    indices: Sequence[int] | None = None,
                    gate: "budget.Gate | None" = None,
                    on_result: Callable[[SubResult], None] | None = None) -> list[SubResult]:
    """Map ``prompts`` over the sub-model concurrently.

    ``indices``  report each result under its ORIGINAL chunk index rather than its
                 position in this list. A resumed batch submits only the unanswered
                 chunks, so position and chunk index stop agreeing — and every caller
                 that labels, persists or re-orders results keys on the chunk index.
    ``gate``     budget.Gate consulted before each dispatch. When it closes, the
                 remaining prompts come back marked ``deferred — …`` instead of being
                 sent: a deferred chunk is resumable work, not a failure.
    ``on_result`` called with each successful result as it lands, so the caller can
                 persist it immediately. An interrupted run keeps what it paid for only
                 if the answer reaches disk before the process dies.
    """
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

    def work(item: tuple[int, str | Callable[[], str]]) -> SubResult:
        pos, entry = item
        idx = indices[pos] if indices is not None else pos
        if fatal:
            return SubResult(idx, "", 0, 0, error=f"skipped — {fatal[0]}")
        if gate is not None and gate.closed:
            return SubResult(idx, "", 0, 0, error="deferred — session budget reached")
        try:
            # Built here, not by the caller: a lazy builder keeps only `concurrency`
            # prompt strings live at once instead of the whole batch (Leaf 3 / option D).
            prompt = entry() if callable(entry) else entry
        except Exception as exc:
            return SubResult(idx, "", 0, 0, error=f"prompt build failed — {exc}")
        # Asked AFTER the prompt is built because only then is its size known, and the
        # gate reserves against the size of the call it is about to admit.
        est_in = estimate_tokens(prompt)
        if gate is not None and not gate.allow(est_in + max_tokens):
            return SubResult(idx, "", 0, 0, error="deferred — session budget reached")
        with bind_rid(parent_rid):
            try:
                text, itok, otok, used = _call(model, prompt, max_tokens, system)
                budget.record(_CFG, used or model, est_in, otok)
                res = SubResult(idx, text, itok, otok, model=used)
                if on_result is not None:
                    try:
                        on_result(res)
                    except Exception:  # persistence must not lose an answer we just paid for
                        pass
                return res
            except Exception as exc:
                if is_fatal_auth(exc):
                    fatal.append(str(exc))
                return SubResult(idx, "", 0, 0, error=str(exc))

    # pool.map yields in submission order, so results already line up with `prompts`.
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        return list(pool.map(work, enumerate(prompts)))

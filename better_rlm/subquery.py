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

from .auth import resolve_auth_mode
from .budget import BudgetStopError, expected_output
from .config import Config, estimate_tokens
from .logsetup import bind_rid, current_rid
from .ratelimit import is_fatal_auth, retry_and_queue_retries
from .transport import get_transport


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
    #: The call hit ``max_tokens``. ``answer`` is then a fragment, or empty when a
    #: reasoning model spent the whole budget thinking. Carried separately from
    #: ``error`` because the call SUCCEEDED and was billed -- the caller has to be
    #: able to tell "produced nothing" from "failed".
    truncated: bool = False


#: Output budget for one sub-model call.
#:
#: Was 4096, chosen when every sub-model was a Claude that answers directly. A
#: reasoning model emits its thinking FIRST and bills it as output: measured on
#: MiniMax-M2.7 over one 12.9k-token log chunk, 13,015 output tokens, of which the
#: thinking block was ~99% and the answer was 286 characters. Every sub-query over a
#: real chunk therefore hit the cap mid-thought and returned EMPTY.
#:
#: 16384 clears that measurement with headroom rather than matching it exactly, since
#: a denser chunk thinks for longer. It is not a cost increase -- output is billed as
#: generated, and a model that answers in 200 tokens still costs 200. It does raise
#: what budget.expected_output reserves per call on the SDK path, which is the honest
#: direction: the old 4096 reservation was under-counting real MiniMax calls 3x.
SUB_MAX_TOKENS = 16384


@retry_and_queue_retries
def _call(cfg: Config, model: str, prompt: str, max_tokens: int,
          system: str | None) -> tuple[str, int, int, str, bool]:
    transport = get_transport(resolve_auth_mode(cfg), cfg)
    res = transport.complete(
        [{"role": "user", "content": prompt}], system, model, max_tokens)
    return res.text, res.input_tokens, res.output_tokens, res.model, res.truncated


def sub_query(cfg: Config, prompt: str, model: str, *, max_tokens: int = SUB_MAX_TOKENS,
              system: str | None = None) -> SubResult:
    try:
        text, itok, otok, used, cut = _call(cfg, model, prompt, max_tokens, system)
        # No budget.record here: transport._LedgeredTransport records EVERY completion,
        # including the engine's, so recording again would double-count this one.
        return SubResult(0, text, itok, otok, model=used, truncated=cut)
    except Exception as exc:  # surfaced to caller, not swallowed
        return SubResult(0, "", 0, 0, error=str(exc))


def sub_query_batch(cfg: Config, prompts: Sequence[str | Callable[[], str]], model: str, *,
                    concurrency: int,
                    max_tokens: int = SUB_MAX_TOKENS, system: str | None = None,
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
    budget_stop: list[bool] = []   # same benign-race shape as `fatal`, same reason
    # What a call EMITS, not the cap we ask for: the CLI takes no output flag, so on the
    # OAuth path `max_tokens` is never sent and reserving it lets the gate admit calls it
    # cannot afford. Resolved once for the batch — it is the same for every chunk, and it
    # reads the ledger.
    out_reserve = expected_output(cfg, max_tokens)

    def work(item: tuple[int, str | Callable[[], str]]) -> SubResult:
        pos, entry = item
        idx = indices[pos] if indices is not None else pos
        if fatal:
            return SubResult(idx, "", 0, 0, error=f"skipped — {fatal[0]}")
        if budget_stop or (gate is not None and gate.closed):
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
        if gate is not None and not gate.allow(est_in + out_reserve):
            return SubResult(idx, "", 0, 0, error="deferred — session budget reached")
        with bind_rid(parent_rid):
            try:
                text, itok, otok, used, cut = _call(cfg, model, prompt, max_tokens, system)
                res = SubResult(idx, text, itok, otok, model=used,   # ledgered in transport
                                truncated=cut)
                if on_result is not None:
                    try:
                        on_result(res)
                    except Exception:  # persistence must not lose an answer we just paid for
                        pass
                return res
            except BudgetStopError:
                # The transport's hard floor fired under us (the batch Gate normally
                # stops first; this is the fallback). Not a failure: scheduled work.
                # Latch the gate if we have one so the rest defer without a call each.
                if gate is not None:
                    gate.closed = True
                budget_stop.append(True)
                return SubResult(idx, "", 0, 0, error="deferred — session budget reached")
            except Exception as exc:
                if is_fatal_auth(exc):
                    fatal.append(str(exc))
                return SubResult(idx, "", 0, 0, error=str(exc))

    # pool.map yields in submission order, so results already line up with `prompts`.
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        return list(pool.map(work, enumerate(prompts)))

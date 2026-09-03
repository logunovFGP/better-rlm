"""Session-window token budget: a spend ledger, a pre-flight estimate, and a mid-run gate.

Why this module exists, concretely: one ``rlm_sub_query_batch`` over a 103-chunk context
made 104 model calls, ran 30 minutes, drew ~60% of a 4-hour subscription window, was
interrupted partway through the *second* pass, and returned nothing. The whole spend
bought no answer, and the obvious next move — run it again — would have spent the same
again for the same nothing. Three separate defects, addressed here and in the callers:

  1. Nobody could see the size of the run before starting it (``estimate_batch``).
  2. Nothing stopped the run before the wall, so it died mid-flight (``Gate``).
  3. Completed chunks were not durable, so a stopped run started over (``results.py``).

A NOTE ON WHAT WE CAN AND CANNOT KNOW. Anthropic publishes no token ceiling for a
subscription and exposes no endpoint to read the remaining balance of the rolling window.
So this module does NOT report "your real remaining quota" — it maintains a LOCAL ledger
of what this server itself spent, over the same rolling window the vendor enforces, and
compares that against a CONFIGURED ceiling. Spend from other Claude Code sessions on the
same account is invisible here, which biases every estimate optimistic. That is stated in
the rendered output rather than hidden, because a budget tool that quietly implies more
headroom than exists is worse than none.

That same blind spot is why a usage limit we actually hit (``note_limit_hit``) is recorded
as EVIDENCE and reported, but never promoted to the ceiling on its own: it is a floor
under the real one, and gating on a floor puts the stop line below the spend that taught
it. See ``ceiling``.

INPUT TOKENS: THE TRANSPORT'S TOTAL WHERE IT HAS ONE, THE LOCAL ESTIMATE OTHERWISE. The
claude-CLI path reports ``usage.input_tokens`` at a count that is not merely low but
useless — the failing run logged ``itok=1027`` for 103 chunks whose real input was ~3M
tokens, three orders of magnitude off — so for a long time this module estimated input
locally and ignored the transport entirely.

That was half right. The number is not missing, it is SPLIT: prompt caching puts the bulk
in ``cache_creation_input_tokens`` and ``cache_read_input_tokens``, and ``input_tokens`` is
only the uncached remainder. Summing all three (``transport._total_input``) gives the real
figure, and the ledger now records ``max(local_estimate, reported_total)`` — which needs no
knowledge of which transport is in play and can only move the recorded number toward the
truth.

The local estimate still matters, for the two places that must run BEFORE the call: the
pre-call floor and the forecast. ``estimate_tokens`` over the bytes we are about to send is
crude (~4 chars/token) and, on its own, ~2x low — it cannot see the harness context the CLI
prepends, measured at ~29k per call. ``input_overhead`` learns that gap from the ledger and
both callers add it.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import Clock, Config, estimate_tokens

#: Ledger lines older than the widest window we would ever ask about are dead weight.
#: Pruned opportunistically on append, not by a scheduled job.
_PRUNE_AFTER_H = 48.0
#: Append is under this lock so concurrent batch workers cannot interleave a partial line.
_LOCK = threading.Lock()


# --------------------------------------------------------------------------- #
# Ledger
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Spend:
    """What this server has spent inside the current rolling window."""

    tokens: int
    calls: int
    window_h: float
    #: Seconds until the OLDEST call in the window ages out — i.e. when headroom next
    #: grows. None when nothing is in the window. This is the honest answer to "when can
    #: I run again": the window rolls continuously, it does not reset on a clock boundary.
    oldest_expires_in_s: float | None = None


def _read_lines(path: Path, since_ts: float) -> list[dict]:
    """Ledger records at or after ``since_ts``. A corrupt line is skipped, never fatal —
    the ledger is an advisory accounting aid and must not be able to break a tool call."""
    try:
        raw = path.read_text(encoding="utf-8").splitlines()
    except (FileNotFoundError, OSError):
        return []
    out: list[dict] = []
    for line in raw:
        try:
            rec = json.loads(line)
            if float(rec.get("ts", 0)) >= since_ts:
                out.append(rec)
        except (ValueError, TypeError):
            continue
    return out


def record(cfg: Config, model: str, itok: int, otok: int, *, est: int = 0,
           now: Clock = time.time) -> None:
    """Append one call's spend. Best-effort: ledger failure must never fail the call.

    ``itok`` is the best available figure for what the call actually consumed on input —
    the transport's total when it reports one (cache included, see
    ``transport._total_input``), otherwise the caller's local estimate. ``est`` is that
    local estimate, kept beside it so ``input_overhead`` can learn the difference. Records
    written before ``est`` existed simply carry none and are skipped there.
    """
    rec = {"ts": now(), "model": model, "itok": int(itok), "otok": int(otok)}
    if est:
        rec["est"] = int(est)
    try:
        cfg.budget_ledger.parent.mkdir(parents=True, exist_ok=True)
        with _LOCK:
            with open(cfg.budget_ledger, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
            _prune(cfg, now)
    except OSError:
        pass


def _prune(cfg: Config, now: Clock) -> None:
    """Drop records older than _PRUNE_AFTER_H once the file grows past a few hundred KB.
    Rewrite via tmp + os.replace so a crash cannot truncate the ledger."""
    try:
        if cfg.budget_ledger.stat().st_size < 512_000:
            return
        keep = _read_lines(cfg.budget_ledger, now() - _PRUNE_AFTER_H * 3600)
        tmp = cfg.budget_ledger.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text("".join(json.dumps(r) + "\n" for r in keep), encoding="utf-8")
        os.replace(tmp, cfg.budget_ledger)
    except OSError:
        pass


def spent(cfg: Config, *, now: Clock = time.time) -> Spend:
    """This server's spend inside the current rolling window."""
    t = now()
    window_s = cfg.session_window_h * 3600
    recs = _read_lines(cfg.budget_ledger, t - window_s)
    if not recs:
        return Spend(0, 0, cfg.session_window_h)
    total = sum(int(r.get("itok", 0)) + int(r.get("otok", 0)) for r in recs)
    oldest = min(float(r.get("ts", t)) for r in recs)
    return Spend(total, len(recs), cfg.session_window_h,
                 oldest_expires_in_s=max(0.0, (oldest + window_s) - t))


# --------------------------------------------------------------------------- #
# Ceiling — configured, or learned by hitting the wall
# --------------------------------------------------------------------------- #
def _state(cfg: Config) -> dict:
    try:
        data = json.loads(cfg.budget_state.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def ceiling(cfg: Config) -> tuple[int | None, str]:
    """The window token ceiling and where it came from: ``(tokens, source)``.

    ``(None, "unknown")`` is a real and common answer — it means the server has not been
    told a budget, so it can estimate but cannot gate. Callers must render that as
    "unknown", never as "unlimited".

    A CONFIGURED NUMBER IS THE ONLY GATE, and an observed wall is deliberately not one.
    A limit hit tells us the account's ceiling C is at least our own window spend S
    (other Claude sessions spent the invisible remainder, so C = S + other >= S). Using S
    AS the ceiling puts the stop line at 0.95*S, below the very spend that produced it, so
    the next call — and every call until the window rolls — is refused. In a large batch a
    transient 429 that the retry layer then handles successfully is the normal case, so
    that inference turned one retried blip into a dead server. The observation is kept and
    reported by ``observed_wall`` as the number to put in ``session_budget_tokens``; it
    does not silently become that number.
    """
    if cfg.session_budget_tokens > 0:
        return cfg.session_budget_tokens, "configured"
    return None, "unknown"


def observed_wall(cfg: Config) -> tuple[int | None, float | None]:
    """The highest local window spend seen at a usage limit, and when — ``(tokens, ts)``.

    Advice for a human, not a gate: this is a FLOOR under the real ceiling (see
    ``ceiling``), which makes it a good starting value for ``session_budget_tokens`` and a
    bad stop line.
    """
    st = _state(cfg)
    tokens = st.get("learned_ceiling_tokens")
    if not isinstance(tokens, (int, float)) or tokens <= 0:
        return None, None
    ts = st.get("observed_at")
    return int(tokens), (float(ts) if isinstance(ts, (int, float)) else None)


def note_limit_hit(cfg: Config, *, now: Clock = time.time) -> None:
    """Record the window spend at the moment a usage limit was hit, as EVIDENCE.

    Read ``ceiling`` for why this is not itself a ceiling. What it is: a floor under the
    real one, and the only number about the account's true limit this server can ever
    measure -- so it is reported back as the value to configure.

    The HIGHEST observation wins. At the wall the account spent its ceiling C while this
    ledger saw only our own S, the rest going to other Claude sessions, so C >= S and a
    quieter window yields a tighter estimate. Keeping the lowest instead let one 429 in a
    busy window pin the number far under the truth, permanently.
    """
    try:
        s = spent(cfg, now=now)
        if s.tokens <= 0:
            return
        st = _state(cfg)
        prev = st.get("learned_ceiling_tokens")
        st["learned_ceiling_tokens"] = (
            max(int(prev), s.tokens) if isinstance(prev, (int, float)) and prev > 0 else s.tokens
        )
        st["observed_at"] = now()
        cfg.budget_state.parent.mkdir(parents=True, exist_ok=True)
        tmp = cfg.budget_state.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(st, indent=2), encoding="utf-8")
        os.replace(tmp, cfg.budget_state)
    except OSError:
        pass


# --------------------------------------------------------------------------- #
# Estimate
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Estimate:
    """A pre-flight forecast for a batch. Every field is derived from what we are about
    to SEND (chunk sizes are already known at chunk time), never from a prior response."""

    calls: int
    input_tokens: int
    output_tokens: int
    #: Wall-clock seconds, from the observed per-call latency and the configured fan-out.
    seconds: float
    chunks_total: int
    chunks_todo: int
    chunks_done: int
    #: The largest SINGLE call in the run, reduce pass included. This is what decides
    #: "possible at all": a run can always be split across windows, but one call cannot,
    #: so a call bigger than the stop line can never be sent -- in this window or any other.
    max_call_tokens: int = 0
    #: The largest single MAP call. Separate from the above because the two failures want
    #: opposite advice: an oversized map call means re-chunk smaller, while an oversized
    #: reduce call means drop the synthesis (reduce=False), which costs no chunk answers.
    max_map_call_tokens: int = 0
    #: The SMALLEST call still to be sent. What decides "is there room for even one" —
    #: comparing the largest against the headroom refused whole batches of small chunks
    #: because one big chunk did not fit, while claiming nothing fit at all.
    min_call_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


#: Fallback per-call latency when the ledger has no history to measure. The failing run
#: averaged ~52s/call on Haiku through the CLI; a cold guess of 45s is the right order.
_DEFAULT_CALL_S = 45.0
#: Ledger records needed before a MEASURED output-per-call is trusted over the cap.
_OUTPUT_SAMPLE_MIN = 8
#: Hard bound on how far a measured output may exceed the requested cap. Without it one
#: runaway completion would inflate every later reservation until the floor refused every
#: call -- the mistake ``ceiling`` documents for a learned ceiling, in another guise.
_MAX_OUTPUT_FACTOR = 8.0
#: Ceiling on the learned per-call input overhead (``input_overhead``). Measured at ~29k
#: for the `claude` CLI's baseline context; 128k leaves room for a much chattier harness
#: while still refusing to let one anomalous ledger record reserve a whole window.
_MAX_INPUT_OVERHEAD = 131_072


def estimate_batch(cfg: Config, chunk_tokens: list[int], *, prompt: str,
                   max_output_tokens: int, reduce: bool, reduce_output_tokens: int = 0,
                   done: set[int] | None = None, system: str = "",
                   now: Clock = time.time) -> Estimate:
    """Forecast a map(-reduce) batch over ``chunk_tokens`` (est_tokens per chunk).

    ``done`` holds the indices (into ``chunk_tokens``) already answered and persisted, so
    a resumed run forecasts only the work that remains — the entire point of forecasting a
    resumable run. It is a SET of indices, not a count: a run stopped by the budget gate
    leaves gaps rather than a clean prefix, because workers finish out of order.

    Output is priced at what a call can actually EMIT — the cap where the transport can
    enforce it, the measured mean where it cannot. See ``expected_output``: pricing at the
    requested cap on the CLI path under-forecast a real 33-chunk run by 1.9x.
    ``reduce_output_tokens`` is the reduce call's own cap, which is NOT the map cap — the
    caller passes both because the estimate is only an estimate of this run if every number
    in it is the number that will actually be sent. ``system`` is here for that same
    sentence: the map sends ``batch.MAP_SYSTEM`` on every call, so leaving it out
    under-counts input by its size times the chunk count.
    """
    skip = done or set()
    todo = [t for i, t in enumerate(chunk_tokens) if i not in skip]
    # prompt + system + "--- CHUNK i/n ---" frame + the harness context the transport adds
    # of its own: every per-call input that is not the chunk itself. Sent on each call, so
    # it is priced per call, not once. The `if system` is load-bearing: estimate_tokens
    # floors at 1 so that a non-empty string never prices at zero, which would otherwise
    # bill an ABSENT system prompt one token on every call.
    per_prompt_overhead = (estimate_tokens(prompt) + 32
                           + (estimate_tokens(system) if system else 0)
                           + input_overhead(cfg, now=now))
    out_per_call = expected_output(cfg, max_output_tokens, now=now)
    itok = sum(todo) + per_prompt_overhead * len(todo)
    otok = out_per_call * len(todo)
    calls = len(todo)
    per_call = per_prompt_overhead + out_per_call
    biggest_map = (max(todo) + per_call) if todo else 0
    #: Every call this run will make, so "largest" and "smallest" are both taken over the
    #: same list and neither can silently omit a call the other counts.
    call_sizes = [t + per_call for t in todo]
    if reduce and chunk_tokens:
        # The reduce pass reads every map answer back in and emits one more. Its INPUT is
        # therefore the whole map output — at 103 chunks that is one ~211K-token call
        # against ~32K for the largest map call, so leaving it out of `biggest` hid the
        # largest call in the run from the one check that asks "can this be sent at all".
        #
        # Keyed on `chunk_tokens`, NOT on the work remaining: the synthesis re-reads the
        # CACHED answers too, so a fully resumed run still pays for it. Keyed on `todo`,
        # this forecast a fully-cached re-run at zero calls and zero tokens while
        # `batch.run` went on issuing the call — an estimate of nothing, for a run that
        # spends.
        red_out = expected_output(cfg, reduce_output_tokens or max_output_tokens, now=now)
        answers_in = out_per_call * len(chunk_tokens)
        call_sizes.append(answers_in + per_prompt_overhead + red_out)
        itok += answers_in
        otok += red_out
        calls += 1
    conc = max(1, cfg.subquery_concurrency)
    return Estimate(calls=calls, input_tokens=itok, output_tokens=otok,
                    seconds=(calls / conc) * _observed_call_s(cfg, now),
                    chunks_total=len(chunk_tokens), chunks_todo=len(todo),
                    chunks_done=len(chunk_tokens) - len(todo),
                    max_call_tokens=max(call_sizes) if call_sizes else 0,
                    max_map_call_tokens=biggest_map,
                    min_call_tokens=min(call_sizes) if call_sizes else 0)


def _observed_call_s(cfg: Config, now: Clock = time.time) -> float:
    """Mean seconds per call, measured from the ledger when it has enough history."""
    recs = _read_lines(cfg.budget_ledger, now() - _PRUNE_AFTER_H * 3600)
    stamps = sorted(float(r.get("ts", 0)) for r in recs)
    if len(stamps) < 8:
        return _DEFAULT_CALL_S
    # Elapsed wall time over the run, scaled by the fan-out that produced it.
    span = stamps[-1] - stamps[0]
    per_call = span / max(1, len(stamps) - 1) * max(1, cfg.subquery_concurrency)
    return per_call if 1.0 <= per_call <= 600.0 else _DEFAULT_CALL_S


def expected_output(cfg: Config, max_tokens: int, *, now: Clock = time.time) -> int:
    """What one call can actually EMIT — ``max_tokens`` only where a transport enforces it.

    THE CAP IS A REQUEST, NOT A BOUND, ON THE OAUTH PATH. The `claude` CLI accepts no
    output-token flag, so ``CliTransport`` is handed ``max_tokens`` and has nowhere to put
    it; ``ApiTransport`` passes it to the SDK, where it is enforced. Measured on a real
    33-chunk batch at ``max_tokens=2048``: 328,453 output tokens, 4.9x the cap that both
    the forecast and the pre-call floor had reserved against, and 1.9x the total estimate.
    An estimate that reads "needs 5% of headroom" for a run that takes 10% is the exact
    failure this module exists to prevent, one level down.

    Taking the max of the cap and the measured mean needs no knowledge of which transport
    is in play: where the cap IS enforced the measurement cannot exceed it, so the cap wins
    by construction and the SDK path is unaffected.

    Bounded on both sides deliberately. Below ``_OUTPUT_SAMPLE_MIN`` records there is no
    measurement worth trusting, so a cold ledger returns the cap and the first run of a
    fresh install is forecast optimistically — the known gap, self-correcting after one
    batch. Above ``_MAX_OUTPUT_FACTOR`` the measurement is clamped, because a reservation
    derived from an unbounded observation is how a learned number stops being an estimate
    and starts refusing every call (see ``ceiling``).

    IT LAGS BY UP TO ``_PRUNE_AFTER_H``, AND THAT IS THE SAFE DIRECTION. The mean is taken
    over a 48-hour window, so a change that makes the sub-model *terser* — adding
    ``batch.MAP_SYSTEM`` cut real output several-fold — keeps being forecast at the old
    volume until the verbose records age out. `rlm_estimate` therefore over-states output
    for up to two days after such a change. Do not read that as the change having failed,
    and do not hand-seed the ledger to hurry it: measure the receipt (the run's own
    ``sub_batch`` record, or the ledger) rather than the forecast.
    """
    recs = _read_lines(cfg.budget_ledger, now() - _PRUNE_AFTER_H * 3600)
    if len(recs) < _OUTPUT_SAMPLE_MIN:
        return max_tokens
    mean = sum(int(r.get("otok", 0)) for r in recs) / len(recs)
    return int(min(max(float(max_tokens), mean), max_tokens * _MAX_OUTPUT_FACTOR))


def input_overhead(cfg: Config, *, now: Clock = time.time) -> int:
    """Input tokens a call costs BEYOND the content we hand it, measured from the ledger.

    THE SAME DEFECT AS ``expected_output``, WITH THE OPERANDS SWAPPED. A local estimate of
    what we send is not what the account is billed: measured on a 30k-token review chunk,
    ``_est_in`` said 30,232 and the call consumed 64,149. The difference is the `claude`
    CLI's own baseline context, which arrives before any of ours — a 22-character prompt
    still created 29,268 cache tokens. Unbudgeted, that is ~960k tokens across a 33-chunk
    batch, and it lands on the one number a floor exists to keep honest.

    Learned rather than hardcoded, and learned as a DIFFERENCE rather than a mean. Input
    per call swings with chunk size, so a mean of ``itok`` would forecast nothing; the
    overhead is the stable part. Each record carries both figures, so the per-call
    overhead is ``itok - est`` and its median is what this returns.

    The median, not the mean: a batch mixes one small tail chunk with several full ones,
    and cache creation vs cache read moves ``itok`` by a factor of its own between the
    first call of a run and the rest. One outlier should not move a floor.

    Bounded like ``expected_output`` and for the same reasons. Below
    ``_OUTPUT_SAMPLE_MIN`` records there is no measurement, so it returns 0 and the floor
    behaves exactly as it did before this existed — the cold-start gap, self-correcting
    after one batch. Negative differences clamp to 0: a transport that reports less than
    we estimated (or none at all, leaving ``itok`` as the estimate itself) has no overhead
    to add, and must never be able to *lower* a reservation. ``_MAX_INPUT_OVERHEAD``
    caps it, because a reservation derived from an unbounded observation stops being an
    estimate and starts refusing every call (see ``ceiling``).
    """
    recs = [r for r in _read_lines(cfg.budget_ledger, now() - _PRUNE_AFTER_H * 3600)
            if r.get("est")]
    if len(recs) < _OUTPUT_SAMPLE_MIN:
        return 0
    deltas = sorted(max(0, int(r.get("itok", 0)) - int(r.get("est", 0))) for r in recs)
    return min(deltas[len(deltas) // 2], _MAX_INPUT_OVERHEAD)


# --------------------------------------------------------------------------- #
# Verdict
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Verdict:
    fits: bool          # completes inside the headroom left in THIS window
    possible: bool      # completes at all, across however many windows
    windows_needed: float
    headroom_tokens: int | None   # None when the ceiling is unknown
    ceiling_tokens: int | None
    ceiling_source: str
    stop_pct: int
    spend: Spend
    estimate: Estimate
    #: Highest local spend seen at a usage limit, or None. Reported as the floor to
    #: configure from — never used as a ceiling. See ``ceiling``.
    observed_wall_tokens: int | None = None

    def as_dict(self) -> dict:
        d = asdict(self)
        d["estimate"]["total_tokens"] = self.estimate.total_tokens
        return d


def judge(cfg: Config, est: Estimate, *, now: Clock = time.time) -> Verdict:
    """Compare an estimate against the window's remaining headroom.

    ``possible`` is False in exactly one case: the largest single call exceeds the stop
    line. A run can be split across windows -- durable per-chunk answers make any SIZE
    completable -- but one call cannot be split, so a call bigger than the line can never
    be sent, now or later. Everything else is SCHEDULED, not refused: "this takes three
    sittings, and each one keeps what it earned" is a different message from "this can
    never run", and the first version of this function could only say the first.

    "Largest call" includes the reduce pass, whose input is every map answer at once. The
    caller is expected to read ``max_map_call_tokens`` to tell the two apart, because an
    oversized map call and an oversized synthesis call call for opposite remedies.
    """
    cap, source = ceiling(cfg)
    s = spent(cfg, now=now)
    stop_pct = round(cfg.budget_stop_fraction * 100)
    if cap is None:
        wall, _ts = observed_wall(cfg)
        return Verdict(fits=True, possible=True, windows_needed=0.0, headroom_tokens=None,
                       ceiling_tokens=None, ceiling_source=source, stop_pct=stop_pct,
                       spend=s, estimate=est, observed_wall_tokens=wall)
    usable = cap * cfg.budget_stop_fraction
    headroom = int(max(0.0, usable - s.tokens))
    possible = est.max_call_tokens <= usable
    return Verdict(
        fits=possible and est.total_tokens <= headroom,
        possible=possible,
        windows_needed=(est.total_tokens / usable) if usable > 0 else float("inf"),
        headroom_tokens=headroom, ceiling_tokens=cap, ceiling_source=source,
        stop_pct=stop_pct, spend=s, estimate=est,
    )


# --------------------------------------------------------------------------- #
# The hard floor: refuse a call that would cross the stop line
# --------------------------------------------------------------------------- #
class BudgetStopError(Exception):
    """Raised by the transport when one more call would cross the session stop line.

    Two duck-typed markers make the engine do the right thing without importing us:

    * ``is_fatal_subcall`` -- the engine's existing contract (rlm.utils.exceptions
      .FATAL_SUBCALL_ATTR): every batched fan-out stops on it instead of reissuing the
      identical doomed call once per prompt.
    * ``is_session_budget_stop`` -- new (STOPS_RUN_ATTR): the ROOT loop converts it into
      a clean limit that carries the best partial answer and a resumable checkpoint,
      rather than letting it escape as a traceback.

    This is the enforcement that belongs where the ledger already is. The batch's Gate
    is the polite early stop that defers remaining chunks; this is the floor under
    EVERY caller -- rlm_query's recursive fan-out included, which had none.
    """

    is_fatal_subcall = True
    is_session_budget_stop = True

    def __init__(self, spent: int, usable: int, next_call: int):
        self.spent, self.usable, self.next_call = spent, usable, next_call
        super().__init__(
            f"session budget stop: ~{spent:,} spent + ~{next_call:,} for this call would "
            f"cross the {usable:,}-token stop line; answers so far are on disk -- resume "
            f"once the window has rolled"
        )


def check_or_raise(cfg: Config, next_call_tokens: int, *, now: Clock = time.time) -> None:
    """Refuse ``next_call_tokens`` if it would cross the stop line. No ceiling -> never.

    Read-and-compare, not reserve: with N concurrent workers the overshoot is bounded
    by N calls (~3 x one call against a 5% margin of ~290k tokens at the configured
    ceiling), which is the price of not serialising every completion through a lock.
    """
    cap, _src = ceiling(cfg)
    if cap is None:
        return
    usable = int(cap * cfg.budget_stop_fraction)
    s = spent(cfg, now=now)
    if s.tokens + next_call_tokens > usable:
        raise BudgetStopError(s.tokens, usable, next_call_tokens)


# --------------------------------------------------------------------------- #
# Mid-run gate
# --------------------------------------------------------------------------- #
class Gate:
    """Stops a running batch BEFORE the wall instead of being killed at it.

    Each worker asks ``allow(next_call_tokens)`` before dispatching. Once the projected
    spend crosses ``budget_stop_fraction`` of the ceiling the gate latches shut, so the
    remaining chunks are reported as deferred rather than each failing at the vendor.
    Stopping at 95% deliberately forfeits the last 5%: that sliver buys a couple more
    chunks, while being killed mid-call costs the run its exit path — which is exactly
    how the original failure produced a 30-minute spend and no answer.

    A gate with no ceiling to enforce (the common ``unknown`` case) never closes. It
    still tracks projected spend, so the caller can report what a run actually cost.
    """

    def __init__(self, cfg: Config, *, now: Clock = time.time):
        self.cfg = cfg
        self._cap, self._source = ceiling(cfg)
        self._lock = threading.Lock()
        self._projected = spent(cfg, now=now).tokens
        self._start = self._projected
        self.closed = False
        self.stopped_at: int | None = None

    @property
    def limit(self) -> float | None:
        return None if self._cap is None else self._cap * self.cfg.budget_stop_fraction

    def allow(self, next_call_tokens: int) -> bool:
        """Reserve budget for one call. False means the batch must stop here."""
        with self._lock:
            if self.closed:
                return False
            lim = self.limit
            if lim is not None and self._projected + next_call_tokens > lim:
                self.closed = True
                self.stopped_at = self._projected
                return False
            self._projected += next_call_tokens
            return True

    @property
    def spent_here(self) -> int:
        """Tokens this gate has reserved since it was created."""
        return self._projected - self._start


# --------------------------------------------------------------------------- #
# rlm_query -- a ceiling, not an estimate
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class QueryCeiling:
    """What rlm_query CAN cost, from config alone. Not a forecast: the root model decides
    at run time how many sub-calls to make, so no pre-flight can predict the count. What
    can be stated is the worst case the limits permit, and the tighter bound the timeout
    imposes in practice."""

    root_calls_max: int
    sub_calls_max: int
    worst_tokens: int         # every permitted call at its maximum size
    timeout_s: int
    timeout_calls: int        # how many calls fit in the timeout at the observed latency
    timeout_tokens: int       # those calls at a typical chunk-sized prompt
    per_call_s: float


def query_ceiling(cfg: Config, *, now: Clock = time.time) -> QueryCeiling:
    """Bound rlm_query from config. Two numbers, both honest about what they are:

    * worst_tokens -- max_iterations root turns, each allowed max_concurrent_subcalls
      sub-calls at the sub-model's full window. This is the limit the config permits, and
      it is normally far past a whole session window.
    * timeout_tokens -- what query_timeout_s actually allows at the observed per-call
      latency. In practice THIS is the bound: rlm_query is stopped by the clock, not by
      tokens, and that is the fact a caller needs before starting one.

    Root-turn INPUT is omitted from both: it is the REPL transcript, which grows per turn
    in a way config cannot bound. Root output is included at max_output_tokens.
    """
    subs = cfg.max_iterations * max(1, cfg.max_concurrent_subcalls)
    sub_out = 2048
    worst = subs * (cfg.sub_context_tokens + sub_out) + cfg.max_iterations * cfg.max_output_tokens
    per_call = _observed_call_s(cfg, now)
    timeout_calls = int(cfg.query_timeout_s / per_call) if cfg.query_timeout_s > 0 else 0
    typical_call = estimate_tokens(cfg.chunk_chars) + sub_out
    return QueryCeiling(
        root_calls_max=cfg.max_iterations, sub_calls_max=subs, worst_tokens=worst,
        timeout_s=cfg.query_timeout_s, timeout_calls=timeout_calls,
        timeout_tokens=timeout_calls * typical_call, per_call_s=per_call,
    )


def render_query_ceiling(q: QueryCeiling, cap: int | None) -> str:
    per_turn = q.sub_calls_max // max(1, q.root_calls_max)
    lines = [
        "## rlm_query on this context -- a ceiling, not an estimate",
        "",
        "The root model decides at run time how many sub-calls to make, so the count cannot "
        "be forecast. What config permits, and what the timeout allows in practice:",
        "",
        f"- permitted: up to **{q.root_calls_max}** root turns x **{per_turn}** sub-calls = "
        f"**{q.sub_calls_max}** sub-calls; worst case **~{q.worst_tokens:,} tokens**"
        + (f" ({q.worst_tokens / cap:.1f}x the whole window)" if cap else ""),
        f"- in practice: `query_timeout_s={q.timeout_s}` at ~{q.per_call_s:.0f}s/call allows "
        f"**~{q.timeout_calls} calls, ~{q.timeout_tokens:,} tokens** before the engine is stopped",
        "",
        "_rlm_query cannot be forecast, but it is bounded on both sides: every model call "
        "passes the session-budget floor, so it stops itself before the wall, and any stop "
        "(budget, timeout, error threshold) checkpoints the transcript and REPL state -- "
        "call it again with the same ctx_id and question to resume. Spend is ledgered, so "
        "the next estimate sees it. Prefer rlm_exec / rlm_grep (free) and "
        "rlm_sub_query_batch (estimable) unless the question truly needs multi-hop "
        "reasoning._",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def fmt_dur(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.1f}h"


def render(v: Verdict, *, what: str) -> str:
    """Human-readable verdict block. Shared by rlm_estimate and the batch pre-flight so
    the two can never describe the same run differently."""
    e = v.estimate
    resume = (f" ({e.chunks_done} of {e.chunks_total} already done — resuming)"
              if e.chunks_done else "")
    lines = [
        f"## Estimate — {what}{resume}",
        "",
        f"- chunks to run: **{e.chunks_todo}**" + (f" of {e.chunks_total}" if e.chunks_done else ""),
        f"- model calls: **{e.calls}**",
        f"- tokens: **~{e.total_tokens:,}** ({e.input_tokens:,} in / {e.output_tokens:,} out)",
        f"- wall time: **~{fmt_dur(e.seconds)}**",
        "",
    ]
    if v.ceiling_tokens is None:
        lines += [
            "**Window budget: unknown.** No `session_budget_tokens` is configured, so this "
            "run cannot be gated against a ceiling — only measured. Set "
            "`session_budget_tokens` in config.yaml to gate it.",
            "",
            f"Spent in the last {v.spend.window_h:g}h by this server: "
            f"~{v.spend.tokens:,} tokens over {v.spend.calls} calls.",
        ]
        if v.observed_wall_tokens:
            lines += [
                "",
                f"_This server has hit a usage limit with ~{v.observed_wall_tokens:,} "
                "tokens of its own spend inside the window. The account's real ceiling is "
                "at least that (other Claude sessions spend against it too and are "
                "invisible here), so that is the floor to start from when you set "
                "`session_budget_tokens`._",
            ]
        return "\n".join(lines)

    pct = 100.0 * e.total_tokens / max(1, v.headroom_tokens or 1)
    lines += [
        f"**Window budget** ({v.ceiling_source}): ~{v.ceiling_tokens:,} tokens per "
        f"{v.spend.window_h:g}h.",
        f"- spent so far this window: ~{v.spend.tokens:,} ({v.spend.calls} calls)",
        f"- headroom to the {v.stop_pct}% stop line: ~{v.headroom_tokens:,}",
        "",
    ]
    if not v.possible:
        usable = int(v.ceiling_tokens * (v.stop_pct / 100))
        lines += [
            f"**Impossible in ANY window.** The largest single chunk needs "
            f"~{e.max_call_tokens:,} tokens in one call, and the {v.stop_pct}% stop line is "
            f"~{usable:,}. Waiting does not help: no window is ever that large.",
            "",
            "Re-chunk smaller (`rlm_chunk_context(ctx_id, strategy='lines', size=<fewer "
            "lines>)`), or raise `session_budget_tokens` if the ceiling is set too low.",
        ]
        return "\n".join(lines)
    if v.fits:
        lines.append(f"**Fits** — this run needs ~{pct:.0f}% of the remaining headroom.")
    else:
        lines += [
            f"**Does not fit this window** — needs ~{v.windows_needed:.1f} full windows "
            f"({v.spend.window_h:g}h each).",
            "",
            "This is not a refusal: per-chunk results are durable, so run it, let it stop "
            "at the budget line, and call the same tool again next window — it resumes at "
            "the first unanswered chunk and never re-pays for a chunk it already has.",
        ]
    if v.spend.oldest_expires_in_s:
        lines.append(f"\n_Headroom next grows in ~{fmt_dur(v.spend.oldest_expires_in_s)} "
                     "as the oldest calls age out of the rolling window._")
    lines.append("\n_Counts this server's own spend only — other Claude sessions on the "
                 "same account are invisible here, so treat headroom as an upper bound._")
    return "\n".join(lines)

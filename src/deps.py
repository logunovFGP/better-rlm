"""The injectable dependency bundle: config, context store, logger.

WHY THIS EXISTS. Every module used to resolve its own config at import
(``server.CFG``, ``subquery._CFG``, ``ratelimit._CFG``), so a test could only reach them
by patching module globals — and that patching was conditional on import order, which is
how a resume test came to pass by reading its own leftovers out of the operator's real
``~/.rlm``. A test that passes because of pollution is worse than one that fails, and the
difficulty of writing it honestly was the design signal.

So the logic takes its collaborators as an argument. ``Deps`` is constructed once at the
composition root — ``server.py``, at import, from the real config — and passed down.
A test constructs its own over ``tmp_path`` and needs no monkeypatching at all.

WHAT DELIBERATELY STAYS GLOBAL. The rate throttle in ``ratelimit`` is process-wide by
design: its job is "at most N calls in flight across this process", which a per-Deps
instance would silently break by handing every caller its own quota. Global is the
correct answer there, not an oversight — see that module's own note.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from . import results
from .config import Clock, Config, cost_usd, load_config
from .context_store import ContextStore
from .logsetup import LOGGER_NAME, configure_logging
from .output import bound_output


@dataclass(frozen=True)
class Deps:
    """What a tool implementation needs from its environment.

    Frozen: a tool must not be able to swap the store or the config out from under a
    concurrent call. Build a variant with ``dataclasses.replace`` instead.
    """

    cfg: Config
    store: ContextStore
    log: logging.Logger
    #: Wall-clock source, threaded into everything that reads a clock POSITION (the spend
    #: ledger's window arithmetic, a persisted answer's timestamp). Defaulted so ordinary
    #: construction is unchanged; a test replaces it to make window boundaries and
    #: measured latencies exact instead of approximately-right-by-wide-margin.
    clock: Clock = field(default=time.time)

    @classmethod
    def create(cls, cfg: Config | None = None) -> "Deps":
        """Build the process's real dependencies. Called once, by the composition root.

        This is the ONLY place that both loads config and installs log handlers, so a
        test constructing its own Deps cannot reconfigure the operator's logging as a
        side effect of wanting a temp store.
        """
        c = cfg or load_config()
        # The answer cache's LRU sweep belongs with the other startup housekeeping (log
        # retention runs inside configure_logging). Cooldown-gated, so the pool of spare
        # daemons does not each walk the cache directory on start.
        results.sweep(c)
        return cls(cfg=c, store=ContextStore(c), log=configure_logging(c))

    @classmethod
    def for_test(cls, cfg: Config, *, clock: Clock = time.time) -> "Deps":
        """Dependencies over an already-redirected config, installing NO log handlers.

        Named here rather than left to each test to assemble, so "a test's Deps never
        touches the real logger" is a property of one line instead of a rule every test
        file has to remember.
        """
        return cls(cfg=cfg, store=ContextStore(cfg),
                   log=logging.getLogger(LOGGER_NAME), clock=clock)

    # -- output bounding ---------------------------------------------------
    def bound(self, text: str) -> str:
        """Raw-content cap (load/inspect/chunk/grep/read/list/exec/status): kept tight so
        file content can never flood the root context."""
        return bound_output(text, self.cfg.output_cap_bytes)

    def answer(self, text: str) -> str:
        """Synthesis cap (query/sub_query[_batch]): the answer IS the deliverable, so it
        is bounded generously rather than at the raw-content cap."""
        return bound_output(text, self.cfg.answer_cap_bytes)

    def cost_note(self, model: str, itok: int, otok: int) -> str:
        """``  |  cost: $x.xxxx`` when report_cost is on, else nothing. Off by default:
        the rate table is Anthropic-only and the CLI path under-counts input tokens, so a
        printed figure would be confidently wrong."""
        if not self.cfg.report_cost:
            return ""
        return f"  |  cost: ${cost_usd(model, itok, otok):.4f}"

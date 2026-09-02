import dataclasses
import logging
from pathlib import Path

import pytest

import src.config as config_mod
from src.deps import Deps


@pytest.fixture
def cfg(tmp_path: Path):
    """Config with every on-disk path redirected under this test's tmp_path.

    All four paths, not just the store: logs, the persisted chunk answers, the spend
    ledger and the learned-ceiling state. Each was written into the operator's real
    ``~/.rlm`` at some point during development, and the ledger was the worst of them —
    fake spend recorded there skews the budget gate of a REAL run afterwards.
    """
    return dataclasses.replace(
        config_mod.load_config(),
        store_dir=tmp_path / "contexts",
        log_dir=tmp_path / "logs",
        budget_ledger=tmp_path / "usage.jsonl",
        budget_state=tmp_path / "budget.json",
    )


class FrozenClock:
    """A wall clock the test moves by hand.

    Callable, so it drops straight into any ``now: Clock`` parameter. Adopted from the
    clinemm suite's fake-timer discipline for a specific reason: three pieces of pure
    clock arithmetic had NO coverage at all, because you cannot test them against a clock
    that moves underneath you.

      * ``budget._observed_call_s`` derives per-call latency from the span between ledger
        timestamps — untestable without control of those timestamps.
      * ``Spend.oldest_expires_in_s`` answers "when does headroom next grow", which is
        arithmetic on now.
      * the rolling-window boundary itself: a record at exactly ``now - window`` is either
        in or out, and asserting which requires writing and reading at one fixed instant.

    The existing window test only worked because its margins were hours wide, which is
    the tell that the boundary was never actually pinned.

    Absolute epoch, not 0: several code paths format or compare timestamps, and a
    1970 clock hides sign errors that a plausible date would surface.
    """

    #: 2026-09-02T12:00:00Z — a fixed, plausible instant.
    START = 1_788_350_400.0

    def __init__(self, start: float = START):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def tick(self, seconds: float) -> "FrozenClock":
        """Advance the clock. Returns self so a test can chain or inline it."""
        self.now += seconds
        return self


@pytest.fixture
def clock():
    """A FrozenClock at a fixed instant. Pair with `deps` to freeze a whole run."""
    return FrozenClock()


@pytest.fixture
def deps(cfg, clock):
    """Injected dependencies over a temp config — the reason there is no longer a fixture
    that reaches into module globals.

    What used to be here: a fixture that monkeypatched ``server.CFG``, ``subquery._CFG``
    and ``ratelimit._CFG``, and could only patch modules already in ``sys.modules``. That
    made the redirect conditional on import order, so it silently skipped for tests whose
    first server import happened later — and a resume test then passed by reading its own
    leftovers out of the real store. The logic now takes ``Deps`` as an argument, so a
    test hands it a temp one and nothing global is touched at all.
    """
    return Deps.for_test(cfg, clock=clock)


@pytest.fixture(autouse=True)
def _no_test_starts_a_real_container(monkeypatch, request):
    """The suite must pass with no Docker daemon running — CI has none, and neither does
    a contributor who has not started Docker Desktop.

    DockerREPL.__init__ calls setup(), which runs `docker run`. A test that constructs one
    without stubbing that took the whole suite down with "failed to connect to the docker
    API" rather than failing by itself. This makes the requirement loud instead: the stub
    raises, so such a test fails alone, pointing at what to do. A test that genuinely wants
    a container marks itself @pytest.mark.docker; a test that only needs the object stubs
    setup itself, and its own monkeypatch replaces this one.
    """
    if "docker" in request.keywords:
        return
    from rlm.environments.docker_repl import DockerREPL

    def refuse(self):
        raise AssertionError(
            "this test constructs a real DockerREPL, whose __init__ starts a container. "
            "Stub setup (monkeypatch.setattr(dr.DockerREPL, 'setup', lambda self: None)) "
            "or mark the test @pytest.mark.docker if it truly needs a daemon."
        )

    monkeypatch.setattr(DockerREPL, "setup", refuse)


@pytest.fixture(autouse=True)
def _no_test_spends_a_real_model_call(monkeypatch, request):
    """No test may reach the live transport. Same shape as the Docker guard above, and
    for a sharper reason: a real call costs the operator's session-window budget.

    This caught one. ``test_probe_reports_ok_on_a_live_transport`` patched sub_query on
    the wrong module — ``batch``, while ``_auth_probe_line`` calls the one imported into
    ``server`` — so every suite run made a live Haiku call and the test PASSED because
    the real transport genuinely replied "ok". Green for the wrong reason, and billed.
    Cheap to detect (one stub) and invisible otherwise (the ledger only grew by ~80
    bytes a run), which is exactly the combination that earns a permanent guard.

    A test that truly wants the network marks itself @pytest.mark.live.
    """
    if "live" in request.keywords:
        return
    import src.subquery as sq

    def refuse(*_a, **_k):
        raise AssertionError(
            "this test reached the real completion transport, which spends the operator's "
            "session budget. Stub the sub_query the code under test actually calls (check "
            "WHICH module imported it), or mark the test @pytest.mark.live."
        )

    monkeypatch.setattr(sq, "_call", refuse)


@pytest.fixture(autouse=True)
def _no_test_writes_to_the_real_log_dir(monkeypatch):
    """Keep pytest out of the operator's ~/.rlm/logs.

    Still needed after the Deps refactor, because importing src.server runs the
    composition root (``Deps.create()``), which installs a RotatingFileHandler on the
    process-global "rlm-mcp" logger against the REAL config. Any test that drives a
    @logged_tool-wrapped tool then logs into the live dir: measured 18 of 20 files there
    as pytest debris, which also supplied 25 of the 26 outcome=error records, and one run
    evicted the real session logs under a 20-file retention cap.

    Stripping is not sufficient on its own, and that gap cost three stray log files: most
    test modules ``import src.server`` INSIDE the test body, so the composition root runs
    AFTER this fixture and installs a fresh handler mid-test. (Same import-order shape as
    the config-global bug that motivated Deps — the lesson is that a fixture which only
    cleans up before the test loses to anything the test imports.) So neutralize the
    cause as well: ``deps.configure_logging`` is the name ``Deps.create`` resolves, and
    patching it there leaves ``logsetup``'s own tests free to exercise the real thing.
    """
    import src.deps as deps_mod

    logger = logging.getLogger("rlm-mcp")
    for h in [h for h in logger.handlers if isinstance(h, logging.FileHandler)]:
        logger.removeHandler(h)
        h.close()
    monkeypatch.setattr(deps_mod, "configure_logging", lambda cfg: logger)


@pytest.fixture
def batch_ctx(deps, monkeypatch):
    """Factory for a stub context wired into a batch Deps, with captured log events.

    Lesson taken from the clinemm suite (mock factories with captured spies; fixture
    builders with partial overrides) after this repo paid for its absence: the same
    ``_Meta``/``_Store`` pair was hand-copied into eight tests, so adding one field the
    production path had always read — ``chunk_strategy`` — meant editing eight stubs, and
    missing one failed as a confusing AttributeError inside a tool rather than as a
    missing-stub error. One factory makes that class of drift impossible.

    Returns ``(deps, events)``: a Deps whose store is the stub, and the list every
    log_event call lands in. Tests that do not care about the log ignore the second
    element — capturing unconditionally costs nothing and means a test can start
    asserting on the log without rewiring its setup.

    Note what is NOT patched any more: no module-level configuration. The stub store is
    swapped by building a new frozen Deps, and ``batch.log_event`` is patched because it
    is a function in the module under test — a different thing from reaching into another
    module's config.
    """
    def _make(n_chunks: int = 3, *, est_tokens: int = 1000, strategy: str = "lines",
              on_read=None):
        import src.batch as batch_mod

        class _Meta:
            chunks = [{"i": i, "est_tokens": est_tokens} for i in range(n_chunks)]
            chunk_strategy = strategy

        class _Store:
            def get(self, ctx_id):
                return _Meta()

            def read_chunk(self, ctx_id, i):
                if on_read is not None:
                    on_read(i)   # lets a test prove WHEN a chunk was read, not just that it was
                return f"chunk{i}"

            def read_text(self, ctx_id):
                return "".join(f"chunk{i}\n" for i in range(n_chunks))

        events: list[tuple[str, dict]] = []
        monkeypatch.setattr(batch_mod, "log_event",
                            lambda log, evt, **f: events.append((evt, f)))
        # models.select resolves the auth mode, which raises on a machine with no CLI and
        # no key — every CI runner. Pin it, as the suite's other fixtures do.
        monkeypatch.setattr(batch_mod.models, "select", lambda cfg, role: "claude-haiku-4-5")
        monkeypatch.setattr(batch_mod.transport, "auth_label", lambda cfg: "test")
        return dataclasses.replace(deps, store=_Store()), events

    return _make

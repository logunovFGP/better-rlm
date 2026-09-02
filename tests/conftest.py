import dataclasses
import logging
import sys
from pathlib import Path

import pytest

import src.config as config_mod
# Imported HERE, at conftest module scope, and not left to whichever test imports it
# first: the redirect fixture below can only patch a module that is already in
# sys.modules, and src.server is imported inside test function bodies. That made the
# redirect conditional on import order — it silently skipped, tests wrote chunk answers
# into the operator's real ~/.rlm/contexts, and a resume test then PASSED by reading its
# own leftovers from a previous run. A test that passes because of pollution is worse
# than one that fails.
import src.server  # noqa: F401 - imported for its side effect on sys.modules



@pytest.fixture
def cfg(tmp_path: Path):
    """Config with store/log dirs redirected to a tmp path."""
    base = config_mod.load_config()
    return dataclasses.replace(base, store_dir=tmp_path / "contexts", log_dir=tmp_path / "logs")


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
def _no_test_writes_to_the_real_rlm_dir(tmp_path, monkeypatch):
    """Redirect every on-disk path the server writes to into this test's tmp_path.

    Modules resolve their config ONCE at import (``server.CFG``, ``subquery._CFG``,
    ``ratelimit._CFG``), so a test that exercises a tool writes wherever the operator's
    real config points — ``~/.rlm/contexts`` for persisted chunk answers and
    ``~/.rlm/usage.jsonl`` for the spend ledger. The log dir already showed what that
    costs: 18 of 20 files in the live log dir were pytest debris. The ledger would be
    worse, because fake spend recorded there then skews the budget gate of a REAL run.

    Redirected at the module globals rather than by passing a tmp config to each test:
    these paths are read through the globals from inside worker threads and result
    callbacks, where no fixture argument reaches.
    """
    tmp_cfg = dataclasses.replace(
        config_mod.load_config(),
        store_dir=tmp_path / "contexts",
        log_dir=tmp_path / "logs",
        budget_ledger=tmp_path / "usage.jsonl",
        budget_state=tmp_path / "budget.json",
    )
    for mod_name, attr in (("src.server", "CFG"), ("src.subquery", "_CFG"),
                           ("src.ratelimit", "_CFG")):
        mod = sys.modules.get(mod_name)
        assert mod is not None, (
            f"{mod_name} is not imported, so its paths were NOT redirected and this test "
            "would write into the operator's real ~/.rlm. Import it at conftest scope."
        )
        monkeypatch.setattr(mod, attr, tmp_cfg, raising=False)


@pytest.fixture(autouse=True)
def _no_test_writes_to_the_real_log_dir():
    """Keep pytest out of the operator's ~/.rlm/logs.

    src/server.py installs a RotatingFileHandler on the process-global "rlm-mcp"
    logger at IMPORT time (``LOG = configure_logging(CFG)``) against the REAL config,
    so every test that imports it logged into the user's live log dir. Measured: 18 of
    20 files there were pytest debris (ctx_x, root-m, the mock "OAuth session expired"),
    which also supplied 25 of the 26 outcome=error records — and since one run creates
    ~15 files against a retention cap of 20, a single `pytest` evicted the real session
    logs it was supposed to sit beside.

    Stripping the handler is enough because it is opened with ``delay=True``: no record
    emitted through it means no file is ever created. Function-scoped so it also undoes
    a re-install by an earlier test; a test that wants a file handler adds its own
    inside the test body (test_logsetup.py), which runs after this.
    """
    logger = logging.getLogger("rlm-mcp")
    for h in [h for h in logger.handlers if isinstance(h, logging.FileHandler)]:
        logger.removeHandler(h)
        h.close()


@pytest.fixture
def batch_ctx(monkeypatch):
    """Factory for a stub context wired into the batch tools, with captured log events.

    Lesson taken from the clinemm suite (mock factories with captured spies; fixture
    builders with partial overrides) after this repo paid for its absence: the same
    ``_Meta``/``_Store`` pair was hand-copied into eight tests, so adding one field the
    production path had always read — ``chunk_strategy`` — meant editing eight stubs, and
    missing one would have failed as a confusing AttributeError inside a tool rather than
    as a missing-stub error. One factory makes that class of drift impossible.

    Returns ``(srv, events)``: the server module with STORE/models/transport/log_event
    stubbed, and the list every log_event call lands in. Tests that do not care about the
    log simply ignore the second element — capturing unconditionally costs nothing and
    means a test can start asserting on it without rewiring its setup.
    """
    def _make(n_chunks: int = 3, *, est_tokens: int = 1000, strategy: str = "lines",
              on_read=None):
        import src.server as srv

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
        monkeypatch.setattr(srv, "STORE", _Store())
        monkeypatch.setattr(srv, "log_event", lambda log, evt, **f: events.append((evt, f)))
        # models.select resolves the auth mode, which raises on a machine with no CLI and
        # no key — every CI runner. Pin it, as the suite's other fixtures do.
        monkeypatch.setattr(srv.models, "select", lambda cfg, role: "claude-haiku-4-5")
        monkeypatch.setattr(srv.transport, "auth_label", lambda cfg: "test")
        return srv, events

    return _make

import dataclasses
import inspect
import logging
import logging.handlers
import os
import sys
from pathlib import Path

from src.config import load_config
import src.logsetup as ls


def _cfg(tmp_path, **over):
    base = load_config()
    d = {"log_dir": tmp_path, "store_dir": tmp_path / "store"}
    d.update(over)
    return dataclasses.replace(base, **d)


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__(logging.INFO)
        self.msgs: list[str] = []

    def emit(self, record):
        self.msgs.append(record.getMessage())


def _capture(logger_name):
    lg = logging.getLogger(logger_name)
    cap = _Capture()
    lg.addHandler(cap)
    lg.setLevel(logging.INFO)
    return lg, cap


# --------------------------- log_event / formatting --------------------------- #
def test_log_event_is_logfmt_and_drops_none():
    lg, cap = _capture("rlm-test-fmt")
    ls.log_event(lg, "tool_call", rid="abc", tool="rlm_query", dur_ms=12, missing=None)
    lg.removeHandler(cap)
    msg = cap.msgs[-1]
    assert msg.startswith("evt=tool_call ")
    assert "rid=abc" in msg and "tool=rlm_query" in msg and "dur_ms=12" in msg
    assert "missing" not in msg  # None-valued fields dropped


def test_fmt_val_quotes_spaces_and_clamps():
    assert ls._fmt_val("plain") == "plain"
    assert ls._fmt_val("has space") == '"has space"'
    assert ls._fmt_val("a\nb") == '"a b"'  # newline neutralized
    out = ls._fmt_val("x" * 1000)
    assert len(out) <= ls._MAX_VAL_LEN + 2  # clamped (+ ellipsis)


# --------------------------- configure_logging --------------------------- #
def test_configure_logging_no_stdout_and_defers_the_file(tmp_path):
    logger = ls.configure_logging(_cfg(tmp_path))
    try:
        for h in logger.handlers:
            assert getattr(h, "stream", None) is not sys.stdout  # never the JSON-RPC channel
        assert any(isinstance(h, logging.handlers.RotatingFileHandler) for h in logger.handlers)
        # delay=True: an idle process must not leave a file behind at all.
        assert not list(tmp_path.glob("rlm-mcp-*.log"))
        ls.log_event(logger, "probe")
        assert list(tmp_path.glob("rlm-mcp-*.log"))  # created on the first record
    finally:
        for h in list(logger.handlers):
            logger.removeHandler(h)
            h.close()


# --------------------------- retention sweep (the disk bound) --------------------------- #
def _make_logs(tmp_path, n, size=100, base_mtime=1000):
    for i in range(n):
        p = tmp_path / f"rlm-mcp-2026-{i:04d}.log"
        p.write_text("x" * size)
        os.utime(p, (base_mtime + i * 10, base_mtime + i * 10))


def test_sweep_enforces_file_count_cap(tmp_path):
    cfg = _cfg(tmp_path, log_retention_files=3, log_retention_total_bytes=10_000_000,
               log_retention_days=3650, log_sweep_cooldown_s=0)
    _make_logs(tmp_path, 8)
    ls._run_retention_sweep(cfg, tmp_path / "rlm-mcp-own.log")
    assert len(list(tmp_path.glob("rlm-mcp-*.log"))) <= 3


def test_sweep_enforces_total_bytes_cap(tmp_path):
    cfg = _cfg(tmp_path, log_retention_files=100, log_retention_total_bytes=250,
               log_retention_days=3650, log_sweep_cooldown_s=0)
    _make_logs(tmp_path, 8, size=100)
    ls._run_retention_sweep(cfg, tmp_path / "rlm-mcp-own.log")
    total = sum(p.stat().st_size for p in tmp_path.glob("rlm-mcp-*.log"))
    assert total <= 300  # keeps only files fitting under the ~250B cap


def test_sweep_never_deletes_own_file(tmp_path):
    cfg = _cfg(tmp_path, log_retention_files=0, log_retention_total_bytes=1,
               log_retention_days=0, log_sweep_cooldown_s=0)
    own = tmp_path / "rlm-mcp-own.log"
    own.write_text("keep")
    old = tmp_path / "rlm-mcp-old.log"
    old.write_text("drop")
    os.utime(old, (1, 1))
    ls._run_retention_sweep(cfg, own)
    assert own.exists()  # own file survives even the most aggressive caps


def test_sweep_respects_cooldown(tmp_path):
    cfg = _cfg(tmp_path, log_retention_files=0, log_retention_total_bytes=1,
               log_retention_days=0, log_sweep_cooldown_s=9999)
    (tmp_path / ".sweep").write_text("now")  # fresh sentinel -> within cooldown
    old = tmp_path / "rlm-mcp-old.log"
    old.write_text("drop")
    os.utime(old, (1, 1))
    ls._run_retention_sweep(cfg, tmp_path / "own.log")
    assert old.exists()  # cooldown skipped the sweep, nothing deleted


# --------------------------- logged_tool decorator --------------------------- #
def test_logged_tool_preserves_signature():
    @ls.logged_tool
    def sample(ctx_id: str, question: str, model_override: str = "") -> str:
        return "## ok"
    # functools.wraps => FastMCP still sees the real signature for the tool schema
    assert list(inspect.signature(sample).parameters) == ["ctx_id", "question", "model_override"]


def test_logged_tool_logs_ok_with_lengths_not_content():
    lg, cap = _capture(ls.LOGGER_NAME)

    @ls.logged_tool
    def sample(ctx_id: str, question: str) -> str:
        return "## answer body"

    out = sample(ctx_id="ctx_1", question="hello?")
    lg.removeHandler(cap)
    assert out == "## answer body"
    joined = " ".join(cap.msgs)
    assert "evt=tool_call" in joined and "tool=sample" in joined and "outcome=ok" in joined
    assert "question_len=6" in joined and "hello?" not in joined  # length, never content


def test_startup_is_held_until_a_tool_actually_runs():
    lg, cap = _capture(ls.LOGGER_NAME)
    ls.note_startup(mode="auto", transport="oauth")
    assert ls.startup_pending()          # an idle spare logs nothing at all
    assert not cap.msgs

    @ls.logged_tool
    def sample(ctx_id: str) -> str:
        return "## ok"

    sample(ctx_id="c")
    lg.removeHandler(cap)
    assert not ls.startup_pending()      # flushed exactly once, before the call
    joined = " ".join(cap.msgs)
    assert "evt=startup" in joined and "transport=oauth" in joined
    assert joined.index("evt=startup") < joined.index("evt=tool_call")


def test_logged_tool_converts_a_raise_into_an_error_string():
    lg, cap = _capture(ls.LOGGER_NAME)

    @ls.logged_tool
    def exploding(ctx_id: str) -> str:
        raise RuntimeError("boom")

    out = exploding(ctx_id="c")
    lg.removeHandler(cap)
    # The tool contract is a string starting with ERROR — never a propagated raise.
    assert out.startswith("ERROR in exploding: boom")
    joined = " ".join(cap.msgs)
    assert "outcome=error" in joined and "RuntimeError" in joined


def test_logged_tool_flags_error_returns():
    lg, cap = _capture(ls.LOGGER_NAME)

    @ls.logged_tool
    def failing(ctx_id: str) -> str:
        return "ERROR in failing: boom"

    failing(ctx_id="c")
    lg.removeHandler(cap)
    assert "outcome=error" in " ".join(cap.msgs)


def test_logged_tool_logs_zero_valued_first_chunk_and_reduce_false():
    """chunk_index=0 is the FIRST chunk and reduce=False is a real choice: both must
    reach the log. A blanket falsy-drop hid them (and False == 0 hid reduce twice over),
    making a supplied value indistinguishable from an absent one. max_chunks=0 IS the
    documented no-limit default and stays dropped."""
    lg, cap = _capture(ls.LOGGER_NAME)

    @ls.logged_tool
    def sample(ctx_id: str, chunk_index: int, reduce: bool, max_chunks: int) -> str:
        return "## ok"

    sample(ctx_id="ctx_1", chunk_index=0, reduce=False, max_chunks=0)
    lg.removeHandler(cap)
    joined = " ".join(cap.msgs)
    assert "chunk_index=0" in joined
    assert "reduce=False" in joined
    assert "max_chunks=" not in joined


def test_sweep_leaves_no_tmp_when_sentinel_replace_fails(tmp_path):
    """A failed os.replace must not orphan .sweep.<pid>.tmp — the prune globs
    rlm-mcp-*.log*, so nothing else ever collects one.

    A directory where the sentinel file belongs fails the replace for real on both
    POSIX and Windows — no process-wide patch of os.replace, and it goes through the
    same OSError path the live orphan came in through.
    """
    cfg = _cfg(tmp_path, log_sweep_cooldown_s=0)
    (tmp_path / ".sweep").mkdir()  # os.replace onto a directory raises OSError
    ls._run_retention_sweep(cfg, tmp_path / "rlm-mcp-own.log")
    assert list(tmp_path.glob(".sweep.*.tmp")) == []


def test_sweep_collects_orphaned_tmps_but_spares_one_in_flight(tmp_path):
    """The unlink on a failed replace cannot fire for a process KILLED between the
    write and the replace — SIGKILL runs no except block, and the pre-warmed spares
    are killed routinely. The sweep collects those strays, and is what retires the
    ones older builds left behind. A fresh tmp is another process mid-write: spare it.
    """
    cfg = _cfg(tmp_path, log_sweep_cooldown_s=0)
    old = tmp_path / ".sweep.29936.tmp"
    old.write_text("1787950531.8465369")
    os.utime(old, (1, 1))  # ancient -> debris
    fresh = tmp_path / ".sweep.111.tmp"
    fresh.write_text("now")  # just written -> could be in flight

    ls._run_retention_sweep(cfg, tmp_path / "rlm-mcp-own.log")

    assert not old.exists()
    assert fresh.exists()


def test_the_suite_never_attaches_a_file_handler_to_the_real_log_dir():
    """Regression guard for the conftest fixture that keeps pytest out of ~/.rlm/logs.

    Importing src.server runs ``configure_logging(CFG)`` against the REAL config, which
    is how 18 of 20 files in the operator's live log dir came to be pytest debris. The
    autouse fixture strips that handler before every test; this asserts it is gone even
    right after the import that installs it.
    """
    import src.server  # noqa: F401 - the import IS the thing under test

    real_dir = load_config().log_dir.resolve()
    for h in logging.getLogger(ls.LOGGER_NAME).handlers:
        base = getattr(h, "baseFilename", None)
        if base is not None:
            assert Path(base).resolve().parent != real_dir, (
                f"a file handler is writing to the real log dir: {base}"
            )


# --------------------------- frozen clock: the age cap --------------------------- #
def test_the_age_cap_keeps_a_file_one_second_inside_the_window(tmp_path, clock):
    """The realistic age boundary, which no existing test covered.

    The two age tests above dodge the clock entirely — one sets log_retention_days=0 so
    everything is expired, the other backdates an mtime to epoch 1970. Neither would
    notice an off-by-one-day error in the cutoff, or a sign flip. With a frozen clock the
    cutoff is exact: 7 days, one file a second inside it and one a second outside.
    """
    cfg = _cfg(tmp_path, log_retention_days=7, log_sweep_cooldown_s=0,
               log_retention_files=100, log_retention_total_bytes=10**9)
    day = 86400
    keep = tmp_path / "rlm-mcp-keep.log"
    drop = tmp_path / "rlm-mcp-drop.log"
    for p, age in ((keep, 7 * day - 1), (drop, 7 * day + 1)):
        p.write_text("x")
        os.utime(p, (clock.now - age, clock.now - age))

    ls._run_retention_sweep(cfg, tmp_path / "rlm-mcp-own.log", now_fn=clock)

    assert keep.exists(), "a file inside the retention window was deleted"
    assert not drop.exists(), "a file past the retention window survived"


def test_the_sweep_cooldown_is_measured_against_the_sentinel_mtime(tmp_path, clock):
    """Two processes starting seconds apart must not both sweep. Pinning this needs a
    fixed now: the assertion is about a 60-second window, and the sentinel's mtime and
    the sweep's clock have to be the same instant for the comparison to mean anything."""
    cfg = _cfg(tmp_path, log_retention_days=0, log_sweep_cooldown_s=60,
               log_retention_files=0, log_retention_total_bytes=0)
    old = tmp_path / "rlm-mcp-old.log"
    old.write_text("x")
    os.utime(old, (clock.now - 99 * 86400, clock.now - 99 * 86400))

    sentinel = tmp_path / ".sweep"
    sentinel.write_text("recent")
    os.utime(sentinel, (clock.now - 59, clock.now - 59))     # inside the cooldown
    ls._run_retention_sweep(cfg, tmp_path / "own.log", now_fn=clock)
    assert old.exists(), "the sweep ran despite a sentinel inside the cooldown"

    os.utime(sentinel, (clock.now - 61, clock.now - 61))     # cooldown elapsed
    ls._run_retention_sweep(cfg, tmp_path / "own.log", now_fn=clock)
    assert not old.exists(), "the sweep did not run after the cooldown elapsed"

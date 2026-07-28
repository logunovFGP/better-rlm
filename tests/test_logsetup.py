import dataclasses
import inspect
import logging
import logging.handlers
import os
import sys

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
def test_configure_logging_no_stdout_and_creates_file(tmp_path):
    logger = ls.configure_logging(_cfg(tmp_path))
    try:
        for h in logger.handlers:
            assert getattr(h, "stream", None) is not sys.stdout  # never the JSON-RPC channel
        assert any(isinstance(h, logging.handlers.RotatingFileHandler) for h in logger.handlers)
        assert list(tmp_path.glob("rlm-mcp-*.log"))  # a per-pid file was opened
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

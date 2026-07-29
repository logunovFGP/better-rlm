"""Structured, disk-bounded file logging for the RLM MCP server.

An MCP stdio server must keep stdout clean (JSON-RPC) and Claude Code tags all
stderr as "error", so detailed logs go to a FILE in ``log_dir`` and stderr stays
WARNING-only. Because Claude Code runs one server per session PLUS a churning pool
of pre-warmed spare daemons (8+ concurrent processes observed), the disk bound can
NOT come from per-process rotation alone — it comes from a race-safe startup sweep
that caps total files/bytes/age across every ``rlm-mcp-*.log*`` in the directory.

Design: one rotated file per PID (no cross-process rotation race), plus a startup
sweep enforcing ``<= log_retention_files`` AND ``<= log_retention_total_bytes`` AND
``<= log_retention_days``. Events are logfmt (``evt=… k=v``); we log lengths/counts
and token/cost — never prompt or context text.
"""

from __future__ import annotations

import contextlib
import contextvars
import functools
import inspect
import logging
import logging.handlers
import os
import time
import uuid
from pathlib import Path

from .config import Config, load_config
from .output import bound_output

LOGGER_NAME = "rlm-mcp"
_MAX_VAL_LEN = 500  # clamp any single logged value so one record can't bloat the file

# Correlation id for the in-flight tool call. logged_tool binds it; log_event reads
# it and stamps every nested event (cli_spawn, rlm_query, retry, sub_batch) with the
# same rid, so a single request's fan-out is traceable across processes/threads.
# Defaults to None (dropped by log_event) for events outside any tool call.
_RID: contextvars.ContextVar[str | None] = contextvars.ContextVar("rlm_rid", default=None)


def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)


def current_rid() -> str | None:
    """The correlation id bound to the current context, or None."""
    return _RID.get()


@contextlib.contextmanager
def bind_rid(rid: str | None):
    """Bind ``rid`` for the duration of the block. Needed to carry the id into worker
    threads (a ThreadPoolExecutor worker starts with a fresh contextvars context)."""
    token = _RID.set(rid)
    try:
        yield
    finally:
        _RID.reset(token)


class _UTCFormatter(logging.Formatter):
    converter = time.gmtime


def _fmt_val(value: object) -> str:
    """Render one logfmt value: clamp length, quote if it contains spaces/specials."""
    s = str(value)
    if len(s) > _MAX_VAL_LEN:
        s = s[:_MAX_VAL_LEN] + "…"
    if s == "" or any(c in s for c in ' ="\n\t'):
        s = '"' + s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ").replace("\t", " ") + '"'
    return s


def log_event(logger: logging.Logger, evt: str, **fields: object) -> None:
    """Emit one logfmt record: ``evt=<evt> [rid=…] k=v ...`` (None-valued fields dropped).

    If a correlation id is bound to the current context and the caller did not pass an
    explicit ``rid``, it is injected right after ``evt`` so nested events (cli_spawn,
    rlm_query, retry, sub_batch) share the originating tool call's id.
    """
    parts = [f"evt={evt}"]
    rid = _RID.get()
    if rid is not None and "rid" not in fields:
        parts.append(f"rid={_fmt_val(rid)}")
    for key, val in fields.items():
        if val is None:
            continue
        parts.append(f"{key}={_fmt_val(val)}")
    logger.info(" ".join(parts))


# --------------------------------------------------------------------------- #
# Retention sweep — the hard disk bound across all processes
# --------------------------------------------------------------------------- #
def _run_retention_sweep(cfg: Config, own_path: Path) -> None:
    """Prune ``rlm-mcp-*.log*`` in log_dir to stay within the file/byte/age caps.

    Best-effort and never fatal. Skips if another process swept within the cooldown
    window (advisory sentinel, not a lock — correctness never depends on it). Never
    deletes ``own_path`` (this process's about-to-be-written log).
    """
    try:
        log_dir = cfg.log_dir
        sentinel = log_dir / ".sweep"
        now = time.time()
        # Cooldown: skip if a recent sweep already ran (avoids redundant concurrent sweeps).
        try:
            if now - sentinel.stat().st_mtime < cfg.log_sweep_cooldown_s:
                return
        except FileNotFoundError:
            pass
        # Touch the sentinel atomically so concurrent starts see the cooldown.
        try:
            tmp = log_dir / f".sweep.{os.getpid()}.tmp"
            tmp.write_text(str(now))
            os.replace(tmp, sentinel)
        except OSError:
            pass

        entries = []
        for p in log_dir.glob("rlm-mcp-*.log*"):
            if p == own_path:
                continue
            try:
                st = p.stat()
            except OSError:
                continue
            entries.append((p, st.st_mtime, st.st_size))

        # 1) Age cap.
        age_cutoff = now - cfg.log_retention_days * 86400
        survivors = []
        for p, mtime, size in entries:
            if mtime < age_cutoff:
                _unlink(p)
            else:
                survivors.append((p, mtime, size))

        # 2) File-count cap + 3) total-bytes cap, newest first.
        survivors.sort(key=lambda t: t[1], reverse=True)
        total = 0
        for i, (p, _mtime, size) in enumerate(survivors):
            total += size
            if i >= cfg.log_retention_files or total > cfg.log_retention_total_bytes:
                _unlink(p)
    except Exception as exc:  # a logging sweep must never take down the server
        try:
            get_logger().warning("evt=sweep_error err=%s", f"{type(exc).__name__}: {exc}"[:_MAX_VAL_LEN])
        except Exception:
            pass


def _unlink(path: Path) -> None:
    try:
        path.unlink()
    except (FileNotFoundError, PermissionError, OSError):
        pass  # another process may hold/have-removed it; harmless


def _pid_log_path(cfg: Config) -> Path:
    stamp = time.strftime("%Y%m%d", time.gmtime())
    return cfg.log_dir / f"rlm-mcp-{stamp}-{os.getpid()}.log"


def configure_logging(cfg: Config) -> logging.Logger:
    """Install the "rlm-mcp" logger: a per-PID rotating file handler (INFO) after a
    retention sweep, plus a WARNING-only stderr handler. Idempotent. Never touches
    stdout (the JSON-RPC channel)."""
    logger = get_logger()
    for h in list(logger.handlers):  # idempotent re-config
        logger.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass
    logger.propagate = False  # don't double-log via the root handler

    level = getattr(logging, cfg.log_level, None)
    if not isinstance(level, int):
        level = logging.INFO
    logger.setLevel(min(level, logging.WARNING))  # let WARNING reach stderr regardless

    fmt = _UTCFormatter(
        "ts=%(asctime)s.%(msecs)03dZ pid=%(process)d lvl=%(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    # stderr: WARNING+ only (Claude Code tags all stderr as "error", so keep it quiet).
    stderr_h = logging.StreamHandler()  # defaults to sys.stderr — NEVER sys.stdout
    stderr_h.setLevel(logging.WARNING)
    stderr_h.setFormatter(fmt)
    logger.addHandler(stderr_h)

    if cfg.log_to_file:
        try:
            cfg.log_dir.mkdir(parents=True, exist_ok=True)
            own = _pid_log_path(cfg)
            _run_retention_sweep(cfg, own)  # sweep BEFORE opening our own file
            file_h = logging.handlers.RotatingFileHandler(
                own, maxBytes=cfg.log_max_bytes, backupCount=cfg.log_backup_count,
                encoding="utf-8",
            )
            file_h.setLevel(level)
            file_h.setFormatter(fmt)
            logger.addHandler(file_h)
        except Exception as exc:  # file logging is best-effort; never block startup
            logger.warning("evt=logfile_error err=%s", f"{type(exc).__name__}: {exc}"[:_MAX_VAL_LEN])
    return logger


# --------------------------------------------------------------------------- #
# Per-tool-call logging decorator (DRY across the 12 @mcp.tool() functions)
# --------------------------------------------------------------------------- #
# Args worth logging (identifiers/flags — never the content itself).
_ID_ARGS = ("ctx_id", "model_override", "reduce", "chunk_index", "max_chunks", "strategy")
_LEN_ARGS = ("question", "prompt", "code", "source")


def logged_tool(fn):
    """Wrap an @mcp.tool() function to emit an ``evt=tool_call`` record (rid, args
    summary, duration, outcome) and to turn an unhandled exception into the bounded
    ``ERROR in <tool>: ...`` string the tools contract on — the tool boundary is the
    one place that conversion belongs. Uses functools.wraps so FastMCP still
    introspects the original signature for the tool schema."""
    sig = inspect.signature(fn)
    tool = fn.__name__
    logger = get_logger()

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        rid = uuid.uuid4().hex[:8]
        fields: dict[str, object] = {"rid": rid, "tool": tool}
        try:
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            named = bound.arguments
        except TypeError:
            named = dict(kwargs)
        for k in _ID_ARGS:
            v = named.get(k)
            if v not in (None, "", -1, 0):
                fields[k] = v
        for k in _LEN_ARGS:
            v = named.get(k)
            if isinstance(v, str) and v:
                fields[f"{k}_len"] = len(v)

        start = time.monotonic()
        rid_token = _RID.set(rid)  # visible to every nested event for this call
        try:
            result = fn(*args, **kwargs)
        except Exception as exc:
            log_event(logger, "tool_call", **fields, dur_ms=round((time.monotonic() - start) * 1000),
                      outcome="error", err=f"{type(exc).__name__}: {exc}")
            return bound_output(f"ERROR in {tool}: {exc}", load_config().output_cap_bytes)
        finally:
            _RID.reset(rid_token)
        dur_ms = round((time.monotonic() - start) * 1000)
        # Tools signal a handled failure by returning a string starting with "ERROR";
        # record the reason (clamped by log_event) so the failure is diagnosable.
        is_err = isinstance(result, str) and result.lstrip().startswith("ERROR")
        log_event(logger, "tool_call", **fields, dur_ms=dur_ms,
                  outcome="error" if is_err else "ok",
                  err=result.strip() if is_err else None,
                  result_bytes=len(result) if isinstance(result, str) else None)
        return result

    return wrapper

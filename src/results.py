"""Durable per-chunk answers, so an interrupted batch resumes instead of restarting.

The failure this exists for: a 103-chunk batch ran 30 minutes, was interrupted during a
second pass, and returned nothing — every completed chunk answer died in memory with the
process. Re-running would have re-paid for all 103. Here each answer is appended to disk
the moment it lands, so the next run reads them back and asks the model only about chunks
it does not already have.

WHY A SEPARATE FILE AND NOT ``ContextMeta``. The meta is one ``meta.json`` rewritten
whole on every save; N batch workers finishing concurrently would race that rewrite and
lose answers. A JSONL appended line-at-a-time under a lock is the shape that survives
both concurrency and a kill -9: a torn final line loses one chunk, not the file, and
``load`` skips it.

WHY THE KEY INCLUDES THE CHUNKING. A cached answer is only valid for the exact bytes it
was asked about. Chunk index 7 under ``lines`` is different text than index 7 under
``files``, and re-chunking at a new size shifts every boundary — so strategy and chunk
count are part of the key. Change either and the old answers are simply not found, which
is the correct behaviour and needs no invalidation pass.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from .config import Clock, Config

_LOCK = threading.Lock()


@dataclass(frozen=True)
class Saved:
    """One chunk's persisted answer."""

    index: int
    answer: str
    itok: int
    otok: int
    model: str


def key_for(prompt: str, model: str, strategy: str, n_chunks: int) -> str:
    """Identity of "this question, over this chunking". See the module docstring for why
    strategy and n_chunks belong in the key rather than in an invalidation step."""
    h = hashlib.sha256()
    h.update(prompt.encode("utf-8", "replace"))
    h.update(b"\x00")
    h.update(model.encode("utf-8", "replace"))
    h.update(b"\x00")
    h.update(f"{strategy}:{n_chunks}".encode())
    return h.hexdigest()[:16]


def path_for(cfg: Config, ctx_id: str, key: str) -> Path:
    return cfg.store_dir / ctx_id / "results" / f"{key}.jsonl"


def load(cfg: Config, ctx_id: str, key: str) -> dict[int, Saved]:
    """Answers already persisted for this (context, question, chunking), by chunk index.

    Tolerant by design: a truncated or corrupt line is skipped rather than raised. The
    cost of skipping is one chunk re-asked; the cost of raising is that a torn write from
    a killed process makes the whole resume file unreadable — which would reintroduce the
    exact restart-from-zero this module prevents.
    """
    p = path_for(cfg, ctx_id, key)
    try:
        raw = p.read_text(encoding="utf-8").splitlines()
    except (FileNotFoundError, OSError):
        return {}
    out: dict[int, Saved] = {}
    for line in raw:
        try:
            r = json.loads(line)
            idx = int(r["index"])
            out[idx] = Saved(idx, str(r.get("answer", "")), int(r.get("itok", 0)),
                             int(r.get("otok", 0)), str(r.get("model", "")))
        except (ValueError, TypeError, KeyError):
            continue
    return out


def append(cfg: Config, ctx_id: str, key: str, saved: Saved, *,
           now: Clock = time.time) -> None:
    """Persist one chunk answer. Best-effort — a storage failure must not fail the batch
    that just successfully produced the answer; it only costs a re-ask on resume."""
    p = path_for(cfg, ctx_id, key)
    rec = {"index": saved.index, "answer": saved.answer, "itok": saved.itok,
           "otok": saved.otok, "model": saved.model, "ts": now()}
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        with _LOCK, open(p, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
            fh.flush()   # a resume must see it even if the process is killed next second
    except OSError:
        pass


def clear(cfg: Config, ctx_id: str, key: str) -> bool:
    """Discard the cached answers for one (question, chunking). Used by an explicit
    ``fresh=True`` — the way to re-ask a question whose answers you no longer trust."""
    try:
        path_for(cfg, ctx_id, key).unlink()
        return True
    except (FileNotFoundError, OSError):
        return False

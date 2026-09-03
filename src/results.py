"""Content-addressed cache of chunk answers, plus a per-run manifest.

THE CACHE IS KEYED BY WHAT THE MODEL WAS ASKED, NOT BY WHICH CONTEXT ASKED IT. The first
version of this module stored answers under ``<ctx_id>/results/``. That worked for a
re-run of the same ctx_id and for nothing else: load the same file again and you get a
new ctx_id, every answer already paid for is orphaned, and the whole batch is re-spent.
An answer's identity is ``(chunk bytes, prompt, model)`` — the ctx_id is an accident of
when you loaded it. So the key is a hash of exactly those three, and a re-loaded file, a
re-bundled repo, or a second context that happens to contain the same file all hit.

This is ordinary content-addressed memoization (git objects, the Nix store, Bazel's
action cache, ccache). Its yield depends entirely on how stable chunk boundaries are:
under the ``files`` strategy an edit to 3 of 1,053 files leaves 1,050 chunks byte-identical
and the re-analysis costs 3 calls; under ``lines`` the same edit shifts every later line
boundary and nearly everything misses. That is why ``batch.default_strategy`` prefers
``files`` for anything that carries file markers — it is content-defined chunking at
file granularity, the same idea rsync and restic use, and it is what makes this cache
worth having.

NO TTL, ON PURPOSE. An entry is correct indefinitely: same bytes, same prompt, same model
give the same answer, and the model id is in the key so a model change is a miss, not a
stale hit. A time-based expiry would throw away work that was paid for and still valid.
Disk is bounded the way the log dir is — a byte cap with LRU eviction in ``sweep`` — and
a cache HIT touches the entry's mtime so recently useful answers survive longest.

THE MANIFEST is separate and per run: ``<store>/<ctx_id>/results/<run_key>.jsonl``, one
line per chunk answered, in COMPLETION order — each line is appended by the pool worker
that produced it, and the workers run concurrently, so the chunk indices arrive shuffled
and a stopped run leaves gaps rather than a clean prefix. Read the ``index`` field; do not
infer position from line number. It is not used for resume (the cache is). It is the
human-readable record of what a run produced, and the file the over-cap reply points at
when 30 chunks of findings will not fit in one tool result.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from .config import Clock, Config

_LOCK = threading.Lock()


@dataclass(frozen=True)
class Saved:
    """One chunk's answer. ``index`` is the position in the context it was read back for
    — a property of the context, not of the answer, which is why the cache entry itself
    does not store it and ``cache_get`` fills it in from the caller."""

    index: int
    answer: str
    itok: int
    otok: int
    model: str


# --------------------------------------------------------------------------- #
# Content-addressed cache
# --------------------------------------------------------------------------- #
def content_key(chunk_digest: str, prompt: str, model: str, system: str = "") -> str:
    """Identity of an answer: a digest of the exact chunk text the model saw, the user's
    prompt, the SYSTEM prompt it ran under, and the model that answered. NOT the assembled
    prompt — that embeds ``CHUNK i/n``, which changes with position and would defeat reuse
    of identical content at a new index. The fidelity trade (the model does not learn it is
    "chunk 5 of 12" this time) is small and deliberate.

    A DIGEST, not the text: ``ContextStore.chunk_digests`` stamps each chunk's sha256 into
    the chunk meta at chunking time, so the cache scan no longer reads every selected chunk
    just to hash it. Callers must hand this the digest that store produces — passing raw
    text still yields a stable key, but not the same one, so it would simply never hit.

    ``system`` is in the key because it changes the answer. The batch map used to send
    none at all and now sends ``batch.MAP_SYSTEM``, which turns a page of prose into a
    JSON envelope: same chunk, same prompt, same model, different answer. Leaving it out
    would serve an old-contract answer under the new one and — with ``reduce=True`` —
    merge prose and envelopes into a single synthesis. The TEXT is hashed, not a version
    tag, so editing that constant self-invalidates with no manual bump."""
    h = hashlib.sha256()
    h.update(chunk_digest.encode("utf-8", "replace"))
    h.update(b"\x00")
    h.update(prompt.encode("utf-8", "replace"))
    h.update(b"\x00")
    h.update(model.encode("utf-8", "replace"))
    h.update(b"\x00")
    h.update(system.encode("utf-8", "replace"))
    return h.hexdigest()


def cache_path(cfg: Config, key: str) -> Path:
    # Two-character fan-out so a directory never holds tens of thousands of entries.
    return cfg.cache_dir / key[:2] / f"{key}.json"


def cache_get(cfg: Config, key: str, *, index: int) -> Saved | None:
    """The cached answer for ``key``, stamped with the caller's chunk ``index``, or None.

    A hit touches the file's mtime: eviction is LRU by mtime (atime is unreliable on
    Windows and often disabled), so the touch is what keeps useful answers alive.
    """
    p = cache_path(cfg, key)
    try:
        r = json.loads(p.read_text(encoding="utf-8"))
        with contextlib.suppress(OSError):
            os.utime(p, None)
        return Saved(index, str(r["answer"]), int(r.get("itok", 0)),
                     int(r.get("otok", 0)), str(r.get("model", "")))
    except (FileNotFoundError, OSError, ValueError, TypeError, KeyError):
        return None


def cache_put(cfg: Config, key: str, saved: Saved, *, now: Clock = time.time) -> None:
    """Persist one answer. Atomic (tmp + replace) so a kill mid-write leaves no torn
    entry to mis-read later. Best-effort: a storage failure only costs a re-ask."""
    p = cache_path(cfg, key)
    rec = {"answer": saved.answer, "itok": saved.itok, "otok": saved.otok,
           "model": saved.model, "ts": now()}
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(rec), encoding="utf-8")
        os.replace(tmp, p)
    except OSError:
        pass


def cache_delete(cfg: Config, key: str) -> bool:
    """Drop one entry. Used by ``fresh=True`` — the deliberate way to re-ask a question
    whose cached answer you no longer trust."""
    try:
        cache_path(cfg, key).unlink()
        return True
    except (FileNotFoundError, OSError):
        return False


# --------------------------------------------------------------------------- #
# LRU byte-cap sweep
# --------------------------------------------------------------------------- #
def sweep(cfg: Config, *, now: Clock = time.time) -> None:
    """Keep the cache under ``cache_max_bytes``, evicting least-recently-used first.

    Same shape as the log retention sweep, for the same reason: Claude Code runs many
    server processes (a pool of pre-warmed spares), so a full directory walk on every
    start is real cost, and the ``.sweep`` sentinel makes only one of them do it per
    ``cache_sweep_cooldown_s``. Advisory, not a lock — correctness never depends on it,
    and a file another process is mid-write on (``*.tmp``) is never taken.

    ponytail: near-duplicate of logsetup._run_retention_sweep minus the age cap. Fold the
    two into one helper when a third caller appears; refactoring a tested sweep for two
    callers is not yet worth its own risk.
    """
    try:
        root = cfg.cache_dir
        if not root.exists():
            return
        t = now()
        sentinel = root / ".sweep"
        with contextlib.suppress(FileNotFoundError):
            if t - sentinel.stat().st_mtime < cfg.cache_sweep_cooldown_s:
                return
        tmp = root / f".sweep.{os.getpid()}.tmp"
        try:
            tmp.write_text(str(t), encoding="utf-8")
            os.replace(tmp, sentinel)
        except OSError:
            with contextlib.suppress(OSError):
                tmp.unlink(missing_ok=True)

        entries: list[tuple[Path, float, int]] = []
        for p in root.glob("*/*.json"):
            with contextlib.suppress(OSError):
                st = p.stat()
                entries.append((p, st.st_mtime, st.st_size))
        entries.sort(key=lambda e: e[1], reverse=True)   # newest (most recently hit) first
        total = 0
        for p, _mtime, size in entries:
            total += size
            if total > cfg.cache_max_bytes:
                with contextlib.suppress(OSError):
                    p.unlink()
    except Exception:  # noqa: BLE001 - a cache sweep must never take down the server
        pass


# --------------------------------------------------------------------------- #
# Per-run manifest
# --------------------------------------------------------------------------- #
def run_key(prompt: str, model: str, strategy: str, n_chunks: int,
            system: str = "") -> str:
    """Identity of one RUN — "this question over this chunking" — for the manifest file.
    Strategy and chunk count are in it because chunk 7 under ``files`` is different text
    than chunk 7 under ``lines``, and a manifest that mixed them would mislabel answers.
    ``system`` for the same reason it is in ``content_key``: a manifest holding answers
    from two different output contracts mislabels them just as badly."""
    h = hashlib.sha256()
    h.update(prompt.encode("utf-8", "replace"))
    h.update(b"\x00")
    h.update(model.encode("utf-8", "replace"))
    h.update(b"\x00")
    h.update(f"{strategy}:{n_chunks}".encode())
    h.update(b"\x00")
    h.update(system.encode("utf-8", "replace"))
    return h.hexdigest()[:16]


def manifest_path(cfg: Config, ctx_id: str, key: str) -> Path:
    return cfg.store_dir / ctx_id / "results" / f"{key}.jsonl"


def manifest_append(cfg: Config, ctx_id: str, key: str, saved: Saved, *,
                    now: Clock = time.time) -> None:
    """Append one answered chunk to the run's manifest. Best-effort, like the cache."""
    p = manifest_path(cfg, ctx_id, key)
    rec = {"index": saved.index, "answer": saved.answer, "itok": saved.itok,
           "otok": saved.otok, "model": saved.model, "ts": now()}
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        with _LOCK, open(p, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
    except OSError:
        pass


def manifest_clear(cfg: Config, ctx_id: str, key: str) -> bool:
    try:
        manifest_path(cfg, ctx_id, key).unlink()
        return True
    except (FileNotFoundError, OSError):
        return False

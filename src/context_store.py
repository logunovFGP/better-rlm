"""On-disk external context store.

Loaded content lives on disk under ``store_dir``; tools return only refs +
metadata, so raw content never enters the root model's context. Single-file
text loads reference the original file in place (no copy); inline text, PDFs,
and directory concatenations are materialized under the context's own dir.
"""

from __future__ import annotations

import hashlib
import codecs
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .config import Config, estimate_tokens

# Separator used when materializing a directory into one context file. The
# "files" chunk strategy splits on this marker (see chunking.py).
FILE_SEP = "\n\n===== FILE: {path} ({size} bytes) =====\n"

# Skipped when loading a directory: dotfiles/dirs, VCS, caches, and likely secrets.
#: Directory names never worth loading. ".venv_windows"/".venv_sh" are THIS repo's own
#: installer-created venvs: without them, load_dir on this checkout ingested 7,383 files /
#: 48.2 MB of which 47.7 MB (99%) was dependency source, burying the ~60 project files.
_SKIP_DIRS = {".git", ".hg", ".svn", "__pycache__", "node_modules",
              ".venv", "venv", ".venv_windows", ".venv_sh"}
_SKIP_NAMES = {".env", ".env.local", "id_rsa", "id_ed25519", ".netrc", ".pgpass"}
_MAX_DIR_FILE_BYTES = 25 * 1024 * 1024  # skip individual files larger than this in dir loads

#: Skips that are the POINT of _SKIP_DIRS / _SKIP_NAMES, not a surprise. Reporting
#: these as loudly as an unexpected drop makes the warning fire on every JS or Python
#: project, and an alarm that always fires is one nobody reads -- which would defeat
#: the fix for the case it exists for (11 source files silently classified binary).
_EXPECTED_SKIPS = frozenset({"skip-dir", "skip-name"})


def _count_reasons(entries: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for e in entries:
        k = e.split(":", 1)[0]
        out[k] = out.get(k, 0) + 1
    return out


@dataclass
class ContextMeta:
    ctx_id: str
    source: str
    source_type: str   # text | file | dir | command
    data_type: str     # text | log | pdf | dir
    content_path: str  # abs path to materialized or original UTF-8 text
    bytes: int
    lines: int
    est_tokens: int
    sha256: str
    file_count: int
    files: list[str]
    created: str
    #: Exact count per skip reason -- always complete, and cheap. Silence here was
    #: the defect: a dir load reported success having dropped 11 of 184 files, and
    #: an answer over the remaining 173 was wrong by omission with nothing to show.
    skipped_counts: dict[str, int] = field(default_factory=dict)
    #: A bounded SAMPLE of skipped paths, biased to the surprising reasons. Bounded
    #: because `skipped` was briefly unbounded and a tree with node_modules wrote
    #: 3,000 entries / 128 KB of meta.json for a context holding one 20-byte file --
    #: and list_metas() parses every meta.json. `files` caps at 100 for the same reason.
    skipped: list[str] = field(default_factory=list)
    chunk_strategy: Optional[str] = None
    chunks: list[dict] = field(default_factory=list)


@dataclass(frozen=True)
class CommandRun:
    """Outcome of ``ContextStore.load_command``: the context plus how the run ended.
    Kept separate from ContextMeta so a stored context never carries transient run
    state — and so a caller cannot ignore a non-zero exit by accident."""
    meta: ContextMeta
    returncode: int
    stderr_tail: str
    truncated: bool     # hit max_bytes
    timed_out: bool     # hit timeout_s

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out


def _stat_path(path: Path) -> tuple[int, int, str]:
    """Stream a file to compute (bytes, line_count, sha256) without loading it."""
    h = hashlib.sha256()
    nbytes = nlines = 0
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
            nbytes += len(block)
            nlines += block.count(b"\n")
    return nbytes, nlines, h.hexdigest()


def _has_carriage_return(path: Path) -> bool:
    """True if the raw file contains any b"\\r" -- i.e. reading it as text translates
    newlines, so char offsets taken on that text do NOT map to raw byte positions.

    Streamed, so it costs one bounded-memory pass and no decode. The byte-count check in
    _add_byte_offsets catches CRLF on its own (each "\\r\\n" collapses to one char, so the
    totals diverge) but NOT a lone "\\r": that translates to "\\n" one-for-one, the totals
    match, and a seek would then return "\\r" where a char slice returns "\\n".
    """
    with open(path, "rb") as fh:
        return any(b"\r" in block for block in iter(lambda: fh.read(1 << 20), b""))


def _looks_binary(path: Path) -> bool:
    """Heuristic: a NUL byte in the first KB means binary.

    The decode uses an INCREMENTAL decoder because a fixed-size byte read can end
    mid-character. A plain ``chunk.decode("utf-8")`` then raises UnicodeDecodeError
    on a perfectly valid UTF-8 file, which load_dir read as "binary" and dropped in
    silence. Measured: a .ts file whose em-dash sat at byte 1022 was classified
    binary; the same file with the character one byte earlier was not. An
    incremental decoder buffers the partial tail instead of failing on it.
    """
    try:
        with open(path, "rb") as fh:
            chunk = fh.read(1024)
        codecs.getincrementaldecoder("utf-8")().decode(chunk)
        return b"\x00" in chunk
    except UnicodeDecodeError:   # genuinely not UTF-8 in the first KB
        return True
    except OSError:
        return True


class ContextStore:
    """Filesystem-backed store. One directory per context under ``store_dir``."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.root = cfg.store_dir
        self.root.mkdir(parents=True, exist_ok=True)

    # -- ids / paths -------------------------------------------------------
    @staticmethod
    def _new_id() -> str:
        return "ctx_" + uuid.uuid4().hex[:8]

    def _dir(self, ctx_id: str) -> Path:
        return self.root / ctx_id

    def _meta_path(self, ctx_id: str) -> Path:
        return self._dir(ctx_id) / "meta.json"

    # -- persistence -------------------------------------------------------
    def _save(self, meta: ContextMeta) -> ContextMeta:
        self._dir(meta.ctx_id).mkdir(parents=True, exist_ok=True)
        with open(self._meta_path(meta.ctx_id), "w") as fh:
            json.dump(asdict(meta), fh, indent=2)
        return meta

    def get(self, ctx_id: str) -> ContextMeta:
        path = self._meta_path(ctx_id)
        if not path.exists():
            raise KeyError(f"Unknown context id: {ctx_id}")
        with open(path) as fh:
            return ContextMeta(**json.load(fh))

    def list_ids(self) -> list[str]:
        return sorted(p.name for p in self.root.glob("ctx_*") if (p / "meta.json").exists())

    def list_metas(self) -> list[ContextMeta]:
        """All stored contexts as metadata, newest-first; unreadable ones are skipped."""
        metas: list[ContextMeta] = []
        for cid in self.list_ids():
            try:
                metas.append(self.get(cid))
            except Exception:
                continue
        metas.sort(key=lambda m: m.created, reverse=True)  # created is ISO8601 -> lexical sort
        return metas

    def drop(self, ctx_id: str) -> bool:
        """Delete a context's store directory (meta.json + any materialized content).
        A file referenced in place lives OUTSIDE this dir and is never touched. Returns
        False if the id was unknown. Guards against path traversal: the target must be a
        direct ``ctx_*`` child of the store root."""
        d = self._dir(ctx_id)
        if d.parent != self.root or not d.name.startswith("ctx_"):
            raise ValueError(f"Invalid context id: {ctx_id!r}")
        if not (d / "meta.json").exists():
            return False
        shutil.rmtree(d, ignore_errors=True)
        return True

    # -- loaders -----------------------------------------------------------
    def _finalize(self, meta: ContextMeta) -> ContextMeta:
        nbytes, nlines, sha = _stat_path(Path(meta.content_path))
        meta.bytes, meta.lines, meta.sha256 = nbytes, nlines, sha
        meta.est_tokens = estimate_tokens(nbytes)
        return self._save(meta)

    def load_text(self, text: str, source: str = "<inline text>") -> ContextMeta:
        ctx_id = self._new_id()
        content_path = self._dir(ctx_id) / "content.txt"
        content_path.parent.mkdir(parents=True, exist_ok=True)
        # newline="" disables newline translation. Without it Windows rewrites every
        # "\n" as "\r\n", which inflates the bytes/est_tokens that _stat_path measures
        # from the raw file, diverges sha256 from POSIX for identical input, and hands
        # the Linux sandbox guest CRLF it never saw on the host.
        content_path.write_text(text, encoding="utf-8", newline="")
        meta = ContextMeta(
            ctx_id=ctx_id, source=source, source_type="text", data_type="text",
            content_path=str(content_path), bytes=0, lines=0, est_tokens=0, sha256="",
            file_count=1, files=[source], created=datetime.now(timezone.utc).isoformat(),
        )
        return self._finalize(meta)

    def load_file(self, path: str, data_type: str = "text") -> ContextMeta:
        src = Path(path).expanduser().resolve()
        if not src.is_file():
            raise FileNotFoundError(f"Not a file: {src}")
        ctx_id = self._new_id()
        if data_type == "pdf":
            content_path = self._dir(ctx_id) / "content.txt"
            content_path.parent.mkdir(parents=True, exist_ok=True)
            # newline="": see load_text — no CRLF translation on the stored copy.
            content_path.write_text(_extract_pdf(src), encoding="utf-8", newline="")
            cp = content_path
        else:
            # Reference the original in place — no copy, so multi-GB logs cost no disk.
            cp = src
            self._dir(ctx_id).mkdir(parents=True, exist_ok=True)
        meta = ContextMeta(
            ctx_id=ctx_id, source=str(src), source_type="file", data_type=data_type,
            content_path=str(cp), bytes=0, lines=0, est_tokens=0, sha256="",
            file_count=1, files=[src.name], created=datetime.now(timezone.utc).isoformat(),
        )
        return self._finalize(meta)

    def load_dir(self, path: str) -> ContextMeta:
        root = Path(path).expanduser().resolve()
        if not root.is_dir():
            raise NotADirectoryError(f"Not a directory: {root}")
        ctx_id = self._new_id()
        content_path = self._dir(ctx_id) / "content.txt"
        content_path.parent.mkdir(parents=True, exist_ok=True)
        files: list[str] = []
        skipped: list[str] = []

        def _rel(p: Path) -> str:
            try:
                return str(p.relative_to(root))
            except ValueError:
                return str(p)

        # newline="": see load_text — the concatenated dump must keep the newlines the
        # source files actually had, not the host's.
        with open(content_path, "w", encoding="utf-8", newline="") as out:
            for fp in sorted(root.glob("**/*")):
                if not fp.is_file():
                    continue
                # Every skip below is RECORDED. Dropping a file silently and still
                # reporting success is how a tenancy question got answered from a
                # tree with tenancy.ts missing.
                if any(part in _SKIP_DIRS for part in fp.parts):
                    skipped.append(f"skip-dir: {_rel(fp)}")
                    continue
                if fp.name in _SKIP_NAMES:
                    skipped.append(f"skip-name: {_rel(fp)}")
                    continue
                try:
                    if fp.stat().st_size > _MAX_DIR_FILE_BYTES:
                        skipped.append(f"too-large: {_rel(fp)}")
                        continue
                    if _looks_binary(fp):
                        skipped.append(f"binary: {_rel(fp)}")
                        continue
                    body = fp.read_text(encoding="utf-8", errors="replace")
                except OSError as exc:
                    skipped.append(f"unreadable ({type(exc).__name__}): {_rel(fp)}")
                    continue
                rel = str(fp.relative_to(root))
                out.write(FILE_SEP.format(path=rel, size=len(body.encode("utf-8"))))
                out.write(body)
                files.append(rel)
        meta = ContextMeta(
            ctx_id=ctx_id, source=str(root), source_type="dir", data_type="dir",
            content_path=str(content_path), bytes=0, lines=0, est_tokens=0, sha256="",
            file_count=len(files), files=files[:100],
            created=datetime.now(timezone.utc).isoformat(),
            skipped_counts=_count_reasons(skipped),
            # surprising reasons first: those are the ones that cost an answer
            skipped=sorted(skipped, key=lambda e: e.split(":", 1)[0] in _EXPECTED_SKIPS)[:50],
        )
        return self._finalize(meta)

    def load_command(self, argv: list[str], *, source: str, timeout_s: int,
                     max_bytes: int, merge_stderr: bool = False) -> "CommandRun":
        """Run ``argv`` (never through a shell) and stream its stdout into a new context.

        Streaming, not capturing: the whole point is inputs bigger than memory, so stdout
        goes block-by-block to disk and is bounded by ``max_bytes`` and ``timeout_s``.
        Both bounds kill the process — a follow/tail command is expected here and must not
        run forever or fill the disk.

        stderr goes to a temporary FILE, not a pipe. With both on pipes, a command that
        writes more than the pipe buffer to stderr while we drain only stdout deadlocks
        both sides forever.

        ``merge_stderr`` sends stderr down the same pipe instead, so it lands *in* the
        context. Some programs log to stderr by convention (postgres does, hence
        ``docker logs`` on one), and the shell answer ``2>&1`` is unavailable here by
        design. There is then no separate ``stderr_tail`` — a failure message arrives
        interleaved with the data.

        Returns the run outcome rather than raising on a non-zero exit: a command can fail
        *after* emitting the output worth analysing. The caller decides what a partial
        result is worth — but it must never be reported as a clean load.
        """
        ctx_id = self._new_id()
        content_path = self._dir(ctx_id) / "content.txt"
        content_path.parent.mkdir(parents=True, exist_ok=True)
        timed_out = threading.Event()
        truncated = False
        written = 0
        with tempfile.TemporaryFile() as errf:
            proc = subprocess.Popen(
                argv, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT if merge_stderr else errf)

            def _on_timeout() -> None:
                if proc.poll() is None:   # don't flag a process that already finished
                    timed_out.set()
                    try:
                        proc.kill()
                    except OSError:
                        pass

            timer = threading.Timer(timeout_s, _on_timeout)
            timer.start()
            try:
                # "wb": raw bytes, so no newline translation and no decode of a stream
                # that may not be valid UTF-8. Readers already open with errors="replace".
                with open(content_path, "wb") as out:
                    for block in iter(lambda: proc.stdout.read(1 << 16), b""):
                        if written + len(block) > max_bytes:
                            out.write(block[:max_bytes - written])
                            written, truncated = max_bytes, True
                            proc.kill()
                            break
                        out.write(block)
                        written += len(block)
            finally:
                # Cancel AFTER reaping, so the watchdog stays armed if the loop above
                # raised — otherwise proc.wait() could block on a live process forever.
                proc.stdout.close()
                rc = proc.wait()
                timer.cancel()
            errf.seek(0, os.SEEK_END)
            errf.seek(max(0, errf.tell() - 2000))   # tail only: stderr can be huge too
            stderr_tail = errf.read().decode("utf-8", errors="replace").strip()
        meta = ContextMeta(
            ctx_id=ctx_id, source=source, source_type="command", data_type="text",
            content_path=str(content_path), bytes=0, lines=0, est_tokens=0, sha256="",
            file_count=1, files=[source], created=datetime.now(timezone.utc).isoformat(),
        )
        return CommandRun(self._finalize(meta), rc, stderr_tail, truncated,
                          timed_out.is_set())

    # -- readers -----------------------------------------------------------
    def read_text(self, ctx_id: str) -> str:
        # Universal-newline translation stays ON, deliberately. It was briefly disabled
        # (newline="") so char offsets would map 1:1 to on-disk bytes for the seek-based
        # read_chunk -- a global change to serve one local need, and it regressed three
        # things: chunk metas written earlier held offsets computed on TRANSLATED text, so
        # the fallback slice shifted and silently truncated their content; chunk_text's
        # paragraph regex (r"\n[ \t]*\n") cannot match "\r\n\r\n", so "paragraphs"/
        # "semantic" degraded to blind fixed-size cuts on any CRLF context; and rlm_exec
        # handed a Linux sandbox guest CRLF. Byte offsets now simply decline to exist for
        # such files (see _add_byte_offsets), which costs those files the seek fast path
        # and nothing else.
        return Path(self.get(ctx_id).content_path).read_text(encoding="utf-8", errors="replace")

    def read_chunk(self, ctx_id: str, index: int) -> str:
        meta = self.get(ctx_id)
        if not meta.chunks:
            raise ValueError(f"{ctx_id} has not been chunked; call rlm_chunk_context first")
        if index < 0 or index >= len(meta.chunks):
            raise IndexError(f"chunk {index} out of range (0..{len(meta.chunks) - 1})")
        ch = meta.chunks[index]
        byte_start, byte_end = ch.get("byte_start", -1), ch.get("byte_end", -1)
        if 0 <= byte_start <= byte_end:
            # Binary seek + bounded read: O(chunk), never decodes the rest of the file.
            # Safe to decode the slice on its own: byte_start/byte_end came from Leaf 1's
            # validated char->byte walk, so they always land on UTF-8 sequence boundaries.
            with open(meta.content_path, "rb") as fh:
                fh.seek(byte_start)
                raw = fh.read(byte_end - byte_start)
            return raw.decode("utf-8", errors="replace")
        return self.read_text(ctx_id)[ch["start"]:ch["end"]]   # legacy/unvalidated metas

    def grep(self, ctx_id: str, pattern: str, *, ignore_case: bool = False,
             max_matches: int = 50, max_line_len: int = 500) -> tuple[list[tuple[int, str]], bool]:
        """Stream the context content line by line and return up to ``max_matches``
        ``(1-based lineno, clamped line)`` tuples plus a ``capped`` flag (True if the scan
        stopped early on reaching ``max_matches`` — more may exist). Streaming keeps this
        bounded on multi-GB contexts. Raises ``re.error`` on a bad pattern."""
        rx = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
        cp = Path(self.get(ctx_id).content_path)
        out: list[tuple[int, str]] = []
        capped = False
        with open(cp, encoding="utf-8", errors="replace") as fh:
            for i, line in enumerate(fh, 1):
                if rx.search(line):
                    s = line.rstrip("\n")
                    if len(s) > max_line_len:
                        s = s[:max_line_len] + "…"
                    out.append((i, s))
                    if len(out) >= max_matches:
                        capped = True
                        break
        return out, capped

    def set_chunks(self, ctx_id: str, strategy: str, chunks: list[dict],
                   *, text: str | None = None) -> ContextMeta:
        """Record the chunk index on a context, computing byte offsets for it.

        ``text`` lets a caller hand over the decode it already holds. Both callers in
        server.py have just read the whole context to chunk it, and without this the
        byte-offset pass decoded it a SECOND time while the caller's copy was still
        live -- two complete copies of a multi-GB context at once, in a tool whose
        premise is contexts too large to hold.
        """
        meta = self.get(ctx_id)
        self._add_byte_offsets(meta, chunks, text)
        meta.chunk_strategy = strategy
        meta.chunks = chunks
        return self._save(meta)

    def _add_byte_offsets(self, meta: ContextMeta, chunks: list[dict],
                          text: str | None = None) -> None:
        """Map each chunk's char offsets to byte offsets in one O(n) prefix-sum pass, so
        read_chunk can seek() instead of re-decoding the whole file per chunk.

        Offsets are stored only when the mapping is PROVEN exact, on two counts:

        * the walked byte total equals meta.bytes (the raw file size). errors="replace"
          decoding is lossy -- an invalid byte becomes U+FFFD, which re-encodes to 3
          bytes -- so the total drifts on invalid UTF-8. This also catches CRLF, where
          each "\\r\\n" read back as one char makes the walk come up short.
        * the file holds no "\\r" at all. read_text applies universal-newline
          translation, and a lone "\\r" translates to "\\n" without changing the byte
          count, so the total alone would not notice.

        Either check failing leaves byte_start/byte_end at their Chunk default of -1 and
        readers take the char-offset slow path -- slower, never wrong.
        """
        if text is None:
            text = self.read_text(meta.ctx_id)
        positions = sorted({p for c in chunks for p in (c["start"], c["end"])})
        byte_at: dict[int, int] = {}
        total = prev = 0
        for p in positions:
            total += len(text[prev:p].encode("utf-8"))
            byte_at[p] = total
            prev = p
        total += len(text[prev:].encode("utf-8"))
        if total != meta.bytes or _has_carriage_return(Path(meta.content_path)):
            return
        for c in chunks:
            c["byte_start"], c["byte_end"] = byte_at[c["start"]], byte_at[c["end"]]


def _extract_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # optional dependency
        raise RuntimeError(
            "PDF support needs the optional dependency: pip install 'rlm-mcp[pdf]' (pypdf)"
        ) from exc
    reader = PdfReader(str(path))
    return "\n\n".join((page.extract_text() or "") for page in reader.pages)

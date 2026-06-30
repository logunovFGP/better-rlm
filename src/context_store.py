"""On-disk external context store.

Loaded content lives on disk under ``store_dir``; tools return only refs +
metadata, so raw content never enters the root model's context. Single-file
text loads reference the original file in place (no copy); inline text, PDFs,
and directory concatenations are materialized under the context's own dir.
"""

from __future__ import annotations

import hashlib
import json
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
_SKIP_DIRS = {".git", ".hg", ".svn", "__pycache__", "node_modules", ".venv", "venv"}
_SKIP_NAMES = {".env", ".env.local", "id_rsa", "id_ed25519", ".netrc", ".pgpass"}
_MAX_DIR_FILE_BYTES = 25 * 1024 * 1024  # skip individual files larger than this in dir loads


@dataclass
class ContextMeta:
    ctx_id: str
    source: str
    source_type: str   # text | file | dir
    data_type: str     # text | log | pdf | dir
    content_path: str  # abs path to materialized or original UTF-8 text
    bytes: int
    lines: int
    est_tokens: int
    sha256: str
    file_count: int
    files: list[str]
    created: str
    chunk_strategy: Optional[str] = None
    chunks: list[dict] = field(default_factory=list)


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


def _looks_binary(path: Path) -> bool:
    try:
        with open(path, "rb") as fh:
            chunk = fh.read(1024)
        chunk.decode("utf-8")
        return b"\x00" in chunk
    except UnicodeDecodeError:
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

    # -- loaders -----------------------------------------------------------
    def _finalize(self, meta: ContextMeta) -> ContextMeta:
        nbytes, nlines, sha = _stat_path(Path(meta.content_path))
        meta.bytes, meta.lines, meta.sha256 = nbytes, nlines, sha
        meta.est_tokens = estimate_tokens(nbytes)
        return self._save(meta)

    def load_text(self, text: str, source: str = "<inline text>", data_type: str = "text") -> ContextMeta:
        ctx_id = self._new_id()
        content_path = self._dir(ctx_id) / "content.txt"
        content_path.parent.mkdir(parents=True, exist_ok=True)
        content_path.write_text(text, encoding="utf-8")
        meta = ContextMeta(
            ctx_id=ctx_id, source=source, source_type="text", data_type=data_type,
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
            content_path.write_text(_extract_pdf(src), encoding="utf-8")
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

    def load_dir(self, path: str, pattern: str = "**/*") -> ContextMeta:
        root = Path(path).expanduser().resolve()
        if not root.is_dir():
            raise NotADirectoryError(f"Not a directory: {root}")
        ctx_id = self._new_id()
        content_path = self._dir(ctx_id) / "content.txt"
        content_path.parent.mkdir(parents=True, exist_ok=True)
        files: list[str] = []
        with open(content_path, "w", encoding="utf-8") as out:
            for fp in sorted(root.glob(pattern)):
                if not fp.is_file():
                    continue
                if any(part in _SKIP_DIRS for part in fp.parts) or fp.name in _SKIP_NAMES:
                    continue
                try:
                    if fp.stat().st_size > _MAX_DIR_FILE_BYTES or _looks_binary(fp):
                        continue
                    body = fp.read_text(encoding="utf-8", errors="replace")
                except OSError:
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
        )
        return self._finalize(meta)

    # -- readers -----------------------------------------------------------
    def read_text(self, ctx_id: str) -> str:
        return Path(self.get(ctx_id).content_path).read_text(encoding="utf-8", errors="replace")

    def preview(self, ctx_id: str, n_bytes: int) -> str:
        cp = Path(self.get(ctx_id).content_path)
        size = cp.stat().st_size
        with open(cp, "rb") as fh:
            head = fh.read(n_bytes)
            if size > 2 * n_bytes:
                fh.seek(-n_bytes, 2)
                tail = fh.read(n_bytes)
                return (head.decode("utf-8", "ignore") + "\n…[middle elided]…\n"
                        + tail.decode("utf-8", "ignore"))
        return head.decode("utf-8", "ignore")

    def read_chunk(self, ctx_id: str, index: int) -> str:
        meta = self.get(ctx_id)
        if not meta.chunks:
            raise ValueError(f"{ctx_id} has not been chunked; call rlm_chunk_context first")
        if index < 0 or index >= len(meta.chunks):
            raise IndexError(f"chunk {index} out of range (0..{len(meta.chunks) - 1})")
        ch = meta.chunks[index]
        return self.read_text(ctx_id)[ch["start"]:ch["end"]]

    def set_chunks(self, ctx_id: str, strategy: str, chunks: list[dict]) -> ContextMeta:
        meta = self.get(ctx_id)
        meta.chunk_strategy = strategy
        meta.chunks = chunks
        return self._save(meta)


def _extract_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # optional dependency
        raise RuntimeError(
            "PDF support needs the optional dependency: pip install 'rlm-mcp[pdf]' (pypdf)"
        ) from exc
    reader = PdfReader(str(path))
    return "\n\n".join((page.extract_text() or "") for page in reader.pages)

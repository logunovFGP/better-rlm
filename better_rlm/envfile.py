"""Write one credential into ``.env`` without it reaching a log, argv, or a terminal.

The installer already does this in bash (``install.sh``'s ``rlm_write_token``). The
setup wizard needs the same thing in Python, under the same rule: the value moves
from the prompt to the file and nowhere else, and the only things reported about it
are its length and a short hash.

Deliberately not YAML: ``.env`` is read by python-dotenv, holds secrets, and must stay
owner-only -- 0600 on POSIX, a single-ACE ACL on Windows, both via ``fsutil.harden``.
``config_writer`` owns config.yaml and has different rules -- its file is meant to be
diffed and committed, this one must never be.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from . import fsutil

# Owner-only mode and the platform-correct way to apply it both live in fsutil now:
# envfile, config_writer and four other writers were each doing their own version.


def fingerprint(value: str) -> str:
    """What may be said out loud about a secret: its size and a short digest.

    Enough to confirm two copies match, or that a paste arrived intact, without
    the value entering a transcript that outlives its rotation.
    """
    raw = value.encode("utf-8")
    return f"{len(raw)} bytes, sha256:{hashlib.sha256(raw).hexdigest()[:16]}"


def set_var(path: Path, key: str, value: str) -> str:
    """Set ``key`` in the env file at ``path``, replacing any existing line.

    Returns the fingerprint, never the value. Writes through a temp file in the same
    directory and ``os.replace``, so a reader sees the old file or the new one and
    never a half-written credential. The temp file is created 0600 by ``mkstemp``
    and the final file is chmod'd every time, not only on create -- an operator who
    once made it group-readable should not keep that after a write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    body = path.read_text(encoding="utf-8") if path.exists() else ""
    kept = [ln for ln in body.splitlines() if not ln.startswith(f"{key}=")]
    new_body = "\n".join(kept + [f"{key}={value}"]) + "\n"

    fsutil.atomic_write(path, new_body, secret=True)
    # And on the FINAL file, every write: an operator who once made it group-readable
    # should not keep that after a key is written into it.
    fsutil.harden(path)
    return fingerprint(value)


def has_var(path: Path, key: str) -> bool:
    """Whether the env file already sets ``key`` to something non-empty.

    Reads the file rather than os.environ: the question is what the next server
    start will load, and an export in this shell never reaches it.
    """
    if not path.exists():
        return False
    for ln in path.read_text(encoding="utf-8").splitlines():
        if ln.startswith(f"{key}=") and ln.split("=", 1)[1].strip():
            return True
    return False

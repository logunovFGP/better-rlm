"""Write one credential into ``.env`` without it reaching a log, argv, or a terminal.

The installer already does this in bash (``install.sh``'s ``rlm_write_token``). The
setup wizard needs the same thing in Python, under the same rule: the value moves
from the prompt to the file and nowhere else, and the only things reported about it
are its length and a short hash.

Deliberately not YAML: ``.env`` is read by python-dotenv, holds secrets, and must stay
owner-only -- 0600 on POSIX, a single-ACE ACL on Windows, both through ``_harden``.
``config_writer`` owns config.yaml and has different rules -- its file is meant to be
diffed and committed, this one must never be.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
from pathlib import Path

_MODE = 0o600


def _harden(path: Path) -> None:
    """Restrict ``path`` to its owner, on either platform.

    On Windows ``os.chmod`` is not merely a weaker control, it is close to no control
    at all: it toggles the read-only bit and nothing else, so a ``.env`` holding an
    API key keeps whatever ACL it inherited from its directory while this module's
    docstring promises 0600. Measured on a freshly created file here: six inherited
    ACEs, one of them a local group holding Modify.

    ``icacls`` is the platform equivalent -- drop inheritance, grant the current user
    alone -- and after it the same file carries exactly one ACE. It has shipped with
    Windows since Vista.

    Failure raises, and that is deliberate. The POSIX ``os.chmod`` this replaces
    raised too, and a credential written to a file we could not protect is precisely
    the thing a caller must not be left believing succeeded.
    """
    if os.name != "nt":
        os.chmod(path, _MODE)
        return
    user = os.environ.get("USERNAME") or os.environ.get("USER") or ""
    if not user:
        raise OSError(f"cannot restrict {path}: no USERNAME in the environment")
    proc = subprocess.run(
        ["icacls", str(path), "/inheritance:r", "/grant:r", f"{user}:F"],
        capture_output=True, text=True, timeout=30,
    )
    if proc.returncode != 0:
        raise OSError(
            f"cannot restrict {path} to {user}: icacls exited {proc.returncode}: "
            f"{(proc.stderr or proc.stdout).strip()[:200]}"
        )


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

    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(new_body)
            fh.flush()
            os.fsync(fh.fileno())
        _harden(Path(tmp))
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    _harden(path)
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

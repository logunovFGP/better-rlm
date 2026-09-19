"""Write a file so a crash cannot leave half of one, and claim a periodic sweep.

Both were written once per caller before this, and the hardening only ever landed on
whichever one the last incident touched. Six writers existed; exactly one had all of it.

  * ``config_writer`` and ``envfile`` used mkstemp + fsync + cleanup-and-re-raise
    (``envfile`` also restricting the result to its owner, because it holds a key);
  * ``budget`` (twice), ``results`` and ``engine`` used a predictable ``<name>.<pid>.tmp``
    with no fsync, no mode and no cleanup -- a failure left the temp behind, and nothing
    collects those.

``engine``'s was the worst of the three: its temp name carries no pid at all, and the
checkpoint path is derived from the question, so two processes resuming the same query
wrote the same temp file. mkstemp removes that by construction rather than by adding a
pid to a name.

Nor is ``fsync`` theoretical here. It is what stops a host crash leaving the renamed file
holding unflushed bytes -- and the ledger and the checkpoint are precisely the files whose
loss costs the work they exist to protect.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

#: Owner-only. The number is POSIX's; ``harden`` implements the same intent on Windows.
MODE = 0o600


def harden(path: Path) -> None:
    """Restrict ``path`` to its owner, on either platform.

    On Windows ``os.chmod`` is not merely a weaker control, it is close to no control at
    all: it toggles the read-only bit and nothing else, so a file holding an API key
    keeps whatever ACL it inherited from its directory while the caller's docstring
    promises 0600. Measured on a freshly created file: six inherited ACEs, one of them a
    local group holding Modify.

    ``icacls`` is the platform equivalent -- drop inheritance, grant the current user
    alone -- and after it the same file carries exactly one ACE. It has shipped with
    Windows since Vista.

    Failure raises, deliberately. A credential written to a file we could not protect is
    precisely the thing a caller must not be left believing succeeded.
    """
    if os.name != "nt":
        os.chmod(path, MODE)
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
            f"{(proc.stderr or proc.stdout or '').strip()[:200]}")


def atomic_write(path: Path, data: str | bytes, *, secret: bool = False) -> None:
    """Write ``data`` to ``path`` via a same-directory temp file and a rename.

    ``mkstemp`` creates the temp O_EXCL at 0600 under an unpredictable name, so there is
    no window where it exists at the umask default and no fixed name for a peer -- or an
    attacker -- to collide with. ``fsync`` before the rename means a host crash cannot
    leave the renamed file holding unflushed bytes. ``os.replace`` is atomic on POSIX and
    Windows alike, so a reader sees the old file or the new one, never a half-written
    intermediate. On any failure the temp is removed and the error re-raised, rather than
    left behind for nothing to collect.

    ``secret=True`` additionally restricts the file to its owner, applied to the TEMP
    before the rename so the content is never readable by anyone else even briefly.

    Text is written with an explicit encoding and ``newline``: these files are tracked,
    .gitattributes pins the repo to ``eol=lf``, and text mode's platform default rewrites
    every line ending on Windows -- turning a one-token change into a whole-file diff.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        mode, kwargs = (("wb", {}) if isinstance(data, bytes)
                        else ("w", {"encoding": "utf-8", "newline": "\n"}))
        with os.fdopen(fd, mode, **kwargs) as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        if secret:
            harden(Path(tmp))
        os.replace(tmp, path)
    except BaseException:
        # BaseException, not Exception: a KeyboardInterrupt mid-write must not leave the
        # temp behind either.
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def claim_sweep(root: Path, cooldown_s: float, t: float) -> bool:
    """True when this process should run the periodic sweep of ``root``.

    A ``.sweep`` sentinel whose mtime is the last run. Advisory, not a lock: N processes
    starting inside the cooldown can each decide to sweep, which costs a redundant pass
    and never corruption -- the sweepers only delete, and deleting an already-deleted
    file is a no-op all of them tolerate.

    The caps are deliberately NOT here. The three sweepers genuinely disagree about them
    -- the log sweep has a file-count cap nothing else wants and must spare its own open
    file, the cache has no age cap on purpose, the store shares one byte budget across
    two globs -- and a parameter list long enough to express all three would be a worse
    abstraction than three loops. What they share is exactly this: the decision to run.
    """
    if not root.exists():
        return False
    sentinel = root / ".sweep"
    try:
        if t - sentinel.stat().st_mtime < cooldown_s:
            return False
    except FileNotFoundError:
        pass
    tmp = root / f".sweep.{os.getpid()}.tmp"
    try:
        tmp.write_text(str(t), encoding="utf-8")
        os.replace(tmp, sentinel)
    except OSError:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        return False
    return True

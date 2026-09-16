"""Tiny config.yaml updater used by the TUI.

We need to update one or two keys in ``config.yaml`` from the picker
without disturbing the rest of the file (which carries comments, blank
lines, and a key order the README and ``config.yaml`` were hand-tuned
for). PyYAML's loader destroys comments and key order; ruamel.yaml keeps
both but is an extra dep.

So this module ships a hand-rolled line-rewriter. The format better-rlm
emits is a flat ``key: value`` document at the top of the file with one
key per line -- see the existing ``config.yaml``. Anything that does not
match ``^key: value$`` is left alone (comments, blanks, the long-form
multi-line keys, the indented body of a nested mapping). Only top-level
scalar ``key: value`` lines are rewritten.

One caveat the regex cannot express: the PARENT line of a nested mapping
(``docker:`` with an indented block under it) does match, with an empty
value. Passing such a key in ``updates`` would flatten it and orphan its
children. The TUI only ever writes the flat scalars it read, so nothing
reaches that case today -- but a caller that writes arbitrary keys must
check first.

Why not ruamel.yaml? Two reasons:

  1. ``pyproject.toml`` keeps a tight, audited dependency surface. Adding
     ruamel.yaml to support a feature that only the TUI uses pulls a
     whole new library in for every install.
  2. The hand-rolled rewriter is easier to audit -- it has one branch per
     line and tests can pin its output exactly.

If the file ever gains nested mappings that the TUI must rewrite, this
module should be replaced by ruamel.yaml. Until then, ~50 lines and no
extra dep.
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

# Match ``key: value`` where the value is a single scalar token (no
# nested mapping, no list, no further colons). Anchored to start-of-line
# so indented continuations of nested mappings are not rewritten.
_SCALAR_LINE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.*?)\s*$")


def _format_scalar(value: str) -> str:
    """Render a scalar value back to YAML -- quoted only when needed.

    The TUI only writes back booleans, enum strings and short free text.
    Booleans are bare; everything else is bare too unless it contains a
    colon, a hash or leading/trailing whitespace -- in which case a quoted
    scalar keeps the loader honest on the next ``load_config()``.
    """
    raw = str(value)
    v = raw.strip()
    if not v:
        return '""'
    if v.lower() in ("true", "false", "null", "yes", "no", "on", "off"):
        return v.lower()
    needs_quote = (
        ":" in v
        or "#" in v
        or '"' in v          # a value that is itself quote-delimited must be quoted
        or v.startswith(("{", "[", "&", "*", "!", "|", ">", "%", "@", "`"))
        or v.endswith((",", "{", "["))
        or any(ch.isspace() for ch in v)
        or raw != v  # leading or trailing whitespace was stripped above
    )
    if needs_quote:
        escaped = raw.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return v


def read_scalar(path: Path, key: str) -> str | None:
    """Return the current string value of a top-level scalar key, or None.

    Used by the TUI's /status line and pickers so they show what's on disk
    rather than what's in the in-memory Config -- those can disagree
    when ``RLM_MODE`` / ``RLM_PROVIDER`` env vars win (see config.load_config).

    Inline ``# ...`` comments after the value are stripped, because the
    existing ``config.yaml`` ships keys like::

        mode: auto   # auto | claude-cli | api. auto: prefer the `claude` CLI

    and we want ``mode == "auto"`` to come back, not the trailing prose.
    """
    if not path.exists():
        return None
    for raw in path.read_text(encoding="utf-8").splitlines():
        m = _SCALAR_LINE.match(raw)
        if m and m.group(1) == key:
            value, _comment = _split_inline_comment(m.group(2))
            if len(value) >= 2 and value.startswith('"') and value.endswith('"'):
                # Quoted: the delimiters are the quotes, so what sits between them is
                # verbatim. Stripping here would discard exactly the whitespace
                # _format_scalar quoted in order to keep.
                inner = value[1:-1].replace('\\"', '"')
                return inner.replace("\\\\", "\\")
            # "" means the key is present and deliberately empty; None means absent.
            return value.strip()
    return None


def _split_inline_comment(value: str) -> tuple[str, str]:
    """Split ``value   # comment`` into ``("value", "# comment")``.

    Handles the common case without breaking on values that legitimately
    contain a hash (e.g. a URL fragment).

    Returns both halves rather than discarding the comment, because
    ``write_scalars`` has to put it back: config.yaml documents half its
    keys in a trailing comment, and rewriting a line without it deletes
    the explanation this module exists to preserve.

    Only a quote at position 0 opens a YAML quoted scalar. The previous version
    toggled quote state on every quote anywhere in the value, so a lone apostrophe
    in a plain scalar -- an ordinary Windows path under a user named O'Brien, say --
    hid the hash that followed it: read_scalar returned the comment as part of the
    value, so /status looked up a path with the comment glued on, and write_scalars
    dropped the comment entirely on rewrite.
    """
    if value[:1] in ("'", '"'):
        quote = value[0]
        i = 1
        while i < len(value):
            if value[i] == "\\" and quote == '"':
                i += 2          # an escaped character inside a double-quoted scalar
                continue
            if value[i] == quote:
                i += 1          # past the closing quote
                break
            i += 1
        head, rest = value[:i], value[i:]
    else:
        head, rest = "", value
    cut = rest.find("#")
    if cut == -1:
        return (head + rest).rstrip(), ""
    return (head + rest[:cut]).rstrip(), rest[cut:]


def write_scalars(path: Path, updates: dict) -> bool:
    """Rewrite the listed top-level scalar keys in ``path``.

    Returns True if any line was changed. Unknown keys are appended at
    the end of the file (with a leading blank line if needed) so a
    config that doesn't yet have, e.g., ``mode:`` still gets one.

    Atomicity: write to ``path.with_suffix(path.suffix + ".tmp")`` first
    and rename, so an interrupted TUI session can never leave the file
    half-written.
    """
    if not updates:
        return False
    if not path.exists():
        lines = [f"{k}: {_format_scalar(v)}" for k, v in updates.items()]
        _atomic_write(path, "\n".join(lines) + "\n")
        return True

    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    seen: set = set()
    out: list = []
    changed = False
    for raw in lines:
        m = _SCALAR_LINE.match(raw)
        if m and m.group(1) in updates:
            key = m.group(1)
            # Carry the inline comment across. group(2) is the tail of raw.rstrip(),
            # so the comment's column is exact -- keep it when the new value still
            # fits, else push it out by two spaces.
            _value, comment = _split_inline_comment(m.group(2))
            new_line = f"{key}: {_format_scalar(updates[key])}"
            if comment:
                col = len(raw.rstrip()) - len(comment)
                new_line = new_line.ljust(max(col, len(new_line) + 2)) + comment
            if raw.rstrip() != new_line:
                changed = True
            out.append(new_line)
            seen.add(key)
        else:
            out.append(raw)

    missing = [k for k in updates if k not in seen]
    if missing:
        if out and out[-1].strip():
            out.append("")
        for key in missing:
            out.append(f"{key}: {_format_scalar(updates[key])}")
        changed = True

    if not changed:
        return False
    _atomic_write(path, "\n".join(out) + "\n")
    return True


def _atomic_write(path: Path, body: str) -> None:
    """Write ``body`` to ``path`` via a same-directory temp file + rename.

    Same shape as ``scripts/install_hook.py::_write_atomic``, deliberately: that one
    already got this right, and two implementations of one operation in one repo
    should not differ in how safe they are. ``mkstemp`` creates the file O_EXCL at
    0600 under an unpredictable name, so there is no window where it exists at the
    umask default and no fixed name to pre-plant -- the previous version wrote a
    predictable ``config.yaml.tmp`` and chmod'd it only afterwards. ``fsync`` before
    the rename means a host crash cannot leave the renamed file holding unflushed
    bytes, and the temp is removed if anything raises instead of being left behind.

    ``os.replace`` is atomic on POSIX and Windows, so a reader sees either the old
    file or the new one, never a half-written intermediate.

    A config the operator deliberately made group-readable still comes back private
    after a TUI write, and on Windows the mode is effectively a no-op (only the
    read-only bit exists there), which is why the mode test skips off POSIX.

    ``encoding`` and ``newline`` are explicit on purpose. config.yaml is
    UTF-8 (em dashes in the mode / provider / sandbox comments) and
    .gitattributes pins the repo to ``eol=lf``; the bare text-mode default
    would read it through the locale codepage and rewrite every line
    ending as CRLF on Windows, churning the whole file against its own
    normalization rule.
    """
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(body)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise

"""Named external sources — operator-declared commands the server can run and load.

RLM's value is reasoning over more than fits in a context window. That applies to a
*running system* — cluster logs, a metrics or trace backend, a journal, an audit API —
not only to a file that already happens to be on disk. Without this, the only route is
"shell out, redirect to a temp file, then load the file", which leaves an unbounded,
uncleaned intermediate on disk and gives the agent nothing to discover.

What those commands ARE is deliberately not this server's business. It ships no vendor
knowledge, no endpoints, no credentials, and no default registry. An operator declares
them in a YAML file **outside the repo** (``sources_file``, default
``~/.rlm/sources.yaml``), so a site's infrastructure never lands in a diff and an
upstream checkout is inert until someone opts in.

Registry shape::

    workload-logs:
      description: Logs for one workload in the current cluster
      command: kubectl logs -n {namespace} -l app={app} --since={since} --tail=-1
      timeout_s: 120
      max_bytes: 268435456

    metrics-range:
      description: Range query against the metrics backend
      command: curl -sS -H "Authorization: Bearer ${METRICS_TOKEN}"
               "${METRICS_URL}/api/v1/query_range?query={query}&start={start}&end={end}"

``command`` is split with ``shlex`` ONCE, at load time, and always run with
``shell=False``. Placeholders are substituted into the already-split argv tokens, so a
parameter value can never introduce a shell metacharacter, a pipe, or a second command —
``app="x; rm -rf /"`` becomes one literal argv token, not two commands. ``${VAR}`` in the
*template* expands from the server's environment, which is where a token belongs: not in
the file, and not in the conversation. Parameter values are never expanded, so a value
containing ``$HOME`` cannot read the environment back out.

``merge_stderr: true`` folds the command's stderr into the loaded content instead of
keeping it as a diagnostic tail. Needed more often than it looks: plenty of programs write
their *logs* to stderr by convention (postgres does, so ``docker logs`` on a postgres
container puts every line there), and the usual fix — ``2>&1`` — is shell syntax that does
not exist here by design. Off by default, because for a well-behaved command stderr is the
error channel and merging it would bury a failure message inside the data.

``{name}`` is a parameter, ``${VAR}`` is an environment reference — the two do not
collide. A template that needs a *literal* brace expression (a PromQL selector, a JSON
body) should take it as a parameter instead: parameter values are substituted verbatim
and never rescanned for placeholders.

The registry is re-read on every call. A running server holds its own copy of ``src/``
(see CLAUDE.md), and the registry is the one thing operators edit routinely — needing a
server reconnect to add a source would make it useless.
"""

from __future__ import annotations

import os
import re
import shlex
from dataclasses import dataclass
from pathlib import Path

import yaml

# Per-source defaults, overridable per entry. Deliberately not config keys: an operator
# who cares sets them on the source that needs it, and nobody tunes a global.
DEFAULT_TIMEOUT_S = 120
DEFAULT_MAX_BYTES = 256 * 1024 * 1024

# The negative lookbehind is load-bearing: "${VAR}" contains "{VAR}", so without it every
# environment reference in a template is parsed as a required parameter and the source
# becomes uncallable.
_PLACEHOLDER = re.compile(r"(?<!\$)\{([A-Za-z_][A-Za-z0-9_]*)\}")
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class Source:
    name: str
    command: tuple[str, ...]     # already shlex-split; run with shell=False
    description: str
    timeout_s: int
    max_bytes: int
    merge_stderr: bool

    @property
    def params(self) -> list[str]:
        """Placeholder names in the template, in first-appearance order."""
        seen: list[str] = []
        for tok in self.command:
            for m in _PLACEHOLDER.finditer(tok):
                if m.group(1) not in seen:
                    seen.append(m.group(1))
        return seen


def _parse_one(name: str, raw: object) -> Source:
    if not _NAME_RE.match(name):
        raise ValueError(f"source name {name!r} must match {_NAME_RE.pattern}")
    if not isinstance(raw, dict):
        raise ValueError(f"source {name!r} must be a mapping, got {type(raw).__name__}")
    cmd = raw.get("command")
    if isinstance(cmd, str):
        argv = shlex.split(cmd)
    elif isinstance(cmd, list) and all(isinstance(x, str) for x in cmd):
        argv = list(cmd)
    else:
        raise ValueError(f"source {name!r}: 'command' must be a string or a list of strings")
    if not argv:
        raise ValueError(f"source {name!r}: 'command' is empty")
    timeout_s = int(raw.get("timeout_s", DEFAULT_TIMEOUT_S))
    max_bytes = int(raw.get("max_bytes", DEFAULT_MAX_BYTES))
    # Zero or negative would not "disable" a bound — it would make every run return an
    # instantly-truncated empty context that still reports as loaded.
    if timeout_s <= 0 or max_bytes <= 0:
        raise ValueError(f"source {name!r}: timeout_s and max_bytes must be positive")
    return Source(
        name=name,
        command=tuple(argv),
        description=str(raw.get("description", "")).strip(),
        timeout_s=timeout_s,
        max_bytes=max_bytes,
        merge_stderr=bool(raw.get("merge_stderr", False)),
    )


def load_sources(path: Path) -> dict[str, Source]:
    """Parse the registry at ``path``. A missing file means "this deployment declares
    none" and is not an error. A malformed one IS an error: quietly serving an empty
    registry looks identical to a typo'd key, and the operator would never learn their
    file is being ignored."""
    path = Path(path).expanduser()
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping of source-name -> definition")
    return {name: _parse_one(name, raw) for name, raw in data.items()}


def _render(token: str, params: dict[str, str]) -> str:
    """Expand ``${VAR}`` in the literal parts of a template token and substitute
    parameters into the placeholder parts. Deliberately asymmetric: the template may
    read the environment, a parameter value never can."""
    out: list[str] = []
    last = 0
    for m in _PLACEHOLDER.finditer(token):
        out.append(os.path.expandvars(token[last:m.start()]))
        out.append(params[m.group(1)])
        last = m.end()
    out.append(os.path.expandvars(token[last:]))
    return "".join(out)


def resolve(src: Source, params: dict[str, str] | None = None) -> list[str]:
    """Substitute ``params`` into the source template and return the argv to run.

    Rejects missing AND unknown parameters. Unknown is not a harmless extra: it is
    almost always a typo for a declared one, which would otherwise run the command with
    the template's own placeholder text and return plausible data about nothing.
    """
    given = {str(k): str(v) for k, v in (params or {}).items()}
    declared = set(src.params)
    missing = sorted(declared - set(given))
    unknown = sorted(set(given) - declared)
    if missing or unknown:
        detail = []
        if missing:
            detail.append(f"missing {missing}")
        if unknown:
            detail.append(f"unknown {unknown}")
        raise ValueError(
            f"source {src.name!r} takes parameters {src.params or '[]'} — " + ", ".join(detail)
        )
    return [_render(tok, given) for tok in src.command]

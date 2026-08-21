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

A source needing a credential should declare ``credential_file`` (plus an optional
``credential_max_age_h``) rather than expecting an exported variable. The server only
stats that path — it never opens it — so the token cannot reach the conversation, a log
or a tool result through this server. A missing or stale file produces an error telling
the *user* to supply a fresh short-lived token, which is the only party that should. The
age limit is enforced here, so it holds regardless of the shortest expiry the upstream
service is willing to issue.

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
import time
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
# Braced form ONLY. Bare "$VAR" is left to expandvars unchecked on purpose: `awk '{print $1}'`
# and `grep 'x$'` would otherwise be misread as environment references and rejected.
_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class Source:
    name: str
    command: tuple[str, ...]     # already shlex-split; run with shell=False
    description: str
    timeout_s: int
    max_bytes: int
    merge_stderr: bool
    credential_file: str         # "" = none required
    credential_max_age_h: float  # 0 = no expiry check

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
        credential_file=str(raw.get("credential_file", "")).strip(),
        credential_max_age_h=float(raw.get("credential_max_age_h", 0) or 0),
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


def check_credential(src: Source) -> None:
    """Pre-flight the source's declared credential file. Raises ValueError with an
    instruction aimed at the USER, because supplying a token is their job, never the
    agent's.

    Only ``exists()`` and ``st_mtime`` are consulted — the server never opens the file,
    so a credential cannot reach the conversation, the logs or a tool result through it.
    The age limit is enforced locally and so holds whatever expiry the upstream service
    happens to offer: a backend that only issues 30-day tokens still gets a 4-hour one
    here, as long as the operator refreshes the file.
    """
    if not src.credential_file:
        return
    path = Path(os.path.expandvars(src.credential_file)).expanduser()
    if not path.exists():
        raise ValueError(
            f"source {src.name!r} needs a credential file that does not exist:\n"
            f"  {path}\n"
            f"ASK THE USER to create it with a short-lived token — do not create, fill or "
            f"invent one, and do not read it back into the conversation."
            + (f" Max age {src.credential_max_age_h:g}h."
               if src.credential_max_age_h else "")
            + (f"\nFormat: {src.description}" if src.description else "")
        )
    age_h = (time.time() - path.stat().st_mtime) / 3600
    if src.credential_max_age_h and age_h > src.credential_max_age_h:
        raise ValueError(
            f"source {src.name!r}: credential {path} is {age_h:.1f}h old, past its "
            f"{src.credential_max_age_h:g}h limit. ASK THE USER for a fresh short-lived "
            f"token and to revoke the old one. Refusing to use it — an expired token "
            f"returns an auth page that reads like data."
        )


def resolve(src: Source, params: dict[str, str] | None = None) -> list[str]:
    """Substitute ``params`` into the source template and return the argv to run.

    Rejects missing AND unknown parameters. Unknown is not a harmless extra: it is
    almost always a typo for a declared one, which would otherwise run the command with
    the template's own placeholder text and return plausible data about nothing.

    Also refuses an unset ``${VAR}``. ``os.path.expandvars`` leaves an unknown name as
    the LITERAL text ``${METRICS_TOKEN}``, so without this the command cheerfully sends
    that string to a remote service as if it were a token and the failure surfaces as a
    confusing 403 from the far end instead of a fixable message here.
    """
    check_credential(src)
    unset = sorted({m.group(1) for tok in src.command for m in _ENV_REF.finditer(tok)
                    if m.group(1) not in os.environ})
    if unset:
        raise ValueError(
            f"source {src.name!r} references environment variable(s) that are not set: "
            f"{unset}. The server would otherwise send the literal text "
            f"'${{{unset[0]}}}' to the far end. Set them where the MCP server is "
            f"launched, or switch the source to a credential_file."
        )
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

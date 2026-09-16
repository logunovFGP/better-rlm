# rlm-mcp

MCP server exposing Recursive Language Models over oversized contexts. Fork of
`eesb99/rlm-mcp`. See `README.md` for the tool surface and `docs/` for the
architecture notes.

## Branching

This repo uses trunk-based development. Read `TRUNK-BASED-PATTERNS.md` before
any code change. Config: `trunk-based.json` (repo root). Worktrees and long-lived
branches are blocked by a PreToolUse guard.

Verify command: `uv run --extra dev pytest -q` — `python` is not on PATH here, so it
must go through uv, and `--extra dev` is what pulls in pytest (it lives in the `dev`
extra, so a plain `uv run` resolves the project without it). Same command the
pre-push hook runs: `scripts/githooks/pre-push`, wired up by the installers via
`core.hooksPath`.

## Patching the vendored engine

The engine is **vendored source at `./rlm`**, not a dependency — copied from
`alexzhang13/rlm` at `v0.1.3`, provenance and the upstream-merge recipe in
`rlm/UPSTREAM.md`. Engine fixes are ordinary edits there; `import rlm` resolves to
this copy (`rlms` is no longer installed).

One import-time patch remains, and it is not a workaround: `better_rlm/auth.py:patch_engine`
rebinds the engine's `AnthropicClient` so completions route through our transport,
throttle and retry. That is dependency injection — folding it into `rlm/` would make
the engine import from `better_rlm/`, inverting the dependency and breaking the engine's
standalone use. It stays. It is idempotent, guarded by an `_rlmmcp_patched` flag.

The Docker REPL exec-protocol fixes that used to live in `src/sandbox_patch.py` are
now the engine's own code (`rlm/environments/docker_repl.py`): the hardened result
marker, the capped `locals` echo, the atomic `state.dill` write, the state-load
warning, UTF-8 host writes, and a real `timeout_s`. `tests/test_sandbox.py::
test_template_still_carries_every_hardening` is the regression guard that replaced
the old fail-loudly string match — an upstream merge that reverts the template fails
there. What is left of that module is `better_rlm/sandbox_reap.py`, host-side housekeeping
that sweeps sandboxes abandoned by a dead server.

**Fix engine defects in `rlm/` directly. Do not add a new monkey-patch.**

## Running server, files on disk

Python imports `better_rlm/` once at startup, so editing files changes nothing for an
already-running MCP server until it is reconnected. Never assume a fix is live
because it is on disk.

## Operator TUI (configure mode and models from the CLI)

The operator TUI (`better_rlm/tui.py`, entry `python -m better_rlm.cli` or `./run_tui.sh`)
mirrors cline-2's mode/model picker (`describe.py` is the data-only twin of
cline-2's `describeMode()`). It writes to `config.yaml` only -- never to
`.env` -- and is purely additive: the MCP server's runtime is unaffected.

It does **not** offer a provider picker, and must not grow one back.
`auth.require_anthropic` raises on every provider but `anthropic` at every
door that leads to a model call, because only its transport passes through
the session-window ledger; a picker writing `provider: openai` would produce
a config.yaml the server refuses. `/status` shows the value and flags a bad
one. `config.PROVIDER_KEY_ENV` was deleted in `f855b9c` for the same reason
-- do not re-add it, here or in `describe.py`.

Same "proxy vs host" terminology as cline-2: `claude-cli` is the **proxy**
path (spawns the `claude` CLI), `api` is the **host** path (talks to the
model endpoint directly), `auto` picks between them per launch. See the
TUI's `/mode-help` for the full comparison.

### Design boundary: TUI edits config; MCP server reads it

The TUI edits `config.yaml`; the next server start reads it. Two properties
hold unconditionally and are not up for trade:

  * diffs of `config.yaml` show every config change that ever shipped --
    there is no `~/.config/better-rlm/` store and must not be one;
  * the explicit `claude mcp restart rlm` is the only restart surface --
    no implicit "alias does the right thing" surprise.

**This section used to ban a console-script entry point outright**, to protect a
third property: that multiple checkouts coexist on one machine. Publishing to
PyPI breaks that property for the global name, deliberately. What replaces the
ban:

  * **the global `better-rlm` name resolves to one install** -- the wheel's, or
    whichever checkout linked it last. The per-checkout scripts
    (`./run_tui.sh`, `./run_server.sh`) stay the way to address a *specific*
    checkout, and each MCP registration still carries its own absolute path, so
    coexistence survives everywhere except the bare word;
  * **`better-rlm` dispatches, it never reimplements.** `better_rlm/cli.py` maps
    a subcommand onto the thing that already does that job. A subcommand growing
    its own copy of installer or server logic is the actual regression here --
    two implementations of auth, drifting apart.

### Endpoints vs providers

`provider` is the **wire protocol**. `auth.require_anthropic` refuses everything but
`anthropic` because `transport._LedgeredTransport` -- the session-window ledger, the
95% floor, ceiling-learning -- is only in the stack when the Anthropic client is.
That guard stays.

It was once read as "Anthropic the company is the only option", and `b7282f9` removed
the TUI's provider picker on that reading. The picker's real defect was different: it
offered four vendors whose selection wrote a config the server then rejected at the
first model call. `cfg.base_url` + `/endpoint` replace it. Every entry is
`provider=anthropic` and differs only in URL, so nothing about the budget changes.

Rules for anything added here:

  * **an endpoint must speak the Anthropic messages format.** A row needing a
    different client belongs behind `require_anthropic`, not in the picker;
  * **`base_url` is passed to `make_client` explicitly**, never left to the SDK's own
    `ANTHROPIC_BASE_URL` lookup. One setting, one visible owner, so `/status` and
    `better-rlm where` can state where calls go;
  * **`base_url` without `mode: api` does nothing** -- the `claude` CLI ignores it.
    `/status` renders that combination as `IGNORED` rather than letting it look live.

### Credentials are per provider

Each provider reads its own variable -- `ANTHROPIC_API_KEY`, `MINIMAX_API_KEY`,
`RLM_API_KEY` for a custom endpoint -- mirroring cline-2's per-provider settings
entries. One shared variable meant configuring a second provider overwrote the
first one's key.

`auth.api_key_for(cfg)` resolves it and deliberately does **not** fall back to
another provider's variable. Falling back would hand the key you issued to
Anthropic to whatever third-party endpoint is configured. A missing key fails as
missing; `test_a_providers_key_is_never_read_for_another` pins it.

When adding a provider, give it a variable nobody else uses
(`test_no_two_providers_share_a_key_variable`). The key name says nothing about the
protocol -- an earlier test asserted every provider used `ANTHROPIC_API_KEY` and
read like a safety check, when it was really just the shared-variable bug wearing a
test's clothes.

### Two shapes: checkout and wheel

`better_rlm.config.IS_CHECKOUT` is the single probe (is there a `pyproject.toml`
next to the package?). Everything that reads a repo-root file must go through
`config_file()` / `env_file()` rather than `PKG_ROOT / "..."`, or it silently
reads nothing in a wheel and falls back to baked-in defaults with no explanation.

| | checkout | `pip install better-rlm` |
|---|---|---|
| `config.yaml` | repo root | `~/.rlm/config.yaml` |
| `.env` | repo root | `~/.rlm/.env` |
| `better-rlm server` | `run_server.sh` | in-process (`server.main()`) |
| `better-rlm auth` | `install.sh --auth` | prints the `claude setup-token` flow |
| `better-rlm install` | `install.sh` | refuses, names the clone command |

`~/.rlm` is **not** a new store -- contexts, logs, cache, the budget ledger and
the source registry already live there (`_DEFAULTS` in `config.py`). Config and
credentials join the directory the tool already owns. A pip install has no
Docker sandbox image either; `sandbox: local` is the fallback.

`claude mcp add rlm -- better-rlm server` is the wheel's registration form. It
is why `server` must not depend on a shell script.

### Publishing

`better-rlm` on PyPI. Cutting a release is two commands:

```bash
# VERSION is the single source; sync_version.py propagates it to plugin.json
echo 0.4.0 > VERSION && uv run python scripts/sync_version.py && git commit -am "chore: 0.4.0"
git tag v0.4.0 && git push origin main v0.4.0
```

The tag push runs verify on both platforms, builds, creates the GitHub release and
publishes to PyPI. `workflow_dispatch` does the same and additionally offers
`dry_run` (verify + build, tag nothing, publish nothing) -- use it first when
anything about the pipeline changed.

One version number is burned per release and can never be reused, even after a
yank. So there is **no `push: branches` and no `schedule` trigger**: pushing a
version tag is an explicit human act, landing a PR is not, and
`tests/test_release.py::test_a_merge_can_never_publish` fails if a branch filter
is ever added. A `concurrency` group serialises releases per version so two runs
cannot race toward the irreversible step. `publish` runs last, after the GitHub
release, because that one is reversible and PyPI is not.

Publishing uses **Trusted Publishing (OIDC)**: no PyPI token exists in this
repository's secrets. The `publish` job holds exactly `id-token: write`, and
ships the artifacts the `release` job built rather than rebuilding, so GitHub
and PyPI serve byte-identical files.

The vendored engine ships as top-level `rlm` because that is the name its own 66
absolute imports use and `rlm/UPSTREAM.md`'s merge recipe depends on the tree
matching upstream. Upstream publishes the same import name (as the `rlms`
distribution), so installing both into one environment collides. Do not "fix"
this by renaming the tree -- it would break every future upstream merge.

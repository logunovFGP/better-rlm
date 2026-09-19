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
cline-2's `describeMode()`). See "It does offer a provider picker" below for what it writes: `config.yaml`,
and `.env` when it takes a credential.

It **does** offer a provider picker, and the rule it must keep is below in
"Endpoints vs providers": every row is `provider: anthropic` and differs only
by endpoint and credential. This paragraph used to ban the picker outright;
that ban was aimed at the wrong thing and contradicted the sections that
follow it.

It writes `config.yaml` and, for a credential, `.env` -- the latter added when
the setup flow took over supplying keys, which is the step being ported from
cline's `runProviderChange`. `better_rlm/envfile.py` owns that write.

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

### The TUI is a port, not an approximation

`better_rlm/searchable_list.py` is cline-2's
`apps/cli/src/tui/components/searchable-list.tsx`: the 100/90/70/30 scoring ladder,
section-first ordering, the three-pass window and the "do not hide a lone section
header" correction. Kept faithful rather than tidied -- a cleaner-looking rewrite
drifts from the thing it mirrors. Upstream's own three section/window cases are
ported into `tests/test_searchable_list.py`, so a divergence fails there.

`better_rlm/picker.py` reproduces the interaction (`useDialogKeyboard`): arrows move
with wrap, typing filters and resets selection to the top, Enter resolves, Esc
dismisses. Raw input is stdlib; a TUI framework would be a large dependency for one
key loop.

Two rules for anything added here:

  * **`os.read(fd)`, never `sys.stdin.read()`.** The buffered text stream pulls a
    whole escape sequence in one syscall and hands back its first byte, leaving the
    rest where `select()` cannot see it -- every arrow key then reads as a bare Esc.
    `test_an_arrow_key_is_not_read_as_escape` drives a real pty because nothing
    smaller reproduces it.
  * **Hold raw mode for the whole loop, never per keypress.** Setting it per key and
    restoring it in a finally leaves the terminal canonical between keys. A paste is
    one burst: the first byte is read raw and the rest land in the line discipline,
    which buffers them until a newline and swallows them as a line -- Cmd+V on a
    40-character key produced exactly one character. `raw_mode()` also enables
    bracketed paste, so pasted content carrying a newline or an escape byte can never
    be read as Enter or Esc. `read_key()` may therefore return TEXT LONGER THAN ONE
    CHARACTER; callers append it as a unit and run it through `clean_paste()`, because
    a key copied out of a file arrives with the newline attached.
  * **Headless is a path, not a fallback.** The suite, CI and `--one-shot` all feed
    stdin from a pipe. Every picker goes through `tui._select`, which chooses the live
    picker or the numbered prompt and returns the same sentinels, so callers never
    branch and the two cannot drift.

### First run asks; later runs offer

`better-rlm` with no working credential shows cline's onboarding
(`views/onboarding/screens.tsx::OnboardingMainMenuScreen`): a centred welcome, one
bordered card per provider, an arrow on the selection. Once a credential exists it
shows the maintenance menu instead. `tui.needs_onboarding(st)` is the switch, and it
asks whether a model call is possible -- not whether config.yaml has values in it. A
config that cannot call a model is not configured.

The flow is `better_rlm/onboard.py`, a port of cline's `views/onboarding/`
(`model.ts` + `controller.ts` + `keyboard.ts` + `view.tsx` collapsed into one module,
because our screens draw through `picker.py` and `tui._select` rather than owning
layout). Its order is cline's:

    vendor -> transport (host/proxy) -> credentials -> connectivity -> models x3

`describe.VENDORS` is the layer above providers, because a provider id pairs a vendor
with a transport -- `claude-cli` and `anthropic` are the same vendor reached two ways.
That flattening is right for the maintenance picker, where one setting is being
changed, and wrong for a first run, where the question is whose account you have and
the transport follows.

**`back()` pops a pushed history, never a static table.** Two edges are
path-dependent and a static table gets both wrong: MiniMax skips the transport screen
going forward, so Escape must skip it coming back or it lands on a question with one
answer; and the model screens are reached from the credential form or from the CLI
check depending on a choice made three screens earlier. `vendor_screen` therefore uses
`replace`, not `goto(Step.MODE)`, when a vendor offers one transport.

**config.yaml is written exactly once, at the end.** Gathering everything before
writing is why a cancel needs no rollback -- `/provider` had one and first-run
onboarding did not, so a cancel used to leave a half-configured file. A cancel at any
screen now leaves the file byte-identical.

  * **Claude** -> OAuth (proxy) or API key (host). Two ways, so it asks.
  * **MiniMax** -> API key only. One way, so it says so and moves on.

The credential screen is cline's `byo_apikey` **form** (`picker.ask`), not a bare
secret prompt: the endpoint is visible and editable beside the key it authenticates.
Tab cycles forward only -- Shift+Tab reads as `""` on both platforms, and naming a key
without adding it to `picker.KEY_NAMES` makes `key_text` return the name itself, typing
it into whatever field has focus, credential included.

The key variable is resolved from the **submitted** base URL, not the vendor picked two
screens earlier. An edited MiniMax URL reverse-maps to `PROVIDER_CUSTOM`, whose variable
is `RLM_API_KEY`; writing to `MINIMAX_API_KEY` there would leave the key unreadable with
every call failing "not set".

`config.py` runs `load_dotenv` at **import**, so a key written to `.env` is invisible to
the process that wrote it. `auth_step` and the wizard export it to `os.environ` as well;
without that the connectivity probe reports a good key as rejected and `/status` prints
MISSING.

cline branches the same way: MAIN_MENU offers vendors, and `runProviderChange` shows
`ModePickerContent` only when the transport is genuinely open rather than always
asking. A screen with one answer is not a question.

`test_every_vendor_offers_only_modes_that_can_reach_it` fails if a vendor offers a
transport it cannot be reached by -- MiniMax under `claude-cli` would route to
Anthropic and silently ignore the endpoint. `test_every_mode_a_vendor_offers_has_a_card`
fails on a mode with no card, which would render a blank row.

API keys echo **masked** -- first two and last two characters, the rest dots. cline
shows them in the clear; hiding them entirely is what we had, and it was worse than
either, because a silent paste gives no way to tell a good clipboard from an empty
one until the first model call fails. `mask_secret` never reveals more than four
characters and preserves length, so a truncated paste is visible.

### Setup mounts itself

`better_rlm/mcpreg.py` registers this install as Claude Code's `rlm` server as the
last step of `onboard.commit()`. Setup that writes a good config, prints Ready, and
leaves the agent talking to a registration from months ago has told the operator
something false -- and when that registration points at a DIFFERENT install it
reads a different `config.yaml` entirely, which is how a checkout kept serving
Claude model ids at a MiniMax endpoint long after the wizard had written correct
ones to `~/.rlm`.

It mirrors `install.sh` / `install.ps1` rather than inventing a second scheme:
`-s user`, probe with `claude mcp get` first (because `claude mcp add` exits 1 on
an existing name), and **replace** a registration whose command differs instead of
leaving it. Outcomes are reported, not silent: `ok` / `added` / `repointed` /
`skipped` / `failed`.

  * **a wheel registers `sys.executable -m better_rlm.cli server`, never the bare
    `better-rlm`.** install.ps1 writes a `better-rlm.cmd` shim that repoints that
    word at a checkout, so a name-based registration can silently flip install;
  * **whichever install you configured is the one that gets mounted.** Running the
    checkout TUI repoints to the checkout, and says so;
  * Claude Code only. cline is a separate target with its own config file and is
    deliberately not attempted.

### A missing sandbox is a routing instruction, not an error

Only `rlm_exec` and `rlm_query` need the Docker REPL; the other thirteen tools do
not. So when it cannot start, `engine.sandbox_diagnosis` separates the three causes
that have three different fixes -- Docker absent, daemon down, image never built
(the normal state of a pip install, which ships none) -- and both tools **return**
`engine.sandbox_guidance` rather than raising it. The message names `rlm_grep`,
`rlm_sub_query`, `rlm_sub_query_batch` and `rlm_estimate` as substitutes and says
not to retry. Raising instead produced `ERROR in rlm_exec: failed to connect to the
docker API at npipe:////./pipe/...` -- true, and useless: an agent reading it
retries the same call.

**It does not auto-fall back to `sandbox: local`.** That runs model-written Python
on the host with no isolation; dropping isolation because a daemon happens to be
down turns an outage into a privilege escalation. It is offered in the message as
an explicit, labelled opt-in and is never taken automatically.

### The model catalogue, and why it is also the price table

`describe.MODELS` is cline's bundled per-provider model table (its generated
`catalog.generated.ts`, reached through `getProviderConfig(id).knownModels`). cline does
not ask the endpoint what it serves either -- for anthropic and minimax the network
refresh is a no-op, because neither declares a `modelsSourceUrl`. A static table is the
faithful port, not a shortcut. Keyed by **vendor**, so `claude-cli` and `anthropic` cannot
hold two copies of one list.

`config.COST_PER_MTOK` is **derived** from it. They were two hand-maintained tables, and
that is how every MiniMax model came to price at $0.00: `cost_usd` returns 0.0 for an id
it does not know. Same rows, one source. A row with no published rate carries
`price_in=None`, stays out of the cost table, and renders as `unpriced` rather than
claiming a free call.

The picker offers the rows of the **configured endpoint**. `tui.pick_model` used to have
one hardcoded list of four Anthropic ids whatever `base_url` said, which is how a MiniMax
install came to run `root_model: claude-sonnet-5` against api.minimax.io.

**Do not "fix" the MiniMax base URL to match cline's.** cline stores
`https://api.minimax.io/anthropic/v1`; the Anthropic SDK appends `/v1/messages` itself, so
that string requests `/v1/v1/messages` and 404s every call. Measured, and pinned by
`test_the_minimax_base_url_does_not_end_in_v1`.

### The connectivity probe is a deliberate divergence

cline has **no** API-path pre-flight -- `controller.ts` says so in a comment, and bad
credentials surface at the first real call. `better_rlm/probe.py` runs one anyway, on both
paths, so a wrong URL or a mistyped key is caught on the screen that produced it. It earns
the divergence twice, because the credential form has no validation (faithful to cline) and
something has to catch a bad key before the model screens.

  * proxy path -> `transport.cli_auth_status`, free, ~215 ms. One connectivity concept,
    not two;
  * host path -> one `messages.create(max_tokens=16)`. Not `models.list()`: a MiniMax
    `/anthropic` endpoint need not implement it, and a 404 would then be
    indistinguishable from a wrong URL -- which is the distinction being bought;
  * it **bypasses `get_transport` on purpose**. The ledger refuses calls past the session
    stop line, so a probe on an exhausted budget would report a budget stop as an auth
    failure;
  * `url_wrong` / `key_rejected` / `network_down` come from disjoint SDK exception
    branches, not string sniffing. The one exception is a 404, where only the body says
    whether the model or the URL is wrong;
  * the key never reaches the result: `_scrub` removes it and clamps to one 200-char line.

A failed probe offers retry / edit / write-anyway. A hard gate would leave someone behind a
proxy unable to finish setup at all.

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

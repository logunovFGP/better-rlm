# Review brief — `leaf/feat/session-budget-and-resume`

Status: **reviewed 2026-09-03; the 13 findings below the line are fixed on this branch.**
Written for a reviewer with no other conversation context. Everything below is verifiable
from the branch; where a claim was NOT verified it says so.

> **Read §9 first.** It records what the self-review found and what changed in response,
> including two corrections to claims this document originally made.

| | |
|---|---|
| Branch | `leaf/feat/session-budget-and-resume` (8 commits, forked from `main`, unpushed) |
| Leaf age | first commit 2026-09-02 21:35 +04 — ~2 h at time of writing (24 h cap) |
| Rebase | no-op — `origin/main` has not moved since the fork |
| Verify | `uv run --extra dev pytest -q` → **354 passed, 1 skipped** (was 288 on `main`) |
| Diff | 29 files, +3393 / −407 |
| Merge | `git merge --no-ff` per `TRUNK-BASED-PATTERNS.md`; a reconnect of the MCP server is part of "deploy" |

## 1. What this leaf is for

One incident: a `rlm_sub_query_batch` over a 103-chunk context made 104 model calls, ran
30 minutes, drew ~60% of a 4-hour subscription window, was interrupted during a second
pass, and returned nothing. The spend bought no result and the obvious next move — run it
again — would have spent the same for the same nothing.

Three sub-causes, each now closed for **every model call the server makes** — which, since
the review, means every call it is willing to make at all: a non-Anthropic provider is
refused outright rather than served by a path the budget cannot see (§9).

| Sub-cause | Mechanism | Where |
|---|---|---|
| No forecast before spending | `rlm_estimate` / `budget.estimate_batch` + a **ceiling** for `rlm_query` (which cannot be forecast) | `src/budget.py`, `src/batch.py` |
| Nothing stopped the run before the wall | a **floor** under every completion at 95% of the window, plus the batch's polite early-stop `Gate` | `src/transport.py`, `src/budget.py` |
| Completed work lived only in memory | content-addressed **answer cache** (batch) and **checkpoint/resume** (`rlm_query`) | `src/results.py`, `rlm/core/rlm.py`, `src/engine.py` |

Plus a `~/.rlm/usage.jsonl` **ledger** of every completion so the next forecast sees
what the last run spent.

## 2. Commit map (oldest first)

| Commit | What | Read first |
|---|---|---|
| `fd95a61` feat | Ledger, estimate, `Gate`, per-chunk persistence, `rlm_estimate`/`rlm_budget`, skill routing; tests stopped writing into the real `~/.rlm` | `src/budget.py` |
| `ab3d161` fix | `bound_output` measured raw bytes; the MCP client counts JSON-encoded chars. A 131,072-byte reply arrived as 134,245 and was refused — a finished 30-call batch surfaced as an error | `src/output.py::encoded_len` |
| `505c333` test | Eight hand-copied `_Meta`/`_Store` stubs → one `batch_ctx` factory | `tests/conftest.py` |
| `8d44108` fix | Ledger moved to the transport (engine calls were invisible to it); the test-isolation redirect had been conditional on import order and a resume test passed by reading its own leftovers from the real store | `src/transport.py::_LedgeredTransport` |
| `e783141` refactor | `Deps(cfg, store, log, clock)` injected; map-reduce logic → `src/batch.py`; `server.py` 996 → 667 lines, tools are one-line adapters | `src/deps.py`, `src/batch.py` |
| `7dd9676` test | `Clock` injected for wall-clock positions only; 11 tests for arithmetic that had zero coverage | `src/config.py::Clock`, `tests/conftest.py::FrozenClock` |
| `2928c59` feat | Cache re-keyed by `(chunk bytes, prompt, model)`; `files`-first chunking; "never fits" vs "wait"; `rlm_query` ceiling | `src/results.py` |
| `c9a9a0c` feat | Floor at the transport; engine converts a refused call into a resumable `SessionBudgetError`; `rlm_query` checkpoints transcript + `state.dill` and resumes | `rlm/core/rlm.py::completion`, `src/engine.py::run_query` |

## 3. Where to look first — risk-ranked

1. **`rlm/core/rlm.py`** (vendored engine, upstream-merge sensitive). `completion(resume=)`,
   the three `except` blocks, `_attach_checkpoint`, `_restore_repl_state`. Check: the
   transcript is cut at `ckpt_len` (BEFORE the failed turn's user prompt) so a resume shows
   no duplicated message; `ckpt_len`/`ckpt_iter` are updated at the top of every turn;
   `state.dill` is read *inside* the `with` (the env is torn down as the exception leaves).
   All local edits carry `# better-rlm:` (eight did not until the review — §9);
   `rlm/UPSTREAM.md` lists them.
2. **`src/transport.py::_LedgeredTransport`**. `check_or_raise` runs *before* dispatch on
   both sync and async paths; the wrapper is rebuilt per call (it binds a cfg) and only the
   inner transport is cached. Check the refusal is not ledgered.
3. **`src/results.py`**. Key = `sha256(chunk_text ⧺ prompt ⧺ model)`; chunk *position* is
   deliberately excluded. Entries are atomic (tmp + `os.replace`); a hit touches mtime;
   `sweep` is LRU by mtime under `cache_max_bytes` with a cooldown sentinel. No TTL.
4. **`src/budget.py::judge`**. `possible = max_call_tokens <= usable`; `fits` requires
   `possible`. `render` has three branches: unknown ceiling, impossible, fits/does-not-fit.
5. **`src/batch.py::run`**. Order: ensure chunked → `_scan_cache` → estimate → refuse
   (impossible) → refuse (no headroom) → `Gate` → fan-out with `on_result=_persist`.
6. **`src/deps.py`** — the only place real config is loaded and log handlers installed.
7. **`tests/conftest.py`** — four autouse guards: no Docker, no live model call, no real log
   dir, plus `cfg` redirecting every `Path` field (enforced by
   `test_config.py::test_every_config_path_is_redirected_by_the_cfg_fixture`).

## 4. Design decisions a reviewer should challenge

- **The ceiling is a local estimate, not the real quota.** Anthropic exposes no endpoint
  for remaining window balance. The ledger counts *this server's* spend; other Claude
  sessions on the account are invisible, so headroom is an upper bound. Stated in every
  rendered output. `config.yaml` seeds `session_budget_tokens: 5800000` from ONE data point
  (the incident: ~3.5M tokens ≈ the user's reported 60%). A configured number is the ONLY
  gate. The server records the local spend it saw at a real usage limit
  (`budget.note_limit_hit`, highest wins) and reports it as the floor to configure from —
  it is not promoted to a ceiling, because that put the stop line below the spend that
  taught it and refused everything after (§9).
- **Input tokens are estimated locally (~4 chars/token), never taken from the transport.**
  The CLI path reported `itok=1027` for ~3M tokens of real input.
- **Stop at 95%, not 100%.** The last 5% buys a couple of chunks; being killed mid-call
  costs the exit path. The floor is read-and-compare, not reserve, so overshoot is bounded
  by concurrency × one call. The batch `Gate` *does* reserve.

  **Re-derived after `expected_output` (2026-09-03, §11).** That fix raised the per-call
  output figure from the 2,048 cap to the ledger's measured mean, so this bound moved and
  the old "~3 × 32k vs a ~290k margin" no longer describes it. Computed from the shipped
  config — `subquery_concurrency` 3, `chunk_chars` 120,000 (30,173 tokens in, plus 173 for
  `MAP_SYSTEM`), `session_budget_tokens` 5,800,000 at `budget_stop_fraction` 0.95, so a
  290,000-token margin:

  | output per call | per call | 3 in flight | share of margin |
  |---|---|---|---|
  | 9,953 (measured 48h mean) | 40,126 | 120,378 | 41.5% |
  | 16,384 (`_MAX_OUTPUT_FACTOR` clamp — worst case) | 46,557 | 139,671 | 48.2% |

  Still inside the margin, so the read-and-compare trade stands and no code changed. But
  the safety factor roughly halved, from ~3x to ~2.1x, and the threshold is now concrete:
  **below about 2.8M configured `session_budget_tokens` three worst-case calls in flight
  can overshoot the entire margin** (139,671 > 5% of the budget once the budget drops
  under 2,793,420). §5 said this "would matter more with a small
  `session_budget_tokens`"; that is the number where it starts to.
- **No TTL on the cache.** Same bytes + prompt + model ⇒ same answer; model id is in the
  key. Disk is bounded by bytes with LRU. A 1-hour TTL was proposed and rejected on purpose.
- **Chunk position is not in the cache key.** `_mk_prompt` frames each chunk as
  `CHUNK i/n`; identical content at a new position reuses the answer. Small fidelity trade.
- **`files` is the automatic chunking default for dir loads and marker-bearing bundles.**
  Content-defined chunking at file granularity: editing 3 of 1,053 files re-asks 3 chunks
  under `files`, nearly all under `lines`. With no markers `files` falls back to a capped
  fixed split, so the default is safe.
- **`Deps` is threaded explicitly; the tools are the composition root.** A tool signature
  *is* its MCP schema, so it cannot take `Deps`; `server.py` holds one `DEPS`. The
  `ratelimit` throttle stays module-global on purpose (it is a process resource).
- **Only wall-clock *positions* take a `Clock`.** `time.monotonic()` durations (tool
  `dur_ms`, throttle spacing, `cli_spawn` timing) deliberately do not — freezing them makes
  every measurement zero and would break the throttle.
- **`rlm_query` is bounded, not forecast.** The root model decides fan-out at run time.
  `rlm_estimate` prints what config permits (~8.7M tokens worst case with current settings)
  and what the timeout allows (~1.3M at default latency). The floor and the checkpoint are
  what make that acceptable.
- **`rlm_query` resume carries `state.dill`.** DockerREPL bind-mounts a host temp dir at
  `/workspace`; the exec runner reloads `state.dill` on every call, so writing the
  checkpointed blob into the fresh container's dir before the first turn restores every
  REPL variable. LocalREPL exposes no `temp_dir` → transcript-only resume. `state.dill`
  includes the `context` variable, so it is roughly context-sized (12.7 MB for the
  clinemm bundle) and is carried in memory on the exception for the moment of the raise.

## 5. Known limitations — please weigh these

- **No retention on `<store>/<ctx>/results/*.jsonl` (manifests) or
  `<store>/<ctx>/query/*` (checkpoints of runs never resumed).** Checkpoints are deleted on
  success and on `fresh=True`; an abandoned stop persists indefinitely. The answer cache
  and log dir have sweeps; these two do not.
- **`_scan_cache` reads every selected chunk to hash it, then the batch reads un-cached
  ones again lazily.** One extra pass over the file; flat memory. The upgrade (store the
  hash in chunk meta at chunking time) is noted in the code and not done.
- **The floor's read-not-reserve overshoot** (see §4). Acceptable at the configured
  ceiling; would matter more with a small `session_budget_tokens`.
- **`query_ceiling` omits root-turn *input*** (the REPL transcript grows per turn in a way
  config cannot bound). Stated in its docstring and output.
- **The lessons doc's coverage** — the clinemm analysis it reports read 15 of 30 chunks
  because of the `bound_output` bug this leaf fixes; the doc says so.

## 6. Verification record

**Automated.** `uv run --extra dev pytest -q` — 354 passed, 1 skipped. Two consecutive
full runs add **no** file to the real `~/.rlm` (cache, contexts, logs, ledger, state).

**Mutation checks** — each mutant fails exactly the test written for it, nothing else:

| Mutant | Caught by |
|---|---|
| window boundary `>=` → `>` | `test_the_window_boundary_is_inclusive_at_exactly_one_window_ago` |
| latency drops the concurrency factor | `test_observed_call_latency_is_measured_from_the_ledger_spacing` |
| log age cutoff off by one day | `test_the_age_cap_keeps_a_file_one_second_inside_the_window` |
| `model` dropped from the content key | `test_the_key_separates_text_prompt_and_model` |
| `possible = True` forced | `test_a_chunk_larger_than_the_stop_line_is_impossible_not_waitable`, `..._refuses_an_impossible_chunk...` |
| dir → `files` default removed | `test_a_dir_load_defaults_to_files_chunking` |
| transport floor removed | `test_the_transport_refuses_a_call_that_would_cross_the_line_without_spending` |
| `stops_run` returns False | `test_the_loop_converts_a_backend_budget_stop_into_a_resumable_limit`, `test_resume_replays_...` |
| checkpoint history not trimmed | same two |
| checkpoint not cleared on success | `test_a_stopped_query_saves_a_checkpoint_and_the_next_call_resumes_from_it` |

**Live (reconnected server, code as of `7dd9676` — before the cache and floor commits).**
`rlm_budget` reported the configured ceiling; `rlm_estimate` on the 128-file sample
returned 30 calls / ~574k tokens / ~8 min / "fits — 10%", matching the offline forecast.

**NOT verified — do before or right after landing:**

- `scripts/validate.py` — the repo's rule: run by hand before anything touching the sandbox
  or context-store path. This leaf touches both. Needs Docker up; takes minutes:
  ```
  uv run --extra dev python scripts/validate.py C:\Users\Logun\AppData\Local\Temp\rlm-val
  ```
- A real end-to-end `rlm_query` stop → resume against Docker (the engine path is tested
  with a fake environment; `state.dill` restore against a live container is not).
- A real floor hit against the live transport (tested with a fake backend only).
- The MCP server has **not** been reconnected since `2928c59`; the running one predates the
  cache, the `files` default, the floor and resume.

## 7. Surface changes

| Kind | Change |
|---|---|
| New tools | `rlm_estimate(ctx_id, prompt, max_chunks, reduce)`, `rlm_budget()` |
| Changed signatures | `rlm_sub_query_batch(+fresh)`, `rlm_query(+fresh)`, `rlm_chunk_context` (empty strategy → content-aware default) |
| `config.yaml` | `session_window_h: 5`, `session_budget_tokens: 5800000` (seed), `budget_stop_fraction: 0.95` |
| `config.py` defaults | `budget_ledger`, `budget_state`, `cache_dir`, `cache_max_bytes` (256 MB), `cache_sweep_cooldown_s` (300) |
| `pyproject.toml` | pytest marker `live` (opt out of the no-live-call guard) |
| New on-disk files | `~/.rlm/usage.jsonl` `{ts, model, itok, otok}` · `~/.rlm/budget.json` `{learned_ceiling_tokens, observed_at}` · `~/.rlm/cache/<k[:2]>/<k>.json` `{answer, itok, otok, model, ts}` + `.sweep` · `<store>/<ctx>/results/<run>.jsonl` (+`index`) · `<store>/<ctx>/query/<h>.json` `{question, root_model, sub_model, history, next_iteration, partial_answer, stopped_on}` + `.state.dill` |
| Engine (`rlm/`) | `SessionBudgetError`, `STOPS_RUN_ATTR`/`stops_run`; `completion(resume=)`; checkpoints on every limit. Listed in `rlm/UPSTREAM.md` |
| Skill | `skills/rlm-large-context/SKILL.md` — estimate-first workflow, Budget section, `files` default, `rlm_query` ceiling/resume |
| Docs | Lessons write-up published as an artifact: https://claude.ai/code/artifact/24397e12-c99b-40c7-8324-48c4b6338c7a |

## 8. Landing checklist (the loop from `TRUNK-BASED-PATTERNS.md`)

- [ ] SYNC — `git switch main && git pull --ff-only` (origin/main unchanged; no-op)
- [ ] FRESHEN — `git rebase main` on the leaf (no-op today)
- [ ] VERIFY — `uv run --extra dev pytest -q` green
- [ ] `scripts/validate.py` run against Docker (see §6)
- [ ] GATE — nothing here needs to land dark; every new path is on by default and gated by config where it spends
- [ ] REVIEW — self-on-green with this brief
- [ ] LAND — `git switch main && git merge --no-ff leaf/feat/session-budget-and-resume && git push`
- [ ] DELETE — `git branch -d leaf/feat/session-budget-and-resume`
- [ ] CONFIRM — `main` green; **reconnect the MCP server**
- [ ] After reconnect: `rlm_budget` → ceiling shown; `rlm_estimate` on an existing ctx → includes the `rlm_query` ceiling section

## 9. Self-review outcome (2026-09-03)

Five parallel reviewers went over `main...HEAD` (project-instruction adherence, a shallow
bug scan, git history, prior-PR comments, comment/doc compliance). Thirteen findings were
confirmed by reading the code, two of them by executing it. All are fixed on this branch;
each fix carries a test, and all 15 mutants written against those tests were caught.
Verify is **368 passed, 1 skipped** (was 354).

**Two claims this document made that were false, and are now corrected in place:**

- §1 said the three sub-causes were closed "for **every** model call the server makes".
  They were closed for Anthropic only. `src/auth.py` returned early for every other
  provider into a throttle-only patch that never resolved our transport, so those runs
  recorded no spend, passed no floor and learned no ceiling. **Non-Anthropic providers now
  raise `NotImplementedError` at the first model call** (`auth.require_anthropic`), read-only
  tools unaffected. `EngineClientTransport` and `_patch_throttle_only` are deleted.
- §3 said "All local edits carry `# better-rlm:`". Eight did not, including the whole
  `SessionBudgetError` class, which sat above the marker banner and so read as upstream
  code to the `grep` recipe in `rlm/UPSTREAM.md`. All eight are marked now.

**The rest, most severe first.** A transient 429 could pin the learned ceiling to local
spend and refuse every later call — the observed wall is now recorded as EVIDENCE and
reported as the number to configure, never promoted to a gate, because local spend at the
wall is a floor under the account's real ceiling, not a cap on it (`budget.ceiling`). The
engine's closing synthesis call sat outside the try, so a refusal there lost the whole run;
it is inside now, with a cursor that resumes only the synthesis. `bound_output` returned
one character of body for escape-heavy content, because it corrected by the overshoot
rather than the ratio. `max_call_tokens` excluded the reduce call, whose input is every map
answer at once. The headroom refusal compared the largest remaining chunk while announcing
that not even one fitted. The checkpoint cursor advanced mid-turn, discarding a completed
turn. `rlm_status` reported a budget stop as an auth failure. The reduce call was forecast
at half the cap it runs with. The manifest was documented in chunk order and written in
completion order. The partial-run note leaked into the reduce-failure fallback.

**Fixed outside the diff.** `scripts/validate.py` could not run at all: the venv still
contained `rlms 0.1.3`, whose `rlm` package shadowed the vendored `./rlm` for any entry
point whose `sys.path[0]` was not the repo root. Every previous run of this mandated
harness had therefore exercised the un-hardened upstream sandbox. `uv pip uninstall rlms`
fixed it; the harness now passes against Docker. `CLAUDE.md`'s claim that `rlms` is no
longer installed was false in that venv, and nothing detects it if it returns.

**Not fixed, deliberately.** The retention gaps in §5 stand. So does the double read in
`_scan_cache`. A live `rlm_query` stop→resume against Docker and a live floor hit are still
unverified.

## 10. Second review round (2026-09-03) — two defects in the budget's own premise

Reviewed again after §9, once by hand and once by driving this branch's own
`rlm_sub_query_batch` over its own diff. **The rlm-driven pass found nothing real** — 33
files mapped, 8 candidates, all 8 refuted under three independent lenses, no cross-file
defect. It also could not rediscover the first finding below, which spans two modules: a
per-file map sees one side of every seam. Worth remembering before treating a clean rlm
pass over a diff as a clean bill of health. Both real findings came from elsewhere.

**The forecast under-counted a fully resumed run to zero.** `estimate_batch` added the
reduce call under `if reduce and calls:`, where `calls` counts only UNCACHED chunks — so a
batch whose every chunk was already answered forecast 0 calls and 0 tokens, while
`batch.run`'s "nothing left to map" early return went on to issue the synthesis over the
cached answers. That return also sat above both budget gates, making the one call such a
run does make the one call nothing checked. The synthesis is now keyed on the chunk count
rather than the remaining work, priced over every answer it reads, and both gates run
before the all-cached return.

**The output cap is unenforceable on the default path, and three places assumed it held.**
`claude` accepts no output-token flag: `CliTransport` is handed `max_tokens` and has
nowhere to put it, so on the OAuth path the cap is never sent. Measured by running this
branch's own 33-chunk review at `max_output_tokens=2048` — **328,453 output tokens, 4.9x
the cap**, and 541,468 total against a 281,461 forecast, 1.9x. The forecast, the
transport's pre-call floor and the batch `Gate`'s reserve all priced output at that cap.
All three now use `budget.expected_output`, the max of the cap and the mean the ledger has
actually seen; where a transport DOES enforce the cap the measurement cannot exceed it, so
the SDK path is untouched and no auth-mode plumbing is needed. Clamped at
`_MAX_OUTPUT_FACTOR` for the reason §9 records for the learned ceiling: a reservation
derived from an unbounded observation stops being an estimate and starts refusing
everything.

Nothing found this by reading. Three static passes — five parallel reviewers, an
adversarial verifier, and the rlm map — all walked past `_prepare(messages, system, model)`
without noticing the argument it drops. It surfaced from spending real tokens and checking
the forecast against the receipt.

**Verify is 377 passed, 1 skipped** (was 368). Ten mutants written against the nine new
tests, all ten caught. Cold-start gap left standing on purpose: with fewer than
`_OUTPUT_SAMPLE_MIN` ledger records there is no measurement, so the first batch after a
fresh install is still forecast at the cap.

## 11. Third round (2026-09-03) — the output contract, and closing §5

Planned end-to-end in `docs/08-plan-subquery-output-contract.md`, which maps every open
item in this document to a leaf or to a stated refusal. Three leaves landed: `b0202a7`,
`fbde525`, `158c432`.

**No sub-model call had ever carried a system prompt.** `sub_query`, `sub_query_batch` and
both transports have always accepted one; `grep -rn "system=" src/ tests/` returned
nothing. So every chunk was answered by the `claude` CLI's default coding-assistant
persona, which explains its work — the 328,453 output tokens §10 measured at a
2,048-per-call cap. The contract sat in the user turn, prompt first and chunk after, ~10K
tokens before the generation point.

Three static passes and the rlm map in §10 all walked past this, as they walked past the
dropped `max_tokens`. Both defects were in the same four lines of `_prepare`/`_argv`, and
both surfaced only from spending real tokens.

The map now sends a SCHEMA rather than a plea — `{"findings": [...]}`, with the empty case
shown as its own example and named correct, a code-fence ban, and a clause making the
envelope beat whatever format the caller's prompt asks for, because two competing format
instructions are what produce hedging. 173 tokens, ~5,709 across a 33-chunk map. The
synthesis, `rlm_sub_query` and the auth probe get a terseness contract without the
envelope: their answers go to a reader, so an array there would destroy the deliverable.

**The cache key had to move with the contract, and that is not a detail.** `content_key`
and `run_key` hashed neither the system prompt nor anything derived from it, so every
answer cached under the old persona would have stayed cache-valid under the new one — a
resumed run serving prose for a JSON contract, and with `reduce=True` folding prose and
envelopes into one synthesis. Four narrower tests all passed with `_scan_cache` keying on
`""` while the map sent the contract; the test that catches it runs a batch, resumes it,
then moves the contract and asserts it re-asks.

**§5's two standing items are closed.** Per-chunk `sha256` is stamped into the chunk meta
at chunk time, following `byte_start`/`byte_end` exactly, so the cache scan costs one probe
read instead of a pass over every selected chunk. Trust is earned twice — the size check
the offsets already use, plus a probe read compared against the stored digest, because
without it a caller handing `set_chunks` text not matching the file could corrupt a cache
key, which it cannot do today. Neither guard is redundant: the probe only proves the chunk
it reads, so a rewrite preserving the opening chunk needs the size check. Manifests and
abandoned checkpoints are now swept on their own caps, kept longer than the cache because
they record work that was paid for and cannot be re-derived.

**Three defects and one false claim came out of the mutation runs, not the reviews.**
Adding the system prompt to the forecast billed an ABSENT one a token per call, because
`estimate_tokens` floors at 1. The two store sweep patterns each got the full byte cap,
together permitting twice the ceiling. `query/*.json` would have swept a checkpoint's
transcript and left its context-sized `state.dill` behind for good. And the age-cap test
passed with the age check deleted — expired files are also the oldest, so the byte cap
evicted them first; the same run showed the expired-file budget skip is unobservable and
its docstring no longer claims it as a safety property.

**§4's overshoot bound is re-derived** (see §4) — 48.2% of the margin at worst against the
~33% it used to claim, still inside, with the threshold that makes it matter now written
down. `query_ceiling`'s missing root-turn input and the cold-start gap stand, with reasons,
in `docs/08`.

**Verify is 394 passed, 1 skipped** (was 377). Twenty-five mutants across the three leaves,
all twenty-five caught — three only after the test that should have caught them was rebuilt.

**Still unverified, and needing real tokens rather than another reading.** The 5x output
reduction is a HYPOTHESIS: no batch has run under the new contract. Re-run the same 33
chunks and read `otok` from the run's own `sub_batch` record or the ledger, never from
`rlm_estimate`, which stays pessimistic for up to 48 hours while the verbose records age
out of `expected_output`'s window. If output does not fall by most of 5x, the diagnosis in
this section is wrong. §9's live gaps — an `rlm_query` stop→resume against Docker, and a
real floor hit — are also still open.

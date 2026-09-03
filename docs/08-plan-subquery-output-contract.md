# Plan — sub-query output contract, and closing out review 07

Covers every open item in `docs/07-review-session-budget-and-resume.md` (§4, §5, §9,
§10) plus three defects found while planning this. Three leaves, in order; each is
independently green and mergeable. Two items get no code and say why.

## Coverage map — every point in 07

| Report item | Source | Where | Action |
|---|---|---|---|
| No system prompt on any call site → 328,453 out, 4.9x cap | §10 | Leaf 1 | fix |
| `content_key`/`run_key` omit the system prompt | new | Leaf 1 | fix |
| `estimate_batch` does not count system-prompt input | new | Leaf 1 | fix |
| 48h ledger mean lags the fix by up to two days | new | Leaf 1 | doc + receipt protocol |
| A per-file map sees one side of every seam | §10 | Leaf 1 | doc (SKILL.md) |
| `_scan_cache` double read | §5 | Leaf 2 | fix |
| No retention on manifests or checkpoints | §5 | Leaf 3 | fix |
| Floor is read-and-compare, not reserve | §4, §5 | Leaf 3 | re-derive + doc, no code |
| `query_ceiling` omits root-turn input | §5 | — | no change (below) |
| Cold-start gap: <8 ledger records forecasts at the cap | §10 | — | no change (below) |
| Lessons doc coverage (15 of 30 chunks) | §5 | — | already closed in 07 |
| Live `rlm_query` stop→resume and a live floor hit unverified | §9 | Final acceptance | live verify |

## Context to load first

- `docs/07-review-session-budget-and-resume.md` §4, §5, §9, §10 — the findings this closes.
- `src/batch.py` — `run`, `one`, `_render`, `_reduce`, `_scan_cache`, `_mk_prompt`.
- `src/subquery.py` — `sub_query`, `sub_query_batch`; both already take `system`.
- `src/transport.py` `_argv`/`_prepare` — the flag plumbing, already tested in both modes.
- `src/results.py` — `content_key`, `run_key`, `sweep`.
- `src/context_store.py` — `_with_byte_offsets`, `_offsets_still_apply`, the `chunks` records.
- `TRUNK-BASED-PATTERNS.md` — the loop. Name the step as you go.

## Defect being fixed (measured)

`grep -rn "system=" src/ tests/` returns **nothing**. Every transport, `_call`,
`sub_query` and `sub_query_batch` accepts a `system` argument and no caller has ever
passed one, so every sub-model call runs under the `claude` CLI's default coding-assistant
persona. The output contract sits in the user turn — the prompt first, then the chunk
(`_mk_prompt`), i.e. ~10K tokens before the generation point, the weakest position it
could hold. Measured consequence on the 33-chunk run in §10: **328,453 output tokens at
`max_output_tokens=2048`**, 4.9x the cap, 541,468 total against a 281,461 forecast.

The cap itself is already handled — `budget.expected_output` (merged `f402914`) prices
output at what a call emits rather than the cap the CLI discards. This plan attacks the
emission, not the accounting.

## Invariants — do not violate

1. **`rlm_sub_query_batch` stays general.** It maps *any* prompt over chunks
   ("summarize each", "extract every IP"). No review-specific vocabulary in a shared
   constant — the envelope field is `findings`, which is already this codebase's word
   (`_render` builds `findings`, `_reduce` consumes it, the reduce prompt says so).
2. **An answer's cache key names everything that produced it.** `content_key`'s docstring
   is "the exact chunk text the model saw, the user's prompt, and the model that
   answered". A system prompt that changes the answer must be in the key, or that sentence
   becomes false.
3. **Only the map gets the schema.** `_reduce` and `one` render prose straight to the
   reader; a findings array there destroys the deliverable.
4. **The system prompt rides in argv, the chunk on stdin.** Keep it short.
5. **Do not "fix" a self-correcting estimate.** The cold-start gap and the 48h lag both
   err safe and resolve on their own. Clamping or hand-seeding them is how a learned
   number stops being an estimate (§9, `budget.ceiling`).

---

## Leaf 1 — `leaf/fix/subquery-system-prompt`

The behavioural fix and everything that must ship *with* it to be correct. The cache-key
change belongs in this leaf and not a later one: the moment the contract changes, every
answer cached under the old persona becomes a wrong answer that still looks valid.

### Steps

1. **Two constants in `src/batch.py`**, next to `BATCH_MAX_TOKENS`. Measured with
   `config.estimate_tokens`: `MAP_SYSTEM` 694 chars / ~173 tokens, `TERSE_SYSTEM` 242
   chars / ~60 tokens. `MAP_SYSTEM` × 33 = ~5,709 input tokens, against the ~300K output
   it targets.

   `MAP_SYSTEM` carries four load-bearing pieces beyond the shape:
   - the envelope, one raw JSON object: `{"findings": ["<one self-contained sentence>"]}`;
   - `{"findings": []}` shown as its own example and named correct and expected — the
     empty case is the one a model resists, and showing it only filled teaches it that
     filled is the goal;
   - "no code fences", or you get fenced JSON, the same padding in a new costume;
   - a precedence clause. The caller still writes "output exactly NO ISSUES"; that now
     competes with ours, and competing instructions are what produce hedging. State that
     the caller's format describes the *content* of a finding string and that the envelope
     always wins.

   `TERSE_SYSTEM` is the obedience contract with no shape: emit only the answer, no
   preamble, no restating, no reasoning, no closing summary; if the request names an
   exact string, reply with that string alone.

2. **Four call sites.**

   | site | file | system |
   |---|---|---|
   | map (×N) | `batch.py` `run` → `sub_query_batch` | `MAP_SYSTEM` |
   | reduce | `batch.py` `_reduce` → `sub_query` | `TERSE_SYSTEM` |
   | single query | `batch.py` `one` → `sub_query` | `TERSE_SYSTEM` |
   | auth probe | `server.py`, the `rlm_status` probe → `sub_query` | `TERSE_SYSTEM` |

   No config change. `cli_system_prompt_mode: "replace"` and `cli_safe_mode: True` are
   already the defaults, and `_argv` emits `--system-prompt` the moment a string arrives.
   `AnthropicTransport._kwargs` already forwards `system` when truthy, so both auth paths
   are covered with no branch.

3. **`system` into both cache keys** (`src/results.py`). `content_key(chunk_text, prompt,
   model)` and `run_key(prompt, model, strategy, n_chunks)` each gain the system text;
   thread it through `_scan_cache` from `batch.run` and `batch.estimate`. Hash the **text**,
   not a version tag — then any future edit to `MAP_SYSTEM` self-invalidates with no
   manual bump. Update `content_key`'s docstring to name the fourth component.

   This invalidates the existing answer cache exactly once, which is correct: those
   answers came from a different contract. It also means the receipt run below needs no
   `fresh=True`.

4. **`estimate_batch` counts it** (`src/budget.py`). Add `system: str = ""`, fold
   `estimate_tokens(system)` into `per_prompt_overhead`; pass it from `batch.estimate`
   and `batch.run`. An explicit parameter, not concatenation into `prompt` — that would
   be a lie the next reader has to decode. Worth two lines because the docstring already
   promises "every number in it is the number that will actually be sent".

5. **`SKILL.md` — two paragraphs, not one.**
   - *Return shape*: what a map answer looks like now, and that prompt authors should
     describe **what** to find, not how to format it. The tool owns the envelope.
   - *The seam*: `grep -n "seam\|cross-file\|per-file"` over the skill returns nothing
     today. Record §10's finding — 33 files mapped, 8 candidates, all 8 refuted, and the
     one real defect spanning two modules was invisible, because a per-chunk map sees one
     side of every seam. **A clean rlm pass over a diff is not a clean bill of health.**
     Reviewing a diff wants a reader that can hold a caller and its callee at once; that
     is not this tool.

6. **The 48h lag, recorded where it will be read.** `_PRUNE_AFTER_H = 48.0`, and
   `expected_output` is `max(cap, 48h mean)` clamped at 8×. The ledger's current mean is
   328,453/33 ≈ **9,953**, under the 16,384 clamp, so that is what gets reserved. Make the
   sub-model terse and real output may fall to a few hundred while `rlm_estimate` keeps
   quoting ~9,953 per call for up to two days. Designed behaviour, errs safe,
   self-corrects. Note it in `expected_output`'s docstring and in the acceptance step below
   so nobody reads a pessimistic forecast as the fix having failed.

### Done when

- Two tests: the map receives `MAP_SYSTEM`, the reduce receives `TERSE_SYSTEM`. Existing
  stubs take `**kw`, so nothing else breaks.
- One test: two `content_key` calls differing only in system text return different digests.
- `uv run --extra dev pytest -q` green (377 passed, 1 skipped is the current floor).
- Mutants: break each of the three fixes, confirm the intended test fails.
- **The receipt.** Re-run the same 33 chunks and compare output tokens against 328,453.
  Read `otok` from the run's own `sub_batch` log record or the ledger — **not** from the
  forecast, which stays pessimistic for 48h per step 6. This is a hypothesis until that
  number exists; 5,709 input tokens is the price of testing it. If output does not fall by
  most of 5x, the diagnosis is wrong and Leaf 1 is the wrong fix, not an incomplete one.
- Reconnect the `rlm` MCP server before measuring. Python imports `src/` once at startup.

---

## Leaf 2 — `leaf/perf/chunk-hash-in-meta`

Closes §5's second bullet: "`_scan_cache` reads every selected chunk to hash it, then the
batch reads un-cached ones again lazily. One extra pass over the file; flat memory. The
upgrade (store the hash in chunk meta at chunking time) is noted in the code and not done."

Lands directly after Leaf 1, before anyone rebuilds a large cache — see the cost note.

### Steps

1. Store `sha256(chunk_text)` per chunk at chunking time, in the existing `chunks`
   records alongside the byte offsets (`context_store._with_byte_offsets`). The pattern,
   the write path and the staleness question all already exist there; this is one more
   field, not a new schema.
2. Gate reads of the stored hash on the existing `_offsets_still_apply(meta)` guard, and
   fall back to reading and hashing when it fails. Without that gate, a file mutated under
   a live context yields a false cache **hit** — an answer served for content the model
   never saw. `read_chunk` already takes exactly this shape.
3. `content_key` consumes the stored digest instead of the chunk text; `_scan_cache` stops
   reading chunks it only needed in order to hash.
4. Absent or stale per-chunk hashes must degrade to today's behaviour, never to an error —
   contexts stored before this leaf have none.

### Known one-time cost

`content_key`'s digest changes again, so the cache invalidates a second time. There is no
way around it that does not defeat the purpose, and no semantic reason for it — which is
why this leaf lands immediately after Leaf 1, while caches are still cold from that
invalidation. Say so in the commit message.

### Done when

- A test proving one pass: count `read_chunk` calls across an `estimate` + `run` over the
  same selection and assert no chunk is read twice merely to hash it.
- A test proving the fallback: break the byte count so `_offsets_still_apply` fails, and
  assert the answer is still correct rather than a false hit.
- Verify green.

---

## Leaf 3 — `leaf/chore/store-retention`

Closes §5's first bullet and re-derives §4's overshoot bound.

### Steps

1. **One sweep helper.** `results.sweep` already carries the note: "near-duplicate of
   `logsetup._run_retention_sweep` minus the age cap. Fold the two into one helper when a
   third caller appears; refactoring a tested sweep for two callers is not yet worth its
   own risk." Manifests and checkpoints are that third caller. Fold them into one helper
   taking (root, glob, caps) and keep the `.sweep` sentinel and cooldown — Claude Code runs
   a pool of pre-warmed servers, so a directory walk per start is real cost.
2. **Sweep the two gaps.** `<store>/<ctx>/results/*.jsonl` (manifests) and
   `<store>/<ctx>/query/*` (checkpoints of runs never resumed). Checkpoints are deleted on
   success and on `fresh=True`; an abandoned stop persists indefinitely. Both live under
   the store, not `cache_dir`, which is why the existing `*/*.json` glob misses them.
3. **Config decision to make, not to guess.** These are not cache bytes and should not
   silently share `cache_max_bytes`. Either add a byte cap plus an age cap for the store's
   own artifacts, or state in config why they ride the cache's. Pick one and write down
   which; a manifest is a record of paid-for work and deserves a longer life than a
   re-derivable cache entry.
4. **Re-derive the floor's overshoot** (§4: "read-and-compare, not reserve, so overshoot
   is bounded by concurrency × one call (~3 × 32k vs a ~290k margin)"). That arithmetic
   predates `expected_output`, which raised the per-call figure from the 2,048 cap to
   ~9,953. From the measured run, 541,468/33 ≈ 16.4k per call, so ~3 × 16.4k ≈ 49k against
   the 290k margin (5% of `session_budget_tokens: 5800000`). Still comfortable, but the
   number moved: recompute it, put the current figures in §4, and change no code. The batch
   `Gate` does reserve; the floor is the fallback.

### Done when

- Manifests and checkpoints are bounded, with a test that a sweep evicts past the cap and
  never takes a `*.tmp` a peer is mid-write on.
- §4's overshoot paragraph carries post-`expected_output` numbers.
- Verify green.

---

## No change, and why

- **`query_ceiling` omits root-turn input** (§5). The REPL transcript grows per turn in a
  way config cannot bound, so there is no honest number to put here. It is disclosed in the
  docstring *and* in the rendered output, and the ledger records the real spend afterwards —
  which is the actual mitigation, and already exists. Instrumenting the engine to measure a
  growing transcript is a larger change than the disclosure is worth.
- **Cold-start gap** (§10). Below `_OUTPUT_SAMPLE_MIN = 8` ledger records there is no
  measurement worth trusting, so a fresh install forecasts the first batch at the cap and
  self-corrects after one run. Seeding it would mean inventing a measurement. Left standing,
  as §10 already decided.

## Final acceptance (after all three leaves)

1. Verify green; the receipt from Leaf 1 recorded with its actual output-token count.
2. **The live gaps from §9**, which no unit test can close: an `rlm_query` stop→resume
   against Docker, and a real floor hit. Both need Docker and real tokens; run them
   deliberately, in their own session, and record the outcome.
3. Add **§11** to `docs/07-review-session-budget-and-resume.md` in the shape of §9 and §10:
   what was found, what was fixed, what stands and why, the verify count, and the mutants
   caught. The seam finding and the receipt belong in it.
4. Reconnect the `rlm` MCP server, or the next measurement measures the old code.

## Explicitly out of scope

- Parsing the JSON envelope. The terseness lands whether or not anything reads it;
  `_render` keeps treating `r.answer` as opaque, `_reduce` folds JSON into prose as it
  already does, and the `not findings` short-circuit is unaffected — it fires only when
  every chunk errored, which `{"findings": []}` is not. A tolerant reader plus a `_render`
  change buys nothing until raw JSON in a `reduce=False` report actually bothers someone.
- A per-request schema parameter. The caller can already state its contract in prose, and
  one shared envelope is what keeps the cache key tractable.
- Building diff-review tooling inside this server. The seam problem is real and is not
  this tool's to solve (Leaf 1 step 5); a map over homogeneous chunks is the paper's
  worst-scoring row for exactly this shape of task.
- Concurrency and model-latency tuning — still out of scope, per `docs/06`.

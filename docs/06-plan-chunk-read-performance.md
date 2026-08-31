# Plan — C+D: seek-based chunk reads + lazy batch prompts

Status: **approved for implementation (C+D chosen; option A skipped — subsumed by C).**
Written 2026-09-01 from a measured investigation. This document is written to be
executed by an agent with no other conversation context.

## Context to load first (read these before editing anything)

- `CLAUDE.md` (repo root) — verify command, vendored-engine rules, server-restart note
- `TRUNK-BASED-PATTERNS.md` — leaf workflow; this plan is 3 sequential leaves
- `src/chunking.py` — `Chunk` dataclass (line ~26), `chunk_text` (line ~90)
- `src/context_store.py` — `read_text` (:368), `read_chunk` (:371), `set_chunks` (:402), `_finalize`/`meta.bytes` (:193), in-place file reference (:227)
- `src/subquery.py` — `sub_query_batch` (:56), `work()` (:70), `pool.map` (:85)
- `src/server.py` — `rlm_chunk_context` (:342, `set_chunks` call :358), `rlm_read_chunk` (:399), `rlm_sub_query` (:502), `rlm_sub_query_batch` (auto-chunk :543, prompt comprehension :550)
- `tests/test_batch_failfast.py` — fake-store pattern to copy for new tests

## Defect being fixed (measured)

`ContextStore.read_chunk` re-reads and re-decodes the **entire** context file per
chunk (`src/context_store.py:378`). `rlm_sub_query_batch` calls it N times →
O(context²) per batch, plus the full prompt list (~= whole context) is held in RAM
for the entire multi-minute batch.

| context | chunks | batch prompt build | peak RSS |
|---|---|---|---|
| 8.4 MB | 105 | 0.75 s | 25 MB |
| 21 MB | 263 | 4.69 s | 63 MB |
| 42 MB | 525 | 25.55 s | 126 MB |

Target after C+D: prompt build O(file) total spread across the batch, per-chunk
read O(chunk), steady-state RAM ≈ `subquery_concurrency × chunk_chars` regardless
of file size. (Chunk *time* still reads the full text once — inherent, out of scope.)

## Invariants — do not violate

1. Verify command is exactly `uv run --extra dev pytest -q`. Green before every merge.
2. Branching: `leaf/<type>/<slug>` off `main`, merge `--no-ff`, delete after. No worktrees.
3. Do not touch `rlm/` (vendored engine) — all changes live in `src/` and `tests/`.
4. Do not add caching of decoded text. `load_file` references user files in place
   (`src/context_store.py:227`); a text cache can go stale. Out of scope.
5. Python floor is 3.11 (`pyproject.toml`): `Path.read_text` has **no** `newline`
   parameter before 3.13 — use `path.open(encoding=..., errors=..., newline="")`.
6. Behavior contract that every leaf preserves: for any already-stored context,
   `read_chunk(ctx, i)` returns *exactly* `read_text(ctx)[start:end]` for that
   chunk's char offsets.
7. Edits to `src/` do not affect a running MCP server until it is reconnected —
   never claim a fix is live from disk state.

---

## Leaf 1 — `leaf/feat/chunk-byte-offsets` (C, write side)

Goal: every newly-chunked context stores validated byte offsets alongside char
offsets. No reader uses them yet — this leaf is inert for behavior.

### Steps

1. **Kill newline translation in `read_text`** (`src/context_store.py:368`) so
   decoded char offsets correspond 1:1 to on-disk bytes for CRLF files:

   ```python
   def read_text(self, ctx_id: str) -> str:
       # newline="": no universal-newline translation, so char offsets computed on
       # this text map to raw bytes (CRLF files keep their \r). Path.read_text has
       # no newline param before 3.13, hence open().
       with open(self.get(ctx_id).content_path, encoding="utf-8",
                 errors="replace", newline="") as fh:
           return fh.read()
   ```

   Consistency requirement: chunk char offsets (produced by `chunk_text` over this
   text) and the byte-offset walk in step 3 MUST both operate on this same
   untranslated text. Nothing else in `src/` may re-introduce translation.
   Impact check: `read_text` callers are `server.py:351,471,517,543,649` and
   `read_chunk` — all feed model prompts or the sandbox; `\r` survival is benign.
   Store-written content files are written `newline=""` already and are unaffected.

2. **Add fields to `Chunk`** (`src/chunking.py`, dataclass at ~:26):

   ```python
   byte_start: int = -1   # -1 = unknown; readers must fall back
   byte_end: int = -1
   ```

   `as_dict` uses `asdict` — picks them up automatically.

3. **Compute offsets in `set_chunks`** (`src/context_store.py:402`), not in
   `chunk_text` — the store knows the file, so this covers every caller
   (`server.py:358`, `server.py:545`, tests). Algorithm: single prefix-sum pass,
   then self-validate against the raw file size:

   ```python
   text = self.read_text(ctx_id)                      # one O(n) read, write time
   positions = sorted({p for c in chunks for p in (c["start"], c["end"])})
   byte_at, total, prev = {}, 0, 0
   for p in positions:
       total += len(text[prev:p].encode("utf-8"))
       byte_at[p] = total
       prev = p
   total += len(text[prev:].encode("utf-8"))
   if total == self.get(ctx_id).bytes:                # lossless round-trip proven
       for c in chunks:
           c["byte_start"], c["byte_end"] = byte_at[c["start"]], byte_at[c["end"]]
   # else: leave -1/-1 — readers fall back to the slow path (see Leaf 2)
   ```

   Why the validation: `errors="replace"` decoding is lossy (a bad byte → U+FFFD,
   which re-encodes to 3 bytes), so re-encoded totals drift on invalid-UTF-8 files.
   Comparing the walked total against `meta.bytes` (raw size from `_stat_path`)
   proves the mapping exact; on mismatch we store nothing and stay correct.
   NEVER compute per-chunk as `len(text[:p].encode())` — that is O(n²) at write time.

4. **Tests** (`tests/test_chunking.py` + `tests/test_context_store.py`, reuse `cfg`
   fixture from `tests/conftest.py`):
   - ASCII file: chunk → every stored chunk dict has `byte_start == start` and
     `byte_end == end` (1 byte per char).
   - Multibyte file (mix `é`, `字`): `byte_at` totals differ from char offsets and
     `raw[bs:be].decode("utf-8") == text[start:end]` for every chunk.
   - CRLF file loaded in place via `load_file`: offsets validate (thanks to step 1)
     and slices round-trip.
   - File with an invalid byte (`b"\xff"` spliced in): offsets are `-1/-1`
     (validation refused), nothing crashes.
   - Old-meta compatibility: a chunk dict *without* the new keys still loads
     (`ContextMeta` round-trip) — guards the migration path.

### Done when

- [ ] `uv run --extra dev pytest -q` green
- [ ] New chunk metas carry validated `byte_start/byte_end`; behavior otherwise unchanged
- [ ] Merged `--no-ff` to `main`, leaf deleted

---

## Leaf 2 — `leaf/feat/seek-read-chunk` (C, read side)

Goal: `read_chunk` becomes O(chunk) when offsets exist; byte-identical fallback
when they don't (old metas, invalid-UTF-8 files).

### Steps

1. **Fast path in `read_chunk`** (`src/context_store.py:371`):

   ```python
   ch = meta.chunks[index]
   bs, be = ch.get("byte_start", -1), ch.get("byte_end", -1)
   if 0 <= bs <= be:
       with open(meta.content_path, "rb") as fh:   # binary: no decode of the rest
           fh.seek(bs)                             # O(1)
           raw = fh.read(be - bs)                  # O(chunk)
       return raw.decode("utf-8", errors="replace")
   return self.read_text(ctx_id)[ch["start"]:ch["end"]]   # legacy metas
   ```

   Safe to decode a slice: boundaries came from char positions, so they always
   land between UTF-8 sequences; Leaf 1's validation already proved the mapping.

2. **Tests** (`tests/test_context_store.py`):
   - Equivalence: for ASCII, multibyte, and CRLF contexts, assert
     `read_chunk(ctx, i) == read_text(ctx)[start:end]` for **every** chunk
     (invariant 6, now exercised through the seek path).
   - Fallback: strip `byte_start/byte_end` from a stored meta (rewrite meta.json),
     assert `read_chunk` still returns the same text.
   - Fast-path proof: monkeypatch `ContextStore.read_text` to raise; `read_chunk`
     on an offset-bearing meta must still succeed (it never full-decodes).

3. **Perf sanity (manual, not a test)**: 20 MB synthetic file, chunk it, time
   `read_chunk` of the last chunk. Expect ~1 ms vs ~50+ ms before. Record the
   number in the merge commit message.

### Done when

- [ ] `uv run --extra dev pytest -q` green
- [ ] `read_chunk` never calls `read_text` for offset-bearing metas
- [ ] Merged `--no-ff` to `main`, leaf deleted

---

## Leaf 3 — `leaf/feat/lazy-batch-prompts` (D)

Goal: the batch never materializes all N prompt strings; peak prompt RAM ≈
`concurrency × chunk_chars`. Depends on Leaf 2 (per-worker `read_chunk` must be
cheap, or laziness re-introduces slow reads *serially* inside workers).

### Steps

1. **`sub_query_batch` accepts callables** (`src/subquery.py:56`) — minimal,
   backward-compatible; do NOT pass a generator to the pool (`pool.map` +
   `enumerate` submits eagerly; tiny callables are fine to materialize, 120 KB
   strings are not):

   ```python
   def sub_query_batch(prompts: Sequence[str | Callable[[], str]], ...)
   ```

   Inside `work()` (:70), after the `fatal` check, before `_call`:

   ```python
   try:
       prompt = item() if callable(item) else item   # built here, freed after call
   except Exception as exc:
       return SubResult(idx, "", 0, 0, error=f"prompt build failed — {exc}")
   ```

   Existing string-list callers (tests) keep working unchanged.

2. **Server passes builders** (`src/server.py:550`) — replace the comprehension:

   ```python
   def _mk_prompt(i: int, _n: int = n, _p: str = prompt) -> str:
       return f"{_p}\n\n--- CHUNK {i + 1}/{_n} ---\n{STORE.read_chunk(ctx_id, i)}"
   prompts = [partial(_mk_prompt, i) for i in sel]
   ```

   `len(prompts)` still works everywhere it's used (`map_start` log :557, error
   accounting, notes). In the auto-chunk branch (:543), `del text` after
   `set_chunks` so the full decode doesn't outlive chunking.

3. **Tests** (`tests/test_batch_failfast.py`, extend the fake-store pattern):
   - Laziness: builders wrap a counter; assert zero builds happen before the pool
     runs and exactly N total after (or ≤ selected when `max_chunks` caps).
   - Build failure: a builder that raises yields a `SubResult` with `error` set,
     batch continues, other chunks succeed (mirrors existing per-chunk error tests).
   - Fatal-auth skip still short-circuits *before* building (assert the counter
     stays 0 for skipped items) — keeps the dead-login fast-abort measured in
     the existing suite.
   - Existing tests still pass with plain string lists (backward compat).

### Done when

- [ ] `uv run --extra dev pytest -q` green
- [ ] No list of full prompt strings exists in `rlm_sub_query_batch`
- [ ] Merged `--no-ff` to `main`, leaf deleted

---

## Final acceptance (after all three leaves)

Synthetic check, ~20 MB line-chunked file via a scratch script (pattern: build file
→ `load_file` → `chunk_text`+`set_chunks` → time/`tracemalloc` the batch prompt
path with a stubbed `_call`):

- [ ] Per-chunk `read_chunk` ~1 ms (was O(file))
- [ ] Peak traced memory of the batch path ≈ a few chunk-sizes, NOT ≈ 3× file size
- [ ] Batch of a legacy (offset-less) meta still returns identical answers (fallback)
- [ ] Reconnect the MCP server before any live verification (imports load once)

## Explicitly out of scope

- Option A (read-once + slice at :550) — superseded by this plan.
- Caching decoded text (invariant 4).
- Streaming the chunker itself (chunk time keeps one full decode; known ceiling).
- Concurrency/model-latency tuning — after C+D the batch is model-bound:
  N calls ÷ `subquery_concurrency` × ~8 s (~23 min at 42 MB; ~$282 at 1 GB).

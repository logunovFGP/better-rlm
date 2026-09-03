# Vendored engine — provenance

This directory is a **vendored copy** of the Recursive Language Models engine.
It is source we now own and edit directly; it is no longer a pip dependency.

| | |
|---|---|
| Upstream | https://github.com/alexzhang13/rlm |
| PyPI name | `rlms` (no longer depended on) |
| Tag | `v0.1.3` |
| Commit | `72d6940142ddfb84ee6be573dc999a37e633e671` |
| Licence | MIT — see `rlm/LICENSE` |
| Copied | only the `rlm/` package; upstream's docs/examples/tests/training/visualizer are not vendored |

At the moment of copying, this tree was byte-identical to the `rlms==0.1.3`
wheel it replaced (verified with `diff -rq`), so the move carried no behaviour
change of its own.

## Merging an upstream release

```bash
git -C <upstream-clone> fetch --tags
git -C <upstream-clone> diff v0.1.3..<new-tag> -- rlm/ > /tmp/engine.patch
git apply --3way --directory=. /tmp/engine.patch   # from the repo root
```

Then update the tag/commit above, re-run `uv run --extra dev pytest -q`, and
re-check the engine's own `pyproject.toml` for dependency changes that need
mirroring into ours.

## Local changes

Every edit this fork has made to the engine is marked in source with a
`# better-rlm:` comment stating what it changes and why. Before and after a merge:

```bash
grep -rn "better-rlm" rlm/
```

The largest local additions, so a merge conflict there is recognised for what it is:

- `core/rlm.py` — `completion(..., resume=)` replays a checkpointed transcript and REPL
  state; every deliberate stop (timeout, error threshold, cancellation, session budget)
  attaches `history` / `next_iteration` / `state_dill` via `_attach_checkpoint`; a
  backend exception flagged `is_session_budget_stop` is converted to `SessionBudgetError`
  instead of escaping as a traceback. The iteration loop starts at `start_iter`, the
  checkpoint cursor advances at the TOP of each turn (so a stop in the timeout check or in
  `_compact_history` does not discard the turn before it), and the closing
  `_default_answer` synthesis is INSIDE the try, so a refusal there checkpoints like any
  other stop instead of escaping every handler.
- `utils/exceptions.py` — `SessionBudgetError`, `STOPS_RUN_ATTR`, `stops_run()`;
  earlier `FATAL_SUBCALL_ATTR`, `aborts_batch()`.
- `environments/docker_repl.py` — the hardened exec protocol (see the repo CLAUDE.md).

`tests/test_engine_limits.py` exercises the resume/conversion path against the real loop
with fakes, so a merge that silently reverts it fails there.

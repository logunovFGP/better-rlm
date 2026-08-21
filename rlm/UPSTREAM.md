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

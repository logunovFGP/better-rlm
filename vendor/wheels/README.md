# Vendored opentui wheels

`opentui` on PyPI is not usable here as published, so these are built from our
own fork and committed as files rather than declared as a dependency.

| | |
|---|---|
| Source | `logunovFGP/opentui-python` @ `f01da3f` (fork of `banditburai/opentui-python`) |
| Version | `0.1.2+scrollextent1` — upstream 0.1.2 plus one patch, PEP 440 local segment |
| Patch | `fix(scrollbox): size scroll_height from the content's real subtree extent` |
| Platform | **macOS arm64 only** |

## Why a file and not a dependency

Two reasons, and either alone is enough:

* **upstream ships no cp314 wheel**, and its sdist cannot self-build — its
  `CMakeLists.txt:91` aborts unless `scripts/download_opentui.py` has already
  fetched the native library. `pip install opentui` therefore fails outright on
  Python 3.14, which `pyproject.toml` supports and CI tests;
* **the fix is not upstream.** `0.1.2` on PyPI carries the defect.

## What the patch fixes

`ScrollBox` measured `scroll_height` from the content node's own layout height.
When a container's children overflow the height Yoga reports for it, those rows
are laid out and painted but sit below `scroll_height`: `max_scroll_y` stops
above them and `sticky_start="bottom"` pins short of the real bottom. A column
declaring `height=20` with 50 children reported `scroll_height` 20, stranding 30
rows.

It originates in the TypeScript core as `logunovFGP/opentui` @ `2837b796`
(pinned at `vendor/opentui`); this binding reimplements the component layer in
Python and so never ran that fix. `vendor/opentui`'s
`packages/core/src/tests/scrollbox-content-extent.test.ts` is the spec — all
three cases are ported into the fork's
`tests/renderables/test_scrollbox_content_extent.py`.

## Rebuilding

The wheel set must cover every Python `requires-python` allows;
`test_a_vendored_opentui_wheel_exists_for_every_supported_python` fails if it
does not.

```bash
git clone git@github.com:logunovFGP/opentui-python.git && cd opentui-python
uv run --no-project python scripts/download_opentui.py   # fetches libopentui.dylib from npm
for V in 3.12 3.13 3.14; do uv build --wheel --python $V --out-dir dist; done
```

Then copy `dist/*.whl` over the files here.

## The platform limit

These are macOS arm64, built on macOS 26 — so the tag is `macosx_26_0_arm64`
and they install nowhere else. A Linux or Windows checkout gets no opentui and
must fall back. Building for those platforms means running the two commands
above on each, since the native `libopentui.dylib`/`.so`/`.dll` differs per
platform.

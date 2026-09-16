"""``python -m src.cli`` -- TUI entry point for configuring better-rlm.

The MCP server itself runs from ``python -m src.server`` (stdio transport)
and is unaffected by this entry point. The TUI is a separate, optional
operator tool -- its only job is to configure the transport mode and the
models, and to run the verify gate (``/test``).

Mirrors the cline-2 ``apps/cli/src/index.ts`` style: a top-level launcher
that delegates to the actual REPL (``src.tui.run_repl``). Kept as a
thin module so ``run_tui.sh`` and ``run_tui.cmd`` have one obvious
``python -m src.cli`` invocation and so ``python -m src.cli --help``
prints the same docstring as ``src/tui.py``.
"""

from __future__ import annotations


def main() -> int:
    # Lazy import: importing src.tui pulls rich + the engine's import
    # graph. A bare ``python -m src.cli --help`` should not pay that cost.
    from .tui import main as tui_main
    return tui_main()


if __name__ == "__main__":
    raise SystemExit(main())

@echo off
REM Launch the RLM MCP server (stdio transport) on Windows.
REM Windows analog of run_server.sh: bash + .venv/bin/python don't exist here
REM (the venv interpreter lives under .venv\Scripts). `@echo off` keeps the
REM JSON-RPC stdout channel clean. Register with:
REM   claude mcp add -s user rlm -- cmd /c "%~dp0run_server.cmd"
setlocal
cd /d "%~dp0"
REM UTF-8 mode: the sandbox guest is Linux and reads every host-written file as
REM UTF-8, but Windows' default locale encoding is cp1252 — without this, a
REM context containing non-ASCII would fail to write. The engine now pins
REM encoding at every host write itself, so this is belt-and-braces.
set PYTHONUTF8=1
REM Windows-only venv (.venv_windows) — kept separate from the POSIX .venv_sh so a
REM WSL-shared checkout doesn't cross-clobber interpreters.
"%~dp0.venv_windows\Scripts\python.exe" -m src.server

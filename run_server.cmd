@echo off
REM Launch the RLM MCP server (stdio transport) on Windows.
REM Windows analog of run_server.sh: bash + .venv/bin/python don't exist here
REM (the venv interpreter lives under .venv\Scripts). `@echo off` keeps the
REM JSON-RPC stdout channel clean. Register with:
REM   claude mcp add -s user rlm -- cmd /c "%~dp0run_server.cmd"
setlocal
cd /d "%~dp0"
REM Windows-only venv (.venv_windows) — kept separate from the POSIX .venv_sh so a
REM WSL-shared checkout doesn't cross-clobber interpreters.
"%~dp0.venv_windows\Scripts\python.exe" -m src.server

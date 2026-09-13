@echo off
REM Launch the better-rlm operator TUI (configure mode / provider / model,
REM run tests via slash commands). Windows analog of run_tui.sh.
REM Does NOT start the MCP server -- that one is run_server.cmd.
REM Mirrors run_server.cmd: `@echo off` keeps the Python output clean,
REM the Windows-only venv lives at .venv_windows\Scripts.
setlocal
cd /d %~dp0
set PYTHONUTF8=1
"%~dp0.venv_windows\Scripts\python.exe" -m src.cli %*

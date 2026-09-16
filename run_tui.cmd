@echo off
REM Launch the better-rlm operator TUI (configure transport mode and models, run
REM tests via slash commands). Windows analog of run_tui.sh.
REM Does NOT start the MCP server -- that one is run_server.cmd.
REM
REM Two shapes ship this file and they build different virtualenvs: install.ps1
REM creates .venv_windows, while /plugin never runs it and launches the server
REM with `uv run --directory %CLAUDE_PLUGIN_ROOT%`, which resolves into .venv.
REM Prefer the installer venv, fall back to uv -- which is already the plugin's
REM stated prerequisite. --extra pdf matches the plugin's own invocation so uv
REM reuses that resolution instead of building a second one.
setlocal
cd /d %~dp0
set PYTHONUTF8=1
if exist "%~dp0.venv_windows\Scripts\python.exe" (
    "%~dp0.venv_windows\Scripts\python.exe" -m better_rlm.cli %*
    exit /b %ERRORLEVEL%
)
where uv >nul 2>&1
if %ERRORLEVEL%==0 (
    uv run --directory "%~dp0" --extra pdf python -m better_rlm.cli %*
    exit /b %ERRORLEVEL%
)
echo run_tui.cmd: no interpreter found.>&2
echo   checkout: run install.ps1 first ^(creates .venv_windows^)>&2
echo   plugin:   install uv - it is already the plugin's prerequisite>&2
exit /b 1

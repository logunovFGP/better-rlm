#Requires -Version 5.1
<#
.SYNOPSIS
    Reverse install.ps1. Idempotent, and never removes anything owned by a DIFFERENT checkout.

.DESCRIPTION
    The hard part of uninstalling is not deletion, it is ownership. install.ps1 writes state
    outside this directory - the `rlm` MCP registration and the ~/.claude/skills junction -
    and it deliberately refuses to hijack either when it already belongs to another checkout,
    which is what lets several checkouts coexist. An uninstaller that just ran
    `claude mcp remove -s user rlm` would break whichever checkout owns the registration
    right now. So every global artefact is compared against THIS directory before it is
    touched, and reported instead when it does not match.

    Removed by default (this checkout only):
      1. The `rlm` MCP registration, if it points at this checkout's run_server.cmd.
      2. The ~/.claude/skills\rlm-large-context junction, if it targets this checkout.
      3. core.hooksPath, if it is set to scripts/githooks.
      4. .venv_windows, .venv and rlm_mcp.egg-info.
      5. .env, but ONLY when byte-identical to .env.example (see below).

    Kept unless asked for, because "uninstall" must not mean "lose data":
      -PurgeData   deletes ~/.rlm - loaded contexts and logs, SHARED by every checkout and
                   the only copy of that data.
      -Image       deletes the rlm-sandbox Docker image, SHARED by every checkout.

    A .env you edited is kept and reported: it may hold CLAUDE_CODE_OAUTH_TOKEN or an API
    key, and silently deleting a credential the operator pasted is worse than leaving a file.

.PARAMETER PurgeData
    Also delete ~/.rlm (loaded contexts and logs). Irreversible.

.PARAMETER Image
    Also delete the rlm-sandbox Docker image, which other checkouts may be using.

.PARAMETER Force
    Stop processes running from .venv_windows (in practice the live rlm MCP server) so the
    directory can be deleted. Without this, a running server makes removal fail - Windows
    locks a loaded .pyd, unlike POSIX where an open file survives rm.

.EXAMPLE
    .\uninstall.ps1 -WhatIf
    Show every change without making one.

.EXAMPLE
    .\uninstall.ps1 -Force
    Uninstall, stopping the running server first if it holds .venv_windows.
#>
[CmdletBinding(SupportsShouldProcess)]
[Diagnostics.CodeAnalysis.SuppressMessageAttribute('PSAvoidUsingWriteHost', '',
    Justification = 'Interactive uninstaller progress is intentionally written to the host.')]
param(
    [switch] $PurgeData,
    [switch] $Image,
    [switch] $Force
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# --- helpers (mirrored from install.ps1 so both read the same) ---------------
function Write-Step { param([Parameter(Mandatory)][string] $Message) Write-Host "==> $Message" -ForegroundColor Cyan }
function Write-Note { param([Parameter(Mandatory)][string] $Message) Write-Host "    $Message" -ForegroundColor DarkGray }
function Test-Tool { param([Parameter(Mandatory)][string] $Name) [bool](Get-Command $Name -ErrorAction SilentlyContinue) }

function Test-DockerRunning {
    # $ErrorActionPreference is scoped: WinPS 5.1 turns a native command's redirected stderr
    # into a terminating error, so a stopped daemon would throw instead of returning $false.
    $ErrorActionPreference = 'Continue'
    if (-not (Test-Tool 'docker')) { return $false }
    & docker info *>$null
    return ($LASTEXITCODE -eq 0)
}

function Get-McpRegistration {
    # Returns `claude mcp get` output for $Name, or $null when it is not registered.
    # Same stderr caveat as Test-DockerRunning.
    param([Parameter(Mandatory)][string] $Name)
    $ErrorActionPreference = 'Continue'
    $out = & 'claude' 'mcp' 'get' $Name 2>&1 | Out-String
    if ($LASTEXITCODE -eq 0) { return $out }
    return $null
}

function Get-VenvHolder {
    # Processes running an executable from inside $Venv - in practice the rlm MCP server's
    # python.exe, which holds Lib\site-packages\*.pyd open and blocks the delete.
    # ponytail: Path-match only, same limitation as install.ps1 - a holder we cannot open
    # is also one we could not stop.
    param([Parameter(Mandatory)][string] $Venv)
    $prefix = $Venv.TrimEnd('\') + '\'
    Get-Process -ErrorAction SilentlyContinue |
        Where-Object { $_.Path -and $_.Path.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase) }
}

# 1) MCP registration --------------------------------------------------------
# First, not last: while the registration stands, Claude Code can relaunch the server at any
# moment, and one started after .venv_windows is gone fails on a missing interpreter.
Write-Step "MCP registration ('rlm', user scope)"
$launcher = Join-Path $PSScriptRoot 'run_server.cmd'
if (-not (Test-Tool 'claude')) {
    Write-Warning "claude CLI not on PATH - cannot check. If 'rlm' is registered, remove it with:`n  claude mcp remove -s user rlm"
} else {
    $reg = Get-McpRegistration 'rlm'
    if (-not $reg) {
        Write-Note 'Not registered - nothing to do.'
    } elseif ($reg -like "*$launcher*") {
        if ($PSCmdlet.ShouldProcess('rlm', 'claude mcp remove -s user')) {
            $ErrorActionPreference = 'Continue'
            & 'claude' 'mcp' 'remove' '-s' 'user' 'rlm'
            if ($LASTEXITCODE -ne 0) { Write-Warning "claude mcp remove failed (exit $LASTEXITCODE) - remove it by hand." }
            else { Write-Note 'Removed (was pointing at this checkout).' }
            $ErrorActionPreference = 'Stop'
        }
    } else {
        Write-Warning ("'rlm' is registered to a DIFFERENT checkout - left as-is. Removing it " +
            'here would uninstall that one. Inspect with: claude mcp get rlm')
    }
}

# 2) Skill junction ----------------------------------------------------------
Write-Step 'Skill junction (~/.claude/skills\rlm-large-context)'
$link = Join-Path (Join-Path $env:USERPROFILE '.claude\skills') 'rlm-large-context'
$target = Join-Path $PSScriptRoot 'skills\rlm-large-context'
$existing = Get-Item $link -ErrorAction SilentlyContinue
if (-not $existing) {
    Write-Note 'Not linked - nothing to do.'
} elseif (-not $existing.LinkType) {
    Write-Warning "$link is a real directory, not a link - left as-is (not ours to delete)."
} elseif (@($existing.Target) -contains $target) {
    if ($PSCmdlet.ShouldProcess($link, 'Delete junction')) {
        # Delete the reparse point ONLY. Remove-Item -Recurse can follow a junction and
        # delete the files it points at - here, this repo's own skills\rlm-large-context.
        # install.ps1 carries the same warning where it re-points a junction.
        [System.IO.Directory]::Delete($link)
        Write-Note "Removed $link"
    }
} else {
    Write-Warning "$link points at another checkout ($(@($existing.Target)[0])) - left as-is."
}

# 3) Verify gate -------------------------------------------------------------
Write-Step 'Verify gate (core.hooksPath)'
if (-not (Test-Tool 'git')) {
    Write-Note 'git not on PATH - skipped.'
} else {
    $ErrorActionPreference = 'Continue'
    $cur = (& git -C $PSScriptRoot config --local --get core.hooksPath 2>$null | Out-String).Trim()
    $ErrorActionPreference = 'Stop'
    if ($cur -eq 'scripts/githooks') {
        if ($PSCmdlet.ShouldProcess('core.hooksPath', 'git config --unset')) {
            & git -C $PSScriptRoot config --local --unset core.hooksPath
            Write-Note "Unset core.hooksPath ('git push' no longer runs the verify gate)."
        }
    } elseif (-not $cur) {
        Write-Note 'Not set - nothing to do.'
    } else {
        Write-Note "core.hooksPath is '$cur', not ours - left as-is."
    }
}

# 4) Python env + build artefacts -------------------------------------------
Write-Step 'Python env + build artefacts'
$venv = Join-Path $PSScriptRoot '.venv_windows'
$holders = @(if (Test-Path $venv) { Get-VenvHolder -Venv $venv } else { @() })
if ($holders.Count -gt 0) {
    $desc = ($holders | ForEach-Object { "$($_.ProcessName) (pid $($_.Id))" }) -join ', '
    if ($Force) {
        if ($PSCmdlet.ShouldProcess($desc, 'Stop-Process')) {
            $holders | Stop-Process -Force
            # Windows releases the image lock asynchronously; without this the Remove-Item
            # below can still fail with "being used by another process".
            Start-Sleep -Milliseconds 500
            Write-Note "Stopped $desc"
        }
    } else {
        Write-Warning ("$desc is running from .venv_windows and will block its removal. " +
            'Re-run with -Force to stop it, or close Claude Code first.')
    }
}
foreach ($name in @('.venv_windows', '.venv', 'rlm_mcp.egg-info')) {
    $path = Join-Path $PSScriptRoot $name
    if (-not (Test-Path $path)) {
        Write-Note "$name absent - nothing to do."
    } elseif ($PSCmdlet.ShouldProcess($path, 'Remove')) {
        try {
            Remove-Item -Recurse -Force $path
            Write-Note "Removed $name"
        } catch {
            Write-Warning "Could not remove $name : $($_.Exception.Message)"
        }
    }
}

# 5) .env -------------------------------------------------------------------
Write-Step '.env'
$envFile = Join-Path $PSScriptRoot '.env'
$envExample = Join-Path $PSScriptRoot '.env.example'
if (-not (Test-Path $envFile)) {
    Write-Note 'Absent - nothing to do.'
} else {
    # Byte comparison so the check means "did the operator add anything", nothing more.
    $same = $false
    if (Test-Path $envExample) {
        $a = [System.IO.File]::ReadAllBytes($envFile)
        $b = [System.IO.File]::ReadAllBytes($envExample)
        $same = ($a.Length -eq $b.Length) -and (-not (Compare-Object $a $b -SyncWindow 0))
    }
    if ($same -and $PSCmdlet.ShouldProcess($envFile, 'Remove')) {
        Remove-Item -Force $envFile
        Write-Note 'Removed (byte-identical to .env.example - no secrets in it).'
    } elseif (-not $same) {
        Write-Warning (".env differs from .env.example - kept, it may hold a token or API key. " +
            "Delete it yourself once you have saved anything you need:`n  Remove-Item '$envFile'")
    }
}

# 6) Loaded contexts + logs (opt-in) ----------------------------------------
Write-Step 'Store dir (~/.rlm)'
$store = Join-Path $env:USERPROFILE '.rlm'
if (-not (Test-Path $store)) {
    Write-Note 'Absent - nothing to do.'
} elseif ($PurgeData) {
    if ($PSCmdlet.ShouldProcess($store, 'Remove (loaded contexts and logs)')) {
        Remove-Item -Recurse -Force $store
        Write-Note "Removed $store (-PurgeData)."
    }
} else {
    $contexts = Join-Path $store 'contexts'
    $n = if (Test-Path $contexts) { @(Get-ChildItem $contexts -ErrorAction SilentlyContinue).Count } else { 0 }
    Write-Note "Kept $store ($n contexts) - SHARED by every checkout, and the only copy of"
    Write-Note 'that data. Delete with: .\uninstall.ps1 -PurgeData'
}

# 7) Sandbox image (opt-in) -------------------------------------------------
Write-Step 'Docker sandbox image (rlm-sandbox)'
if (-not (Test-DockerRunning)) {
    Write-Note 'Docker unavailable - skipped.'
} else {
    $ErrorActionPreference = 'Continue'
    & docker image inspect rlm-sandbox *>$null
    $present = ($LASTEXITCODE -eq 0)
    $ErrorActionPreference = 'Stop'
    if (-not $present) {
        Write-Note 'No rlm-sandbox image - nothing to do.'
    } elseif ($Image) {
        if ($PSCmdlet.ShouldProcess('rlm-sandbox', 'docker image rm')) {
            $ErrorActionPreference = 'Continue'
            & docker image rm rlm-sandbox
            if ($LASTEXITCODE -ne 0) {
                Write-Warning 'Could not remove (in use?) - try: docker image rm -f rlm-sandbox'
            }
            $ErrorActionPreference = 'Stop'
        }
    } else {
        Write-Note 'Kept - SHARED by every checkout. Delete with: .\uninstall.ps1 -Image'
    }
}

Write-Host ''
Write-Host 'Done. This checkout is no longer registered or linked; its files are still here.'
Write-Note 'Other gitignored working state (.rlm_workspace, __pycache__, .pytest_cache) is left'
Write-Note 'to git, which does it better than a hand-rolled list that would drift:'
Write-Note "  git -C '$PSScriptRoot' clean -xdn    # preview"
Write-Note "  git -C '$PSScriptRoot' clean -xdf    # delete"
Write-Note "To remove it entirely: Remove-Item -Recurse -Force '$PSScriptRoot'"

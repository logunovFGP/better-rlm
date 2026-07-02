#Requires -Version 5.1
<#
.SYNOPSIS
    One-shot Windows setup for the RLM MCP server (PowerShell equivalent of install.sh).

.DESCRIPTION
    Setup steps:
      1. (Re)creates the Windows virtual environment (.venv_windows) from scratch.
      2. Installs pinned dependencies (editable install).
      3. Builds the Docker sandbox image (rlm-sandbox) unless skipped.
      4. Creates .env from .env.example if missing (optional; mode=auto needs no key).
      5. Links the rlm-large-context skill into ~/.claude/skills (directory junction).
      6. Prints - or, with -Register, runs - the `claude mcp add` command.

    The venv is rebuilt fresh every run: reusing an existing venv proved unreliable on a
    Windows+WSL shared checkout, whereas a clean create is deterministic. Per-platform
    names (.venv_windows here, .venv_sh for install.sh) let both OSes coexist in one folder.

.PARAMETER PythonVersion
    Python version for the venv. The rlms engine requires >=3.11,<3.14. Default: 3.13.

.PARAMETER Sandbox
    Sandbox backend the server expects: 'docker' (default) or 'local'. 'local' skips the image build.

.PARAMETER SkipDocker
    Do not build the Docker sandbox image.

.PARAMETER SkipSkill
    Do not create the rlm-large-context skill link.

.PARAMETER Register
    Run `claude mcp add -s user rlm ...` after setup (requires the claude CLI on PATH).

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File install.ps1

.EXAMPLE
    .\install.ps1 -Register -Verbose

.EXAMPLE
    .\install.ps1 -PythonVersion 3.12 -Sandbox local

.NOTES
    Windows counterpart of install.sh. Pair with run_server.cmd (the MCP stdio launcher).
    Run:  powershell -ExecutionPolicy Bypass -File install.ps1
#>
[CmdletBinding(SupportsShouldProcess)]
[Diagnostics.CodeAnalysis.SuppressMessageAttribute('PSAvoidUsingWriteHost', '',
    Justification = 'Interactive installer progress is intentionally written to the host.')]
param(
    [ValidatePattern('^3\.(11|12|13)$')]
    [string] $PythonVersion = '3.13',

    [ValidateSet('docker', 'local')]
    [string] $Sandbox = 'docker',

    [switch] $SkipDocker,
    [switch] $SkipSkill,
    [switch] $Register
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# --- helpers ---------------------------------------------------------------
function Write-Step { param([Parameter(Mandatory)][string] $Message) Write-Host "==> $Message" -ForegroundColor Cyan }
function Write-Note { param([Parameter(Mandatory)][string] $Message) Write-Host "    $Message" -ForegroundColor DarkGray }
function Test-Tool { param([Parameter(Mandatory)][string] $Name) [bool](Get-Command $Name -ErrorAction SilentlyContinue) }

function Invoke-Native {
    # Run a native command and fail loudly on a non-zero exit code. $ErrorActionPreference
    # does NOT catch native (non-cmdlet) failures on Windows PowerShell / pre-7.4 pwsh.
    param(
        [Parameter(Mandatory)][scriptblock] $Command,
        [Parameter(Mandatory)][string] $What
    )
    & $Command
    if ($LASTEXITCODE -ne 0) { throw "$What failed (exit code $LASTEXITCODE)." }
}

# --- setup -----------------------------------------------------------------
Push-Location $PSScriptRoot
try {
    # Windows-only venv name. This checkout may be shared with WSL/Linux, whose POSIX
    # venv (.venv_sh, bin/python) is NOT interchangeable with a Windows venv. Separate
    # names let both coexist in one folder without clobbering each other.
    $venv = Join-Path $PSScriptRoot '.venv_windows'
    $python = Join-Path $venv 'Scripts\python.exe'

    # 1) Virtual environment + dependencies -------------------------------
    Write-Step "Python env (.venv_windows, $PythonVersion) + dependencies"

    $hasUv = Test-Tool 'uv'
    if (-not $hasUv -and -not (Test-Tool 'py') -and -not (Test-Tool 'python')) {
        throw "Neither 'uv' nor Python found on PATH. Install uv (https://docs.astral.sh/uv/) or Python $PythonVersion."
    }

    # Always rebuild the venv fresh. Reusing an existing venv proved flaky on this
    # WSL-shared checkout; a clean create is reliable and deterministic.
    if (Test-Path $venv) {
        if ($PSCmdlet.ShouldProcess($venv, 'Remove existing virtual environment')) {
            try {
                Remove-Item -Recurse -Force $venv
            } catch {
                throw ("Could not remove '$venv' - it is likely in use by the running rlm MCP " +
                    "server. Stop Claude Code (or disconnect 'rlm' via /mcp), then re-run. " +
                    "Details: $($_.Exception.Message)")
            }
        }
    }
    if ($PSCmdlet.ShouldProcess($venv, "Create Python $PythonVersion virtual environment")) {
        if ($hasUv) { Invoke-Native { uv venv --python $PythonVersion $venv } 'uv venv' }
        else { Invoke-Native { & 'py' "-$PythonVersion" -m venv $venv } "py -$PythonVersion -m venv" }
    }
    if ($PSCmdlet.ShouldProcess('dependencies', 'Install (editable)')) {
        if ($hasUv) {
            Invoke-Native { uv pip install --python $python -e '.[dev,pdf]' } 'uv pip install'
        } else {
            Invoke-Native { & $python -m pip install --upgrade pip } 'pip upgrade'
            Invoke-Native { & $python -m pip install -e '.[dev,pdf]' } 'pip install'
        }
    }

    # 2) Docker sandbox image ---------------------------------------------
    if ($SkipDocker -or $Sandbox -eq 'local') {
        Write-Step "Skipping Docker image (Sandbox=$Sandbox, SkipDocker=$SkipDocker)"
        Write-Note "Set 'sandbox: local' in config.yaml to run on the host (see README Security)."
    } else {
        Write-Step 'Docker sandbox image (rlm-sandbox)'
        if (Test-Tool 'docker') {
            docker info *> $null
            if ($LASTEXITCODE -eq 0) {
                if ($PSCmdlet.ShouldProcess('rlm-sandbox', 'docker build')) {
                    Invoke-Native { docker build -t rlm-sandbox -f 'docker/Dockerfile.sandbox' 'docker/' } 'docker build'
                }
            } else {
                Write-Warning 'Docker is installed but not running. Start Docker Desktop and re-run, or use -Sandbox local.'
            }
        } else {
            Write-Warning 'docker not found on PATH. Install Docker Desktop, or use -Sandbox local.'
        }
    }

    # 3) .env (optional) ---------------------------------------------------
    Write-Step '.env (optional - mode=auto reuses your Claude Code login)'
    if (Test-Path '.env') {
        Write-Note '.env already present - left unchanged.'
    } elseif ($PSCmdlet.ShouldProcess('.env', 'Create from .env.example')) {
        Copy-Item '.env.example' '.env'
        Write-Note 'Created .env (only needed for mode: api - add ANTHROPIC_API_KEY there).'
    }

    # 4) Skill link --------------------------------------------------------
    if ($SkipSkill) {
        Write-Step 'Skipping skill link (-SkipSkill)'
    } else {
        Write-Step 'Skill (rlm-large-context)'
        $skillsDir = Join-Path $env:USERPROFILE '.claude\skills'
        $null = New-Item -ItemType Directory -Force -Path $skillsDir
        $link = Join-Path $skillsDir 'rlm-large-context'
        $target = Join-Path $PSScriptRoot 'skills\rlm-large-context'
        $existing = Get-Item $link -ErrorAction SilentlyContinue
        if ($existing -and -not $existing.LinkType) {
            Write-Warning "$link exists and is not a link - left as-is. Remove it and re-run to link."
        } elseif ($existing) {
            Write-Note 'Skill link already present.'
        } elseif ($PSCmdlet.ShouldProcess($link, 'Create junction to repo skill')) {
            $null = New-Item -ItemType Junction -Path $link -Target $target
            Write-Note "Linked $link -> $target"
        }
    }

    # 5) Register with Claude Code ----------------------------------------
    Write-Step 'Register with Claude Code'
    $launcher = Join-Path $PSScriptRoot 'run_server.cmd'
    $registerCmd = "claude mcp add -s user rlm -- cmd /c `"$launcher`""
    if ($Register) {
        if (-not (Test-Tool 'claude')) {
            Write-Warning "claude CLI not found on PATH - cannot auto-register. Run manually:`n  $registerCmd"
        } elseif ($PSCmdlet.ShouldProcess('rlm', 'claude mcp add')) {
            Invoke-Native { & 'claude' 'mcp' 'add' '-s' 'user' 'rlm' '--' 'cmd' '/c' $launcher } 'claude mcp add'
            Write-Note 'Registered. Restart Claude Code to load the server and skill.'
        }
    } else {
        Write-Note 'Run this to register (or re-run install.ps1 with -Register):'
        Write-Host "  $registerCmd" -ForegroundColor Green
        Write-Note "Then restart Claude Code so the 'rlm' server and 'rlm-large-context' skill load."
    }

    Write-Host ''
    Write-Host 'RLM MCP setup complete.' -ForegroundColor Green
} finally {
    Pop-Location
}

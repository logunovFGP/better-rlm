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
    [switch] $Register,

    # Stop processes holding .venv_windows (the running rlm MCP server) so a rebuild can
    # proceed. Only consulted when dependencies actually changed. Also pre-answers the
    # interactive prompt for that case.
    [switch] $Force,

    # Never prompt; take the documented default for every decision. Set this in CI. The
    # script also detects a non-interactive host on its own, so this is belt-and-braces.
    [switch] $NonInteractive
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# --- helpers ---------------------------------------------------------------
function Write-Step { param([Parameter(Mandatory)][string] $Message) Write-Host "==> $Message" -ForegroundColor Cyan }
function Write-Note { param([Parameter(Mandatory)][string] $Message) Write-Host "    $Message" -ForegroundColor DarkGray }
function Test-Tool { param([Parameter(Mandatory)][string] $Name) [bool](Get-Command $Name -ErrorAction SilentlyContinue) }

function Test-DockerRunning {
    # Probe the daemon WITHOUT letting a stopped daemon abort the install. Windows
    # PowerShell 5.1 turns a native command's *redirected* stderr into a terminating
    # NativeCommandError while $ErrorActionPreference is 'Stop' (pwsh 7 does not), so
    # `docker info *> $null` used inline would kill the script instead of falling
    # through to the warning below. Scoping the preference to this function keeps a
    # down daemon a plain $false on both editions.
    $ErrorActionPreference = 'Continue'
    docker info *> $null
    return ($LASTEXITCODE -eq 0)
}

function Get-Choice {
    # One prompt for every decision this script used to settle by printing a warning and
    # moving on. Returns the chosen option index.
    #
    # Falls back to $DefaultChoice whenever asking is impossible or wrong: -NonInteractive,
    # a service/redirected host, CI, or -WhatIf. Each call site sets $DefaultChoice to the
    # behaviour this script had BEFORE prompting existed, so a scripted or CI run is
    # unchanged and can never block on a question nobody is there to answer.
    param(
        [Parameter(Mandatory)][string] $Title,
        [Parameter(Mandatory)][string] $Message,
        [Parameter(Mandatory)][string[]] $Options,   # '&Yes' - & marks the hotkey
        [int] $DefaultChoice = 0
    )
    if (-not $script:CanPrompt) {
        Write-Note "$Title - assuming '$($Options[$DefaultChoice] -replace '&', '')' (not interactive)."
        return $DefaultChoice
    }
    $descs = @($Options | ForEach-Object {
        New-Object System.Management.Automation.Host.ChoiceDescription $_, $_
    })
    try {
        return $Host.UI.PromptForChoice($Title, $Message, $descs, $DefaultChoice)
    } catch {
        # Host advertises UI but cannot actually prompt - treat as non-interactive.
        return $DefaultChoice
    }
}

function Get-DepHash {
    # Fingerprint of everything that decides what ends up in the venv. Stored inside the
    # venv so a re-run with unchanged dependencies skips the rebuild entirely - which is
    # what keeps a running server from turning `install.ps1` into a hard failure.
    param(
        [Parameter(Mandatory)][string] $Root,
        [Parameter(Mandatory)][string] $PyVersion
    )
    $parts = @($PyVersion)
    foreach ($f in 'pyproject.toml', 'uv.lock') {
        $p = Join-Path $Root $f
        if (Test-Path $p) { $parts += (Get-FileHash -LiteralPath $p -Algorithm SHA256).Hash }
    }
    $bytes = [System.Text.Encoding]::UTF8.GetBytes(($parts -join "`n"))
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try { return ([System.BitConverter]::ToString($sha.ComputeHash($bytes)) -replace '-', '') }
    finally { $sha.Dispose() }
}

function Get-VenvHolder {
    # Processes running an executable from inside $Venv - in practice the rlm MCP server's
    # python.exe, which is what holds Lib\site-packages\*.pyd open.
    # ponytail: Path-match only. .Path is empty for processes we cannot open, but a holder
    # we cannot see is also one we could not stop; use Sysinternals handle.exe if a
    # non-python holder ever matters.
    param([Parameter(Mandatory)][string] $Venv)
    $prefix = $Venv.TrimEnd('\') + '\'
    Get-Process -ErrorAction SilentlyContinue |
        Where-Object { $_.Path -and $_.Path.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase) }
}

function Get-McpRegistration {
    # Returns `claude mcp get` output for $Name, or $null when it is not registered.
    # Two callers need this: -Register must probe before adding (`claude mcp add` exits 1
    # on an existing name, and Invoke-Native turns that into a throw, which aborted every
    # re-run), and the status report needs the text to tell WHICH checkout is registered.
    # $ErrorActionPreference is scoped for the same reason as in Test-DockerRunning:
    # WinPS 5.1 turns a native command's redirected stderr into a terminating error.
    param([Parameter(Mandatory)][string] $Name)
    $ErrorActionPreference = 'Continue'
    $out = & 'claude' 'mcp' 'get' $Name 2>&1 | Out-String
    if ($LASTEXITCODE -eq 0) { return $out }
    return $null
}

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
# Can we actually ask a human? A prompt in a service, a redirected pipeline, or CI would
# hang forever, and -WhatIf must stay side-effect free, so refuse to prompt in all of them.
# [Environment]::UserInteractive stays $true under a redirected pipeline, where
# PromptForChoice throws instead of asking, so test the stream itself as well.
$script:CanPrompt = -not $NonInteractive -and -not $WhatIfPreference -and
    [Environment]::UserInteractive -and $null -ne $Host.UI -and
    -not [Console]::IsInputRedirected -and
    -not $env:CI -and -not $env:TF_BUILD -and -not $env:GITHUB_ACTIONS

# `local` means the sandbox is skipped AND the server must be told at registration time,
# otherwise rlm_exec/rlm_query try to reach a Docker daemon that was never used here. The
# Docker prompt can flip this on the fly.
$script:UseLocalSandbox = ($Sandbox -eq 'local')

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

    # Rebuild only when the inputs that decide venv contents actually changed. The old
    # unconditional rebuild deleted a venv it usually did not need to, which failed
    # outright whenever the registered rlm MCP server held the interpreter open - the
    # common case, since installing is exactly when a server is already running.
    $stamp = Join-Path $venv '.rlm-deps-sha256'
    $wantHash = Get-DepHash -Root $PSScriptRoot -PyVersion $PythonVersion
    $haveHash = if (Test-Path $stamp) { (Get-Content $stamp -Raw).Trim() } else { $null }

    if ((Test-Path $python) -and $haveHash -eq $wantHash) {
        Write-Note 'Dependencies unchanged (deps hash matches) - venv reused, nothing to rebuild.'
    } else {
        if (Test-Path $venv) {
            # Free the interpreter first when asked, so a running server is a handled
            # condition rather than a hard stop. Without -Force this reports the holders
            # instead of killing anything: stopping a live Claude Code session is the
            # operator's call, not an installer's.
            $holders = @(Get-VenvHolder -Venv $venv)
            if ($holders.Count) {
                $list = ($holders | ForEach-Object { "PID $($_.Id) ($($_.ProcessName))" }) -join ', '
                # -Force pre-answers this; otherwise ask. Default is Cancel, which keeps the
                # old behaviour for scripted runs: never kill a session nobody approved.
                $stop = $Force -or 0 -eq (Get-Choice -Title 'Virtual environment is in use' `
                    -Message ("Dependencies changed, so '$venv' must be rebuilt, but it is held by: $list`n" +
                        "This is the running rlm MCP server. Stopping it ends the connection until " +
                        "Claude Code restarts it.") `
                    -Options '&Stop them and rebuild', '&Cancel' -DefaultChoice 1)
                if ($stop) {
                    foreach ($h in $holders) {
                        if ($PSCmdlet.ShouldProcess("PID $($h.Id) ($($h.ProcessName))", 'Stop process holding the venv')) {
                            Stop-Process -Id $h.Id -Force -ErrorAction SilentlyContinue
                            Write-Note "Stopped PID $($h.Id) ($($h.ProcessName))"
                        }
                    }
                    Start-Sleep -Milliseconds 750   # let Windows release the file handles
                } else {
                    $msg = ("Dependencies changed, so '$venv' must be rebuilt - but it is in use by: " +
                        "$list. Stop Claude Code (or disconnect 'rlm' via /mcp) and re-run, or re-run " +
                        "with -Force to stop those processes automatically.")
                    # -WhatIf must describe, never fail: nothing is being deleted in a dry run.
                    if ($WhatIfPreference) { Write-Warning $msg } else { throw $msg }
                }
            }
            if ($PSCmdlet.ShouldProcess($venv, 'Remove existing virtual environment')) {
                try {
                    Remove-Item -Recurse -Force $venv
                } catch {
                    throw ("Could not remove '$venv' - something still holds a file in it. Stop " +
                        "Claude Code (or disconnect 'rlm' via /mcp), then re-run; -Force stops the " +
                        "processes this script can see. Details: $($_.Exception.Message)")
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
            # Written last: a stamp only means anything if the install above succeeded.
            Set-Content -LiteralPath $stamp -Value $wantHash -Encoding ASCII
        }
    }

    # 2) Docker sandbox image ---------------------------------------------
    if ($SkipDocker -or $script:UseLocalSandbox) {
        Write-Step "Skipping Docker image (Sandbox=$Sandbox, SkipDocker=$SkipDocker)"
        Write-Note 'Model-written Python will run on the HOST (see README Security).'
    } else {
        Write-Step 'Docker sandbox image (rlm-sandbox)'
        if (-not (Test-Tool 'docker')) {
            Write-Warning 'docker not found on PATH. Install Docker Desktop, or use -Sandbox local.'
        } else {
            # Retry in place rather than making the operator re-run the whole installer just
            # because Docker Desktop was still starting.
            $asking = $true
            while ($asking) {
                $asking = $false
                if (Test-DockerRunning) {
                    if ($PSCmdlet.ShouldProcess('rlm-sandbox', 'docker build')) {
                        Invoke-Native { docker build -t rlm-sandbox -f 'docker/Dockerfile.sandbox' 'docker/' } 'docker build'
                    }
                } else {
                    switch (Get-Choice -Title 'Docker is installed but not running' `
                        -Message ("The sandbox image cannot be built. Start Docker Desktop and retry, " +
                            "or run without a sandbox - which executes model-written Python on this host.") `
                        -Options '&Retry (I started Docker)', 'Run with &local sandbox', '&Skip the image build' `
                        -DefaultChoice 2) {
                        0 { $asking = $true }
                        1 {
                            $script:UseLocalSandbox = $true
                            Write-Note 'Using the local sandbox: registration will set RLM_SANDBOX=local.'
                            Write-Warning 'Local sandbox runs model-written Python on this host - trusted inputs only.'
                        }
                        default {
                            Write-Warning ('No rlm-sandbox image: rlm_exec/rlm_query will fail until Docker ' +
                                'is running and install.ps1 is re-run, or RLM_SANDBOX=local is set.')
                        }
                    }
                }
            }
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
            # A link may point at ANOTHER checkout. Reporting a bare "already present"
            # served a stale SKILL.md while looking like success. .Target is string[] on
            # WinPS 5.1 and a string on pwsh 7, so normalise before comparing.
            if (@($existing.Target) -contains $target) {
                Write-Note 'Skill link already present (this checkout).'
            } else {
                $other = @($existing.Target)[0]
                # Default Leave = the old behaviour, so scripted runs never silently
                # steal a link another checkout owns.
                if (0 -eq (Get-Choice -Title 'Skill link points at another checkout' `
                        -Message ("It targets $other, so Claude is served that checkout's SKILL.md, " +
                            "not this one.") `
                        -Options '&Re-point it here', '&Leave it' -DefaultChoice 1)) {
                    if ($PSCmdlet.ShouldProcess($link, 'Re-point junction to this checkout')) {
                        # Delete the reparse point only. Remove-Item -Recurse on a junction
                        # can follow it and delete the OTHER checkout's files.
                        [System.IO.Directory]::Delete($link)
                        $null = New-Item -ItemType Junction -Path $link -Target $target
                        Write-Note "Re-pointed $link -> $target"
                    }
                } else {
                    Write-Warning "Skill link left at $other - a stale skill is being served."
                }
            }
        } elseif ($PSCmdlet.ShouldProcess($link, 'Create junction to repo skill')) {
            $null = New-Item -ItemType Junction -Path $link -Target $target
            Write-Note "Linked $link -> $target"
        }
    }

    # 5) Verify gate ------------------------------------------------------
    Write-Step 'Verify gate (git pre-push hook)'
    # A hook in .git/hooks is untracked and never reaches a clone, which is why the
    # "enforced by .git/hooks/pre-push" claim was false for every fresh checkout.
    # Point git at the version-controlled directory instead.
    if (Test-Path (Join-Path $PSScriptRoot '.git')) {
        if ($PSCmdlet.ShouldProcess('core.hooksPath', 'git config')) {
            Invoke-Native { git -C $PSScriptRoot config core.hooksPath scripts/githooks } 'git config core.hooksPath'
            Write-Note "core.hooksPath -> scripts/githooks ('git push' now runs the verify gate)"
        }
    } else {
        Write-Note 'Not a git checkout - skipped.'
    }

    # 6) Register with Claude Code ----------------------------------------
    Write-Step 'Register with Claude Code'
    $launcher = Join-Path $PSScriptRoot 'run_server.cmd'
    # A local sandbox is useless unless the server is told at launch, so carry the choice
    # made above (or -Sandbox local) into the registration itself.
    $envArgs = if ($script:UseLocalSandbox) { @('-e', 'RLM_SANDBOX=local') } else { @() }
    $registerCmd = 'claude mcp add -s user rlm ' +
        (($envArgs -join ' ') + ' ').TrimStart() + "-- cmd /c `"$launcher`""
    $addArgs = @('mcp', 'add', '-s', 'user', 'rlm') + $envArgs + @('--', 'cmd', '/c', $launcher)

    $hasClaude = Test-Tool 'claude'
    $reg = if ($hasClaude) { Get-McpRegistration 'rlm' } else { $null }

    if (-not $hasClaude) {
        Write-Warning "claude CLI not found on PATH. Once installed, run:`n  $registerCmd"
    } elseif ($reg -like "*$launcher*") {
        Write-Note "'rlm' already registered to THIS checkout - nothing to do."
    } elseif (-not $reg) {
        # Nothing registered: the server cannot load however often Claude Code restarts.
        # -Register pre-answers; default No keeps scripted runs from touching global state.
        if ($Register -or 0 -eq (Get-Choice -Title "'rlm' is not registered" `
                -Message "Without it the server never loads. Register this checkout now?`n  $registerCmd" `
                -Options '&Register now', '&Not now' -DefaultChoice 1)) {
            if ($PSCmdlet.ShouldProcess('rlm', 'claude mcp add')) {
                Invoke-Native { & 'claude' @addArgs } 'claude mcp add'
                Write-Note 'Registered.'
            }
        } else {
            Write-Warning "'rlm' is NOT registered - the server will not load. Run:`n  $registerCmd"
        }
    } else {
        # Registered, but to another checkout - this one will not be used. Default Leave,
        # so a scripted run never hijacks a registration it does not own.
        if (0 -eq (Get-Choice -Title "'rlm' points at a different checkout" `
                -Message "This checkout will not be used. Re-point 'rlm' here?" `
                -Options '&Re-point here', '&Leave it' -DefaultChoice 1)) {
            if ($PSCmdlet.ShouldProcess('rlm', 'claude mcp remove + add')) {
                Invoke-Native { & 'claude' 'mcp' 'remove' '-s' 'user' 'rlm' } 'claude mcp remove'
                Invoke-Native { & 'claude' @addArgs } 'claude mcp add'
                Write-Note 'Re-pointed to this checkout.'
            }
        } else {
            Write-Warning ("'rlm' stays registered to a DIFFERENT checkout - this one will not " +
                "be used. To switch:`n  claude mcp remove -s user rlm`n  $registerCmd")
        }
    }
    Write-Note "Restart Claude Code so the 'rlm' server and 'rlm-large-context' skill load."

    Write-Host ''
    Write-Host 'RLM MCP setup complete.' -ForegroundColor Green
} finally {
    Pop-Location
}

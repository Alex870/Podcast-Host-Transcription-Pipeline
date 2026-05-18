param(
    [ValidateSet("Prompt", "Run", "Debug")]
    [string]$Action = "Prompt"
)

$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$RunScript = Join-Path $ScriptRoot "scripts\Convert-AudioToDiarizedText.ps1"
$DebugScript = Join-Path $ScriptRoot "scripts\Debug-PodcastTranscribeEnvironment.ps1"

function Invoke-LauncherScript {
    param(
        [string]$Path
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Launcher script not found: $Path"
    }

    & $Path
}

if ($Action -eq "Prompt") {
    Write-Host ""
    Write-Host "Podcast Host Transcription Pipeline"
    Write-Host "Choose what to run:"
    Write-Host "  1. Run environment validation (debug)"
    Write-Host "  2. Run transcription pipeline"
    Write-Host "  Q. Quit"
    $selection = (Read-Host "Enter 1, 2, or Q").Trim()

    switch ($selection.ToUpperInvariant()) {
        "1" { $Action = "Debug" }
        "2" { $Action = "Run" }
        "Q" { return }
        default {
            Write-Host "Unrecognized selection. Exiting."
            return
        }
    }
}

switch ($Action) {
    "Debug" { Invoke-LauncherScript -Path $DebugScript }
    "Run" { Invoke-LauncherScript -Path $RunScript }
}

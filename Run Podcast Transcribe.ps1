param(
    [ValidateSet("Prompt", "Run", "Debug", "Migrate", "Benchmark", "Workbench")]
    [string]$Action = "Prompt"
)

$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$RunScript = Join-Path $ScriptRoot "scripts\Convert-AudioToDiarizedText.ps1"
$DebugScript = Join-Path $ScriptRoot "scripts\Debug-PodcastTranscribeEnvironment.ps1"
$MigrateScript = Join-Path $ScriptRoot "scripts\Migrate-LegacyPodcastTranscribeState.ps1"
$WorkbenchScript = Join-Path $ScriptRoot "scripts\Launch-PodcastTranscribeWorkbench.ps1"

function Invoke-LauncherScript {
    param(
        [string]$Path,
        [switch]$ReviewBenchmark
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Launcher script not found: $Path"
    }

    if ($ReviewBenchmark) {
        & $Path -ReviewBenchmark
    } else {
        & $Path
    }
}

if ($Action -eq "Prompt") {
    Write-Host ""
    Write-Host "Podcast Host Transcription Pipeline"
    Write-Host "Choose what to run:"
    Write-Host "  1. Run environment validation (debug)"
    Write-Host "  2. Run transcription pipeline"
    Write-Host "  3. Migrate settings and state from a legacy directory"
    Write-Host "  4. Run review benchmark"
    Write-Host "  5. Launch transcript review workbench"
    Write-Host "  Q. Quit"
    $selection = (Read-Host "Enter 1, 2, 3, 4, 5, or Q").Trim()

    switch ($selection.ToUpperInvariant()) {
        "1" { $Action = "Debug" }
        "2" { $Action = "Run" }
        "3" { $Action = "Migrate" }
        "4" { $Action = "Benchmark" }
        "5" { $Action = "Workbench" }
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
    "Migrate" { Invoke-LauncherScript -Path $MigrateScript }
    "Benchmark" { Invoke-LauncherScript -Path $RunScript -ReviewBenchmark }
    "Workbench" { Invoke-LauncherScript -Path $WorkbenchScript }
}

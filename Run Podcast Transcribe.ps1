param(
    [ValidateSet("Prompt", "Run", "Debug", "Migrate", "Benchmark", "PipelineBenchmark", "DownloadModels", "AnonymousMeeting", "Workbench", "ConfigureLLM", "PartitionManager")]
    [string]$Action = "Prompt"
)

$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$RunScript = Join-Path $ScriptRoot "scripts\Convert-AudioToDiarizedText.ps1"
$DebugScript = Join-Path $ScriptRoot "scripts\Debug-PodcastTranscribeEnvironment.ps1"
$MigrateScript = Join-Path $ScriptRoot "scripts\Migrate-LegacyPodcastTranscribeState.ps1"
$WorkbenchScript = Join-Path $ScriptRoot "scripts\Launch-PodcastTranscribeWorkbench.ps1"
$ConfigureLlmScript = Join-Path $ScriptRoot "scripts\Configure-PodcastTranscribeReviewBackend.ps1"

function Invoke-LauncherScript {
    param(
        [string]$Path,
        [switch]$ReviewBenchmark,
        [switch]$PipelineBenchmark,
        [switch]$DownloadProviderModels,
        [ValidateSet("podcast", "anonymous_meeting")]
        [string]$WorkflowProfile = "podcast"
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Launcher script not found: $Path"
    }

    if ($ReviewBenchmark) {
        & $Path -ReviewBenchmark
    } elseif ($PipelineBenchmark) {
        & $Path -PipelineBenchmark
    } elseif ($DownloadProviderModels) {
        & $Path -DownloadProviderModels
    } elseif ($WorkflowProfile -eq "anonymous_meeting") {
        & $Path -WorkflowProfile anonymous_meeting
    } else {
        & $Path
    }
}

function Read-LauncherAction {
    Write-Host ""
    Write-Host "Podcast Host Transcription Pipeline"
    Write-Host "Choose what to run:"
    Write-Host "  1. Run environment validation (debug)"
    Write-Host "  2. Run transcription pipeline"
    Write-Host "  3. Migrate settings and state from a legacy directory"
    Write-Host "  4. Run review benchmark"
    Write-Host "  5. Launch transcript review workbench"
    Write-Host "  6. Run pipeline quality benchmark"
    Write-Host "  7. Configure external review LLM"
    Write-Host "  8. Download pinned transcription models"
    Write-Host "  9. Transcribe committee meeting (anonymous speakers)"
    Write-Host " 10. Manage processing spaces"
    Write-Host "  Q. Quit"
    $selection = (Read-Host "Enter 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, or Q").Trim()

    switch ($selection.ToUpperInvariant()) {
        "1" { return "Debug" }
        "2" { return "Run" }
        "3" { return "Migrate" }
        "4" { return "Benchmark" }
        "5" { return "Workbench" }
        "6" { return "PipelineBenchmark" }
        "7" { return "ConfigureLLM" }
        "8" { return "DownloadModels" }
        "9" { return "AnonymousMeeting" }
        "10" { return "PartitionManager" }
        "Q" { return "Quit" }
        default {
            Write-Host "Unrecognized selection. Please try again." -ForegroundColor Yellow
            return "Prompt"
        }
    }
}

function Invoke-SelectedLauncherAction {
    param(
        [Parameter(Mandatory = $true)]
        [string]$SelectedAction
    )

    switch ($SelectedAction) {
        "Debug" { Invoke-LauncherScript -Path $DebugScript }
        "Run" { Invoke-LauncherScript -Path $RunScript }
        "Migrate" { Invoke-LauncherScript -Path $MigrateScript }
        "Benchmark" { Invoke-LauncherScript -Path $RunScript -ReviewBenchmark }
        "PipelineBenchmark" { Invoke-LauncherScript -Path $RunScript -PipelineBenchmark }
        "DownloadModels" { Invoke-LauncherScript -Path $RunScript -DownloadProviderModels }
        "AnonymousMeeting" { Invoke-LauncherScript -Path $RunScript -WorkflowProfile anonymous_meeting }
        "Workbench" { Invoke-LauncherScript -Path $WorkbenchScript }
        "ConfigureLLM" { Invoke-LauncherScript -Path $ConfigureLlmScript }
        "PartitionManager" {
            $pythonWrapper = Join-Path $ScriptRoot "podcast_transcribe_host.py"
            & python $pythonWrapper --partition-manager --project-root $ScriptRoot
        }
        default { throw "Unsupported launcher action: $SelectedAction" }
    }
}

if ($Action -ne "Prompt") {
    Invoke-SelectedLauncherAction -SelectedAction $Action
    return
}

while ($true) {
    $selectedAction = Read-LauncherAction
    if ($selectedAction -eq "Quit") {
        break
    }
    if ($selectedAction -eq "Prompt") {
        continue
    }

    try {
        Invoke-SelectedLauncherAction -SelectedAction $selectedAction
    } catch {
        Write-Host ""
        Write-Host ("The selected action ended with an error: {0}" -f $_.Exception.Message) -ForegroundColor Red
    }

    Write-Host ""
    Write-Host "Returning to the main menu..."
}

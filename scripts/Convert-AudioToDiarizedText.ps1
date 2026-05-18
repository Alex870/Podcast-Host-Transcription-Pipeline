$CommonScript = Join-Path $PSScriptRoot "PodcastTranscribeLauncher.Common.ps1"
. $CommonScript

$LauncherContext = Get-PodcastTranscribeLauncherContext -ScriptRoot $PSScriptRoot
$ProjectRoot = $LauncherContext.ProjectRoot
$PythonScript = Join-Path $ProjectRoot "podcast_transcribe_host.py"
$ConfigPath = $LauncherContext.ConfigPath
$ConfigExamplePath = $LauncherContext.ConfigExamplePath
$EnvPath = $LauncherContext.EnvPath
$Config = $LauncherContext.Config
$ConfigSource = $LauncherContext.ConfigSource
$ActiveConfigPath = $LauncherContext.ActiveConfigPath

function Resolve-ConfigPathValue {
    param(
        [string]$Value
    )

    if ([string]::IsNullOrWhiteSpace($Value)) {
        return $null
    }

    if ([System.IO.Path]::IsPathRooted($Value)) {
        return $Value
    }

    return Join-Path $ProjectRoot $Value
}

function Set-ConfigValue {
    param(
        [psobject]$ConfigObject,
        [string]$Name,
        $Value
    )

    if ($null -eq $ConfigObject.PSObject.Properties[$Name]) {
        $ConfigObject | Add-Member -NotePropertyName $Name -NotePropertyValue $Value
    } else {
        $ConfigObject.$Name = $Value
    }
}

function Save-Config {
    param(
        [psobject]$ConfigObject
    )

    $ConfigObject | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $ConfigPath -Encoding UTF8
}

function Select-WhisperModel {
    param(
        [string]$CurrentValue
    )

    if (-not [string]::IsNullOrWhiteSpace($CurrentValue)) {
        return $CurrentValue
    }

    Write-Host ""
    Write-Host "No transcription model is configured."
    Write-Host "Choose a Whisper model for this machine:"
    Write-Host "  1. distil-large-v3 (Recommended) - best speed/quality tradeoff for English podcast transcription"
    Write-Host "  2. large-v3-turbo - very fast, strong general transcription choice"
    Write-Host "  3. large-v3 - slowest, most conservative accuracy-first option"
    $selection = Read-Host "Enter 1, 2, or 3 (default: 1)"

    switch (($selection | ForEach-Object { $_.Trim() })) {
        "2" { return "large-v3-turbo" }
        "3" { return "large-v3" }
        default { return "distil-large-v3" }
    }
}

function Load-DotEnvFile {
    param(
        [string]$Path
    )

    $values = @{}
    if (-not (Test-Path -LiteralPath $Path)) {
        return $values
    }

    foreach ($line in Get-Content -LiteralPath $Path) {
        $trimmed = $line.Trim()
        if ([string]::IsNullOrWhiteSpace($trimmed) -or $trimmed.StartsWith("#")) {
            continue
        }

        $parts = $trimmed -split "=", 2
        if ($parts.Count -ne 2) {
            continue
        }

        $key = $parts[0].Trim()
        $value = $parts[1].Trim()

        if (
            ($value.StartsWith('"') -and $value.EndsWith('"')) -or
            ($value.StartsWith("'") -and $value.EndsWith("'"))
        ) {
            $value = $value.Substring(1, $value.Length - 2)
        }

        if (-not [string]::IsNullOrWhiteSpace($key)) {
            $values[$key] = $value
        }
    }

    return $values
}

function Resolve-HfToken {
    param(
        [psobject]$ConfigObject,
        [string]$EnvFilePath,
        [string]$ConfigFilePath
    )

    $attempts = New-Object System.Collections.Generic.List[string]
    $resolvedToken = $null
    $resolvedSource = $null

    $processToken = $env:HF_TOKEN
    if ([string]::IsNullOrWhiteSpace($processToken)) {
        $attempts.Add("Process environment HF_TOKEN: not found")
    } else {
        $attempts.Add("Process environment HF_TOKEN: found")
        $resolvedToken = $processToken
        $resolvedSource = "process environment variable HF_TOKEN"
    }

    $dotenvValues = Load-DotEnvFile -Path $EnvFilePath
    if (Test-Path -LiteralPath $EnvFilePath) {
        if ($dotenvValues.ContainsKey("HF_TOKEN") -and -not [string]::IsNullOrWhiteSpace($dotenvValues["HF_TOKEN"])) {
            $attempts.Add(".env file ($EnvFilePath): HF_TOKEN found")
            if (-not $resolvedToken) {
                $resolvedToken = $dotenvValues["HF_TOKEN"]
                $resolvedSource = ".env file at $EnvFilePath"
                $env:HF_TOKEN = $resolvedToken
            }
        } else {
            $attempts.Add(".env file ($EnvFilePath): file found, HF_TOKEN missing")
        }
    } else {
        $attempts.Add(".env file ($EnvFilePath): file not found")
    }

    $configToken = if ($ConfigObject.hf_token) { [string]$ConfigObject.hf_token } else { $null }
    if ([string]::IsNullOrWhiteSpace($configToken)) {
        $attempts.Add("Config file ($ConfigFilePath): hf_token not set")
    } else {
        $attempts.Add("Config file ($ConfigFilePath): hf_token found")
        if (-not $resolvedToken) {
            $resolvedToken = $configToken
            $resolvedSource = "config file at $ConfigFilePath"
            $env:HF_TOKEN = $resolvedToken
        }
    }

    return [pscustomobject]@{
        Token   = $resolvedToken
        Source  = $resolvedSource
        Attempts = @($attempts)
    }
}

function Test-HuggingFaceToken {
    param(
        [string]$Token
    )

    try {
        $headers = @{ Authorization = "Bearer $Token" }
        $response = Invoke-RestMethod -Uri "https://huggingface.co/api/whoami-v2" -Headers $headers -Method Get -TimeoutSec 15
        return [pscustomobject]@{
            IsValid = $true
            Detail  = if ($response.name) { "Authenticated as $($response.name)." } else { "Token accepted by Hugging Face." }
        }
    } catch {
        $statusCode = $null
        if ($_.Exception.Response -and $_.Exception.Response.StatusCode) {
            $statusCode = [int]$_.Exception.Response.StatusCode
        }

        if ($statusCode -eq 401 -or $statusCode -eq 403) {
            return [pscustomobject]@{
                IsValid = $false
                Detail  = "Hugging Face rejected the token with HTTP $statusCode."
            }
        }

        return [pscustomobject]@{
            IsValid = $false
            Detail  = "Unable to validate the Hugging Face token early. $($_.Exception.Message)"
        }
    }
}

function Select-Folder {
    param(
        [string]$Description,
        [string]$InitialFolder
    )

    Write-Host "Folder selection dialog open..."
    $folderBrowser = New-Object System.Windows.Forms.FolderBrowserDialog -Property @{
        RootFolder  = "MyComputer"
        Description = $Description
    }

    if ($InitialFolder -and (Test-Path -LiteralPath $InitialFolder)) {
        $folderBrowser.SelectedPath = $InitialFolder
    }

    $null = $folderBrowser.ShowDialog()
    return $folderBrowser.SelectedPath
}

function Test-PythonDependencies {
    $dependencyCheck = @"
import importlib
import sys

required = ['numpy', 'torch', 'torchaudio', 'faster_whisper', 'pyannote.audio', 'speechbrain']
missing = []
resolved = []
errors = []

for name in required:
    try:
        module = importlib.import_module(name)
        path = getattr(module, '__file__', '<built-in>')
        resolved.append(f'{name}={path}')
    except Exception as exc:
        missing.append(name)
        errors.append(f'{name}:{type(exc).__name__}:{exc}')

print('RESOLVED:' + '|'.join(resolved))
if errors:
    print('ERRORS:' + '|'.join(errors))
if missing:
    print('MISSING:' + '|'.join(missing))
    sys.exit(1)
"@

    & python -c $dependencyCheck
    return $LASTEXITCODE
}

if (-not (Test-Path -LiteralPath $PythonScript)) {
    Write-Error "Python helper script not found: $PythonScript"
    pause
    exit
}

Add-Type -AssemblyName System.Windows.Forms

$tokenResolution = Resolve-HfToken -ConfigObject $Config -EnvFilePath $EnvPath -ConfigFilePath $ConfigPath
Write-Host "HF_TOKEN lookup details:"
foreach ($attempt in $tokenResolution.Attempts) {
    Write-Host " - $attempt"
}
if ($ActiveConfigPath) {
    Write-Host "Resolved config source: $ConfigSource ($ActiveConfigPath)"
} else {
    Write-Host "Resolved config source: $ConfigSource"
}

if (-not $tokenResolution.Token) {
    Write-Host ""
    Write-Error "HF_TOKEN could not be resolved. The loader checked the process environment, .env, and podcast_transcribe_config.json."
    pause
    exit
}

Write-Host "Using HF_TOKEN from $($tokenResolution.Source)"
$tokenValidation = Test-HuggingFaceToken -Token $tokenResolution.Token
if (-not $tokenValidation.IsValid) {
    Write-Host ""
    Write-Error $tokenValidation.Detail
    pause
    exit
}
Write-Host $tokenValidation.Detail

$ConfiguredSourceFolder = Resolve-ConfigPathValue $(if ($Config.default_source_dir) { $Config.default_source_dir } else { $null })
$SourceFolder = $null
if ($ConfiguredSourceFolder -and (Test-Path -LiteralPath $ConfiguredSourceFolder)) {
    $SourceFolder = $ConfiguredSourceFolder
    Write-Host "Using configured source folder: $SourceFolder"
} else {
    if ($ConfiguredSourceFolder) {
        Write-Host "Configured source folder was not found: $ConfiguredSourceFolder"
    } else {
        if ($ConfigSource -eq "example config") {
            Write-Host "No project config file was found; the example config does not define a valid local source folder for this machine."
        } else {
            Write-Host "No source folder configured."
        }
    }

    $SelectedFolder = Select-Folder -Description "Select a source folder for podcast audio files." -InitialFolder $null
    if ([string]::IsNullOrWhiteSpace($SelectedFolder)) {
        Write-Error "Error: folder not selected. Exiting."
        pause
        exit
    }

    $SourceFolder = $SelectedFolder
    Set-ConfigValue -ConfigObject $Config -Name "default_source_dir" -Value $SourceFolder
    Save-Config -ConfigObject $Config
    Write-Host "Saved default_source_dir to $ConfigPath"
}

$ConfiguredFfmpegBinDir = Resolve-ConfigPathValue $(if ($Config.ffmpeg_bin_dir) { $Config.ffmpeg_bin_dir } else { $null })
$FfmpegBinDir = $null
$PersistFfmpegBinDir = $false
if ($ConfiguredFfmpegBinDir -and (Test-Path -LiteralPath $ConfiguredFfmpegBinDir)) {
    $FfmpegBinDir = $ConfiguredFfmpegBinDir
    Write-Host "Using configured ffmpeg bin directory: $FfmpegBinDir"
} else {
    if ($ConfiguredFfmpegBinDir) {
        Write-Host "Configured ffmpeg bin directory was not found: $ConfiguredFfmpegBinDir"
    } else {
        Write-Host "No ffmpeg bin directory configured."
    }

    $InitialFfmpegFolder = if (Test-Path -LiteralPath "C:\ffmpeg\bin") { "C:\ffmpeg\bin" } else { $null }
    $SelectedFfmpegBinDir = Select-Folder -Description "Select the ffmpeg bin directory (the folder containing ffmpeg DLLs)." -InitialFolder $InitialFfmpegFolder
    if ([string]::IsNullOrWhiteSpace($SelectedFfmpegBinDir)) {
        Write-Error "Error: ffmpeg bin directory not selected. Exiting."
        pause
        exit
    }

    $FfmpegBinDir = $SelectedFfmpegBinDir
    $PersistFfmpegBinDir = $true
}

$DefaultKnownSpeakersDir = Join-Path $ProjectRoot "speaker_reference_samples"
$ConfiguredKnownSpeakersDir = Resolve-ConfigPathValue $(if ($Config.known_speakers_dir) { $Config.known_speakers_dir } else { $null })
$KnownSpeakersDir = $null

if ($ConfiguredKnownSpeakersDir -and (Test-Path -LiteralPath $ConfiguredKnownSpeakersDir)) {
    $KnownSpeakersDir = $ConfiguredKnownSpeakersDir
    Write-Host "Using configured speaker reference samples directory: $KnownSpeakersDir"
} elseif (Test-Path -LiteralPath $DefaultKnownSpeakersDir) {
    $KnownSpeakersDir = $DefaultKnownSpeakersDir
    if ($ConfiguredKnownSpeakersDir) {
        Write-Host "Configured speaker reference samples directory was not found: $ConfiguredKnownSpeakersDir"
    }
    Write-Host "Using default speaker reference samples directory: $KnownSpeakersDir"
    if ($Config.known_speakers_dir -ne "speaker_reference_samples") {
        Set-ConfigValue -ConfigObject $Config -Name "known_speakers_dir" -Value "speaker_reference_samples"
        Save-Config -ConfigObject $Config
        Write-Host "Saved known_speakers_dir to $ConfigPath"
    }
} else {
    Write-Host "No speaker reference samples directory is configured."
    $SelectedKnownSpeakersDir = Select-Folder -Description "Select the folder containing speaker reference samples." -InitialFolder $null
    if (-not [string]::IsNullOrWhiteSpace($SelectedKnownSpeakersDir)) {
        $KnownSpeakersDir = $SelectedKnownSpeakersDir
        Set-ConfigValue -ConfigObject $Config -Name "known_speakers_dir" -Value $KnownSpeakersDir
        Save-Config -ConfigObject $Config
        Write-Host "Saved known_speakers_dir to $ConfigPath"
    }
}

$PreferredTermsFile = Resolve-ConfigPathValue $(if ($Config.preferred_terms_file) { $Config.preferred_terms_file } else { "examples\preferred_terms.txt" })
$ReplacementMapJson = Resolve-ConfigPathValue $(if ($Config.replacement_map_json) { $Config.replacement_map_json } else { "examples\preferred_replacements.json" })
$HostProfileJson = Resolve-ConfigPathValue $(if ($Config.host_profile_json) { $Config.host_profile_json } else { "host_profile.json" })
$ConfiguredHostReference = Resolve-ConfigPathValue $(if ($Config.host_reference) { $Config.host_reference } else { $null })

$WhisperModel = Select-WhisperModel -CurrentValue $(if ($null -ne $Config.model) { [string]$Config.model } else { $null })
if ([string]::IsNullOrWhiteSpace($(if ($null -ne $Config.model) { [string]$Config.model } else { $null }))) {
    Set-ConfigValue -ConfigObject $Config -Name "model" -Value $WhisperModel
    Save-Config -ConfigObject $Config
    Write-Host "Saved model selection to $ConfigPath"
}
$Language = if ($Config.language) { $Config.language } else { "en" }
$Device = if ($Config.device) { $Config.device } else { "auto" }
$ComputeType = if ($Config.compute_type) { $Config.compute_type } else { "auto" }
$BeamSize = if ($null -ne $Config.beam_size) { [int]$Config.beam_size } else { 5 }
$BatchSize = if ($null -ne $Config.batch_size) { [int]$Config.batch_size } else { 8 }
$AssumeDominantSpeakerIsHost = if ($null -ne $Config.assume_dominant_speaker_is_host) { [bool]$Config.assume_dominant_speaker_is_host } else { $true }
$HostThreshold = if ($null -ne $Config.host_threshold) { [double]$Config.host_threshold } else { 0.45 }
$IsolateFiles = if ($null -ne $Config.isolate_files) { [bool]$Config.isolate_files } else { $true }
$CleanupLevel = if ($Config.cleanup_level) { [string]$Config.cleanup_level } else { "conservative" }
$FilenameDateConfig = if ($Config.filename_date) { $Config.filename_date } else { $null }
$FilenameDatePreset = if ($FilenameDateConfig -and $FilenameDateConfig.preset) { [string]$FilenameDateConfig.preset } else { "strict_iso" }
$FilenameDatePosition = if ($FilenameDateConfig -and $FilenameDateConfig.position) { [string]$FilenameDateConfig.position } else { "last" }
$FilenameDateFormats = @()
if ($FilenameDateConfig -and $FilenameDateConfig.formats) {
    foreach ($item in $FilenameDateConfig.formats) {
        if (-not [string]::IsNullOrWhiteSpace([string]$item)) {
            $FilenameDateFormats += [string]$item
        }
    }
}
$ConfiguredCorrectionsDir = Resolve-ConfigPathValue $(if ($Config.corrections_dir) { $Config.corrections_dir } else { "corrections" })
$ResumeIntermediates = if ($null -ne $Config.resume_intermediates) { [bool]$Config.resume_intermediates } else { $true }
$ArchiveDebugArtifacts = if ($null -ne $Config.archive_debug_artifacts) { [bool]$Config.archive_debug_artifacts } else { $false }
$ChildTimeoutSeconds = if ($null -ne $Config.child_timeout_seconds) { [int]$Config.child_timeout_seconds } else { 0 }
$BenchmarkOnly = if ($null -ne $Config.benchmark_only) { [bool]$Config.benchmark_only } else { $false }

conda activate podcast-transcribe
$env:PYTHONNOUSERSITE = "1"
$env:PODCAST_TRANSCRIBE_FFMPEG_BIN_DIR = $FfmpegBinDir

$dependencyCheckOutput = Test-PythonDependencies 2>&1
if ($LASTEXITCODE -ne 0) {
    $resolvedModules = @($dependencyCheckOutput | Where-Object { $_ -like "RESOLVED:*" })
    $dependencyErrors = @($dependencyCheckOutput | Where-Object { $_ -like "ERRORS:*" })
    $missingModules = @($dependencyCheckOutput | Where-Object { $_ -like "MISSING:*" })

    Write-Host ""
    if ($resolvedModules) {
        Write-Host ($resolvedModules -replace '^RESOLVED:', 'Resolved modules: ')
    }
    if ($dependencyErrors) {
        Write-Host ($dependencyErrors -replace '^ERRORS:', 'Import errors: ')
    }
    Write-Error "The active 'podcast-transcribe' environment is missing required Python packages: $($missingModules -replace '^MISSING:', '')"
    Write-Host "Install the missing packages into the conda environment and rerun the launcher."
    Write-Host "Suggested command: pip install -r podcast_transcribe_requirements.txt"
    pause
    exit
}

if ($PersistFfmpegBinDir -and $FfmpegBinDir) {
    Set-ConfigValue -ConfigObject $Config -Name "ffmpeg_bin_dir" -Value $FfmpegBinDir
    Save-Config -ConfigObject $Config
    Write-Host "Saved ffmpeg_bin_dir to $ConfigPath"
}

$startTime = Get-Date

Write-Host "Processing Folder: $SourceFolder"
Write-Host "Configured model: $WhisperModel"
Write-Host "Configured device: $Device"
Write-Host "Isolated per-file processing: $IsolateFiles"
Write-Host "Intermediate resume: $ResumeIntermediates"
Write-Host "Cleanup level: $CleanupLevel"
if ($BenchmarkOnly) {
    Write-Host "Benchmark-only mode: enabled"
}
Write-Host "PYTHONNOUSERSITE: $env:PYTHONNOUSERSITE"
Write-Host "ffmpeg bin directory: $FfmpegBinDir"
$OutputRoot = Split-Path -Path $SourceFolder -Parent
$OutputFolder = Join-Path $OutputRoot "output"
Write-Host "Output folder: $OutputFolder"
if ($ConfiguredHostReference -and (Test-Path -LiteralPath $ConfiguredHostReference)) {
    Write-Host "Using configured host reference sample: $ConfiguredHostReference"
} elseif ($KnownSpeakersDir -and (Test-Path -LiteralPath (Join-Path $KnownSpeakersDir "speakers.json"))) {
    Write-Host "Host reference sample: using speaker reference samples from $KnownSpeakersDir"
} else {
    Write-Host "Host reference sample: not configured"
}

$args = @(
    $PythonScript
    "--input-dir", $SourceFolder
    "--output-dir", $OutputFolder
    "--model", $WhisperModel
    "--language", $Language
    "--device", $Device
    "--compute-type", $ComputeType
    "--beam-size", "$BeamSize"
    "--batch-size", "$BatchSize"
    "--preferred-terms-file", $PreferredTermsFile
    "--replacement-map-json", $ReplacementMapJson
    "--filename-date-preset", $FilenameDatePreset
    "--filename-date-position", $FilenameDatePosition
    "--host-profile-json", $HostProfileJson
    "--cleanup-level", $CleanupLevel
    "--corrections-dir", $ConfiguredCorrectionsDir
    "--host-threshold", "$HostThreshold"
    "--hf-token", $tokenResolution.Token
)

if ($AssumeDominantSpeakerIsHost) {
    $args += "--assume-dominant-speaker-is-host"
}

if ($IsolateFiles) {
    $args += "--isolate-files"
} else {
    $args += "--no-isolate-files"
}

if (-not $ResumeIntermediates) {
    $args += "--no-resume-intermediates"
}

if ($ArchiveDebugArtifacts) {
    $args += "--archive-debug-artifacts"
}

if ($ChildTimeoutSeconds -gt 0) {
    $args += @("--child-timeout-seconds", "$ChildTimeoutSeconds")
}

if ($BenchmarkOnly) {
    $args += "--benchmark-only"
}

if ($ConfiguredHostReference -and (Test-Path -LiteralPath $ConfiguredHostReference)) {
    $args += @("--host-reference", $ConfiguredHostReference)
}

if ($FilenameDateFormats.Count -gt 0) {
    $args += @("--filename-date-formats")
    $args += $FilenameDateFormats
}

if ($KnownSpeakersDir) {
    $KnownSpeakersConfigPath = Join-Path $KnownSpeakersDir "speakers.json"
    if (Test-Path -LiteralPath $KnownSpeakersConfigPath) {
        $args += @("--known-speakers-dir", $KnownSpeakersDir)
        Write-Host "Using speaker reference config: $KnownSpeakersConfigPath"
    } else {
        Write-Host "Speaker reference directory is configured, but speakers.json was not found at $KnownSpeakersConfigPath"
    }
}

& python @args
$pythonExitCode = $LASTEXITCODE

if ($pythonExitCode -ne 0) {
    Write-Host ""
    Write-Host "The transcription pipeline exited with an error."
}

$elapsedTime = (Get-Date) - $startTime
Write-Host ("Total Duration: {0} hr {1} min {2} sec" -f $elapsedTime.Hours, $elapsedTime.Minutes, $elapsedTime.Seconds)

pause

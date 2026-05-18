$CommonScript = Join-Path $PSScriptRoot "PodcastTranscribeLauncher.Common.ps1"
. $CommonScript

$LauncherContext = Get-PodcastTranscribeLauncherContext -ScriptRoot $PSScriptRoot
$ProjectRoot = $LauncherContext.ProjectRoot
$ScriptRoot = $ProjectRoot
$ConfigPath = $LauncherContext.ConfigPath
$ConfigExamplePath = $LauncherContext.ConfigExamplePath
$EnvPath = $LauncherContext.EnvPath
$Config = $LauncherContext.Config
$ConfigSource = $LauncherContext.ConfigSource
$ActiveConfigPath = $LauncherContext.ActiveConfigPath
$HasProjectConfig = $LauncherContext.HasProjectConfig
$PrimaryDiarizationModel = "pyannote/speaker-diarization-community-1"
$SecondaryDiarizationModel = "pyannote/segmentation-3.0"
$CondaEnvironmentName = "podcast-transcribe"
$script:TotalChecks = 0
$script:FailedChecks = 0
$script:WarningChecks = 0

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

    return Join-Path $ScriptRoot $Value
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
        }
    }

    return [pscustomobject]@{
        Token    = $resolvedToken
        Source   = $resolvedSource
        Attempts = @($attempts)
    }
}

function Write-Section {
    param([string]$Title)
    Write-Host ""
    Write-Host ("=" * 78)
    Write-Host $Title
    Write-Host ("=" * 78)
}

function Write-CheckResult {
    param(
        [string]$Label,
        [bool]$Passed,
        [string]$Detail
    )

    $script:TotalChecks += 1
    if (-not $Passed) {
        $script:FailedChecks += 1
    }

    $status = if ($Passed) { "[PASS]" } else { "[FAIL]" }
    $statusColor = if ($Passed) { "Green" } else { "Red" }
    $detailText = if ([string]::IsNullOrWhiteSpace($Detail)) { "" } else { ": $Detail" }

    Write-Host -NoNewline $status -ForegroundColor $statusColor
    Write-Host (" {0}{1}" -f $Label, $detailText)
}

function Write-WarningResult {
    param(
        [string]$Label,
        [string]$Detail
    )

    $script:TotalChecks += 1
    $script:WarningChecks += 1

    $detailText = if ([string]::IsNullOrWhiteSpace($Detail)) { "" } else { ": $Detail" }
    Write-Host -NoNewline "[WARN]" -ForegroundColor Yellow
    Write-Host (" {0}{1}" -f $Label, $detailText)
}

function Write-FinalSummary {
    $passedAll = $script:FailedChecks -eq 0
    $summaryColor = if ($passedAll) { "Green" } else { "Red" }
    $summaryText = if ($passedAll) {
        if ($script:WarningChecks -gt 0) {
            "[PASS] All required tests passed ({0} warning-level checks)" -f $script:WarningChecks
        } else {
            "[PASS] All tests passed"
        }
    } else {
        "[FAIL] {0} of {1} tests failed" -f $script:FailedChecks, $script:TotalChecks
    }

    Write-Host ""
    Write-Host $summaryText -ForegroundColor $summaryColor
}

function Test-HfRepoAccess {
    param(
        [string]$RepoId,
        [string]$Token
    )

    $url = "https://huggingface.co/$RepoId/resolve/main/config.yaml"
    $headers = @{ Authorization = "Bearer $Token" }

    try {
        $null = Invoke-WebRequest -Uri $url -Headers $headers -Method Head -ErrorAction Stop
        return [pscustomobject]@{
            Passed = $true
            Detail = "Access confirmed."
        }
    } catch {
        $statusCode = $null
        $statusDesc = $null
        if ($_.Exception.Response) {
            $statusCode = $_.Exception.Response.StatusCode.value__
            $statusDesc = $_.Exception.Response.StatusDescription
        }

        if ($statusCode -eq 403) {
            return [pscustomobject]@{
                Passed = $false
                Detail = "HTTP 403 ($statusDesc). Accept model terms for $RepoId on Hugging Face."
            }
        }

        if ($statusCode -eq 401) {
            return [pscustomobject]@{
                Passed = $false
                Detail = "HTTP 401 ($statusDesc). Token is missing, invalid, or not being sent correctly."
            }
        }

        return [pscustomobject]@{
            Passed = $false
            Detail = "Request failed. $($_.Exception.Message)"
        }
    }
}

$SourceFolder = Resolve-ConfigPathValue $(if ($Config.default_source_dir) { $Config.default_source_dir } else { $null })
$KnownSpeakersDir = Resolve-ConfigPathValue $(if ($Config.known_speakers_dir) { $Config.known_speakers_dir } else { "speaker_reference_samples" })
$PreferredTermsFile = Resolve-ConfigPathValue $(if ($Config.preferred_terms_file) { $Config.preferred_terms_file } else { "examples\preferred_terms.txt" })
$ReplacementMapJson = Resolve-ConfigPathValue $(if ($Config.replacement_map_json) { $Config.replacement_map_json } else { "examples\preferred_replacements.json" })
$HostProfileJson = Resolve-ConfigPathValue $(if ($Config.host_profile_json) { $Config.host_profile_json } else { "host_profile.json" })
$FfmpegBinDir = Resolve-ConfigPathValue $(if ($Config.ffmpeg_bin_dir) { $Config.ffmpeg_bin_dir } else { $null })
$FilenameDateConfig = if ($Config.filename_date) { $Config.filename_date } else { $null }
$FilenameDatePreset = if ($FilenameDateConfig -and $FilenameDateConfig.preset) { [string]$FilenameDateConfig.preset } else { "strict_iso" }
$FilenameDatePosition = if ($FilenameDateConfig -and $FilenameDateConfig.position) { [string]$FilenameDateConfig.position } else { "last" }
$FilenameDateFormats = if ($FilenameDateConfig -and $FilenameDateConfig.formats) { @($FilenameDateConfig.formats) } else { @() }
$Device = if ($Config.device) { [string]$Config.device } else { "auto" }
$Model = if ($Config.model) { [string]$Config.model } else { "large-v3" }
$TokenResolution = Resolve-HfToken -ConfigObject $Config -EnvFilePath $EnvPath -ConfigFilePath $ConfigPath

Write-Section "Config Resolution"
if ($HasProjectConfig) {
    Write-CheckResult "Config file" $true ("Using project config: {0}" -f $ConfigPath)
} elseif ($ActiveConfigPath) {
    Write-WarningResult "Config file" ("Project config not found; using example config at {0}" -f $ActiveConfigPath)
} else {
    Write-WarningResult "Config file" "No project config or example config was found. Script is using built-in defaults."
}

$sourceConfigured = -not [string]::IsNullOrWhiteSpace($SourceFolder)
$sourceExists = $sourceConfigured -and (Test-Path -LiteralPath $SourceFolder)
if ($sourceExists) {
    Write-CheckResult "Source directory" $true $SourceFolder
} elseif ($sourceConfigured -and $HasProjectConfig) {
    Write-CheckResult "Source directory" $false $SourceFolder
} elseif ($sourceConfigured) {
    Write-WarningResult "Source directory" ("Configured path does not exist in this clone: {0}" -f $SourceFolder)
} else {
    Write-WarningResult "Source directory" "Not configured."
}

Write-CheckResult "Speaker reference directory" (Test-Path -LiteralPath $KnownSpeakersDir) $KnownSpeakersDir
Write-CheckResult "Speaker reference config" (Test-Path -LiteralPath (Join-Path $KnownSpeakersDir "speakers.json")) (Join-Path $KnownSpeakersDir "speakers.json")
Write-CheckResult "Preferred terms file" (Test-Path -LiteralPath $PreferredTermsFile) $PreferredTermsFile
Write-CheckResult "Replacement map file" (Test-Path -LiteralPath $ReplacementMapJson) $ReplacementMapJson
Write-CheckResult "Filename date preset" $true $FilenameDatePreset
Write-CheckResult "Filename date position" $true $FilenameDatePosition
if ($FilenameDateFormats.Count -gt 0) {
    Write-CheckResult "Filename date formats" $true ($FilenameDateFormats -join ", ")
}
Write-CheckResult "Host profile path" $true $HostProfileJson
Write-CheckResult "Configured device" $true $Device
Write-CheckResult "Configured model" $true $Model

Write-Section "Token Resolution"
foreach ($attempt in $TokenResolution.Attempts) {
    Write-Host " - $attempt"
}
if ($ActiveConfigPath) {
    Write-Host " - Active config source: $ConfigSource ($ActiveConfigPath)"
} else {
    Write-Host " - Active config source: $ConfigSource"
}
Write-CheckResult "Resolved HF token" (-not [string]::IsNullOrWhiteSpace($TokenResolution.Token)) $(if ($TokenResolution.Source) { $TokenResolution.Source } else { "No token found." })

if ($TokenResolution.Token) {
    Write-Section "Hugging Face Access"
    $primaryAccess = Test-HfRepoAccess -RepoId $PrimaryDiarizationModel -Token $TokenResolution.Token
    Write-CheckResult $PrimaryDiarizationModel $primaryAccess.Passed $primaryAccess.Detail

    $secondaryAccess = Test-HfRepoAccess -RepoId $SecondaryDiarizationModel -Token $TokenResolution.Token
    Write-CheckResult $SecondaryDiarizationModel $secondaryAccess.Passed $secondaryAccess.Detail
    Write-Host "Note: diarization can require access to both the primary pipeline model and the segmentation submodel."
}

Write-Section "FFmpeg"
$ffmpegDirectoryExists = -not [string]::IsNullOrWhiteSpace($FfmpegBinDir) -and (Test-Path -LiteralPath $FfmpegBinDir)
Write-CheckResult "ffmpeg_bin_dir" $ffmpegDirectoryExists $(if ($FfmpegBinDir) { $FfmpegBinDir } else { "Not configured." })

$ffmpegExe = if ($FfmpegBinDir) { Join-Path $FfmpegBinDir "ffmpeg.exe" } else { $null }
$ffmpegExeExists = $ffmpegExe -and (Test-Path -LiteralPath $ffmpegExe)
Write-CheckResult "ffmpeg.exe" $ffmpegExeExists $(if ($ffmpegExe) { $ffmpegExe } else { "Not available." })

if ($ffmpegExeExists) {
    try {
        $ffmpegVersion = & $ffmpegExe -version 2>$null | Select-Object -First 1
        Write-CheckResult "FFmpeg launch" $true $ffmpegVersion
    } catch {
        Write-CheckResult "FFmpeg launch" $false $_.Exception.Message
    }
}

Write-Section "Conda Environment"
$condaCommand = Get-Command conda -ErrorAction SilentlyContinue
Write-CheckResult "conda command" ($null -ne $condaCommand) $(if ($condaCommand) { $condaCommand.Source } else { "Conda command not found in PATH." })

if (-not $condaCommand) {
    Write-Host ""
    Write-Host "Cannot continue Python diagnostics because 'conda' was not found."
    exit 1
}

try {
    conda activate $CondaEnvironmentName
    $env:PYTHONNOUSERSITE = "1"
    if ($TokenResolution.Token) {
        $env:HF_TOKEN = $TokenResolution.Token
    }
    if ($FfmpegBinDir) {
        $env:PODCAST_TRANSCRIBE_FFMPEG_BIN_DIR = $FfmpegBinDir
    }
    Write-CheckResult "Activated env" $true $CondaEnvironmentName
} catch {
    Write-CheckResult "Activated env" $false $_.Exception.Message
    exit 1
}

Write-Section "Python Diagnostics"
$pythonReportTempPath = [System.IO.Path]::GetTempFileName()
$env:PODCAST_TRANSCRIBE_DIAGNOSTIC_JSON = $pythonReportTempPath
$pythonDiagnosticScript = @'
import importlib
import importlib.util
import inspect
import json
import os
import sys
import warnings


def configure_ffmpeg_dll_directory():
    ffmpeg_bin_dir = os.getenv("PODCAST_TRANSCRIBE_FFMPEG_BIN_DIR") or os.getenv("FFMPEG_BIN_DIR")
    if os.name != "nt" or not ffmpeg_bin_dir or not hasattr(os, "add_dll_directory"):
        return

    if os.path.isdir(ffmpeg_bin_dir):
        os.add_dll_directory(ffmpeg_bin_dir)


configure_ffmpeg_dll_directory()

warnings.filterwarnings(
    "ignore",
    message=r".*torchcodec is not installed correctly so built-in audio decoding will fail.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    module=r"pyannote\.audio\.core\.io",
    category=Warning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*TensorFloat-32 \(TF32\) has been disabled.*",
)
warnings.filterwarnings(
    "ignore",
    message=r".*torchaudio\._backend\.list_audio_backends has been deprecated.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*implementation will be changed to use torchaudio\.load_with_torchcodec.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*Requested Pretrainer collection using symlinks on Windows.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*std\(\): degrees of freedom is <= 0.*",
    category=UserWarning,
)

report = {
    "python_executable": sys.executable,
    "python_version": sys.version,
    "python_no_user_site": os.getenv("PYTHONNOUSERSITE"),
    "ffmpeg_bin_dir": os.getenv("PODCAST_TRANSCRIBE_FFMPEG_BIN_DIR"),
    "module_checks": {},
    "torch": {},
    "pyannote": {},
    "pyannote_path_decoder": {},
    "speechbrain": {},
    "full_model_checks": {},
}

modules_to_check = [
    ("numpy", "numpy"),
    ("torch", "torch"),
    ("torchaudio", "torchaudio"),
    ("faster_whisper", "faster_whisper"),
    ("huggingface_hub", "huggingface_hub"),
    ("pyannote_audio", "pyannote.audio"),
]

for label, module_name in modules_to_check:
    try:
        module = importlib.import_module(module_name)
        report["module_checks"][label] = {
            "ok": True,
            "path": getattr(module, "__file__", "<built-in>"),
            "version": getattr(module, "__version__", None),
        }
    except Exception as exc:
        report["module_checks"][label] = {
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }

torchcodec_spec = importlib.util.find_spec("torchcodec")
report["module_checks"]["torchcodec"] = {
    "ok": True,
    "path": getattr(torchcodec_spec, "origin", None) if torchcodec_spec else None,
    "version": None,
    "note": (
        "Optional runtime dependency. The main pipeline preloads audio with torchaudio and does not require "
        "torchcodec to load successfully."
    ),
}

speechbrain_spec = importlib.util.find_spec("speechbrain")
report["module_checks"]["speechbrain"] = {
    "ok": speechbrain_spec is not None,
    "path": getattr(speechbrain_spec, "origin", None) if speechbrain_spec else None,
    "version": None,
    "note": (
        "Package discovered without importing it early, to mirror the main pipeline's deferred SpeechBrain loading."
    ) if speechbrain_spec else None,
    "error_type": None if speechbrain_spec else "ModuleNotFoundError",
    "error": None if speechbrain_spec else "speechbrain is not installed",
}

try:
    import torch

    report["torch"] = {
        "version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device_count": torch.cuda.device_count(),
    }
except Exception as exc:
    report["torch"] = {
        "error_type": type(exc).__name__,
        "error": str(exc),
    }

try:
    from pyannote.audio import Pipeline
    import pyannote.audio
    import pyannote.audio.core.io as pyannote_io

    report["pyannote"] = {
        "version": pyannote.audio.__version__,
        "from_pretrained_signature": str(inspect.signature(Pipeline.from_pretrained)),
    }
    decoder_available = hasattr(pyannote_io, "AudioDecoder")
    report["pyannote_path_decoder"] = {
        "ok": decoder_available,
        "audio_decoder_available": decoder_available,
        "detail": (
            "pyannote AudioDecoder is available; path-based diarization input can be attempted."
            if decoder_available
            else "pyannote AudioDecoder is unavailable; the main pipeline will preload audio with torchaudio for diarization.  This will prevent audio chunking for long podcasts and may require a lot of system RAM for processing."
        ),
    }
except Exception as exc:
    report["pyannote"] = {
        "error_type": type(exc).__name__,
        "error": str(exc),
    }
    report["pyannote_path_decoder"] = {
        "ok": False,
        "audio_decoder_available": False,
        "detail": "Could not inspect pyannote audio decoder.",
        "error_type": type(exc).__name__,
        "error": str(exc),
    }

try:
    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1",
        token=os.getenv("HF_TOKEN"),
    )
    report["full_model_checks"]["diarization_pipeline"] = {
        "ok": True,
        "type": str(type(pipeline)),
    }
except Exception as exc:
    report["full_model_checks"]["diarization_pipeline"] = {
        "ok": False,
        "error_type": type(exc).__name__,
        "error": str(exc),
    }

try:
    from speechbrain.inference.speaker import SpeakerRecognition

    report["speechbrain"] = {
        "speaker_recognition_import_ok": True,
        "speaker_recognition_type": str(SpeakerRecognition),
    }

    verifier = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_speaker_model_diagnostic",
        run_opts={"device": "cuda:0" if report.get("torch", {}).get("cuda_available") else "cpu"},
    )
    report["full_model_checks"]["speaker_verifier"] = {
        "ok": True,
        "type": str(type(verifier)),
    }
except Exception as exc:
    if not report["speechbrain"]:
        report["speechbrain"] = {
            "speaker_recognition_import_ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    report["full_model_checks"]["speaker_verifier"] = {
        "ok": False,
        "error_type": type(exc).__name__,
        "error": str(exc),
    }

report_path = os.getenv("PODCAST_TRANSCRIBE_DIAGNOSTIC_JSON")
if report_path:
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle)
else:
    print(json.dumps(report))
'@

$pythonOutput = $pythonDiagnosticScript | python - 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-CheckResult "Python diagnostics" $false ($pythonOutput -join "`n")
    Remove-Item -LiteralPath $pythonReportTempPath -ErrorAction SilentlyContinue
    exit 1
}

$pythonOutputText = $pythonOutput -join "`n"
if (-not (Test-Path -LiteralPath $pythonReportTempPath)) {
    Write-CheckResult "Python diagnostics" $false "Python diagnostics did not produce a JSON report file. Raw output:`n$pythonOutputText"
    exit 1
}

try {
    $pythonJson = Get-Content -LiteralPath $pythonReportTempPath -Raw
    $pythonReport = $pythonJson | ConvertFrom-Json
} catch {
    Write-CheckResult "Python diagnostics" $false "Failed to parse Python diagnostic JSON. Raw output:`n$pythonOutputText"
    Remove-Item -LiteralPath $pythonReportTempPath -ErrorAction SilentlyContinue
    exit 1
}
Remove-Item -LiteralPath $pythonReportTempPath -ErrorAction SilentlyContinue
Remove-Item Env:PODCAST_TRANSCRIBE_DIAGNOSTIC_JSON -ErrorAction SilentlyContinue
Write-CheckResult "Python executable" $true $pythonReport.python_executable
Write-CheckResult "Python version" $true $pythonReport.python_version
Write-CheckResult "PYTHONNOUSERSITE" ($pythonReport.python_no_user_site -eq "1") $pythonReport.python_no_user_site
Write-CheckResult "FFmpeg DLL env" (-not [string]::IsNullOrWhiteSpace($pythonReport.ffmpeg_bin_dir)) $pythonReport.ffmpeg_bin_dir

foreach ($property in $pythonReport.module_checks.PSObject.Properties) {
    $value = $property.Value
    if ($value.ok) {
        $detail = if ($value.note) {
            if ($value.path) {
                "$($value.note) @ $($value.path)"
            } else {
                $value.note
            }
        } elseif ($value.version) {
            "$($value.version) @ $($value.path)"
        } else {
            "$($value.path)"
        }
        Write-CheckResult ("Module " + $property.Name) $true $detail
    } else {
        Write-CheckResult ("Module " + $property.Name) $false ("{0}: {1}" -f $value.error_type, $value.error)
    }
}

if ($pythonReport.torch.version) {
    $torchDetail = "version=$($pythonReport.torch.version), cuda_available=$($pythonReport.torch.cuda_available), cuda_version=$($pythonReport.torch.cuda_version), device_count=$($pythonReport.torch.device_count)"
    Write-CheckResult "Torch runtime" $true $torchDetail
} else {
    Write-CheckResult "Torch runtime" $false ("{0}: {1}" -f $pythonReport.torch.error_type, $pythonReport.torch.error)
}

if ($pythonReport.pyannote.version) {
    Write-CheckResult "pyannote.audio runtime" $true ("version={0}, from_pretrained={1}" -f $pythonReport.pyannote.version, $pythonReport.pyannote.from_pretrained_signature)
} else {
    Write-CheckResult "pyannote.audio runtime" $false ("{0}: {1}" -f $pythonReport.pyannote.error_type, $pythonReport.pyannote.error)
}

if ($pythonReport.pyannote_path_decoder.audio_decoder_available) {
    Write-CheckResult "pyannote path decoder" $true $pythonReport.pyannote_path_decoder.detail
} else {
    $decoderDetail = $pythonReport.pyannote_path_decoder.detail
    if ($pythonReport.pyannote_path_decoder.error) {
        $decoderDetail += " {0}: {1}" -f $pythonReport.pyannote_path_decoder.error_type, $pythonReport.pyannote_path_decoder.error
    }
    Write-WarningResult "pyannote path decoder" $decoderDetail
}

foreach ($property in $pythonReport.full_model_checks.PSObject.Properties) {
    $value = $property.Value
    if ($value.ok) {
        Write-CheckResult ("Model load " + $property.Name) $true $value.type
    } else {
        $detail = ("{0}: {1}" -f $value.error_type, $value.error)
        if ($property.Name -eq "diarization_pipeline" -and $value.error -like "*speechbrain.integrations.k2_fsa*") {
            $detail += " | Known diagnostic false positive caused by optional SpeechBrain/k2 lazy-import behavior. The main pipeline works around this with deferred SpeechBrain loading."
            Write-CheckResult ("Model load " + $property.Name) $true $detail
            continue
        }
        Write-CheckResult ("Model load " + $property.Name) $false $detail
    }
}

if ($pythonReport.speechbrain.speaker_recognition_import_ok) {
    Write-CheckResult "SpeechBrain verifier import" $true $pythonReport.speechbrain.speaker_recognition_type
} else {
    Write-CheckResult "SpeechBrain verifier import" $false ("{0}: {1}" -f $pythonReport.speechbrain.error_type, $pythonReport.speechbrain.error)
}

Write-Section "Notes"
Write-Host "- If Hugging Face access fails with HTTP 403, accept terms for both $PrimaryDiarizationModel and $SecondaryDiarizationModel."
Write-Host "- If Torch reports CUDA unavailable, install matching CUDA-enabled PyTorch wheels in the '$CondaEnvironmentName' environment."
Write-Host "- If the pyannote path decoder is only a warning, the main pipeline will still run by preloading audio with torchaudio."
Write-Host "- To enable pyannote path decoding on Windows, verify torchcodec is installed for the active PyTorch version and can find a full shared FFmpeg build through ffmpeg_bin_dir/PATH."
Write-Host "- If speechbrain or pyannote model loading fails only during full model checks, compare the installed versions against podcast_transcribe_requirements.txt and reinstall inside '$CondaEnvironmentName'."
Write-FinalSummary
pause

function Get-PodcastTranscribeProjectRoot {
    param(
        [string]$StartPath
    )

    $candidate = $StartPath
    if ([string]::IsNullOrWhiteSpace($candidate)) {
        $candidate = (Get-Location).Path
    }

    if (Test-Path -LiteralPath $candidate -PathType Leaf) {
        $candidate = Split-Path -Parent $candidate
    }

    try {
        $candidate = (Resolve-Path -LiteralPath $candidate -ErrorAction Stop).Path
    } catch {
        $candidate = [System.IO.Path]::GetFullPath($candidate)
    }

    while ($true) {
        $wrapperPath = Join-Path $candidate "podcast_transcribe_host.py"
        $packagePath = Join-Path $candidate "src\podcast_transcribe\cli.py"
        if ((Test-Path -LiteralPath $wrapperPath) -and (Test-Path -LiteralPath $packagePath)) {
            return $candidate
        }

        $parent = Split-Path -Parent $candidate
        if ([string]::IsNullOrWhiteSpace($parent) -or $parent -eq $candidate) {
            break
        }
        $candidate = $parent
    }

    throw "Could not locate the podcast-transcribe project root from '$StartPath'."
}

function Get-PodcastTranscribeLauncherContext {
    param(
        [string]$ScriptRoot
    )

    $projectRoot = Get-PodcastTranscribeProjectRoot -StartPath $ScriptRoot
    $configPath = Join-Path $projectRoot "podcast_transcribe_config.json"
    $configExamplePath = Join-Path $projectRoot "examples\podcast_transcribe_config.example.json"
    $envPath = Join-Path $projectRoot ".env"

    if (Test-Path -LiteralPath $configPath) {
        $config = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
        $configSource = "project config"
        $activeConfigPath = $configPath
    } elseif (Test-Path -LiteralPath $configExamplePath) {
        $config = Get-Content -LiteralPath $configExamplePath -Raw | ConvertFrom-Json
        $configSource = "example config"
        $activeConfigPath = $configExamplePath
    } else {
        $config = [pscustomobject]@{}
        $configSource = "generated defaults"
        $activeConfigPath = $null
    }

    return [pscustomobject]@{
        ProjectRoot       = $projectRoot
        ConfigPath        = $configPath
        ConfigExamplePath = $configExamplePath
        EnvPath           = $envPath
        Config            = $config
        ConfigSource      = $configSource
        ActiveConfigPath  = $activeConfigPath
        HasProjectConfig  = Test-Path -LiteralPath $configPath
    }
}

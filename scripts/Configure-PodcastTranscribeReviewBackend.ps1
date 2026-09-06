param(
    [string]$ProjectRoot = "",
    [switch]$LibraryOnly,
    [switch]$NoPause
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = "Stop"

function Test-CancelValue {
    param([AllowNull()][string]$Value)
    return $null -eq $Value -or $Value.Trim().ToUpperInvariant() -in @("Q", "QUIT", "CANCEL")
}

function Get-PropertyValue {
    param(
        [AllowNull()][object]$Object,
        [string]$Name,
        [AllowNull()][object]$DefaultValue = $null
    )
    if ($null -eq $Object) {
        return $DefaultValue
    }
    $property = $Object.PSObject.Properties[$Name]
    if ($null -eq $property) {
        return $DefaultValue
    }
    return $property.Value
}

function Set-JsonProperty {
    param(
        [Parameter(Mandatory = $true)][object]$Object,
        [Parameter(Mandatory = $true)][string]$Name,
        [AllowNull()][object]$Value
    )
    $property = $Object.PSObject.Properties[$Name]
    if ($null -eq $property) {
        $Object | Add-Member -NotePropertyName $Name -NotePropertyValue $Value
    } else {
        $property.Value = $Value
    }
}

function Normalize-ReviewServerAddress {
    param([Parameter(Mandatory = $true)][string]$Address)

    $value = $Address.Trim()
    if ([string]::IsNullOrWhiteSpace($value)) {
        throw "Server address is empty."
    }
    if ($value -notmatch "^[A-Za-z][A-Za-z0-9+.-]*://") {
        $value = "http://$value"
    }

    $uri = $null
    if (-not [System.Uri]::TryCreate($value, [System.UriKind]::Absolute, [ref]$uri)) {
        throw "Server address is not a valid absolute URL: $Address"
    }
    if ($uri.Scheme -notin @("http", "https")) {
        throw "Only http and https review servers are supported."
    }
    if ([string]::IsNullOrWhiteSpace($uri.Host)) {
        throw "Server address does not contain a hostname or IP address."
    }
    if (-not [string]::IsNullOrWhiteSpace($uri.UserInfo)) {
        throw "Credentials must not be embedded in the server URL."
    }
    if (-not [string]::IsNullOrWhiteSpace($uri.Query) -or -not [string]::IsNullOrWhiteSpace($uri.Fragment)) {
        throw "The server URL must not contain a query string or fragment."
    }

    $builder = New-Object System.UriBuilder($uri)
    $path = $builder.Path.TrimEnd("/")
    if ($path -eq "/v1") {
        $path = ""
    }
    if (-not [string]::IsNullOrWhiteSpace($path) -and $path -ne "/") {
        throw "The server URL must point to the server root, not '$path'."
    }
    $builder.Path = ""
    $builder.Query = ""
    $builder.Fragment = ""
    return $builder.Uri.AbsoluteUri.TrimEnd("/")
}

function Test-AddressHasExplicitPort {
    param([Parameter(Mandatory = $true)][string]$Address)
    $value = $Address.Trim()
    if ($value -notmatch "^[A-Za-z][A-Za-z0-9+.-]*://") {
        $value = "http://$value"
    }
    $authority = ([System.Uri]$value).Authority
    if ($authority.StartsWith("[")) {
        return $authority -match "\]:\d+$"
    }
    return $authority -match ":\d+$"
}

function Get-ReviewServerCandidates {
    param(
        [Parameter(Mandatory = $true)][string]$Address,
        [string]$CurrentBaseUrl = ""
    )

    $normalized = Normalize-ReviewServerAddress -Address $Address
    if (Test-AddressHasExplicitPort -Address $Address) {
        return @($normalized)
    }

    $inputUri = [System.Uri]$normalized
    $ports = New-Object System.Collections.Generic.List[int]
    if (-not [string]::IsNullOrWhiteSpace($CurrentBaseUrl)) {
        try {
            $currentNormalized = Normalize-ReviewServerAddress -Address $CurrentBaseUrl
            $currentUri = [System.Uri]$currentNormalized
            if ($currentUri.Host -ieq $inputUri.Host) {
                $ports.Add($currentUri.Port)
            }
        } catch {
            # An invalid old setting should not prevent discovery of the new server.
        }
    }
    $ports.Add(8000)
    $ports.Add(1234)

    $results = New-Object System.Collections.Generic.List[string]
    foreach ($port in ($ports | Select-Object -Unique)) {
        $builder = New-Object System.UriBuilder($inputUri)
        $builder.Port = $port
        $candidate = $builder.Uri.AbsoluteUri.TrimEnd("/")
        if (-not $results.Contains($candidate)) {
            $results.Add($candidate)
        }
    }
    return @($results)
}

function Invoke-ReviewJsonRequest {
    param(
        [Parameter(Mandatory = $true)][ValidateSet("GET", "POST")][string]$Method,
        [Parameter(Mandatory = $true)][string]$Uri,
        [AllowNull()][object]$Body = $null,
        [int]$TimeoutSec = 12
    )

    $arguments = @{
        Method      = $Method
        Uri         = $Uri
        TimeoutSec  = $TimeoutSec
        ErrorAction = "Stop"
    }
    if ($null -ne $Body) {
        $arguments.ContentType = "application/json"
        $arguments.Body = $Body | ConvertTo-Json -Depth 20 -Compress
    }
    return Invoke-RestMethod @arguments
}

function Get-ReviewModelIds {
    param([AllowNull()][object]$Response)

    $models = New-Object System.Collections.Generic.List[string]
    $items = @()
    if ($null -eq $Response) {
        return @()
    } elseif ($null -ne $Response.PSObject.Properties["data"]) {
        $items = @($Response.data)
    } elseif ($null -ne $Response.PSObject.Properties["models"]) {
        $items = @($Response.models)
    } elseif ($Response -is [System.Array]) {
        $items = @($Response)
    }

    foreach ($item in $items) {
        $modelType = [string](Get-PropertyValue -Object $item -Name "type" -DefaultValue "")
        if (-not [string]::IsNullOrWhiteSpace($modelType) -and $modelType -notin @("llm", "chat")) {
            continue
        }
        foreach ($field in @("id", "key", "model", "name")) {
            $value = Get-PropertyValue -Object $item -Name $field -DefaultValue ""
            if (-not [string]::IsNullOrWhiteSpace([string]$value)) {
                $modelId = ([string]$value).Trim()
                if (-not $models.Contains($modelId)) {
                    $models.Add($modelId)
                }
                break
            }
        }
    }
    return @($models | Sort-Object)
}

function Invoke-OptionalProbe {
    param(
        [Parameter(Mandatory = $true)][scriptblock]$RequestInvoker,
        [Parameter(Mandatory = $true)][string]$Uri
    )
    try {
        return [pscustomobject]@{
            Succeeded = $true
            Response  = & $RequestInvoker "GET" $Uri $null
            Error     = ""
            Category  = ""
        }
    } catch {
        $category = "connection_error"
        $exception = $_.Exception
        if ($null -ne $exception.PSObject.Properties["Response"] -and $null -ne $exception.Response) {
            $category = "http_error"
        } elseif ($exception.Message -match "(?i)name.*resolv|dns|no such host|remote name") {
            $category = "dns_error"
        }
        return [pscustomobject]@{
            Succeeded = $false
            Response  = $null
            Error     = $exception.Message
            Category  = $category
        }
    }
}

function Find-ReviewServer {
    param(
        [Parameter(Mandatory = $true)][string[]]$Candidates,
        [scriptblock]$RequestInvoker = ${function:Invoke-ReviewJsonRequest}
    )

    $attempts = New-Object System.Collections.Generic.List[object]
    foreach ($baseUrl in $Candidates) {
        $modelsProbe = Invoke-OptionalProbe -RequestInvoker $RequestInvoker -Uri "$baseUrl/v1/models"
        if (-not $modelsProbe.Succeeded) {
            $attempts.Add([pscustomobject]@{
                BaseUrl = $baseUrl
                Result  = $modelsProbe.Category
                Detail  = $modelsProbe.Error
            })
            continue
        }

        $modelIds = @(Get-ReviewModelIds -Response $modelsProbe.Response)
        if ($modelIds.Count -eq 0) {
            $hasRecognizedList = (
                $null -ne $modelsProbe.Response.PSObject.Properties["data"] -or
                $null -ne $modelsProbe.Response.PSObject.Properties["models"] -or
                $modelsProbe.Response -is [System.Array]
            )
            $attempts.Add([pscustomobject]@{
                BaseUrl = $baseUrl
                Result  = if ($hasRecognizedList) { "no_models_returned" } else { "invalid_model_list_response" }
                Detail  = if ($hasRecognizedList) {
                    "The model-list endpoint returned no models."
                } else {
                    "The model-list response did not have a recognized data or models collection."
                }
            })
            continue
        }

        $versionProbe = Invoke-OptionalProbe -RequestInvoker $RequestInvoker -Uri "$baseUrl/version"
        $lmProbe = Invoke-OptionalProbe -RequestInvoker $RequestInvoker -Uri "$baseUrl/api/v1/models"
        $vllmDetected = (
            $versionProbe.Succeeded -and
            -not [string]::IsNullOrWhiteSpace(
                [string](Get-PropertyValue -Object $versionProbe.Response -Name "version" -DefaultValue "")
            )
        )
        $lmStudioDetected = (
            $lmProbe.Succeeded -and
            $null -ne $lmProbe.Response.PSObject.Properties["models"]
        )
        $backend = "ambiguous"
        if ($vllmDetected -and -not $lmStudioDetected) {
            $backend = "vllm"
        } elseif ($lmStudioDetected -and -not $vllmDetected) {
            $backend = "lm_studio"
        }

        $nativeModels = @()
        if ($lmProbe.Succeeded) {
            $nativeModels = @(Get-ReviewModelIds -Response $lmProbe.Response)
        }
        if ($lmStudioDetected -and $nativeModels.Count -gt 0) {
            $allModels = @($nativeModels | Sort-Object -Unique)
        } else {
            $allModels = @($modelIds + $nativeModels | Sort-Object -Unique)
        }
        return [pscustomobject]@{
            Succeeded        = $true
            BaseUrl          = $baseUrl
            Backend          = $backend
            ModelIds         = $allModels
            VllmDetected     = $vllmDetected
            LmStudioDetected = $lmStudioDetected
            Attempts         = @($attempts | ForEach-Object { $_ })
        }
    }

    return [pscustomobject]@{
        Succeeded        = $false
        BaseUrl          = ""
        Backend          = ""
        ModelIds         = @()
        VllmDetected     = $false
        LmStudioDetected = $false
        Attempts         = @($attempts | ForEach-Object { $_ })
    }
}

function Test-ReviewChatCompletion {
    param(
        [Parameter(Mandatory = $true)][string]$BaseUrl,
        [Parameter(Mandatory = $true)][string]$ModelId,
        [scriptblock]$RequestInvoker = ${function:Invoke-ReviewJsonRequest},
        [string]$Backend = ""
    )

    $body = [ordered]@{
        model       = $ModelId
        messages    = @(
            [ordered]@{
                role    = "user"
                content = "Reply with exactly OK."
            }
        )
        temperature = 0
        max_tokens  = 16
    }
    if ($Backend.Trim().ToLowerInvariant() -eq "vllm") {
        $body.chat_template_kwargs = [ordered]@{
            enable_thinking = $false
        }
    }
    try {
        $response = & $RequestInvoker "POST" "$BaseUrl/v1/chat/completions" $body
        $choices = @(Get-PropertyValue -Object $response -Name "choices" -DefaultValue @())
        if ($choices.Count -eq 0) {
            return [pscustomobject]@{
                Succeeded = $false
                Detail    = "The chat-completions response did not contain a choices array."
            }
        }
        return [pscustomobject]@{
            Succeeded = $true
            Detail    = "The selected model returned a valid chat-completions response."
        }
    } catch {
        return [pscustomobject]@{
            Succeeded = $false
            Detail    = $_.Exception.Message
        }
    }
}

function Get-ReviewStageSettings {
    param([Parameter(Mandatory = $true)][object]$Config)
    return [ordered]@{
        runtime_profile              = [string](Get-PropertyValue $Config "runtime_profile" "baseline_16gb")
        transcript_cleanup_review    = [bool](Get-PropertyValue $Config "transcript_cleanup_review" $false)
        glossary_correction_review   = [bool](Get-PropertyValue $Config "glossary_correction_review" $false)
        speaker_consistency_review   = [bool](Get-PropertyValue $Config "speaker_consistency_review" $false)
        episode_qa_review            = [bool](Get-PropertyValue $Config "episode_qa_review" $false)
    }
}

function Get-ConfigurationChanges {
    param(
        [Parameter(Mandatory = $true)][object]$Config,
        [Parameter(Mandatory = $true)][ValidateSet("vllm", "lm_studio")][string]$Backend,
        [Parameter(Mandatory = $true)][string]$BaseUrl,
        [Parameter(Mandatory = $true)][string]$ModelId,
        [Parameter(Mandatory = $true)][ValidateSet("Keep", "Local", "All")][string]$ReviewMode
    )

    $changes = [ordered]@{
        backend          = $Backend
        review_base_url  = $BaseUrl
        review_model_name = $ModelId
    }
    if ($ReviewMode -eq "Local") {
        $changes.transcript_cleanup_review = $true
        $changes.glossary_correction_review = $true
        $changes.speaker_consistency_review = $true
        $changes.episode_qa_review = $false
    } elseif ($ReviewMode -eq "All") {
        $changes.runtime_profile = "high_context_5090"
        $changes.transcript_cleanup_review = $true
        $changes.glossary_correction_review = $true
        $changes.speaker_consistency_review = $true
        $changes.episode_qa_review = $true
    }
    return $changes
}

function Write-ReviewBackendConfiguration {
    param(
        [Parameter(Mandatory = $true)][string]$ConfigPath,
        [Parameter(Mandatory = $true)][System.Collections.IDictionary]$Changes
    )

    if (-not (Test-Path -LiteralPath $ConfigPath -PathType Leaf)) {
        throw "Configuration file not found: $ConfigPath"
    }
    $raw = [System.IO.File]::ReadAllText($ConfigPath)
    try {
        $config = $raw | ConvertFrom-Json
    } catch {
        throw "Invalid JSON in configuration file '$ConfigPath': $($_.Exception.Message)"
    }

    foreach ($key in $Changes.Keys) {
        Set-JsonProperty -Object $config -Name ([string]$key) -Value $Changes[$key]
    }

    $json = $config | ConvertTo-Json -Depth 100
    try {
        $null = $json | ConvertFrom-Json
    } catch {
        throw "The updated configuration failed JSON validation: $($_.Exception.Message)"
    }

    $directory = Split-Path -Parent $ConfigPath
    $baseName = [System.IO.Path]::GetFileNameWithoutExtension($ConfigPath)
    $extension = [System.IO.Path]::GetExtension($ConfigPath)
    $timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $backupPath = Join-Path $directory "$baseName.backup-$timestamp$extension"
    $counter = 1
    while (Test-Path -LiteralPath $backupPath) {
        $backupPath = Join-Path $directory "$baseName.backup-$timestamp-$counter$extension"
        $counter += 1
    }
    $tempPath = Join-Path $directory ".$baseName.$([System.Guid]::NewGuid().ToString('N')).tmp"
    $replaceBackupPath = Join-Path $directory ".$baseName.$([System.Guid]::NewGuid().ToString('N')).replace-backup"
    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)

    try {
        [System.IO.File]::Copy($ConfigPath, $backupPath, $false)
        [System.IO.File]::WriteAllText($tempPath, $json + [Environment]::NewLine, $utf8NoBom)
        $null = [System.IO.File]::ReadAllText($tempPath, $utf8NoBom) | ConvertFrom-Json
        [System.IO.File]::Replace($tempPath, $ConfigPath, $replaceBackupPath, $true)
        Remove-Item -LiteralPath $replaceBackupPath -Force -ErrorAction SilentlyContinue
    } catch {
        if (Test-Path -LiteralPath $tempPath) {
            Remove-Item -LiteralPath $tempPath -Force -ErrorAction SilentlyContinue
        }
        if (Test-Path -LiteralPath $replaceBackupPath) {
            Remove-Item -LiteralPath $replaceBackupPath -Force -ErrorAction SilentlyContinue
        }
        throw "Configuration write failed: $($_.Exception.Message)"
    }

    return $backupPath
}

function Format-SettingValue {
    param([AllowNull()][object]$Value)
    if ($Value -is [bool]) {
        return $Value.ToString().ToLowerInvariant()
    }
    if ($null -eq $Value) {
        return "<not set>"
    }
    return [string]$Value
}

function Read-WizardValue {
    param(
        [Parameter(Mandatory = $true)][string]$Prompt,
        [string]$DefaultValue = ""
    )
    $label = $Prompt
    if (-not [string]::IsNullOrWhiteSpace($DefaultValue)) {
        $label += " [$DefaultValue]"
    }
    $value = Read-Host $label
    if ([string]::IsNullOrWhiteSpace($value)) {
        return $DefaultValue
    }
    return $value.Trim()
}

function Invoke-ReviewBackendWizard {
    param([Parameter(Mandatory = $true)][string]$RootPath)

    $configPath = Join-Path $RootPath "podcast_transcribe_config.json"
    if (-not (Test-Path -LiteralPath $configPath -PathType Leaf)) {
        throw "Project configuration not found: $configPath"
    }
    try {
        $config = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
    } catch {
        throw "Invalid JSON in configuration file '$configPath': $($_.Exception.Message)"
    }

    $currentBackend = [string](Get-PropertyValue $config "backend" "none")
    $currentBaseUrl = [string](Get-PropertyValue $config "review_base_url" "")
    $currentModel = [string](Get-PropertyValue $config "review_model_name" "")
    $currentStages = Get-ReviewStageSettings -Config $config

    Write-Host ""
    Write-Host "External Review LLM Configuration"
    Write-Host "Config: $configPath"
    Write-Host "Current backend: $(Format-SettingValue $currentBackend)"
    Write-Host "Current URL: $(Format-SettingValue $currentBaseUrl)"
    Write-Host "Current model: $(Format-SettingValue $currentModel)"
    Write-Host "Current profile: $($currentStages.runtime_profile)"
    Write-Host ("Review stages: cleanup={0}, glossary={1}, speaker={2}, episode QA={3}" -f
        (Format-SettingValue $currentStages.transcript_cleanup_review),
        (Format-SettingValue $currentStages.glossary_correction_review),
        (Format-SettingValue $currentStages.speaker_consistency_review),
        (Format-SettingValue $currentStages.episode_qa_review))
    Write-Host "Enter Q at any prompt to cancel without changing the config."
    Write-Host ""

    $address = Read-WizardValue -Prompt "Server address" -DefaultValue $currentBaseUrl
    if (Test-CancelValue $address) {
        Write-Host "Configuration cancelled. No changes were made."
        return $false
    }

    try {
        $candidates = @(Get-ReviewServerCandidates -Address $address -CurrentBaseUrl $currentBaseUrl)
    } catch {
        throw "Invalid server address: $($_.Exception.Message)"
    }
    Write-Host "Discovering models..."
    $discovery = Find-ReviewServer -Candidates $candidates
    if (-not $discovery.Succeeded) {
        Write-Host "Model discovery failed." -ForegroundColor Red
        foreach ($attempt in $discovery.Attempts) {
            Write-Host ("  {0}: {1} - {2}" -f $attempt.BaseUrl, $attempt.Result, $attempt.Detail)
        }
        throw "No compatible OpenAI-style model endpoint was found."
    }

    $backend = $discovery.Backend
    if ($backend -eq "ambiguous") {
        Write-Host "Backend detection is ambiguous; both or neither identifying endpoint responded." -ForegroundColor Yellow
        Write-Host "  1. vLLM"
        Write-Host "  2. LM Studio"
        $backendChoice = Read-WizardValue -Prompt "Choose backend (1, 2, or Q)"
        if (Test-CancelValue $backendChoice) {
            Write-Host "Configuration cancelled. No changes were made."
            return $false
        }
        if ($backendChoice -eq "1") {
            $backend = "vllm"
        } elseif ($backendChoice -eq "2") {
            $backend = "lm_studio"
        } else {
            throw "Unrecognized backend selection."
        }
    } else {
        $backendLabel = if ($backend -eq "vllm") { "vLLM" } else { "LM Studio" }
        $confirmation = Read-WizardValue -Prompt "Detected $backendLabel. Use this backend? (Y/n)" -DefaultValue "Y"
        if (Test-CancelValue $confirmation) {
            Write-Host "Configuration cancelled. No changes were made."
            return $false
        }
        if ($confirmation.ToUpperInvariant() -notin @("Y", "YES")) {
            Write-Host "  1. vLLM"
            Write-Host "  2. LM Studio"
            $backendChoice = Read-WizardValue -Prompt "Choose backend (1, 2, or Q)"
            if (Test-CancelValue $backendChoice) {
                Write-Host "Configuration cancelled. No changes were made."
                return $false
            }
            if ($backendChoice -eq "1") {
                $backend = "vllm"
            } elseif ($backendChoice -eq "2") {
                $backend = "lm_studio"
            } else {
                throw "Unrecognized backend selection."
            }
        }
    }

    Write-Host ""
    Write-Host "Available models:"
    $defaultModelIndex = 1
    for ($index = 0; $index -lt $discovery.ModelIds.Count; $index += 1) {
        $marker = ""
        if ($discovery.ModelIds[$index] -eq $currentModel) {
            $defaultModelIndex = $index + 1
            $marker = " (current)"
        }
        Write-Host ("  {0}. {1}{2}" -f ($index + 1), $discovery.ModelIds[$index], $marker)
    }
    $modelChoice = Read-WizardValue -Prompt "Select model number" -DefaultValue ([string]$defaultModelIndex)
    if (Test-CancelValue $modelChoice) {
        Write-Host "Configuration cancelled. No changes were made."
        return $false
    }
    $modelNumber = 0
    if (-not [int]::TryParse($modelChoice, [ref]$modelNumber) -or
        $modelNumber -lt 1 -or $modelNumber -gt $discovery.ModelIds.Count) {
        throw "Model selection must be a number from 1 to $($discovery.ModelIds.Count)."
    }
    $selectedModel = $discovery.ModelIds[$modelNumber - 1]

    Write-Host "Testing chat completions for '$selectedModel'..."
    $completionTest = Test-ReviewChatCompletion -BaseUrl $discovery.BaseUrl -ModelId $selectedModel -Backend $backend
    if (-not $completionTest.Succeeded) {
        Write-Host "Selected-model completion test failed: $($completionTest.Detail)" -ForegroundColor Red
        throw "The configuration was not changed because the selected model could not serve chat completions."
    }
    Write-Host $completionTest.Detail -ForegroundColor Green

    Write-Host ""
    Write-Host "Review-stage settings:"
    Write-Host "  1. Keep current settings (default)"
    Write-Host "  2. Enable local stages (cleanup, glossary, speaker consistency)"
    Write-Host "  3. Enable all stages with high_context_5090 (includes episode QA)"
    $stageChoice = Read-WizardValue -Prompt "Choose 1, 2, 3, or Q" -DefaultValue "1"
    if (Test-CancelValue $stageChoice) {
        Write-Host "Configuration cancelled. No changes were made."
        return $false
    }
    $reviewMode = switch ($stageChoice) {
        "1" { "Keep" }
        "2" { "Local" }
        "3" { "All" }
        default { throw "Unrecognized review-stage selection." }
    }

    $changes = Get-ConfigurationChanges -Config $config -Backend $backend -BaseUrl $discovery.BaseUrl `
        -ModelId $selectedModel -ReviewMode $reviewMode
    Write-Host ""
    Write-Host "Configuration preview"
    Write-Host ("{0,-34} {1,-35} {2}" -f "Setting", "Before", "After")
    Write-Host ("{0,-34} {1,-35} {2}" -f ("-" * 30), ("-" * 30), ("-" * 30))
    foreach ($key in $changes.Keys) {
        $before = Get-PropertyValue -Object $config -Name ([string]$key) -DefaultValue $null
        Write-Host ("{0,-34} {1,-35} {2}" -f $key, (Format-SettingValue $before), (Format-SettingValue $changes[$key]))
    }

    $saveChoice = Read-WizardValue -Prompt "Write these changes? (y/N)" -DefaultValue "N"
    if ((Test-CancelValue $saveChoice) -or $saveChoice.ToUpperInvariant() -notin @("Y", "YES")) {
        Write-Host "Configuration cancelled. No changes were made."
        return $false
    }

    $backupPath = Write-ReviewBackendConfiguration -ConfigPath $configPath -Changes $changes
    $updated = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
    $enabledStages = Get-ReviewStageSettings -Config $updated
    $backendLabel = if ($backend -eq "vllm") { "vLLM" } else { "LM Studio" }

    Write-Host ""
    Write-Host "External review LLM configured successfully." -ForegroundColor Green
    Write-Host "Backend: $backendLabel"
    Write-Host "Server URL: $($discovery.BaseUrl)"
    Write-Host "Model: $selectedModel"
    Write-Host ("Review stages: cleanup={0}, glossary={1}, speaker={2}, episode QA={3}" -f
        (Format-SettingValue $enabledStages.transcript_cleanup_review),
        (Format-SettingValue $enabledStages.glossary_correction_review),
        (Format-SettingValue $enabledStages.speaker_consistency_review),
        (Format-SettingValue $enabledStages.episode_qa_review))
    Write-Host "Config: $configPath"
    Write-Host "Backup: $backupPath"
    Write-Host "Use launcher option 4 to benchmark the selected model."
    return $true
}

if (-not $LibraryOnly) {
    $exitCode = 0
    try {
        if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
            $CommonScript = Join-Path $PSScriptRoot "PodcastTranscribeLauncher.Common.ps1"
            . $CommonScript
            $ProjectRoot = Get-PodcastTranscribeProjectRoot -StartPath $PSScriptRoot
        } else {
            $ProjectRoot = (Resolve-Path -LiteralPath $ProjectRoot -ErrorAction Stop).Path
        }
        $null = Invoke-ReviewBackendWizard -RootPath $ProjectRoot
    } catch {
        Write-Error $_.Exception.Message
        $exitCode = 1
    } finally {
        if (-not $NoPause) {
            Write-Host ""
            Read-Host "Press Enter to close"
        }
    }
    if ($exitCode -ne 0) {
        exit $exitCode
    }
}

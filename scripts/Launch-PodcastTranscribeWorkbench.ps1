param()

$CommonScript = Join-Path $PSScriptRoot "PodcastTranscribeLauncher.Common.ps1"
. $CommonScript

$LauncherContext = Get-PodcastTranscribeLauncherContext -ScriptRoot $PSScriptRoot
$ProjectRoot = $LauncherContext.ProjectRoot
$Config = $LauncherContext.Config

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

function Select-Folder {
    param(
        [string]$Description,
        [string]$InitialFolder
    )

    Add-Type -AssemblyName System.Windows.Forms
    $dialog = New-Object System.Windows.Forms.FolderBrowserDialog -Property @{
        RootFolder  = "MyComputer"
        Description = $Description
    }
    if ($InitialFolder -and (Test-Path -LiteralPath $InitialFolder)) {
        $dialog.SelectedPath = $InitialFolder
    }
    $null = $dialog.ShowDialog()
    return $dialog.SelectedPath
}

function Test-CommandAvailable {
    param([string]$Name)
    return $null -ne (Get-Command $Name -ErrorAction SilentlyContinue)
}

function Resolve-NpmCommand {
    $npmCmd = Get-Command "npm.cmd" -ErrorAction SilentlyContinue
    if ($npmCmd) {
        return $npmCmd.Source
    }

    $npm = Get-Command "npm" -ErrorAction SilentlyContinue
    if ($npm) {
        return $npm.Source
    }

    return $null
}

function Get-FrontendTrackedInputFiles {
    param(
        [string]$WorkbenchUiRoot
    )

    $trackedFiles = @()
    $trackedFiles += Get-ChildItem -LiteralPath (Join-Path $WorkbenchUiRoot "src") -Recurse -File -ErrorAction SilentlyContinue
    $trackedFiles += Get-ChildItem -LiteralPath $WorkbenchUiRoot -Filter "package.json" -File -ErrorAction SilentlyContinue
    $trackedFiles += Get-ChildItem -LiteralPath $WorkbenchUiRoot -Filter "tsconfig*.json" -File -ErrorAction SilentlyContinue
    $trackedFiles += Get-ChildItem -LiteralPath $WorkbenchUiRoot -Filter "vite.config.ts" -File -ErrorAction SilentlyContinue
    $trackedFiles += Get-ChildItem -LiteralPath $WorkbenchUiRoot -Filter "index.html" -File -ErrorAction SilentlyContinue
    return $trackedFiles | Sort-Object FullName -Unique
}

function Get-FrontendBuildState {
    param(
        [string]$WorkbenchUiRoot,
        [string]$FrontendDistPath
    )

    if (-not (Test-Path -LiteralPath $FrontendDistPath)) {
        return "missing"
    }

    $distItem = Get-Item -LiteralPath $FrontendDistPath -ErrorAction Stop
    $trackedFiles = Get-FrontendTrackedInputFiles -WorkbenchUiRoot $WorkbenchUiRoot
    foreach ($file in $trackedFiles) {
        if ($file.LastWriteTimeUtc -gt $distItem.LastWriteTimeUtc) {
            return "stale"
        }
    }

    return "ready"
}

function Invoke-FrontendCommand {
    param(
        [string]$NpmExe,
        [string]$WorkbenchUiRoot,
        [string[]]$Arguments,
        [string]$LogPath
    )

    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $LogPath) | Out-Null
    if (Test-Path -LiteralPath $LogPath) {
        Remove-Item -LiteralPath $LogPath -Force -ErrorAction SilentlyContinue
    }

    $previousLocation = Get-Location
    try {
        Set-Location $WorkbenchUiRoot
        & $NpmExe @Arguments *> $LogPath
        return $LASTEXITCODE
    }
    finally {
        Set-Location $previousLocation
    }
}

function Show-FrontendLogTail {
    param(
        [string]$LogPath,
        [int]$LineCount = 40
    )

    if (Test-Path -LiteralPath $LogPath) {
        Get-Content -LiteralPath $LogPath | Select-Object -Last $LineCount
    }
}

if (-not ("LauncherProcessJob" -as [type])) {
    Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;

public static class LauncherProcessJob
{
    [StructLayout(LayoutKind.Sequential)]
    struct JOBOBJECT_BASIC_LIMIT_INFORMATION
    {
        public long PerProcessUserTimeLimit;
        public long PerJobUserTimeLimit;
        public uint LimitFlags;
        public UIntPtr MinimumWorkingSetSize;
        public UIntPtr MaximumWorkingSetSize;
        public uint ActiveProcessLimit;
        public long Affinity;
        public uint PriorityClass;
        public uint SchedulingClass;
    }

    [StructLayout(LayoutKind.Sequential)]
    struct IO_COUNTERS
    {
        public ulong ReadOperationCount;
        public ulong WriteOperationCount;
        public ulong OtherOperationCount;
        public ulong ReadTransferCount;
        public ulong WriteTransferCount;
        public ulong OtherTransferCount;
    }

    [StructLayout(LayoutKind.Sequential)]
    struct JOBOBJECT_EXTENDED_LIMIT_INFORMATION
    {
        public JOBOBJECT_BASIC_LIMIT_INFORMATION BasicLimitInformation;
        public IO_COUNTERS IoInfo;
        public UIntPtr ProcessMemoryLimit;
        public UIntPtr JobMemoryLimit;
        public UIntPtr PeakProcessMemoryUsed;
        public UIntPtr PeakJobMemoryUsed;
    }

    const int JobObjectExtendedLimitInformation = 9;
    const uint JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000;

    [DllImport("kernel32.dll", CharSet = CharSet.Unicode)]
    static extern IntPtr CreateJobObject(IntPtr lpJobAttributes, string lpName);

    [DllImport("kernel32.dll", SetLastError = true)]
    static extern bool SetInformationJobObject(
        IntPtr hJob,
        int JobObjectInfoClass,
        IntPtr lpJobObjectInfo,
        uint cbJobObjectInfoLength);

    [DllImport("kernel32.dll", SetLastError = true)]
    static extern bool AssignProcessToJobObject(IntPtr hJob, IntPtr hProcess);

    static IntPtr _jobHandle = IntPtr.Zero;

    public static void EnsureCreated()
    {
        if (_jobHandle != IntPtr.Zero)
        {
            return;
        }

        var handle = CreateJobObject(IntPtr.Zero, null);
        if (handle == IntPtr.Zero)
        {
            throw new InvalidOperationException("Unable to create launcher job object.");
        }

        var info = new JOBOBJECT_EXTENDED_LIMIT_INFORMATION();
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        int length = Marshal.SizeOf(typeof(JOBOBJECT_EXTENDED_LIMIT_INFORMATION));
        IntPtr pointer = Marshal.AllocHGlobal(length);
        try
        {
            Marshal.StructureToPtr(info, pointer, false);
            if (!SetInformationJobObject(handle, JobObjectExtendedLimitInformation, pointer, (uint)length))
            {
                throw new InvalidOperationException("Unable to set launcher job object limits.");
            }
        }
        finally
        {
            Marshal.FreeHGlobal(pointer);
        }

        _jobHandle = handle;
    }

    public static void AddProcess(int processId)
    {
        EnsureCreated();
        var process = System.Diagnostics.Process.GetProcessById(processId);
        if (!AssignProcessToJobObject(_jobHandle, process.Handle))
        {
            throw new InvalidOperationException("Unable to assign child process to launcher job object.");
        }
    }
}
"@
}

function Start-SessionBoundProcess {
    param(
        [string]$FilePath,
        [string[]]$ArgumentList
    )

    [LauncherProcessJob]::EnsureCreated()
    $process = Start-Process -FilePath $FilePath -ArgumentList $ArgumentList -WindowStyle Hidden -PassThru
    [LauncherProcessJob]::AddProcess($process.Id)
    return $process
}

function Stop-SessionBoundProcess {
    param(
        $Process
    )

    if ($null -eq $Process) {
        return
    }

    try {
        if (-not $Process.HasExited) {
            Stop-Process -Id $Process.Id -Force -ErrorAction SilentlyContinue
        }
    } catch {
    }
}

function Test-WorkbenchPythonDependencies {
    $dependencyCheck = @"
import importlib
required = ['fastapi', 'uvicorn']
missing = []
for name in required:
    try:
        importlib.import_module(name)
    except Exception:
        missing.append(name)
if missing:
    raise SystemExit('MISSING:' + '|'.join(missing))
"@
    & python -c $dependencyCheck 2>&1
    return $LASTEXITCODE
}

function Wait-ForHttpEndpoint {
    param(
        [string]$Url,
        [int]$TimeoutSeconds = 30
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        try {
            Invoke-RestMethod -Uri $Url -Method Get -TimeoutSec 3 | Out-Null
            return $true
        } catch {
            Start-Sleep -Milliseconds 750
        }
    }
    return $false
}

$ConfiguredSourceFolder = Resolve-ConfigPathValue $(if ($Config.default_source_dir) { $Config.default_source_dir } else { $null })
$OutputFolder = if ($ConfiguredSourceFolder -and (Test-Path -LiteralPath $ConfiguredSourceFolder)) {
    Join-Path (Split-Path -Path $ConfiguredSourceFolder -Parent) "output"
} else {
    Join-Path $ProjectRoot "output"
}

if (-not (Test-Path -LiteralPath $OutputFolder)) {
    $SelectedOutputFolder = Select-Folder -Description "Select the processed output folder for the transcript review workbench." -InitialFolder $OutputFolder
    if ([string]::IsNullOrWhiteSpace($SelectedOutputFolder)) {
        Write-Error "Output folder not selected."
        pause
        exit 1
    }
    $OutputFolder = $SelectedOutputFolder
}

$WorkbenchUiRoot = Join-Path $ProjectRoot "workbench-ui"
$FrontendDistPath = Join-Path $WorkbenchUiRoot "dist\index.html"
$FrontendNodeModulesPath = Join-Path $WorkbenchUiRoot "node_modules"
$BackendLogDir = Join-Path $ProjectRoot ".workbench\logs"
$BackendLogPath = Join-Path $BackendLogDir "workbench-backend.log"
$FrontendLogPath = Join-Path $BackendLogDir "workbench-frontend.log"
$BackendPort = 8765
$BackendUrl = "http://127.0.0.1:$BackendPort"

New-Item -ItemType Directory -Force -Path $BackendLogDir | Out-Null

conda activate podcast-transcribe
$env:PYTHONNOUSERSITE = "1"
$env:PYTHONPATH = "$(Join-Path $ProjectRoot 'src');$env:PYTHONPATH"
$PythonExe = (Get-Command python -ErrorAction Stop).Source

$dependencyCheckOutput = Test-WorkbenchPythonDependencies
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Error "The active 'podcast-transcribe' environment is missing workbench Python packages."
    Write-Host "Suggested command: pip install -r podcast_transcribe_requirements.txt"
    if ($dependencyCheckOutput) {
        Write-Host $dependencyCheckOutput
    }
    pause
    exit 1
}

Write-Host ""
Write-Host "Transcript Review Workbench"
Write-Host "Project root: $ProjectRoot"
Write-Host "Output folder: $OutputFolder"
Write-Host "Backend URL: $BackendUrl"
$backendProcess = $null

try {
    $frontendState = Get-FrontendBuildState -WorkbenchUiRoot $WorkbenchUiRoot -FrontendDistPath $FrontendDistPath
    if ($frontendState -ne "ready") {
        $NpmExe = Resolve-NpmCommand
        if ([string]::IsNullOrWhiteSpace($NpmExe)) {
            Write-Host ""
            Write-Error "Workbench frontend is $frontendState and Node.js/npm is not available. Install Node.js so option 5 can set up the workbench frontend automatically."
            pause
            exit 1
        }

        if (-not (Test-Path -LiteralPath $FrontendNodeModulesPath)) {
            Write-Host "Installing workbench frontend dependencies..."
            $installExitCode = Invoke-FrontendCommand -NpmExe $NpmExe -WorkbenchUiRoot $WorkbenchUiRoot -Arguments @("install") -LogPath $FrontendLogPath
            if ($installExitCode -ne 0) {
                Write-Host ""
                Write-Error "Workbench frontend dependency installation failed."
                Show-FrontendLogTail -LogPath $FrontendLogPath
                pause
                exit 1
            }
        }

        Write-Host "Building workbench frontend..."
        $buildExitCode = Invoke-FrontendCommand -NpmExe $NpmExe -WorkbenchUiRoot $WorkbenchUiRoot -Arguments @("run", "build") -LogPath $FrontendLogPath
        if ($buildExitCode -ne 0 -or -not (Test-Path -LiteralPath $FrontendDistPath)) {
            Write-Host ""
            Write-Error "Workbench frontend build failed."
            Show-FrontendLogTail -LogPath $FrontendLogPath
            pause
            exit 1
        }
    }

    $backendCommand = "Set-Location '$ProjectRoot'; `$env:PYTHONNOUSERSITE='1'; `$env:PYTHONPATH='$(Join-Path $ProjectRoot 'src')'; & '$PythonExe' -m podcast_transcribe.workbench_api --host 127.0.0.1 --port $BackendPort --project-root '$ProjectRoot' --output-dir '$OutputFolder' *> '$BackendLogPath'; exit `$LASTEXITCODE"
    $backendProcess = Start-SessionBoundProcess -FilePath "powershell" -ArgumentList @("-NoProfile", "-Command", $backendCommand)

    if (-not (Wait-ForHttpEndpoint -Url "$BackendUrl/api/health" -TimeoutSeconds 30)) {
        Write-Host ""
        Write-Error "Workbench backend did not start successfully."
        if (Test-Path -LiteralPath $BackendLogPath) {
            Get-Content -LiteralPath $BackendLogPath | Select-Object -Last 40
        }
        pause
        exit 1
    }

    Start-Process $BackendUrl | Out-Null
    Write-Host "Workbench launched: $BackendUrl"
    Write-Host "Backend log: $BackendLogPath"
    if (Test-Path -LiteralPath $FrontendLogPath) {
        Write-Host "Frontend log: $FrontendLogPath"
    }
    Write-Host "Closing this launcher window will stop the workbench backend."
    pause
}
finally {
    Stop-SessionBoundProcess -Process $backendProcess
}

param(
    [string] $Task = "Inventory-Leveling-v0",
    [int] $NumEnvs = 1024,
    [int] $ChunkIterations = 1500,
    [double] $MinDurationHours = 5.5,
    [string] $ExperimentName = "inventory_leveling",
    [string] $RunNamePrefix = "auto_extend",
    [string] $InitialRunDir = "",
    [int] $ActiveProcessId = 0,
    [int] $PollSeconds = 60,
    [int] $ArtifactVideoLength = 600,
    [switch] $GenerateArtifacts
)

$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$runnerScript = Join-Path $projectRoot "scripts\tools\run_python.ps1"
$trainScript = Join-Path $projectRoot "scripts\rsl_rl\train.py"
$artifactScript = Join-Path $projectRoot "scripts\generate_comparison_artifacts.py"
$logRoot = Join-Path $projectRoot ("logs\rsl_rl\" + $ExperimentName)
$launchLogDir = Join-Path $projectRoot "artifacts\launch_logs"
New-Item -ItemType Directory -Force -Path $launchLogDir | Out-Null

$stamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
$supervisorLog = Join-Path $launchLogDir ("auto_extend_" + $stamp + ".log")
$latestStatusFile = Join-Path $launchLogDir "auto_extend_latest.txt"

function Write-Log {
    param([string] $Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Add-Content -Path $supervisorLog -Value $line -Encoding UTF8
    Set-Content -Path $latestStatusFile -Value $line -Encoding UTF8
    Write-Output $line
}

function Get-RunName {
    param([string] $RunDir)
    if (-not $RunDir) {
        return $null
    }
    return Split-Path -Leaf $RunDir
}

function Get-LatestRunDir {
    if (-not (Test-Path $logRoot)) {
        return $null
    }
    $run = Get-ChildItem $logRoot -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($null -eq $run) {
        return $null
    }
    return $run.FullName
}

function Get-LatestCheckpoint {
    param([string] $RunDir)

    if (-not $RunDir -or -not (Test-Path $RunDir)) {
        return $null
    }

    $checkpoint = Get-ChildItem $RunDir -File -Filter "model_*.pt" |
        Sort-Object @{Expression = {
            if ($_.BaseName -match '^model_(\d+)$') {
                [int]$matches[1]
            } else {
                -1
            }
        }} |
        Select-Object -Last 1

    if ($null -eq $checkpoint) {
        return $null
    }
    return $checkpoint.Name
}

function Wait-ForProcess {
    param(
        [int] $ProcessId,
        [datetime] $Deadline,
        [string] $KnownRunDir = ""
    )

    while ($true) {
        $proc = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
        if ($null -eq $proc) {
            Write-Log ("PID {0} finished." -f $ProcessId)
            break
        }

        $remaining = $Deadline - (Get-Date)
        $remainingText = if ($remaining.TotalSeconds -gt 0) {
            "{0:hh\:mm\:ss}" -f $remaining
        } else {
            "deadline passed"
        }
        $checkpoint = Get-LatestCheckpoint -RunDir $KnownRunDir
        if ($checkpoint) {
            Write-Log ("PID {0} still running. Remaining target window: {1}. Latest checkpoint: {2}" -f $ProcessId, $remainingText, $checkpoint)
        } else {
            Write-Log ("PID {0} still running. Remaining target window: {1}." -f $ProcessId, $remainingText)
        }

        Start-Sleep -Seconds $PollSeconds
    }
}

function Start-ResumeChunk {
    param(
        [string] $ResumeRunDir,
        [string] $CheckpointName,
        [int] $ChunkIndex
    )

    $resumeRunName = Get-RunName -RunDir $ResumeRunDir
    $timestamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
    $runName = "{0}_chunk_{1}_{2}" -f $RunNamePrefix, $ChunkIndex, $timestamp
    $stdoutLog = Join-Path $launchLogDir ("resume_" + $timestamp + ".stdout.log")
    $stderrLog = Join-Path $launchLogDir ("resume_" + $timestamp + ".stderr.log")

    $argList = @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", $runnerScript,
        $trainScript,
        "--task", $Task,
        "--num_envs", "$NumEnvs",
        "--max_iterations", "$ChunkIterations",
        "--run_name", $runName,
        "--headless",
        "--resume",
        "--load_run", $resumeRunName,
        "--checkpoint", $CheckpointName
    )

    $proc = Start-Process -FilePath "powershell.exe" -ArgumentList $argList -WorkingDirectory $projectRoot `
        -RedirectStandardOutput $stdoutLog -RedirectStandardError $stderrLog -PassThru

    Write-Log ("Started resume chunk {0}. PID={1}, resume_run={2}, checkpoint={3}" -f $ChunkIndex, $proc.Id, $resumeRunName, $CheckpointName)
    Start-Sleep -Seconds 20

    $newRunDir = Get-LatestRunDir
    if ($newRunDir) {
        Write-Log ("Newest run directory after launch: {0}" -f $newRunDir)
    }

    return @{
        Process = $proc
        StdoutLog = $stdoutLog
        StderrLog = $stderrLog
        RunDir = $newRunDir
    }
}

function Generate-LearnedArtifacts {
    param([string] $RunDir)

    $checkpoint = Get-LatestCheckpoint -RunDir $RunDir
    if (-not $checkpoint) {
        Write-Log ("Artifact generation skipped: no checkpoint found in {0}" -f $RunDir)
        return
    }

    $checkpointPath = Join-Path $RunDir $checkpoint
    Write-Log ("Generating learned artifacts from {0}" -f $checkpointPath)
    & $runnerScript $artifactScript `
        --mode learned `
        --task Inventory-Leveling-Play-v0 `
        --video_length $ArtifactVideoLength `
        --num_envs 1 `
        --checkpoint $checkpointPath `
        --headless
}

$deadline = (Get-Date).AddHours($MinDurationHours)
Write-Log ("Supervisor started. Target minimum duration: {0} hours, deadline: {1}" -f $MinDurationHours, $deadline.ToString("yyyy-MM-dd HH:mm:ss"))

$currentRunDir = if ($InitialRunDir) { $InitialRunDir } else { Get-LatestRunDir }
if (-not $currentRunDir) {
    throw "Could not resolve a starting run directory under $logRoot"
}

Write-Log ("Initial run directory: {0}" -f $currentRunDir)

if ($ActiveProcessId -gt 0) {
    Write-Log ("Watching existing training process PID={0}" -f $ActiveProcessId)
    Wait-ForProcess -ProcessId $ActiveProcessId -Deadline $deadline -KnownRunDir $currentRunDir
}

$chunkIndex = 0
while ((Get-Date) -lt $deadline) {
    $checkpoint = Get-LatestCheckpoint -RunDir $currentRunDir
    if (-not $checkpoint) {
        Write-Log ("No checkpoint found yet in {0}. Waiting {1}s." -f $currentRunDir, $PollSeconds)
        Start-Sleep -Seconds $PollSeconds
        continue
    }

    $chunkIndex += 1
    $launch = Start-ResumeChunk -ResumeRunDir $currentRunDir -CheckpointName $checkpoint -ChunkIndex $chunkIndex
    $currentRunDir = $launch.RunDir
    Wait-ForProcess -ProcessId $launch.Process.Id -Deadline $deadline -KnownRunDir $currentRunDir

    $latestRunDir = Get-LatestRunDir
    if ($latestRunDir) {
        $currentRunDir = $latestRunDir
        Write-Log ("Current resume source updated to latest run: {0}" -f $currentRunDir)
    }
}

Write-Log "Minimum wall-clock duration reached. No new resume chunks will be launched."

if ($GenerateArtifacts) {
    Generate-LearnedArtifacts -RunDir $currentRunDir
}

Write-Log "Supervisor finished."

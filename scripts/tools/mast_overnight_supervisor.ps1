param(
    [string] $Stage1RunDir = "R:\Frbot_Lift\logs\rsl_rl\inventory_leveling_mast_flat\2026-03-24_22-46-02",
    [int] $Stage1ProcessId = 55596,
    [string] $DeadlineLocal = "2026-03-25 09:00:00",
    [int] $PrimaryNumEnvs = 4096,
    [int] $FallbackNumEnvs = 2048,
    [int] $Stage1TargetIteration = 3000,
    [int] $Stage2TargetIteration = 7000,
    [int] $Stage1ExtensionIterations = 1000,
    [int] $Stage2ExtensionIterations = 2000,
    [int] $PollSeconds = 120,
    [int] $CheckpointStaleMinutes = 10,
    [int] $ArtifactVideoLength = 600
)

$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$runnerScript = Join-Path $projectRoot "scripts\tools\run_python.ps1"
$scalarReaderScript = Join-Path $projectRoot "scripts\tools\read_tensorboard_scalars.py"
$trainScript = Join-Path $projectRoot "scripts\rsl_rl\train.py"
$artifactScript = Join-Path $projectRoot "scripts\generate_comparison_artifacts.py"
$condaPython = "C:\Users\HYUNSEO\miniconda3\envs\env_isaaclab_2.3.1\python.exe"
$stage1ExpRoot = Join-Path $projectRoot "logs\rsl_rl\inventory_leveling_mast_flat"
$stage2ExpRoot = Join-Path $projectRoot "logs\rsl_rl\inventory_leveling_mast_adaptive"
$defaultAssetPath = Join-Path $projectRoot "assets\robots\lift_amr_fixedlift_runtime.usda"
$rigidAssetPath = Join-Path $projectRoot "assets\robots\lift_amr_fixedlift_rigidsusp_runtime.usda"
$deadline = [datetime]::Parse($DeadlineLocal)
$continuationGate = [datetime]::Parse("2026-03-25 08:00:00")

$launchLogDir = Join-Path $projectRoot "artifacts\launch_logs"
New-Item -ItemType Directory -Force -Path $launchLogDir | Out-Null
$resultsRoot = Join-Path $projectRoot ("results\" + $deadline.ToString("yyyy-MM-dd") + "\overnight_mast")
New-Item -ItemType Directory -Force -Path $resultsRoot | Out-Null

$stamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
$supervisorLog = Join-Path $launchLogDir ("mast_overnight_" + $stamp + ".log")
$latestStatusFile = Join-Path $launchLogDir "mast_overnight_latest.txt"
$summaryFile = Join-Path $resultsRoot "summary.txt"

function Write-Log {
    param([string] $Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Add-Content -Path $supervisorLog -Value $line -Encoding UTF8
    Write-Output $line
}

function Update-Status {
    param([string[]] $Lines)
    Set-Content -Path $latestStatusFile -Value $Lines -Encoding UTF8
}

function Format-Scalar {
    param($Value)
    if ($null -eq $Value) {
        return "n/a"
    }
    return ("{0:N6}" -f [double] $Value)
}

function Get-RunName {
    param([string] $RunDir)
    if (-not $RunDir) {
        return $null
    }
    return Split-Path -Leaf $RunDir
}

function Get-LatestRunDir {
    param([string] $ExperimentRoot)
    if (-not (Test-Path $ExperimentRoot)) {
        return $null
    }
    $run = Get-ChildItem $ExperimentRoot -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($null -eq $run) {
        return $null
    }
    return $run.FullName
}

function Get-CheckpointInfo {
    param([string] $RunDir)

    if (-not $RunDir -or -not (Test-Path $RunDir)) {
        return $null
    }

    $checkpoint = Get-ChildItem $RunDir -File -Filter "model_*.pt" |
        Sort-Object @{Expression = {
            if ($_.BaseName -match '^model_(\d+)$') {
                [int] $matches[1]
            } else {
                -1
            }
        }} |
        Select-Object -Last 1

    if ($null -eq $checkpoint) {
        return $null
    }

    $iteration = -1
    if ($checkpoint.BaseName -match '^model_(\d+)$') {
        $iteration = [int] $matches[1]
    }

    return [pscustomobject] @{
        Name = $checkpoint.Name
        Path = $checkpoint.FullName
        Iteration = $iteration
        LastWriteTime = $checkpoint.LastWriteTime
    }
}

function Get-LatestEventFile {
    param([string] $RunDir)
    if (-not $RunDir -or -not (Test-Path $RunDir)) {
        return $null
    }
    $eventFile = Get-ChildItem $RunDir -File -Filter "events.out.tfevents.*" |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if ($null -eq $eventFile) {
        return $null
    }
    return $eventFile.FullName
}

function Get-LatestScalarValues {
    param(
        [string] $EventFile,
        [string[]] $Tags
    )

    if (-not $EventFile -or -not (Test-Path $EventFile)) {
        return @{}
    }

    $pyCode = @'
import sys
from tensorboard.backend.event_processing import event_accumulator

event_file = sys.argv[1]
tags = sys.argv[2:]
acc = event_accumulator.EventAccumulator(event_file)
acc.Reload()
for tag in tags:
    try:
        ev = acc.Scalars(tag)[-1]
        print(f"{tag}\t{ev.step}\t{ev.value}")
    except Exception:
        pass
'@
    $lines = & $condaPython -c $pyCode $EventFile @Tags
    if ($LASTEXITCODE -ne 0 -or -not $lines) {
        return @{}
    }
    $result = @{}
    foreach ($line in $lines) {
        $parts = $line -split "`t"
        if ($parts.Length -ge 3) {
            $nested = @{}
            $nested["step"] = [double] $parts[1]
            $nested["value"] = [double] $parts[2]
            $result[$parts[0]] = $nested
        }
    }
    return $result
}

function Get-RunMetrics {
    param([string] $RunDir)

    $eventFile = Get-LatestEventFile -RunDir $RunDir
    if (-not $eventFile) {
        return $null
    }

    $tagMap = [ordered] @{
        MeanReward = "Train/mean_reward"
        MeanEpisodeLength = "Train/mean_episode_length"
        ForwardVelocityReward = "Episode_Reward/forward_velocity"
        MastTopXYDisplacement = "Episode_Reward/mast_top_xy_displacement"
        MastTopXYVelocity = "Episode_Reward/mast_top_xy_velocity"
        MastTopVerticalVelocity = "Episode_Reward/mast_top_vertical_velocity"
        TimeOut = "Episode_Termination/time_out"
        CatastrophicBadOrientation = "Episode_Termination/catastrophic_bad_orientation"
        TaskExcessiveMotion = "Episode_Termination/task_excessive_motion"
        TerrainOutOfBounds = "Episode_Termination/terrain_out_of_bounds"
    }

    $pairArgs = @()
    foreach ($entry in $tagMap.GetEnumerator()) {
        $pairArgs += ("{0}={1}" -f $entry.Key, $entry.Value)
    }

    $lines = & $condaPython $scalarReaderScript $eventFile @pairArgs
    if ($LASTEXITCODE -ne 0 -or -not $lines) {
        return $null
    }

    $steps = @{}
    $values = @{}
    foreach ($line in $lines) {
        $parts = $line -split "`t"
        if ($parts.Length -ge 3) {
            $steps[$parts[0]] = [double] $parts[1]
            $values[$parts[0]] = [double] $parts[2]
        }
    }

    return [pscustomobject] @{
        Step = $(if ($steps.ContainsKey("MeanReward")) { $steps["MeanReward"] } else { $null })
        MeanReward = $(if ($values.ContainsKey("MeanReward")) { $values["MeanReward"] } else { $null })
        MeanEpisodeLength = $(if ($values.ContainsKey("MeanEpisodeLength")) { $values["MeanEpisodeLength"] } else { $null })
        ForwardVelocityReward = $(if ($values.ContainsKey("ForwardVelocityReward")) { $values["ForwardVelocityReward"] } else { $null })
        MastTopXYDisplacement = $(if ($values.ContainsKey("MastTopXYDisplacement")) { $values["MastTopXYDisplacement"] } else { $null })
        MastTopXYVelocity = $(if ($values.ContainsKey("MastTopXYVelocity")) { $values["MastTopXYVelocity"] } else { $null })
        MastTopVerticalVelocity = $(if ($values.ContainsKey("MastTopVerticalVelocity")) { $values["MastTopVerticalVelocity"] } else { $null })
        TimeOut = $(if ($values.ContainsKey("TimeOut")) { $values["TimeOut"] } else { $null })
        CatastrophicBadOrientation = $(if ($values.ContainsKey("CatastrophicBadOrientation")) { $values["CatastrophicBadOrientation"] } else { $null })
        TaskExcessiveMotion = $(if ($values.ContainsKey("TaskExcessiveMotion")) { $values["TaskExcessiveMotion"] } else { $null })
        TerrainOutOfBounds = $(if ($values.ContainsKey("TerrainOutOfBounds")) { $values["TerrainOutOfBounds"] } else { $null })
    }
}

function Update-RunStatus {
    param(
        [string] $StageName,
        [string] $RunDir,
        [int] $ProcessId
    )

    $ckpt = Get-CheckpointInfo -RunDir $RunDir
    $metrics = Get-RunMetrics -RunDir $RunDir
    $lines = @(
        "stage=$StageName",
        "run_dir=$RunDir",
        "process_id=$ProcessId",
        "checkpoint=$($ckpt.Name)",
        "checkpoint_iteration=$($ckpt.Iteration)",
        "checkpoint_time=$($ckpt.LastWriteTime)",
        "metric_step=$($metrics.Step)",
        "mean_reward=$(Format-Scalar $metrics.MeanReward)",
        "mean_episode_length=$(Format-Scalar $metrics.MeanEpisodeLength)",
        "forward_velocity_reward=$(Format-Scalar $metrics.ForwardVelocityReward)",
        "mast_top_xy_displacement=$(Format-Scalar $metrics.MastTopXYDisplacement)",
        "mast_top_xy_velocity=$(Format-Scalar $metrics.MastTopXYVelocity)",
        "mast_top_vertical_velocity=$(Format-Scalar $metrics.MastTopVerticalVelocity)",
        "timeout=$(Format-Scalar $metrics.TimeOut)",
        "catastrophic_bad_orientation=$(Format-Scalar $metrics.CatastrophicBadOrientation)",
        "task_excessive_motion=$(Format-Scalar $metrics.TaskExcessiveMotion)",
        "terrain_out_of_bounds=$(Format-Scalar $metrics.TerrainOutOfBounds)",
        "updated_at=$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    )
    Update-Status -Lines $lines
    return [pscustomobject] @{
        Checkpoint = $ckpt
        Metrics = $metrics
    }
}

function Stop-TrainingProcessTree {
    param(
        [int] $PythonPid,
        [int] $PowerShellPid = 0,
        [int] $CmdPid = 0
    )

    foreach ($procId in @($PythonPid, $PowerShellPid, $CmdPid)) {
        if ($procId -gt 0 -and (Get-Process -Id $procId -ErrorAction SilentlyContinue)) {
            try {
                taskkill /PID $procId /F | Out-Null
            } catch {
            }
        }
    }
}

function Wait-For-CheckpointProgress {
    param(
        [string] $StageName,
        [string] $RunDir,
        [int] $PythonPid,
        [int] $TargetIteration,
        [datetime] $DeadlineTime
    )

    while ($true) {
        $snapshot = Update-RunStatus -StageName $StageName -RunDir $RunDir -ProcessId $PythonPid
        $ckpt = $snapshot.Checkpoint
        $metrics = $snapshot.Metrics
        Write-Log ("{0}: iter={1}, reward={2}, timeout={3}, mast_xy_disp={4}, mast_xy_vel={5}" -f `
            $StageName, `
            $(if ($ckpt) { $ckpt.Iteration } else { "n/a" }), `
            (Format-Scalar $metrics.MeanReward), `
            (Format-Scalar $metrics.TimeOut), `
            (Format-Scalar $metrics.MastTopXYDisplacement), `
            (Format-Scalar $metrics.MastTopXYVelocity))

        if ((Get-Date) -ge $DeadlineTime) {
            Write-Log ("{0}: deadline reached. Stopping training and using latest checkpoint." -f $StageName)
            if (Get-Process -Id $PythonPid -ErrorAction SilentlyContinue) {
                Stop-TrainingProcessTree -PythonPid $PythonPid
            }
            return [pscustomobject] @{
                Result = "deadline"
                RunDir = $RunDir
                Checkpoint = $ckpt
                Metrics = $metrics
            }
        }

        if ($null -eq (Get-Process -Id $PythonPid -ErrorAction SilentlyContinue)) {
            if ($ckpt -and $ckpt.Iteration -ge $TargetIteration) {
                return [pscustomobject] @{
                    Result = "completed"
                    RunDir = $RunDir
                    Checkpoint = $ckpt
                    Metrics = $metrics
                }
            }
            return [pscustomobject] @{
                Result = "crashed"
                RunDir = $RunDir
                Checkpoint = $ckpt
                Metrics = $metrics
            }
        }

        if ($ckpt -and ((Get-Date) - $ckpt.LastWriteTime).TotalMinutes -ge $CheckpointStaleMinutes) {
            Write-Log ("{0}: checkpoint stale for >= {1} minutes. Stopping current run." -f $StageName, $CheckpointStaleMinutes)
            Stop-TrainingProcessTree -PythonPid $PythonPid
            return [pscustomobject] @{
                Result = "stale"
                RunDir = $RunDir
                Checkpoint = $ckpt
                Metrics = $metrics
            }
        }

        Start-Sleep -Seconds $PollSeconds
    }
}

function Start-VisibleTrainingJob {
    param(
        [string] $Task,
        [string] $ExperimentRoot,
        [int] $NumEnvs,
        [int] $MaxIterations,
        [string] $WindowTitle,
        [string] $RunName = "",
        [string] $ResumeRun = "",
        [string] $Checkpoint = ""
    )

    $cmdLine = "title $WindowTitle && cd /d R:\Frbot_Lift && call scripts\tools\run_python_visible.cmd scripts\rsl_rl\train.py --task $Task --num_envs $NumEnvs --max_iterations $MaxIterations --headless"
    if ($RunName) {
        $cmdLine += " --run_name $RunName"
    }
    if ($ResumeRun -and $Checkpoint) {
        $cmdLine += " --resume --load_run $ResumeRun --checkpoint $Checkpoint"
    }

    $cmdProc = Start-Process -FilePath "cmd.exe" -ArgumentList "/k $cmdLine" -PassThru
    Start-Sleep -Seconds 8
    $psChild = Get-CimInstance Win32_Process | Where-Object {
        $_.ParentProcessId -eq $cmdProc.Id -and $_.Name -eq "powershell.exe"
    } | Select-Object -First 1
    Start-Sleep -Seconds 8
    $pythonChild = if ($psChild) {
        Get-CimInstance Win32_Process | Where-Object {
            $_.ParentProcessId -eq $psChild.ProcessId -and $_.Name -eq "python.exe"
        } | Select-Object -First 1
    }

    $runDir = Get-LatestRunDir -ExperimentRoot $ExperimentRoot
    return [pscustomobject] @{
        CmdPid = $cmdProc.Id
        PowerShellPid = $(if ($psChild) { $psChild.ProcessId } else { 0 })
        PythonPid = $(if ($pythonChild) { $pythonChild.ProcessId } else { 0 })
        RunDir = $runDir
    }
}

function Import-CheckpointIntoExperiment {
    param(
        [string] $CheckpointPath,
        [string] $DestinationExperimentRoot,
        [string] $Prefix
    )

    $importRunName = "{0}_{1}" -f $Prefix, (Get-Date -Format "yyyy-MM-dd_HH-mm-ss")
    $destRunDir = Join-Path $DestinationExperimentRoot $importRunName
    New-Item -ItemType Directory -Force -Path $destRunDir | Out-Null
    $destCheckpoint = Join-Path $destRunDir (Split-Path -Leaf $CheckpointPath)
    Copy-Item -Path $CheckpointPath -Destination $destCheckpoint -Force
    return [pscustomobject] @{
        RunName = $importRunName
        RunDir = $destRunDir
        CheckpointName = Split-Path -Leaf $destCheckpoint
        CheckpointPath = $destCheckpoint
    }
}

function Test-Stage1Gate {
    param($Metrics)
    if ($null -eq $Metrics) {
        return $false
    }
    return (
        ([double] $Metrics.TimeOut -ge 0.85) -and
        ([double] $Metrics.CatastrophicBadOrientation -le 0.03) -and
        ([double] $Metrics.TaskExcessiveMotion -le 0.12)
    )
}

function Invoke-ArtifactGeneration {
    param(
        [string] $Mode,
        [string] $Task,
        [string] $CheckpointPath = "",
        [string] $AssetPath = ""
    )

    $previousAsset = $env:FRBOT_LIFT_ASSET_PATH
    try {
        if ($AssetPath) {
            $env:FRBOT_LIFT_ASSET_PATH = $AssetPath
        } else {
            $env:FRBOT_LIFT_ASSET_PATH = $defaultAssetPath
        }

        $args = @(
            $artifactScript,
            "--mode", $Mode,
            "--task", $Task,
            "--video_length", "$ArtifactVideoLength",
            "--num_envs", "1",
            "--headless"
        )
        if ($CheckpointPath) {
            $args += @("--checkpoint", $CheckpointPath)
        }
        & $runnerScript @args
        if ($LASTEXITCODE -ne 0) {
            throw "Artifact generation failed for mode=$Mode task=$Task"
        }
    } finally {
        if ([string]::IsNullOrWhiteSpace($previousAsset)) {
            Remove-Item Env:\FRBOT_LIFT_ASSET_PATH -ErrorAction SilentlyContinue
        } else {
            $env:FRBOT_LIFT_ASSET_PATH = $previousAsset
        }
    }
}

function Copy-ScenarioArtifacts {
    param(
        [string] $ScenarioName,
        [string] $OutputRoot
    )

    $scenarioDir = Join-Path $OutputRoot $ScenarioName
    New-Item -ItemType Directory -Force -Path $scenarioDir | Out-Null

    $srcVideoDir = Join-Path $projectRoot "artifacts\videos"
    $srcMetricsDir = Join-Path $projectRoot "artifacts\metrics"

    Copy-Item (Join-Path $srcVideoDir "baseline_straight.mp4") (Join-Path $scenarioDir "baseline_straight.mp4") -Force
    Copy-Item (Join-Path $srcVideoDir "learned_straight.mp4") (Join-Path $scenarioDir "learned_straight.mp4") -Force
    Copy-Item (Join-Path $srcMetricsDir "tilt_metrics.csv") (Join-Path $scenarioDir "tilt_metrics.csv") -Force
    Copy-Item (Join-Path $srcMetricsDir "baseline_per_env_metrics.csv") (Join-Path $scenarioDir "baseline_per_env_metrics.csv") -Force
    Copy-Item (Join-Path $srcMetricsDir "learned_per_env_metrics.csv") (Join-Path $scenarioDir "learned_per_env_metrics.csv") -Force
    return $scenarioDir
}

function Read-ScenarioMetrics {
    param([string] $ScenarioDir)
    $csvPath = Join-Path $ScenarioDir "tilt_metrics.csv"
    if (-not (Test-Path $csvPath)) {
        return $null
    }
    $rows = Import-Csv $csvPath
    $baseline = $rows | Where-Object { $_.mode -eq "baseline" } | Select-Object -First 1
    $learned = $rows | Where-Object { $_.mode -eq "learned" } | Select-Object -First 1
    return [pscustomobject] @{
        Baseline = $baseline
        Learned = $learned
    }
}

function Invoke-ScenarioEvaluation {
    param(
        [string] $ScenarioName,
        [string] $Task,
        [string] $CheckpointPath
    )

    Write-Log ("Generating {0} baseline artifacts with rigid suspension asset." -f $ScenarioName)
    Invoke-ArtifactGeneration -Mode "baseline" -Task $Task -AssetPath $rigidAssetPath
    Write-Log ("Generating {0} learned artifacts from checkpoint {1}" -f $ScenarioName, $CheckpointPath)
    Invoke-ArtifactGeneration -Mode "learned" -Task $Task -CheckpointPath $CheckpointPath -AssetPath $defaultAssetPath
    $scenarioDir = Copy-ScenarioArtifacts -ScenarioName $ScenarioName -OutputRoot $resultsRoot
    return Read-ScenarioMetrics -ScenarioDir $scenarioDir
}

function Test-WeakPerformance {
    param(
        $FlatMetrics,
        $RoughMetrics
    )

    if ($null -eq $FlatMetrics -or $null -eq $RoughMetrics) {
        return $true
    }

    $flatBaseline = $FlatMetrics.Baseline
    $flatLearned = $FlatMetrics.Learned
    $roughBaseline = $RoughMetrics.Baseline
    $roughLearned = $RoughMetrics.Learned

    if ($null -eq $flatBaseline -or $null -eq $flatLearned -or $null -eq $roughBaseline -or $null -eq $roughLearned) {
        return $true
    }

    $flatBetter = (
        ([double] $flatLearned.mast_top_xy_rms_m -lt [double] $flatBaseline.mast_top_xy_rms_m) -and
        ([double] $flatLearned.mast_top_xy_vel_rms_mps -lt [double] $flatBaseline.mast_top_xy_vel_rms_mps)
    )
    $roughBetter = (
        ([double] $roughLearned.mast_top_xy_rms_m -lt [double] $roughBaseline.mast_top_xy_rms_m) -and
        ([double] $roughLearned.mast_top_xy_vel_rms_mps -lt [double] $roughBaseline.mast_top_xy_vel_rms_mps)
    )
    $speedOk = ([double] $roughLearned.mean_forward_velocity_mps) -ge ([double] $roughBaseline.mean_forward_velocity_mps * 0.5)

    return -not ($flatBetter -and $roughBetter -and $speedOk)
}

function Write-Summary {
    param(
        [string] $Stage1RunDirFinal,
        [string] $Stage1CheckpointPath,
        [string] $Stage2RunDirFinal,
        [string] $Stage2CheckpointPath,
        $FlatMetrics,
        $RoughMetrics
    )

    $lines = @(
        "completed_at=$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')",
        "deadline_local=$DeadlineLocal",
        "stage1_run_dir=$Stage1RunDirFinal",
        "stage1_checkpoint=$Stage1CheckpointPath",
        "stage2_run_dir=$Stage2RunDirFinal",
        "stage2_checkpoint=$Stage2CheckpointPath",
        "flat_metrics_dir=$(Join-Path $resultsRoot 'flat')",
        "mild_rough_metrics_dir=$(Join-Path $resultsRoot 'mild_rough')"
    )

    if ($FlatMetrics -and $FlatMetrics.Baseline -and $FlatMetrics.Learned) {
        $lines += "flat_baseline_mast_top_xy_rms_m=$($FlatMetrics.Baseline.mast_top_xy_rms_m)"
        $lines += "flat_learned_mast_top_xy_rms_m=$($FlatMetrics.Learned.mast_top_xy_rms_m)"
        $lines += "flat_baseline_mast_top_xy_vel_rms_mps=$($FlatMetrics.Baseline.mast_top_xy_vel_rms_mps)"
        $lines += "flat_learned_mast_top_xy_vel_rms_mps=$($FlatMetrics.Learned.mast_top_xy_vel_rms_mps)"
        $lines += "flat_baseline_mean_forward_velocity_mps=$($FlatMetrics.Baseline.mean_forward_velocity_mps)"
        $lines += "flat_learned_mean_forward_velocity_mps=$($FlatMetrics.Learned.mean_forward_velocity_mps)"
    }

    if ($RoughMetrics -and $RoughMetrics.Baseline -and $RoughMetrics.Learned) {
        $lines += "rough_baseline_mast_top_xy_rms_m=$($RoughMetrics.Baseline.mast_top_xy_rms_m)"
        $lines += "rough_learned_mast_top_xy_rms_m=$($RoughMetrics.Learned.mast_top_xy_rms_m)"
        $lines += "rough_baseline_mast_top_xy_vel_rms_mps=$($RoughMetrics.Baseline.mast_top_xy_vel_rms_mps)"
        $lines += "rough_learned_mast_top_xy_vel_rms_mps=$($RoughMetrics.Learned.mast_top_xy_vel_rms_mps)"
        $lines += "rough_baseline_mean_forward_velocity_mps=$($RoughMetrics.Baseline.mean_forward_velocity_mps)"
        $lines += "rough_learned_mean_forward_velocity_mps=$($RoughMetrics.Learned.mean_forward_velocity_mps)"
    }

    Set-Content -Path $summaryFile -Value $lines -Encoding UTF8
}

Write-Log ("Overnight mast supervisor started. Deadline: {0}" -f $deadline.ToString("yyyy-MM-dd HH:mm:ss"))
Update-Status -Lines @(
    "mode=mast_overnight_supervisor",
    "status=running",
    "stage=initializing",
    "deadline=$DeadlineLocal",
    "stage1_run_dir=$Stage1RunDir",
    "stage1_process_id=$Stage1ProcessId",
    "updated_at=$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
)

$stage1Outcome = Wait-For-CheckpointProgress -StageName "stage1_flat" -RunDir $Stage1RunDir -PythonPid $Stage1ProcessId -TargetIteration $Stage1TargetIteration -DeadlineTime $deadline
$stage1RunDirFinal = $stage1Outcome.RunDir
$stage1Checkpoint = $stage1Outcome.Checkpoint
$stage1Metrics = $stage1Outcome.Metrics

if (($stage1Outcome.Result -in @("crashed", "stale")) -and $stage1Checkpoint -and $stage1Checkpoint.Iteration -lt $Stage1TargetIteration) {
    Write-Log ("Stage 1 fallback resume triggered due to {0}. Switching to {1} envs." -f $stage1Outcome.Result, $FallbackNumEnvs)
    $resumeRun = Get-RunName -RunDir $stage1RunDirFinal
    $resumeTitle = "Frbot Lift MastFlat Stage1 Overnight Resume"
    $resumeLaunch = Start-VisibleTrainingJob `
        -Task "Inventory-Leveling-MastFlat-v0" `
        -ExperimentRoot $stage1ExpRoot `
        -NumEnvs $FallbackNumEnvs `
        -MaxIterations $Stage1TargetIteration `
        -WindowTitle $resumeTitle `
        -RunName "overnight_stage1_resume" `
        -ResumeRun $resumeRun `
        -Checkpoint $stage1Checkpoint.Name
    $stage1Outcome = Wait-For-CheckpointProgress -StageName "stage1_flat_resume" -RunDir $resumeLaunch.RunDir -PythonPid $resumeLaunch.PythonPid -TargetIteration $Stage1TargetIteration -DeadlineTime $deadline
    $stage1RunDirFinal = $stage1Outcome.RunDir
    $stage1Checkpoint = $stage1Outcome.Checkpoint
    $stage1Metrics = $stage1Outcome.Metrics
}

if (-not (Test-Stage1Gate -Metrics $stage1Metrics)) {
    Write-Log "Stage 1 gate not met. Extending flat training by 1000 iterations."
    $resumeRun = Get-RunName -RunDir $stage1RunDirFinal
    $extendedTarget = $stage1Checkpoint.Iteration + $Stage1ExtensionIterations
    $extensionLaunch = Start-VisibleTrainingJob `
        -Task "Inventory-Leveling-MastFlat-v0" `
        -ExperimentRoot $stage1ExpRoot `
        -NumEnvs $PrimaryNumEnvs `
        -MaxIterations $extendedTarget `
        -WindowTitle "Frbot Lift MastFlat Stage1 Overnight Extend" `
        -RunName "overnight_stage1_extend" `
        -ResumeRun $resumeRun `
        -Checkpoint $stage1Checkpoint.Name
    $stage1Outcome = Wait-For-CheckpointProgress -StageName "stage1_flat_extend" -RunDir $extensionLaunch.RunDir -PythonPid $extensionLaunch.PythonPid -TargetIteration $extendedTarget -DeadlineTime $deadline
    $stage1RunDirFinal = $stage1Outcome.RunDir
    $stage1Checkpoint = $stage1Outcome.Checkpoint
    $stage1Metrics = $stage1Outcome.Metrics
}

Write-Log ("Stage 1 complete. Using checkpoint {0}" -f $stage1Checkpoint.Path)
$flatMetrics = Invoke-ScenarioEvaluation -ScenarioName "flat" -Task "Inventory-Leveling-MastFlat-v0" -CheckpointPath $stage1Checkpoint.Path

$stage2RunDirFinal = ""
$stage2Checkpoint = $null
$roughMetrics = $null

if ((Get-Date) -lt $deadline) {
    $imported = Import-CheckpointIntoExperiment -CheckpointPath $stage1Checkpoint.Path -DestinationExperimentRoot $stage2ExpRoot -Prefix "stage1_import"
    Write-Log ("Stage 2 import checkpoint prepared: {0}" -f $imported.CheckpointPath)
    $stage2Launch = Start-VisibleTrainingJob `
        -Task "Inventory-Leveling-MastAdaptive-v0" `
        -ExperimentRoot $stage2ExpRoot `
        -NumEnvs $PrimaryNumEnvs `
        -MaxIterations $Stage2TargetIteration `
        -WindowTitle "Frbot Lift MastAdaptive Stage2 Overnight" `
        -RunName "overnight_stage2" `
        -ResumeRun $imported.RunName `
        -Checkpoint $imported.CheckpointName

    $stage2Outcome = Wait-For-CheckpointProgress -StageName "stage2_adaptive" -RunDir $stage2Launch.RunDir -PythonPid $stage2Launch.PythonPid -TargetIteration $Stage2TargetIteration -DeadlineTime $deadline
    $stage2RunDirFinal = $stage2Outcome.RunDir
    $stage2Checkpoint = $stage2Outcome.Checkpoint

    if (($stage2Outcome.Result -in @("crashed", "stale")) -and $stage2Checkpoint -and $stage2Checkpoint.Iteration -lt $Stage2TargetIteration) {
        Write-Log ("Stage 2 fallback resume triggered due to {0}. Switching to {1} envs." -f $stage2Outcome.Result, $FallbackNumEnvs)
        $resumeRun = Get-RunName -RunDir $stage2RunDirFinal
        $stage2ResumeLaunch = Start-VisibleTrainingJob `
            -Task "Inventory-Leveling-MastAdaptive-v0" `
            -ExperimentRoot $stage2ExpRoot `
            -NumEnvs $FallbackNumEnvs `
            -MaxIterations $Stage2TargetIteration `
            -WindowTitle "Frbot Lift MastAdaptive Stage2 Overnight Resume" `
            -RunName "overnight_stage2_resume" `
            -ResumeRun $resumeRun `
            -Checkpoint $stage2Checkpoint.Name
        $stage2Outcome = Wait-For-CheckpointProgress -StageName "stage2_adaptive_resume" -RunDir $stage2ResumeLaunch.RunDir -PythonPid $stage2ResumeLaunch.PythonPid -TargetIteration $Stage2TargetIteration -DeadlineTime $deadline
        $stage2RunDirFinal = $stage2Outcome.RunDir
        $stage2Checkpoint = $stage2Outcome.Checkpoint
    }

    if ($stage2Checkpoint) {
        Write-Log ("Stage 2 checkpoint selected: {0}" -f $stage2Checkpoint.Path)
        $roughMetrics = Invoke-ScenarioEvaluation -ScenarioName "mild_rough" -Task "Inventory-Leveling-MastPlay-v0" -CheckpointPath $stage2Checkpoint.Path

        if ((Get-Date) -lt $continuationGate -and (Test-WeakPerformance -FlatMetrics $flatMetrics -RoughMetrics $roughMetrics)) {
            Write-Log "Stage 2 performance is still weak before 08:00. Launching one 2000-iteration continuation at 2048 envs."
            $resumeRun = Get-RunName -RunDir $stage2RunDirFinal
            $extraTarget = $stage2Checkpoint.Iteration + $Stage2ExtensionIterations
            $stage2ExtraLaunch = Start-VisibleTrainingJob `
                -Task "Inventory-Leveling-MastAdaptive-v0" `
                -ExperimentRoot $stage2ExpRoot `
                -NumEnvs $FallbackNumEnvs `
                -MaxIterations $extraTarget `
                -WindowTitle "Frbot Lift MastAdaptive Stage2 Overnight Extra" `
                -RunName "overnight_stage2_extra" `
                -ResumeRun $resumeRun `
                -Checkpoint $stage2Checkpoint.Name
            $stage2Outcome = Wait-For-CheckpointProgress -StageName "stage2_adaptive_extra" -RunDir $stage2ExtraLaunch.RunDir -PythonPid $stage2ExtraLaunch.PythonPid -TargetIteration $extraTarget -DeadlineTime $deadline
            $stage2RunDirFinal = $stage2Outcome.RunDir
            $stage2Checkpoint = $stage2Outcome.Checkpoint
            $roughMetrics = Invoke-ScenarioEvaluation -ScenarioName "mild_rough" -Task "Inventory-Leveling-MastPlay-v0" -CheckpointPath $stage2Checkpoint.Path
        }
    }
}

Write-Summary `
    -Stage1RunDirFinal $stage1RunDirFinal `
    -Stage1CheckpointPath $stage1Checkpoint.Path `
    -Stage2RunDirFinal $stage2RunDirFinal `
    -Stage2CheckpointPath $(if ($stage2Checkpoint) { $stage2Checkpoint.Path } else { "" }) `
    -FlatMetrics $flatMetrics `
    -RoughMetrics $roughMetrics

Update-Status -Lines @(
    "mode=mast_overnight_supervisor",
    "status=completed",
    "stage=done",
    "completed_at=$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')",
    "stage1_run_dir=$stage1RunDirFinal",
    "stage1_checkpoint=$($stage1Checkpoint.Path)",
    "stage2_run_dir=$stage2RunDirFinal",
    "stage2_checkpoint=$(if ($stage2Checkpoint) { $stage2Checkpoint.Path } else { '' })",
    "results_root=$resultsRoot",
    "summary_file=$summaryFile"
)
Write-Log "Overnight mast supervisor finished."

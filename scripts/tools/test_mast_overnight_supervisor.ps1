param(
    [string] $CondaPython = "C:\Users\HYUNSEO\miniconda3\envs\env_isaaclab_2.3.1\python.exe"
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$supervisorScript = Join-Path $repoRoot "scripts\tools\mast_overnight_supervisor.ps1"
$scalarReaderSource = Join-Path $repoRoot "scripts\tools\read_tensorboard_scalars.py"

$stamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
$mockRoot = Join-Path $repoRoot ("artifacts\selftest\mast_overnight_" + $stamp)
$resultsSubdirName = "overnight_mast_selftest"
$deadline = (Get-Date).AddMinutes(15)
$continuationGate = (Get-Date).AddMinutes(10)

Write-Host "[selftest] mock root: $mockRoot"

$dirs = @(
    "assets\robots",
    "artifacts\launch_logs",
    "artifacts\metrics",
    "artifacts\videos",
    "logs\rsl_rl\inventory_leveling_mast_flat",
    "logs\rsl_rl\inventory_leveling_mast_adaptive",
    "results",
    "scripts",
    "scripts\rsl_rl",
    "scripts\tools"
)

foreach ($dir in $dirs) {
    New-Item -ItemType Directory -Force -Path (Join-Path $mockRoot $dir) | Out-Null
}

Set-Content -Path (Join-Path $mockRoot "assets\robots\lift_amr_fixedlift_runtime.usda") -Value "#usda 1.0`n" -Encoding UTF8
Set-Content -Path (Join-Path $mockRoot "assets\robots\lift_amr_fixedlift_rigidsusp_runtime.usda") -Value "#usda 1.0`n" -Encoding UTF8
Copy-Item -Path $scalarReaderSource -Destination (Join-Path $mockRoot "scripts\tools\read_tensorboard_scalars.py") -Force

$runPythonContent = @'
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $Args
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $projectRoot
$python = "C:\Users\HYUNSEO\miniconda3\envs\env_isaaclab_2.3.1\python.exe"
& $python @Args
exit $LASTEXITCODE
'@
Set-Content -Path (Join-Path $mockRoot "scripts\tools\run_python.ps1") -Value $runPythonContent -Encoding UTF8

$runPythonVisibleCmd = @'
@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..\..") do set "PROJECT_ROOT=%%~fI"
cd /d "%PROJECT_ROOT%"
title Mock Visible Runner
powershell.exe -NoLogo -ExecutionPolicy Bypass -File "%SCRIPT_DIR%run_python.ps1" %*
set "EXIT_CODE=%ERRORLEVEL%"
echo.
echo [Mock Runner] Process exited with code %EXIT_CODE%.
exit /b %EXIT_CODE%
'@
Set-Content -Path (Join-Path $mockRoot "scripts\tools\run_python_visible.cmd") -Value $runPythonVisibleCmd -Encoding ASCII

$dummyTrainPy = @'
import argparse
import os
import re
import sys
import time
from pathlib import Path

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    from tensorboardX import SummaryWriter


def experiment_name(task: str) -> str:
    if "MastFlat" in task:
        return "inventory_leveling_mast_flat"
    if "MastAdaptive" in task:
        return "inventory_leveling_mast_adaptive"
    return "inventory_leveling_misc"


def parse_iter(name: str) -> int:
    match = re.search(r"model_(\d+)\.pt$", name)
    return int(match.group(1)) if match else 0


def resolve_resume_checkpoint(project_root: Path, exp_name: str, load_run: str, checkpoint: str) -> int:
    if not checkpoint:
        return 0
    ckpt_path = project_root / "logs" / "rsl_rl" / exp_name / load_run / checkpoint
    if ckpt_path.exists():
        return parse_iter(ckpt_path.name)
    return parse_iter(checkpoint)


def metrics_for(task: str, step: int):
    if "MastFlat" in task:
        return {
            "Train/mean_reward": -0.6 + (step * 0.05),
            "Train/mean_episode_length": 640.0 + step,
            "Episode_Reward/forward_velocity": 0.18 + (step * 0.002),
            "Episode_Reward/mast_top_xy_displacement": -0.80 + (step * 0.005),
            "Episode_Reward/mast_top_xy_velocity": -0.11 + (step * 0.004),
            "Episode_Reward/mast_top_vertical_velocity": -0.08 + (step * 0.003),
            "Episode_Termination/time_out": 0.88,
            "Episode_Termination/catastrophic_bad_orientation": 0.02,
            "Episode_Termination/task_excessive_motion": 0.08,
            "Episode_Termination/terrain_out_of_bounds": 0.0,
        }
    return {
        "Train/mean_reward": 0.10 + (step * 0.03),
        "Train/mean_episode_length": 670.0 + step,
        "Episode_Reward/forward_velocity": 0.22 + (step * 0.003),
        "Episode_Reward/mast_top_xy_displacement": -0.70 + (step * 0.004),
        "Episode_Reward/mast_top_xy_velocity": -0.09 + (step * 0.004),
        "Episode_Reward/mast_top_vertical_velocity": -0.07 + (step * 0.003),
        "Episode_Termination/time_out": 0.90,
        "Episode_Termination/catastrophic_bad_orientation": 0.015,
        "Episode_Termination/task_excessive_motion": 0.07,
        "Episode_Termination/terrain_out_of_bounds": 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--max_iterations", type=int, required=True)
    parser.add_argument("--run_name", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--load_run", default="")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--headless", action="store_true")
    args, _ = parser.parse_known_args()

    project_root = Path.cwd()
    exp_name = experiment_name(args.task)
    exp_root = project_root / "logs" / "rsl_rl" / exp_name
    exp_root.mkdir(parents=True, exist_ok=True)

    run_name = args.run_name or f"run_{int(time.time())}"
    run_dir = exp_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    start_iter = 0
    if args.resume:
        start_iter = resolve_resume_checkpoint(project_root, exp_name, args.load_run, args.checkpoint)

    writer = SummaryWriter(str(run_dir))
    fail_stage2_once = (
        "MastAdaptive" in args.task
        and args.run_name == "overnight_stage2"
        and args.num_envs >= 64
    )

    stop_iter = args.max_iterations
    if fail_stage2_once:
        stop_iter = min(args.max_iterations, max(start_iter + 1, args.max_iterations - 1))

    for step in range(start_iter + 1, stop_iter + 1):
        metrics = metrics_for(args.task, step)
        for tag, value in metrics.items():
            writer.add_scalar(tag, value, step)
        writer.flush()
        (run_dir / f"model_{step}.pt").write_bytes(f"checkpoint-{step}".encode("utf-8"))
        time.sleep(1.1)

    writer.close()
    if fail_stage2_once and stop_iter < args.max_iterations:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'@
Set-Content -Path (Join-Path $mockRoot "scripts\rsl_rl\train.py") -Value $dummyTrainPy -Encoding UTF8

$dummyArtifactPy = @'
import argparse
import csv
import re
from pathlib import Path


def parse_iter(path: str) -> int:
    if not path:
        return 0
    match = re.search(r"model_(\d+)\.pt$", path)
    return int(match.group(1)) if match else 0


def rows_for(task: str, mode: str, checkpoint_iter: int):
    if "MastFlat" in task:
        baseline = {
            "mode": "baseline",
            "mast_top_xy_rms_m": 1.00,
            "mast_top_xy_vel_rms_mps": 0.90,
            "mast_top_peak_xy_disp_m": 1.60,
            "base_roll_rms_deg": 5.0,
            "base_pitch_rms_deg": 6.0,
            "mean_forward_velocity_mps": 0.20,
        }
        learned = {
            "mode": "learned",
            "mast_top_xy_rms_m": 0.65,
            "mast_top_xy_vel_rms_mps": 0.55,
            "mast_top_peak_xy_disp_m": 1.00,
            "base_roll_rms_deg": 3.5,
            "base_pitch_rms_deg": 4.0,
            "mean_forward_velocity_mps": 0.22,
        }
        return baseline if mode == "baseline" else learned

    baseline = {
        "mode": "baseline",
        "mast_top_xy_rms_m": 1.20,
        "mast_top_xy_vel_rms_mps": 1.00,
        "mast_top_peak_xy_disp_m": 1.80,
        "base_roll_rms_deg": 6.0,
        "base_pitch_rms_deg": 7.0,
        "mean_forward_velocity_mps": 0.20,
    }
    if checkpoint_iter >= 9:
        learned = {
            "mode": "learned",
            "mast_top_xy_rms_m": 0.80,
            "mast_top_xy_vel_rms_mps": 0.70,
            "mast_top_peak_xy_disp_m": 1.10,
            "base_roll_rms_deg": 4.2,
            "base_pitch_rms_deg": 4.8,
            "mean_forward_velocity_mps": 0.18,
        }
    else:
        learned = {
            "mode": "learned",
            "mast_top_xy_rms_m": 1.35,
            "mast_top_xy_vel_rms_mps": 1.10,
            "mast_top_peak_xy_disp_m": 1.90,
            "base_roll_rms_deg": 6.4,
            "base_pitch_rms_deg": 7.2,
            "mean_forward_velocity_mps": 0.05,
        }
    return baseline if mode == "baseline" else learned


def write_csv(path: Path, rows):
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--video_length", type=int, default=0)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    project_root = Path.cwd()
    videos_dir = project_root / "artifacts" / "videos"
    metrics_dir = project_root / "artifacts" / "metrics"
    videos_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_iter = parse_iter(args.checkpoint)
    row = rows_for(args.task, args.mode, checkpoint_iter)

    video_path = videos_dir / ("baseline_straight.mp4" if args.mode == "baseline" else "learned_straight.mp4")
    video_path.write_bytes(b"mock-video")

    tilt_csv = metrics_dir / "tilt_metrics.csv"
    rows = []
    if tilt_csv.exists():
        with tilt_csv.open("r", newline="", encoding="utf-8") as fp:
            reader = csv.DictReader(fp)
            rows = list(reader)
        rows = [r for r in rows if r.get("mode") != args.mode]
    rows.append(row)
    write_csv(tilt_csv, rows)

    per_env_path = metrics_dir / ("baseline_per_env_metrics.csv" if args.mode == "baseline" else "learned_per_env_metrics.csv")
    per_env_row = dict(row)
    per_env_row["env_id"] = 0
    write_csv(per_env_path, [per_env_row])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'@
Set-Content -Path (Join-Path $mockRoot "scripts\generate_comparison_artifacts.py") -Value $dummyArtifactPy -Encoding UTF8

$initialStage1RunName = "stage1_seed"
$stage1RunDir = Join-Path $mockRoot ("logs\rsl_rl\inventory_leveling_mast_flat\" + $initialStage1RunName)

$stage1Proc = Start-Process -FilePath $CondaPython -ArgumentList @(
    (Join-Path $mockRoot "scripts\rsl_rl\train.py"),
    "--task", "Inventory-Leveling-MastFlat-v0",
    "--num_envs", "64",
    "--max_iterations", "6",
    "--run_name", $initialStage1RunName,
    "--headless"
) -WorkingDirectory $mockRoot -PassThru

Start-Sleep -Seconds 2

Write-Host "[selftest] launching supervisor"
& $supervisorScript `
    -Stage1RunDir $stage1RunDir `
    -Stage1ProcessId $stage1Proc.Id `
    -DeadlineLocal $deadline.ToString("yyyy-MM-dd HH:mm:ss") `
    -ContinuationGateLocal $continuationGate.ToString("yyyy-MM-dd HH:mm:ss") `
    -ProjectRootOverride $mockRoot `
    -CondaPythonOverride $CondaPython `
    -ResultsSubdirName $resultsSubdirName `
    -PrimaryNumEnvs 64 `
    -FallbackNumEnvs 32 `
    -Stage1TargetIteration 6 `
    -Stage2TargetIteration 8 `
    -Stage1ExtensionIterations 2 `
    -Stage2ExtensionIterations 2 `
    -PollSeconds 1 `
    -LaunchProbeDelaySeconds 1 `
    -UseVisibleRunner:$false `
    -CheckpointStaleMinutes 10 `
    -ArtifactVideoLength 16
if ($LASTEXITCODE -ne 0) {
    throw "supervisor process failed with exit code $LASTEXITCODE"
}

$statusFile = Join-Path $mockRoot "artifacts\launch_logs\mast_overnight_latest.txt"
$summaryFile = Join-Path $mockRoot ("results\" + $deadline.ToString("yyyy-MM-dd") + "\" + $resultsSubdirName + "\summary.txt")
$latestLog = Get-ChildItem -Path (Join-Path $mockRoot "artifacts\launch_logs") -Filter "mast_overnight_*.log" |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

if (-not (Test-Path $statusFile)) {
    throw "status file missing: $statusFile"
}
if (-not (Test-Path $summaryFile)) {
    throw "summary file missing: $summaryFile"
}

$status = Get-Content -Path $statusFile
$summary = Get-Content -Path $summaryFile
$flatDir = Join-Path (Split-Path $summaryFile -Parent) "flat"
$roughDir = Join-Path (Split-Path $summaryFile -Parent) "mild_rough"

if ($status -notcontains "status=completed") {
    throw "supervisor did not complete successfully"
}
if (-not (Test-Path (Join-Path $flatDir "tilt_metrics.csv"))) {
    throw "flat artifacts missing"
}
if (-not (Test-Path (Join-Path $roughDir "tilt_metrics.csv"))) {
    throw "rough artifacts missing"
}

$logText = Get-Content -Path $latestLog.FullName -Raw
if ($logText -notmatch "Stage 2 fallback resume triggered due to crashed") {
    throw "stage2 crash fallback path was not exercised"
}
if ($logText -notmatch "Stage 2 performance is still weak before 08:00") {
    throw "stage2 extra continuation path was not exercised"
}

$roughRows = Import-Csv (Join-Path $roughDir "tilt_metrics.csv")
$roughBaseline = $roughRows | Where-Object { $_.mode -eq "baseline" } | Select-Object -First 1
$roughLearned = $roughRows | Where-Object { $_.mode -eq "learned" } | Select-Object -First 1

if ([double] $roughLearned.mast_top_xy_rms_m -ge [double] $roughBaseline.mast_top_xy_rms_m) {
    throw "rough learned mast_top_xy_rms_m did not beat baseline"
}
if ([double] $roughLearned.mast_top_xy_vel_rms_mps -ge [double] $roughBaseline.mast_top_xy_vel_rms_mps) {
    throw "rough learned mast_top_xy_vel_rms_mps did not beat baseline"
}

Write-Host "[selftest] PASS"
Write-Host "[selftest] status: $statusFile"
Write-Host "[selftest] summary: $summaryFile"
Write-Host "[selftest] log: $($latestLog.FullName)"

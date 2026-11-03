# Frbot Lift Leveling

Manager-based Isaac Lab project for an inventory robot that must drive forward while keeping the chassis level with the lift fully extended.

[Project overview](https://home.hyunseo.kr/projects/inventory-inspection-adaptive-suspension/) · [Published evaluation snapshot](results/2026-03-24/README.md)

## Setup and scope

This is a research workspace for suspension-policy training and stability evaluation in simulation. The checked-in Windows launchers contain machine-specific paths. Before running them, update the Conda Python and Isaac Sim/Isaac Lab paths in [run_python.ps1](scripts/tools/run_python.ps1) and [run_python_visible.cmd](scripts/tools/run_python_visible.cmd) for your installation. The `R:` drive alias is part of those launchers, not a GitHub file location.

Clone the repository, configure those launchers, then use the environment-listing command below before starting training. Existing training/play commands assume that the required simulator and robot assets are installed.

## Reference environment

- Conda: `env_isaaclab_2.3.1`
- Isaac Sim: `E:\isaac-sim-5.1.0`
- Isaac Lab: `E:\IsaacLab`
- Workspace root: `E:\...\2603_...\`
- Project root: `R:\Frbot_Lift`

This repo does not modify the original Isaac Lab or Isaac Sim folders.
The local `E:\IsaacLab\isaac_wheel` folder is intentionally ignored.

## Windows Launching

Always use [run_python.ps1](scripts/tools/run_python.ps1) or [run_python_visible.cmd](scripts/tools/run_python_visible.cmd).

These launchers:

- use the ASCII-safe alias `R:\Frbot_Lift`
- set Isaac Sim / Isaac Lab environment variables
- set `PYTHONPATH` for the local project package
- allow overriding `FRBOT_LIFT_ASSET_PATH`
- trim accidental trailing spaces from `FRBOT_LIFT_ASSET_PATH`

## Robot Assets

Source asset:

- [lift_amr_new.usd](assets/robots/lift_amr_new.usd): copied original robot USD

Runtime assets:

- [lift_amr_runtime.usda](assets/robots/lift_amr_runtime.usda): early wrapper asset
- [lift_amr_fixedlift_runtime.usda](assets/robots/lift_amr_fixedlift_runtime.usda): lift fixed at maximum extension, steer locked forward
- `assets/robots/lift_amr_fixedlift_rigidsusp_runtime.usda`: lift fixed, steer locked, suspension joints rigid-locked near midpoint

Asset generation tool:

- [generate_fixedlift_runtime.py](scripts/tools/generate_fixedlift_runtime.py)

Example:

```powershell
.\scripts\tools\run_python.ps1 .\scripts\tools\generate_fixedlift_runtime.py --source .\assets\robots\lift_amr_new.usd --output .\assets\robots\lift_amr_fixedlift_runtime.usda
.\scripts\tools\run_python.ps1 .\scripts\tools\generate_fixedlift_runtime.py --source .\assets\robots\lift_amr_new.usd --output .\assets\robots\lift_amr_fixedlift_rigidsusp_runtime.usda --lock-suspension
```

## Control Assumptions

Current stable setup:

- lift joints are fixed at maximum extension
- steer joints are locked to forward-facing values
- wheel joints provide scripted straight driving
- suspension joints are the only learned DOFs in the stable RL tasks

Wheel-only debug setup:

- no learned policy
- no suspension action term
- rigid suspension USD
- wheel actuator only

## Task Set

Legacy tasks kept for reference:

- `Inventory-Leveling-v0`
- `Inventory-Leveling-Play-v0`

Stable v2 tasks:

- `Inventory-Leveling-StableFlat-v0`
  - flat terrain only
  - manager-based RL training
  - suspension policy + scripted drivetrain
- `Inventory-Leveling-StableAdaptive-v0`
  - flat + mild rough terrain
  - manager-based RL training
  - suspension policy + scripted drivetrain
- `Inventory-Leveling-StablePlay-v0`
  - GUI / playback task for the stable adaptive setup
  - no curriculum during play
  - safe reset path for reliable GUI launch
- `Inventory-Leveling-WheelOnlyPlay-v0`
  - GUI / playback task with rigid suspension USD
  - no learned policy
  - wheel drive only

Task registration lives in [`__init__.py`](source/Frbot_Lift/Frbot_Lift/tasks/manager_based/frbot_lift_stable/__init__.py).

## Main Config Files

- [frbot_lift_asset.py](source/Frbot_Lift/Frbot_Lift/assets/frbot_lift_asset.py)
- [frbot_lift_stable_env_cfg.py](source/Frbot_Lift/Frbot_Lift/tasks/manager_based/frbot_lift_stable/frbot_lift_stable_env_cfg.py)
- [actions.py](source/Frbot_Lift/Frbot_Lift/tasks/manager_based/frbot_lift_stable/mdp/actions.py)
- [bootstrap.py](source/Frbot_Lift/Frbot_Lift/tasks/manager_based/frbot_lift_stable/mdp/bootstrap.py)
- [events.py](source/Frbot_Lift/Frbot_Lift/tasks/manager_based/frbot_lift_stable/mdp/events.py)
- [rewards.py](source/Frbot_Lift/Frbot_Lift/tasks/manager_based/frbot_lift_stable/mdp/rewards.py)
- [terminations.py](source/Frbot_Lift/Frbot_Lift/tasks/manager_based/frbot_lift_stable/mdp/terminations.py)
- [rsl_rl_flat_ppo_cfg.py](source/Frbot_Lift/Frbot_Lift/tasks/manager_based/frbot_lift_stable/agents/rsl_rl_flat_ppo_cfg.py)
- [rsl_rl_adaptive_ppo_cfg.py](source/Frbot_Lift/Frbot_Lift/tasks/manager_based/frbot_lift_stable/agents/rsl_rl_adaptive_ppo_cfg.py)

## What Was Implemented

Implemented over the course of stabilization:

- external Isaac Lab project layout under `R:\Frbot_Lift`
- copied robot USD under version control
- manager-based stable training tasks separated from the broken early task
- fixed-lift runtime USD generation
- steer lock in stable training
- slow scripted drivetrain with long ramp
- mild-rough adaptive curriculum
- GUI-safe play reset path
- debug velocity arrows for drivetrain target/current velocity
- rigid-suspension runtime USD for baseline inspection
- wheel-only play task for non-learning drivetrain-only inspection

## Useful Commands

List registered environments:

```powershell
.\scripts\tools\run_python.ps1 .\scripts\list_envs.py
```

Train stable flat:

```powershell
.\scripts\tools\run_python.ps1 .\scripts\rsl_rl\train.py --task Inventory-Leveling-StableFlat-v0 --num_envs 256 --max_iterations 10000 --headless
```

Train stable adaptive:

```powershell
.\scripts\tools\run_python.ps1 .\scripts\rsl_rl\train.py --task Inventory-Leveling-StableAdaptive-v0 --num_envs 4096 --max_iterations 10000 --headless
```

GUI train with visible CMD:

```powershell
.\scripts\tools\run_python_visible.cmd scripts\rsl_rl\train.py --task Inventory-Leveling-StableFlat-v0 --num_envs 8 --max_iterations 10000
```

Latest-checkpoint GUI play:

```powershell
.\scripts\tools\run_python_visible.cmd scripts\rsl_rl\play.py --task Inventory-Leveling-StablePlay-v0 --num_envs 8 --checkpoint R:\Frbot_Lift\logs\rsl_rl\inventory_leveling_stable_adaptive\<run>\model_<n>.pt
```

Rigid-suspension baseline play:

```powershell
$env:FRBOT_LIFT_ASSET_PATH = "R:\Frbot_Lift\assets\robots\lift_amr_fixedlift_rigidsusp_runtime.usda"
.\scripts\tools\run_python_visible.cmd scripts\zero_agent.py --task Inventory-Leveling-StablePlay-v0 --num_envs 8
```

Wheel-only play on the current training floor:

```powershell
.\scripts\tools\run_python_visible.cmd scripts\zero_agent.py --task Inventory-Leveling-WheelOnlyPlay-v0 --num_envs 8
```

The `scripted_stabilized` presentation mode demonstrates scripted stabilization. Use the `--mode learned` comparison command with a trained checkpoint when evaluating a learned policy.

Presentation baseline GUI:

```powershell
cd R:\Frbot_Lift
.\scripts\tools\run_python_visible.cmd scripts\generate_presentation_comparison.py --mode baseline_uncorrected --task Inventory-Leveling-StableFlat-v0 --num_envs 64 --video_length 5000 --speed_scale 10.0
```

Presentation stabilized GUI:

```powershell
cd R:\Frbot_Lift
.\scripts\tools\run_python_visible.cmd scripts\generate_presentation_comparison.py --mode scripted_stabilized --task Inventory-Leveling-StableFlat-v0 --num_envs 64 --video_length 5000 --speed_scale 10.0
```

Presentation baseline render:

```powershell
cd R:\Frbot_Lift
.\scripts\tools\run_python.ps1 scripts\generate_presentation_comparison.py --mode baseline_uncorrected --task Inventory-Leveling-StableFlat-v0 --num_envs 64 --video_length 500 --speed_scale 10.0 --headless
```

Presentation stabilized render:

```powershell
cd R:\Frbot_Lift
.\scripts\tools\run_python.ps1 scripts\generate_presentation_comparison.py --mode scripted_stabilized --task Inventory-Leveling-StableFlat-v0 --num_envs 64 --video_length 500 --speed_scale 10.0 --headless
```

Presentation demo camera:

- fixed camera eye: `(52, -33, 10)`
- target is still computed from the center of the demo cluster
- both baseline and stabilized use the same camera setup

Generate learned comparison artifacts:

```powershell
.\scripts\tools\run_python.ps1 .\scripts\generate_comparison_artifacts.py --mode learned --task Inventory-Leveling-StablePlay-v0 --video_length 600 --num_envs 1 --checkpoint R:\Frbot_Lift\logs\rsl_rl\inventory_leveling_stable_adaptive\<run>\model_<n>.pt --headless
```

Generate baseline comparison artifacts:

```powershell
.\scripts\tools\run_python.ps1 .\scripts\generate_comparison_artifacts.py --mode baseline --task Inventory-Leveling-StablePlay-v0 --video_length 600 --num_envs 1 --headless
```

## Training snapshot (March 2026)

A recorded run used during tuning is:

- `logs/rsl_rl/inventory_leveling_stable_adaptive/2026-03-24_18-42-27_quickconv_adaptive_mildrough_4096_10k`

At one inspected point around iteration `2097 / 10000`, the run showed:

- mean reward around `0.49`
- mean episode length around `698`
- timeout termination around `96.5%`
- catastrophic bad orientation around `0.1%`
- terrain out of bounds around `0.02%`

This means the stable adaptive task is no longer failing immediately and is mostly surviving full episodes on flat + mild rough terrain.

## Outputs

The committed [March 24 evaluation snapshot](results/2026-03-24/README.md) includes [baseline video](results/2026-03-24/baseline_straight.mp4), [learned-policy video](results/2026-03-24/learned_straight.mp4), and [tilt metrics](results/2026-03-24/tilt_metrics.csv). Generated files under `artifacts/` and checkpoints under `logs/` are local outputs and are not included in the repository.

Generated videos and metrics:

- `artifacts/videos/baseline_straight.mp4`
- `artifacts/videos/learned_straight.mp4`
- `artifacts/metrics/tilt_metrics.csv`

Published result snapshot:

- [results/2026-03-24](results/2026-03-24)

Logs and checkpoints:

- `logs/rsl_rl/inventory_leveling/`
- `logs/rsl_rl/inventory_leveling_stable_flat/`
- `logs/rsl_rl/inventory_leveling_stable_adaptive/`

Launch metadata:

- `artifacts/launch_logs/`

## Notes

- Generated logs and most artifacts are gitignored.
- Result snapshots under `results/` are intended for commit/push.
- GUI play should use `num_envs=8`.
- Long training is usually run headless, but visible CMD windows are preferred for user monitoring.

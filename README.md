# Frbot Lift Leveling

Manager-based Isaac Lab task for keeping the inventory robot level while the lift stays fully extended.

## Environment

- Conda: `env_isaaclab_2.3.1`
- Isaac Sim: `E:\isaac-sim-5.1.0`
- Isaac Lab: `E:\IsaacLab`
- Project root: `E:\재고조사로봇\2603_재고조사로봇_학습\Frbot_Lift`

This project does not modify the original Isaac Lab or Isaac Sim folders.
The robot USD is copied into this repo at `assets/robots/lift_amr_new.usd`.

## Windows Path Note

USD loading on Windows was unreliable on the original Korean path. Use `scripts\tools\run_python.ps1` for all runs.
It creates and uses the ASCII-safe alias `R:\Frbot_Lift` and exports the required Isaac Sim environment variables.

## Tasks

- `Inventory-Leveling-v0`: training task
- `Inventory-Leveling-Play-v0`: playback and recording task

Policy actions control only:

- `FL_suspenstion`
- `FR_suspenstion`
- `RL_suspenstion`
- `RR_suspenstion`

Fixed controls:

- Lift joints stay at maximum extension
- `*_steer` joints stay at the USD forward-fixed values
- `*_Wheel` joints provide constant straight driving velocity

## Install

```powershell
.\scripts\tools\install_project.ps1
```

## Useful Commands

List registered environments:

```powershell
.\scripts\tools\run_python.ps1 .\scripts\list_envs.py
```

Smoke-test zero actions:

```powershell
.\scripts\tools\run_python.ps1 .\scripts\zero_agent.py --task Inventory-Leveling-Play-v0 --num_envs 1 --headless
```

Train PPO:

```powershell
.\scripts\tools\run_python.ps1 .\scripts\rsl_rl\train.py --task Inventory-Leveling-v0 --num_envs 1024 --max_iterations 1500 --headless
```

Short training smoke test:

```powershell
.\scripts\tools\run_python.ps1 .\scripts\rsl_rl\train.py --task Inventory-Leveling-v0 --num_envs 16 --max_iterations 2 --headless
```

Generate baseline video and metrics:

```powershell
.\scripts\tools\run_python.ps1 .\scripts\generate_comparison_artifacts.py --mode baseline --task Inventory-Leveling-Play-v0 --video_length 600 --num_envs 1 --headless
```

Generate learned video and metrics with a checkpoint:

```powershell
.\scripts\tools\run_python.ps1 .\scripts\generate_comparison_artifacts.py --mode learned --task Inventory-Leveling-Play-v0 --video_length 600 --num_envs 1 --checkpoint R:\Frbot_Lift\logs\rsl_rl\inventory_leveling\<run>\model_<n>.pt --headless
```

## Outputs

Generated comparison artifacts are written to:

- `artifacts/videos/baseline_straight.mp4`
- `artifacts/videos/learned_straight.mp4`
- `artifacts/metrics/tilt_metrics.csv`

Training logs and checkpoints are written under:

- `logs/rsl_rl/inventory_leveling/`

## Notes

- The copied USD is version-controlled in this repository.
- Generated artifacts and logs are ignored by git.
- The local `isaac_wheel` folder inside `E:\IsaacLab` is intentionally not referenced.

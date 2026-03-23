# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Record baseline or learned straight-driving videos and tilt metrics."""

import argparse
import csv
import os
import shutil
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Generate comparison artifacts for inventory leveling.")
parser.add_argument("--mode", choices=("baseline", "learned"), required=True, help="Artifact generation mode.")
parser.add_argument("--task", type=str, default="Inventory-Leveling-Play-v0", help="Task name.")
parser.add_argument("--video_length", type=int, default=600, help="Number of environment steps to record.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path for learned mode.")
parser.add_argument("--run", type=str, default=".*", help="Run regex when resolving the latest checkpoint.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for playback.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric for debugging.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import Frbot_Lift.tasks  # noqa: F401
import isaaclab.utils.math as math_utils
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner


def resolve_project_root() -> Path:
    env_root = os.environ.get("FRBOT_LIFT_ROOT")
    if env_root:
        return Path(env_root)
    alias_root = Path("R:/Frbot_Lift")
    if alias_root.exists():
        return alias_root
    return Path(__file__).resolve().parents[1]


def resolve_output_paths(project_root: Path, mode: str) -> tuple[Path, Path]:
    video_dir = project_root / "artifacts" / "videos"
    metrics_dir = project_root / "artifacts" / "metrics"
    video_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    return video_dir / f"{mode}_straight.mp4", metrics_dir / "tilt_metrics.csv"


def resolve_checkpoint(project_root: Path, task_name: str, checkpoint: str | None, run_pattern: str) -> str:
    if checkpoint:
        return checkpoint
    agent_cfg = load_cfg_from_registry(task_name.replace("-Play", ""), "rsl_rl_cfg_entry_point")
    log_root = project_root / "logs" / "rsl_rl" / agent_cfg.experiment_name
    return get_checkpoint_path(str(log_root), run_dir=run_pattern, checkpoint="model_.*.pt")


def extract_metrics(base_env) -> tuple[float, float, float]:
    quat = base_env.scene["robot"].data.root_quat_w[0:1]
    roll, pitch, _ = math_utils.euler_xyz_from_quat(quat)
    roll_deg = torch.rad2deg(roll).item()
    pitch_deg = torch.rad2deg(pitch).item()
    peak_tilt_deg = max(abs(roll_deg), abs(pitch_deg))
    return roll_deg, pitch_deg, peak_tilt_deg


def finalize_video(video_dir: Path, target_path: Path):
    mp4_files = sorted(video_dir.glob("*.mp4"))
    if not mp4_files:
        raise FileNotFoundError(f"No mp4 files were generated in {video_dir}.")
    shutil.move(str(mp4_files[0]), target_path)
    shutil.rmtree(video_dir, ignore_errors=True)


def upsert_metrics(
    csv_path: Path,
    mode: str,
    roll_values: list[float],
    pitch_values: list[float],
    peak_values: list[float],
    forward_velocity_values: list[float],
    abs_forward_velocity_values: list[float],
):
    rows = []
    if csv_path.exists():
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        rows = [row for row in rows if row.get("mode") != mode]

    roll_tensor = torch.tensor(roll_values)
    pitch_tensor = torch.tensor(pitch_values)
    peak_tensor = torch.tensor(peak_values)
    forward_tensor = torch.tensor(forward_velocity_values)
    abs_forward_tensor = torch.tensor(abs_forward_velocity_values)

    rows.append(
        {
            "mode": mode,
            "steps": str(len(roll_values)),
            "roll_rms_deg": f"{torch.sqrt(torch.mean(torch.square(roll_tensor))).item():.6f}",
            "pitch_rms_deg": f"{torch.sqrt(torch.mean(torch.square(pitch_tensor))).item():.6f}",
            "peak_tilt_deg": f"{torch.max(peak_tensor).item():.6f}",
            "mean_forward_velocity_mps": f"{torch.mean(forward_tensor).item():.6f}",
            "mean_abs_forward_velocity_mps": f"{torch.mean(abs_forward_tensor).item():.6f}",
        }
    )

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "mode",
                "steps",
                "roll_rms_deg",
                "pitch_rms_deg",
                "peak_tilt_deg",
                "mean_forward_velocity_mps",
                "mean_abs_forward_velocity_mps",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    project_root = resolve_project_root()
    target_video_path, metrics_csv_path = resolve_output_paths(project_root, args_cli.mode)
    temp_video_dir = target_video_path.parent / f"_{args_cli.mode}_tmp"
    shutil.rmtree(temp_video_dir, ignore_errors=True)

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.seed = args_cli.seed
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.log_dir = str(project_root / "artifacts" / "logs" / args_cli.mode)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=str(temp_video_dir),
        step_trigger=lambda step: step == 0,
        video_length=args_cli.video_length,
        disable_logger=True,
    )

    base_env = env.unwrapped
    roll_values = []
    pitch_values = []
    peak_values = []
    forward_velocity_values = []
    abs_forward_velocity_values = []

    if args_cli.mode == "baseline":
        env.reset()
        action_tensor = torch.zeros(env.action_space.shape, device=base_env.device)
        for _ in range(args_cli.video_length):
            if not simulation_app.is_running():
                break
            with torch.inference_mode():
                env.step(action_tensor)
            roll_deg, pitch_deg, peak_tilt_deg = extract_metrics(base_env)
            roll_values.append(roll_deg)
            pitch_values.append(pitch_deg)
            peak_values.append(peak_tilt_deg)
            forward_velocity = base_env.scene["robot"].data.root_lin_vel_b[0, 0].item()
            forward_velocity_values.append(forward_velocity)
            abs_forward_velocity_values.append(abs(forward_velocity))
        env.close()
    else:
        checkpoint_path = resolve_checkpoint(project_root, args_cli.task, args_cli.checkpoint, args_cli.run)
        agent_cfg = load_cfg_from_registry(args_cli.task.replace("-Play", ""), "rsl_rl_cfg_entry_point")
        vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(vec_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(checkpoint_path)
        policy = runner.get_inference_policy(device=vec_env.unwrapped.device)
        try:
            policy_nn = runner.alg.policy
        except AttributeError:
            policy_nn = runner.alg.actor_critic

        observations = vec_env.get_observations()
        for _ in range(args_cli.video_length):
            if not simulation_app.is_running():
                break
            with torch.inference_mode():
                actions = policy(observations)
                observations, _, dones, _ = vec_env.step(actions)
                policy_nn.reset(dones)
            roll_deg, pitch_deg, peak_tilt_deg = extract_metrics(base_env)
            roll_values.append(roll_deg)
            pitch_values.append(pitch_deg)
            peak_values.append(peak_tilt_deg)
            forward_velocity = base_env.scene["robot"].data.root_lin_vel_b[0, 0].item()
            forward_velocity_values.append(forward_velocity)
            abs_forward_velocity_values.append(abs(forward_velocity))
        vec_env.close()

    finalize_video(temp_video_dir, target_video_path)
    upsert_metrics(
        metrics_csv_path,
        args_cli.mode,
        roll_values,
        pitch_values,
        peak_values,
        forward_velocity_values,
        abs_forward_velocity_values,
    )


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()

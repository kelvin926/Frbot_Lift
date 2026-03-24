from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Sweep wheel target sign patterns for the stable flat task.")
parser.add_argument("--task", type=str, default="Inventory-Leveling-StableFlat-v0")
parser.add_argument("--artifact", type=str, required=True)
parser.add_argument("--steps", type=int, default=200)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import isaaclab.utils.math as math_utils
import torch

import Frbot_Lift.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

from Frbot_Lift.assets.frbot_lift_asset import WHEEL_JOINTS, WHEEL_TARGET_RAD_S
from Frbot_Lift.tasks.manager_based.frbot_lift.mdp.robot_frame import robot_roll_pitch_from_gravity, robot_to_body_frame
from Frbot_Lift.tasks.manager_based.frbot_lift_stable.mdp.reference_body import (
    reference_body_lin_vel_robot,
    reference_body_pos_w,
    reference_body_projected_gravity_robot,
    reference_body_quat_w,
)


def run_pattern(signs: tuple[int, int, int, int]) -> dict:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1, use_fabric=not args_cli.disable_fabric)
    env_cfg.actions.drivetrain.debug_vis = False
    env_cfg.actions.drivetrain.wheel_velocity_targets = {
        joint_name: float(sign * WHEEL_TARGET_RAD_S) for joint_name, sign in zip(WHEEL_JOINTS, signs, strict=True)
    }
    env = gym.make(args_cli.task, cfg=env_cfg)
    try:
        env.reset()
        base_env = env.unwrapped
        initial_pos_w = reference_body_pos_w(base_env).clone()
        forward_body = robot_to_body_frame(torch.tensor([[1.0, 0.0, 0.0]], device=base_env.device, dtype=torch.float32))
        initial_forward_dir_w = math_utils.quat_apply(reference_body_quat_w(base_env), forward_body)
        max_tilt_deg = 0.0
        actions = torch.zeros(env.action_space.shape, device=base_env.device)
        for _ in range(args_cli.steps):
            _, _, terminated, truncated, _ = env.step(actions)
            gravity_robot = reference_body_projected_gravity_robot(base_env)
            roll, pitch = robot_roll_pitch_from_gravity(gravity_robot)
            peak_tilt_deg = torch.maximum(torch.abs(torch.rad2deg(roll)), torch.abs(torch.rad2deg(pitch)))
            max_tilt_deg = max(max_tilt_deg, float(peak_tilt_deg.max().item()))
            if bool((terminated | truncated).any().item()):
                break
        final_pos_w = reference_body_pos_w(base_env)
        delta_pos_w = final_pos_w - initial_pos_w
        forward_progress_m = float(torch.sum(delta_pos_w * initial_forward_dir_w, dim=1)[0].item())
        final_forward_velocity_mps = float(reference_body_lin_vel_robot(base_env)[0, 0].item())
        return {
            "signs": dict(zip(WHEEL_JOINTS, signs, strict=True)),
            "forward_progress_m": forward_progress_m,
            "final_forward_velocity_mps": final_forward_velocity_mps,
            "max_tilt_deg": max_tilt_deg,
        }
    finally:
        env.close()


def main() -> None:
    rows = [run_pattern(signs) for signs in itertools.product((-1, 1), repeat=4)]
    rows.sort(key=lambda row: (row["forward_progress_m"], row["final_forward_velocity_mps"]), reverse=True)
    Path(args_cli.artifact).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()

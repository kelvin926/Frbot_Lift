# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run a short smoke test for the stable Frbot Lift tasks and dump summary metrics to JSON."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Smoke-check a stable Frbot Lift environment.")
parser.add_argument("--task", type=str, required=True, help="Task to test.")
parser.add_argument("--num_envs", type=int, default=8, help="Number of environments.")
parser.add_argument("--steps", type=int, default=400, help="Number of env steps to simulate.")
parser.add_argument("--env_spacing", type=float, default=None, help="Override environment spacing.")
parser.add_argument("--artifact", type=str, required=True, help="Path to output JSON artifact.")
parser.add_argument("--policy", choices=("zero", "random"), default="zero", help="Action policy used in the smoke test.")
parser.add_argument("--seed", type=int, default=42, help="Environment seed.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric for debugging.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import isaaclab.utils.math as math_utils
import torch

import Frbot_Lift.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

from Frbot_Lift.assets.frbot_lift_asset import WHEEL_JOINTS
from Frbot_Lift.tasks.manager_based.frbot_lift.mdp.robot_frame import (
    robot_roll_pitch_from_gravity,
    robot_to_body_frame,
)
from Frbot_Lift.tasks.manager_based.frbot_lift_stable.mdp.bootstrap import phase_index, phase_name_from_index
from Frbot_Lift.tasks.manager_based.frbot_lift_stable.mdp.reference_body import (
    reference_body_height,
    reference_body_lin_vel_robot,
    reference_body_pos_w,
    reference_body_projected_gravity_robot,
    reference_body_quat_w,
)


def main() -> None:
    progress_path = Path(args_cli.artifact).with_suffix(".progress.log")
    error_path = Path(args_cli.artifact).with_suffix(".error.log")
    progress_path.parent.mkdir(parents=True, exist_ok=True)

    def log_progress(message: str) -> None:
        with progress_path.open("a", encoding="utf-8") as f:
            f.write(message + "\n")

    log_progress("start")
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.seed = args_cli.seed
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.env_spacing is not None:
        env_cfg.scene.env_spacing = args_cli.env_spacing
    if getattr(env_cfg.actions, "drivetrain", None) is not None:
        env_cfg.actions.drivetrain.debug_vis = False

    log_progress("cfg_ready")
    env = gym.make(args_cli.task, cfg=env_cfg)
    try:
        log_progress("env_created")
        env.reset()
        log_progress("env_reset")
        base_env = env.unwrapped
        initial_root_pos_w = reference_body_pos_w(base_env).clone()
        forward_body = robot_to_body_frame(
            torch.tensor([[1.0, 0.0, 0.0]], device=base_env.device, dtype=torch.float32).repeat(base_env.num_envs, 1)
        )
        initial_forward_dir_w = math_utils.quat_apply(reference_body_quat_w(base_env), forward_body)
        initial_forward_dir_w = torch.nn.functional.normalize(initial_forward_dir_w, dim=1)
        wheel_joint_ids, _ = base_env.scene["robot"].find_joints(WHEEL_JOINTS, preserve_order=True)
        first_two_sec_steps = int(round(2.0 / base_env.step_dt))
        min_root_height = float("inf")
        max_root_height = float("-inf")
        min_lowest_body_height = float("inf")
        max_lowest_body_height = float("-inf")
        max_peak_tilt_deg = 0.0
        first_done_step = None
        catastrophic_first_two_sec = 0
        aggregate_term_counts = {term_name: 0 for term_name in base_env.termination_manager.active_terms}
        step_trace = []

        if args_cli.policy == "random":
            torch.manual_seed(args_cli.seed)

        terrain = getattr(base_env.scene, "terrain", None)
        has_terrain_grid = terrain is not None and hasattr(terrain, "terrain_levels") and hasattr(terrain, "terrain_types")

        def terrain_levels_list() -> list[int] | list[None]:
            if not has_terrain_grid:
                return [None] * base_env.num_envs
            return terrain.terrain_levels.detach().cpu().tolist()

        def terrain_types_list() -> list[int] | list[None]:
            if not has_terrain_grid:
                return [None] * base_env.num_envs
            return terrain.terrain_types.detach().cpu().tolist()

        for step in range(args_cli.steps):
            if args_cli.policy == "zero":
                actions = torch.zeros(env.action_space.shape, device=base_env.device)
            else:
                actions = torch.rand(env.action_space.shape, device=base_env.device) * 2.0 - 1.0

            _, _, terminated, truncated, _ = env.step(actions)
            if step == 0:
                log_progress("first_step_done")
            done = terminated | truncated

            root_height = reference_body_height(base_env)
            lowest_body_height = torch.min(base_env.scene["robot"].data.body_pos_w[..., 2], dim=1).values - base_env.scene.env_origins[:, 2]
            min_root_height = min(min_root_height, float(root_height.min().item()))
            max_root_height = max(max_root_height, float(root_height.max().item()))
            min_lowest_body_height = min(min_lowest_body_height, float(lowest_body_height.min().item()))
            max_lowest_body_height = max(max_lowest_body_height, float(lowest_body_height.max().item()))

            projected_gravity_robot = reference_body_projected_gravity_robot(base_env)
            roll, pitch = robot_roll_pitch_from_gravity(projected_gravity_robot)
            peak_tilt_deg = torch.maximum(torch.abs(torch.rad2deg(roll)), torch.abs(torch.rad2deg(pitch)))
            max_peak_tilt_deg = max(max_peak_tilt_deg, float(peak_tilt_deg.max().item()))
            step_forward_velocity = reference_body_lin_vel_robot(base_env)[:, 0]

            if step < first_two_sec_steps:
                catastrophic = (
                    base_env.termination_manager.get_term("catastrophic_bad_orientation")
                    | base_env.termination_manager.get_term("catastrophic_root_height")
                )
                catastrophic_first_two_sec += int(catastrophic.sum().item())
            for term_name in base_env.termination_manager.active_terms:
                aggregate_term_counts[term_name] += int(base_env.termination_manager.get_term(term_name).sum().item())

            if first_done_step is None and bool(done.any().item()):
                first_done_step = step
            if step < min(args_cli.steps, 120):
                term_envs = {
                    term_name: (
                        base_env.termination_manager.get_term(term_name).nonzero(as_tuple=True)[0].detach().cpu().tolist()
                    )
                    for term_name in base_env.termination_manager.active_terms
                    if bool(base_env.termination_manager.get_term(term_name).any().item())
                }
                step_trace.append(
                    {
                        "step": step,
                        "done_count": int(done.sum().item()),
                        "done_env_ids": done.nonzero(as_tuple=True)[0].detach().cpu().tolist(),
                        "terrain_levels": terrain_levels_list(),
                        "terrain_types": terrain_types_list(),
                        "root_height_min": float(root_height.min().item()),
                        "root_height_max": float(root_height.max().item()),
                        "lowest_body_height_min": float(lowest_body_height.min().item()),
                        "lowest_body_height_max": float(lowest_body_height.max().item()),
                        "roll_deg": torch.rad2deg(roll).detach().cpu().tolist(),
                        "pitch_deg": torch.rad2deg(pitch).detach().cpu().tolist(),
                        "forward_velocity_mps": step_forward_velocity.detach().cpu().tolist(),
                        "phase_indices": phase_index(base_env).detach().cpu().tolist(),
                        "catastrophic_root_height_count": int(
                            base_env.termination_manager.get_term("catastrophic_root_height").sum().item()
                        ),
                        "active_terms": term_envs,
                    }
                )

        phase_names = [phase_name_from_index(v.item()) for v in phase_index(base_env)]
        phase_counts: dict[str, int] = {}
        for phase_name in phase_names:
            phase_counts[phase_name] = phase_counts.get(phase_name, 0) + 1

        term_counts = {
            term_name: int(base_env.termination_manager.get_term(term_name).sum().item())
            for term_name in base_env.termination_manager.active_terms
        }

        forward_velocity = reference_body_lin_vel_robot(base_env)[:, 0]
        delta_pos_w = reference_body_pos_w(base_env) - initial_root_pos_w
        forward_progress_m = torch.sum(delta_pos_w * initial_forward_dir_w, dim=1)
        artifact = {
            "task": args_cli.task,
            "policy": args_cli.policy,
            "num_envs": args_cli.num_envs,
            "steps": args_cli.steps,
            "env_spacing": float(env_cfg.scene.env_spacing),
            "step_dt": float(base_env.step_dt),
            "first_two_sec_steps": first_two_sec_steps,
            "catastrophic_first_two_sec": catastrophic_first_two_sec,
            "first_done_step": first_done_step,
            "min_root_height_m": min_root_height,
            "max_root_height_m": max_root_height,
            "min_lowest_body_height_m": min_lowest_body_height,
            "max_lowest_body_height_m": max_lowest_body_height,
            "max_peak_tilt_deg": max_peak_tilt_deg,
            "mean_forward_velocity_mps": float(forward_velocity.mean().item()),
            "final_forward_velocity_mps": forward_velocity.detach().cpu().tolist(),
            "final_forward_progress_m": forward_progress_m.detach().cpu().tolist(),
            "final_wheel_velocity_rad_s": (
                base_env.scene["robot"].data.joint_vel[:, wheel_joint_ids]
            ).detach().cpu().tolist(),
            "final_phase_counts": phase_counts,
            "final_term_counts": term_counts,
            "aggregate_term_counts": aggregate_term_counts,
            "terrain_levels": terrain_levels_list(),
            "terrain_types": terrain_types_list(),
            "step_trace": step_trace,
        }

        artifact_path = Path(args_cli.artifact)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
        log_progress("artifact_written")
        print(json.dumps(artifact, indent=2))
    except Exception:
        error_path.write_text(traceback.format_exc(), encoding="utf-8")
        log_progress("exception_logged")
        raise
    finally:
        log_progress("env_close")
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()

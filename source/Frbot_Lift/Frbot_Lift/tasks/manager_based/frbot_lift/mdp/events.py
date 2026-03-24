# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def _axis_vector(axis: tuple[float, float, float], count: int, device: str) -> torch.Tensor:
    return torch.tensor(axis, device=device, dtype=torch.float32).repeat(count, 1)


def reset_root_state_uniform_robot_frame(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the root pose using roll/pitch/yaw defined in the robot-aligned frame."""

    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    root_states = asset.data.default_root_state[env_ids].clone()

    pose_keys = ["x", "y", "z", "roll", "pitch", "yaw"]
    pose_ranges = torch.tensor([pose_range.get(key, (0.0, 0.0)) for key in pose_keys], device=asset.device)
    pose_samples = math_utils.sample_uniform(
        pose_ranges[:, 0], pose_ranges[:, 1], (len(env_ids), len(pose_keys)), device=asset.device
    )

    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + pose_samples[:, 0:3]

    # Robot frame axes expressed in the imported raw body frame:
    # roll  -> +X_raw
    # pitch -> -Z_raw
    # yaw   -> +Y_raw
    q_roll = math_utils.quat_from_angle_axis(
        pose_samples[:, 3], _axis_vector((1.0, 0.0, 0.0), len(env_ids), asset.device)
    )
    q_pitch = math_utils.quat_from_angle_axis(
        pose_samples[:, 4], _axis_vector((0.0, 0.0, -1.0), len(env_ids), asset.device)
    )
    q_yaw = math_utils.quat_from_angle_axis(
        pose_samples[:, 5], _axis_vector((0.0, 1.0, 0.0), len(env_ids), asset.device)
    )
    orientation_delta = math_utils.quat_mul(math_utils.quat_mul(q_yaw, q_pitch), q_roll)
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientation_delta)

    velocity_keys = ["x", "y", "z", "roll", "pitch", "yaw"]
    velocity_ranges = torch.tensor([velocity_range.get(key, (0.0, 0.0)) for key in velocity_keys], device=asset.device)
    velocity_samples = math_utils.sample_uniform(
        velocity_ranges[:, 0], velocity_ranges[:, 1], (len(env_ids), len(velocity_keys)), device=asset.device
    )
    velocities = root_states[:, 7:13] + velocity_samples

    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)

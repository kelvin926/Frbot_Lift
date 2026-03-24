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
from isaaclab.terrains import TerrainImporter

from Frbot_Lift.tasks.manager_based.frbot_lift.mdp.events import _axis_vector

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def assign_terrain_types(
    env: "ManagerBasedEnv",
    env_ids,
    terrain_types: tuple[int, ...],
):
    """Assign terrain columns and refresh environment origins."""

    terrain: TerrainImporter = env.scene.terrain
    if env_ids is None or isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    else:
        env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)

    terrain_type_choices = torch.tensor(terrain_types, device=env.device, dtype=torch.long)
    assigned_types = terrain_type_choices[env_ids % len(terrain_type_choices)]
    terrain.terrain_types[env_ids] = assigned_types
    env.scene.env_origins[env_ids] = terrain.terrain_origins[terrain.terrain_levels[env_ids], terrain.terrain_types[env_ids]]


def reset_root_state_from_terrain_robot_frame(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    root_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the root state from terrain flat patches while sampling orientation deltas in the robot frame."""

    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    root_states = asset.data.default_root_state[env_ids].clone()

    valid_positions: torch.Tensor = terrain.flat_patches.get("init_pos")
    if valid_positions is None:
        raise ValueError(
            "The event term 'reset_root_state_from_terrain_robot_frame' requires valid flat patches under 'init_pos'."
            f" Found: {list(terrain.flat_patches.keys())}"
        )

    ids = torch.randint(0, valid_positions.shape[2], size=(len(env_ids),), device=env.device)
    positions = valid_positions[terrain.terrain_levels[env_ids], terrain.terrain_types[env_ids], ids].clone()
    positions[:, 2] += float(root_height)

    pose_keys = ["roll", "pitch", "yaw"]
    pose_ranges = torch.tensor([pose_range.get(key, (0.0, 0.0)) for key in pose_keys], device=asset.device)
    pose_samples = math_utils.sample_uniform(
        pose_ranges[:, 0], pose_ranges[:, 1], (len(env_ids), len(pose_keys)), device=asset.device
    )

    q_roll = math_utils.quat_from_angle_axis(
        pose_samples[:, 0], _axis_vector((1.0, 0.0, 0.0), len(env_ids), asset.device)
    )
    q_pitch = math_utils.quat_from_angle_axis(
        pose_samples[:, 1], _axis_vector((0.0, 0.0, -1.0), len(env_ids), asset.device)
    )
    q_yaw = math_utils.quat_from_angle_axis(
        pose_samples[:, 2], _axis_vector((0.0, 1.0, 0.0), len(env_ids), asset.device)
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


def reset_root_state_uniform_robot_frame_height(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    root_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the root pose using a fixed safe spawn height and roll/pitch/yaw in the robot-aligned frame."""

    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    root_states = asset.data.default_root_state[env_ids].clone()

    pose_keys = ["x", "y", "z", "roll", "pitch", "yaw"]
    pose_ranges = torch.tensor([pose_range.get(key, (0.0, 0.0)) for key in pose_keys], device=asset.device)
    pose_samples = math_utils.sample_uniform(
        pose_ranges[:, 0], pose_ranges[:, 1], (len(env_ids), len(pose_keys)), device=asset.device
    )

    positions = env.scene.env_origins[env_ids].clone()
    positions[:, :2] += pose_samples[:, :2]
    positions[:, 2] += float(root_height) + pose_samples[:, 2]

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

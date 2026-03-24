# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from .robot_frame import body_to_robot_frame, robot_tilt_angle_from_gravity

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def track_forward_speed_abs_exp(
    env: "ManagerBasedRLEnv",
    target: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward body-frame forward speed magnitude tracking with sign invariance."""

    asset: Articulation = env.scene[asset_cfg.name]
    root_lin_vel_robot = body_to_robot_frame(asset.data.root_lin_vel_b)
    error = torch.square(torch.abs(root_lin_vel_robot[:, 0]) - target)
    return torch.exp(-error / (std**2))


def track_forward_speed_exp(
    env: "ManagerBasedRLEnv",
    target: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward signed robot-frame forward speed tracking."""

    asset: Articulation = env.scene[asset_cfg.name]
    root_lin_vel_robot = body_to_robot_frame(asset.data.root_lin_vel_b)
    error = torch.square(root_lin_vel_robot[:, 0] - target)
    return torch.exp(-error / (std**2))


def forward_speed_abs(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward absolute body-frame forward speed."""

    asset: Articulation = env.scene[asset_cfg.name]
    root_lin_vel_robot = body_to_robot_frame(asset.data.root_lin_vel_b)
    return torch.abs(root_lin_vel_robot[:, 0])


def forward_speed_deficit_l1(
    env: "ManagerBasedRLEnv",
    target: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize moving slower than the desired forward speed magnitude."""

    asset: Articulation = env.scene[asset_cfg.name]
    root_lin_vel_robot = body_to_robot_frame(asset.data.root_lin_vel_b)
    return (target - torch.abs(root_lin_vel_robot[:, 0])).clip(min=0.0)


def forward_speed_deficit_signed_l1(
    env: "ManagerBasedRLEnv",
    target: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize moving slower than the desired signed forward speed."""

    asset: Articulation = env.scene[asset_cfg.name]
    root_lin_vel_robot = body_to_robot_frame(asset.data.root_lin_vel_b)
    return (target - root_lin_vel_robot[:, 0]).clip(min=0.0)


def backward_speed_l1(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize reverse motion in the robot-aligned frame."""

    asset: Articulation = env.scene[asset_cfg.name]
    root_lin_vel_robot = body_to_robot_frame(asset.data.root_lin_vel_b)
    return (-root_lin_vel_robot[:, 0]).clip(min=0.0)


def flat_orientation_robot_l2(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize non-flat orientation in the robot-aligned frame."""

    asset: Articulation = env.scene[asset_cfg.name]
    projected_gravity_robot = body_to_robot_frame(asset.data.projected_gravity_b)
    return torch.sum(torch.square(projected_gravity_robot[:, :2]), dim=1)


def tilt_margin_l2(
    env: "ManagerBasedRLEnv",
    threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize tilt angle once it exceeds a small upright margin."""

    asset: Articulation = env.scene[asset_cfg.name]
    projected_gravity_robot = body_to_robot_frame(asset.data.projected_gravity_b)
    tilt_angle = robot_tilt_angle_from_gravity(projected_gravity_robot)
    return torch.square((tilt_angle - threshold).clip(min=0.0))


def robot_ang_vel_xy_l2(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize roll/pitch rates in the robot-aligned frame."""

    asset: Articulation = env.scene[asset_cfg.name]
    root_ang_vel_robot = body_to_robot_frame(asset.data.root_ang_vel_b)
    return torch.sum(torch.square(root_ang_vel_robot[:, :2]), dim=1)


def robot_lin_vel_z_l2(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize vertical motion in the robot-aligned frame."""

    asset: Articulation = env.scene[asset_cfg.name]
    root_lin_vel_robot = body_to_robot_frame(asset.data.root_lin_vel_b)
    return torch.square(root_lin_vel_robot[:, 2])


def robot_lateral_velocity_l2(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize lateral slip in the robot-aligned frame."""

    asset: Articulation = env.scene[asset_cfg.name]
    root_lin_vel_robot = body_to_robot_frame(asset.data.root_lin_vel_b)
    return torch.square(root_lin_vel_robot[:, 1])

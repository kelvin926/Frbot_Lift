# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

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
    error = torch.square(torch.abs(asset.data.root_lin_vel_b[:, 0]) - target)
    return torch.exp(-error / (std**2))


def forward_speed_abs(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward absolute body-frame forward speed."""

    asset: Articulation = env.scene[asset_cfg.name]
    return torch.abs(asset.data.root_lin_vel_b[:, 0])


def forward_speed_deficit_l1(
    env: "ManagerBasedRLEnv",
    target: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize moving slower than the desired forward speed magnitude."""

    asset: Articulation = env.scene[asset_cfg.name]
    return (target - torch.abs(asset.data.root_lin_vel_b[:, 0])).clip(min=0.0)


def tilt_margin_l2(
    env: "ManagerBasedRLEnv",
    threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize tilt angle once it exceeds a small upright margin."""

    asset: Articulation = env.scene[asset_cfg.name]
    projected_gravity = asset.data.projected_gravity_b
    tilt_angle = torch.atan2(torch.linalg.vector_norm(projected_gravity[:, :2], dim=1), (-projected_gravity[:, 2]).clip(min=1.0e-6))
    return torch.square((tilt_angle - threshold).clip(min=0.0))


def lateral_velocity_l2(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize lateral slip in the body frame."""

    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 1])

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from .robot_frame import body_to_robot_frame

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def robot_base_lin_vel(env: "ManagerBasedEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root linear velocity in the robot-aligned frame."""

    asset: Articulation = env.scene[asset_cfg.name]
    return body_to_robot_frame(asset.data.root_lin_vel_b)


def robot_base_ang_vel(env: "ManagerBasedEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root angular velocity in the robot-aligned frame."""

    asset: Articulation = env.scene[asset_cfg.name]
    return body_to_robot_frame(asset.data.root_ang_vel_b)


def robot_projected_gravity(env: "ManagerBasedEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Projected gravity in the robot-aligned frame."""

    asset: Articulation = env.scene[asset_cfg.name]
    return body_to_robot_frame(asset.data.projected_gravity_b)


def constant_forward_command(env: "ManagerBasedEnv", value: float) -> torch.Tensor:
    """Return a constant scalar command used as context for the policy."""

    return torch.full((env.num_envs, 1), value, device=env.device, dtype=torch.float32)

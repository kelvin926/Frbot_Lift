# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, SceneEntityCfg, TerminationTermCfg

from .robot_frame import body_to_robot_frame, robot_to_body_frame

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def bad_orientation_robot_frame(
    env: "ManagerBasedRLEnv",
    limit_angle: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the robot tilt exceeds the limit in the robot-aligned frame."""

    asset: Articulation = env.scene[asset_cfg.name]
    projected_gravity_robot = body_to_robot_frame(asset.data.projected_gravity_b)
    return torch.acos((-projected_gravity_robot[:, 2]).clip(min=-1.0, max=1.0)).abs() > limit_angle


def terrain_out_of_bounds(
    env: "ManagerBasedRLEnv",
    distance_limit: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the robot leaves the active terrain patch."""

    asset: Articulation = env.scene[asset_cfg.name]
    relative_pos = asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    return torch.any(torch.abs(relative_pos) > distance_limit, dim=1)


def root_height_out_of_bounds(
    env: "ManagerBasedRLEnv",
    min_height: float,
    max_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the base root leaves the expected near-ground height band."""

    asset: Articulation = env.scene[asset_cfg.name]
    relative_height = asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return (relative_height < min_height) | (relative_height > max_height)


def excessive_motion(
    env: "ManagerBasedRLEnv",
    linear_speed_limit: float,
    angular_speed_limit: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate unstable launches before they corrupt training."""

    asset: Articulation = env.scene[asset_cfg.name]
    linear_speed = torch.linalg.norm(asset.data.root_lin_vel_b, dim=1)
    angular_speed = torch.linalg.norm(asset.data.root_ang_vel_b, dim=1)
    return (linear_speed > linear_speed_limit) | (angular_speed > angular_speed_limit)


class stalled_progress(ManagerTermBase):
    """Terminate if the robot fails to make enough forward progress from its episode start pose."""

    def __init__(self, cfg: TerminationTermCfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        self._asset: Articulation = env.scene[asset_cfg.name]
        self._start_pos = torch.zeros(env.num_envs, 3, device=env.device)
        self._start_forward_dir = torch.zeros(env.num_envs, 3, device=env.device)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._start_pos[env_ids] = self._asset.data.root_pos_w[env_ids]
        forward_body = robot_to_body_frame(
            torch.tensor([[1.0, 0.0, 0.0]], device=self._asset.device, dtype=torch.float32).repeat(
                self._asset.data.root_quat_w[env_ids].shape[0], 1
            )
        )
        self._start_forward_dir[env_ids] = math_utils.quat_apply(self._asset.data.root_quat_w[env_ids], forward_body)

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        min_progress: float,
        min_steps: int,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        delta_pos = self._asset.data.root_pos_w - self._start_pos
        delta_forward = torch.abs(torch.sum(delta_pos * self._start_forward_dir, dim=1))
        return (env.episode_length_buf > min_steps) & (delta_forward < min_progress)

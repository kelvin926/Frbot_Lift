# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CurriculumTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.terrains import TerrainImporter

from .robot_frame import robot_to_body_frame

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class terrain_levels_progress(ManagerTermBase):
    """Advance terrain difficulty based on forward progress from the episode start pose."""

    def __init__(self, cfg: CurriculumTermCfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        self._asset: Articulation = env.scene[asset_cfg.name]
        self._start_pos = torch.zeros(env.num_envs, 3, device=env.device)
        self._start_forward_dir = torch.zeros(env.num_envs, 3, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
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
        env_ids: Sequence[int],
        progress_threshold: float,
        regression_threshold: float,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        terrain: TerrainImporter = env.scene.terrain
        delta_pos = self._asset.data.root_pos_w[env_ids] - self._start_pos[env_ids]
        delta_forward = torch.abs(torch.sum(delta_pos * self._start_forward_dir[env_ids], dim=1))
        episode_steps = env.episode_length_buf[env_ids]

        move_up = delta_forward > progress_threshold
        move_down = delta_forward < regression_threshold
        move_down &= episode_steps > (env.max_episode_length // 2)
        move_down &= ~move_up

        terrain.update_env_origins(env_ids, move_up, move_down)
        return torch.mean(terrain.terrain_levels.float())

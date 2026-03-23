# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CurriculumTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class terrain_levels_progress(ManagerTermBase):
    """Advance terrain difficulty based on x progress from the episode start pose."""

    def __init__(self, cfg: CurriculumTermCfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        self._asset: Articulation = env.scene[asset_cfg.name]
        self._start_x = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._start_x[env_ids] = self._asset.data.root_pos_w[env_ids, 0]

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        env_ids: Sequence[int],
        progress_threshold: float,
        regression_threshold: float,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        terrain: TerrainImporter = env.scene.terrain
        delta_x = torch.abs(self._asset.data.root_pos_w[env_ids, 0] - self._start_x[env_ids])
        episode_steps = env.episode_length_buf[env_ids]

        move_up = delta_x > progress_threshold
        move_down = delta_x < regression_threshold
        move_down &= episode_steps > (env.max_episode_length // 2)
        move_down &= ~move_up

        terrain.update_env_origins(env_ids, move_up, move_down)
        return torch.mean(terrain.terrain_levels.float())

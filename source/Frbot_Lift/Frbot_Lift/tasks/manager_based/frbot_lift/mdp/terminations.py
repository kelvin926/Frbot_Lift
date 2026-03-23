# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, SceneEntityCfg, TerminationTermCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_out_of_bounds(
    env: "ManagerBasedRLEnv",
    distance_limit: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the robot leaves the active terrain patch."""

    asset: Articulation = env.scene[asset_cfg.name]
    relative_pos = asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    return torch.any(torch.abs(relative_pos) > distance_limit, dim=1)
class stalled_progress(ManagerTermBase):
    """Terminate if the robot fails to make enough x progress from its episode start pose."""

    def __init__(self, cfg: TerminationTermCfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        self._asset: Articulation = env.scene[asset_cfg.name]
        self._start_x = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._start_x[env_ids] = self._asset.data.root_pos_w[env_ids, 0]

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        min_progress: float,
        min_steps: int,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        delta_x = torch.abs(self._asset.data.root_pos_w[:, 0] - self._start_x)
        return (env.episode_length_buf > min_steps) & (delta_x < min_progress)

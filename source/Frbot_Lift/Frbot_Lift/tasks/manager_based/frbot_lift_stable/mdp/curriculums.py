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

from Frbot_Lift.tasks.manager_based.frbot_lift.mdp.robot_frame import robot_to_body_frame

from . import bootstrap

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class phase_aware_terrain_levels_progress(ManagerTermBase):
    """Advance terrain difficulty based on progress made after the bootstrap phases complete."""

    def __init__(self, cfg: CurriculumTermCfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        self._asset: Articulation = env.scene[asset_cfg.name]
        self._task_started = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        self._task_start_step = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        self._start_pos = torch.zeros(env.num_envs, 3, device=env.device)
        self._start_forward_dir = torch.zeros(env.num_envs, 3, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._task_started[env_ids] = False
        self._task_start_step[env_ids] = 0
        self._start_pos[env_ids] = 0.0
        self._start_forward_dir[env_ids] = 0.0

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        env_ids: Sequence[int],
        progress_threshold: float,
        regression_threshold: float,
        plane_terrain_types: tuple[int, ...] | None = None,
        rough_terrain_types: tuple[int, ...] | None = None,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        terrain: TerrainImporter = env.scene.terrain
        task_phase_mask = bootstrap.task_phase_mask(env)
        newly_armed = task_phase_mask & (~self._task_started)
        if torch.any(newly_armed):
            armed_ids = newly_armed.nonzero(as_tuple=True)[0]
            self._start_pos[armed_ids] = self._asset.data.root_pos_w[armed_ids]
            self._task_start_step[armed_ids] = env.episode_length_buf[armed_ids]
            forward_body = robot_to_body_frame(
                torch.tensor([[1.0, 0.0, 0.0]], device=self._asset.device, dtype=torch.float32).repeat(len(armed_ids), 1)
            )
            self._start_forward_dir[armed_ids] = math_utils.quat_apply(self._asset.data.root_quat_w[armed_ids], forward_body)
            self._task_started[armed_ids] = True

        env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
        delta_pos = self._asset.data.root_pos_w[env_ids] - self._start_pos[env_ids]
        delta_forward = torch.abs(torch.sum(delta_pos * self._start_forward_dir[env_ids], dim=1))
        task_steps = env.episode_length_buf[env_ids] - self._task_start_step[env_ids]
        task_active = self._task_started[env_ids] & task_phase_mask[env_ids]

        move_up = task_active & (delta_forward > progress_threshold)
        move_down = task_active & (delta_forward < regression_threshold)
        move_down &= task_steps > (env.max_episode_length // 2)
        move_down &= ~move_up

        if rough_terrain_types is not None:
            rough_types = torch.tensor(rough_terrain_types, device=env.device, dtype=torch.long)
            rough_env_ids = env_ids[move_up]
            if len(rough_env_ids) > 0:
                terrain.terrain_types[rough_env_ids] = rough_types[rough_env_ids % len(rough_types)]

        if plane_terrain_types is not None:
            plane_types = torch.tensor(plane_terrain_types, device=env.device, dtype=torch.long)
            plane_env_ids = env_ids[move_down]
            if len(plane_env_ids) > 0:
                terrain.terrain_types[plane_env_ids] = plane_types[plane_env_ids % len(plane_types)]

        terrain.update_env_origins(env_ids, move_up, move_down)
        return torch.mean(terrain.terrain_levels.float())

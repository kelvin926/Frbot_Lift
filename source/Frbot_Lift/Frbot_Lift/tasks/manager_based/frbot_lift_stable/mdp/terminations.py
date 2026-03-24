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
from isaaclab.managers import ManagerTermBase, SceneEntityCfg, TerminationTermCfg

from Frbot_Lift.tasks.manager_based.frbot_lift.mdp import terminations as base_terms
from Frbot_Lift.tasks.manager_based.frbot_lift.mdp.robot_frame import robot_to_body_frame

from . import bootstrap

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def catastrophic_bad_orientation_robot_frame(
    env: "ManagerBasedRLEnv",
    limit_angle: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    return base_terms.bad_orientation_robot_frame(env, limit_angle=limit_angle, asset_cfg=asset_cfg)


def catastrophic_root_height(
    env: "ManagerBasedRLEnv",
    min_height: float,
    max_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    return base_terms.root_height_out_of_bounds(env, min_height=min_height, max_height=max_height, asset_cfg=asset_cfg)


def task_phase_bad_orientation_robot_frame(
    env: "ManagerBasedRLEnv",
    limit_angle: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    return bootstrap.task_phase_mask(env) & base_terms.bad_orientation_robot_frame(
        env, limit_angle=limit_angle, asset_cfg=asset_cfg
    )


def task_phase_terrain_out_of_bounds(
    env: "ManagerBasedRLEnv",
    distance_limit: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    return bootstrap.task_phase_mask(env) & base_terms.terrain_out_of_bounds(
        env, distance_limit=distance_limit, asset_cfg=asset_cfg
    )


def task_phase_excessive_motion(
    env: "ManagerBasedRLEnv",
    linear_speed_limit: float,
    angular_speed_limit: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    return bootstrap.task_phase_mask(env) & base_terms.excessive_motion(
        env,
        linear_speed_limit=linear_speed_limit,
        angular_speed_limit=angular_speed_limit,
        asset_cfg=asset_cfg,
    )


class phase_aware_stalled_progress(ManagerTermBase):
    """Terminate only if the robot fails to make enough progress after the bootstrap phases finish."""

    def __init__(self, cfg: TerminationTermCfg, env: "ManagerBasedRLEnv"):
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
        min_progress: float,
        min_steps: int,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        task_phase_mask = bootstrap.task_phase_mask(env)
        newly_armed = task_phase_mask & (~self._task_started)
        if torch.any(newly_armed):
            env_ids = newly_armed.nonzero(as_tuple=True)[0]
            self._start_pos[env_ids] = self._asset.data.root_pos_w[env_ids]
            self._task_start_step[env_ids] = env.episode_length_buf[env_ids]
            forward_body = robot_to_body_frame(
                torch.tensor([[1.0, 0.0, 0.0]], device=self._asset.device, dtype=torch.float32).repeat(len(env_ids), 1)
            )
            self._start_forward_dir[env_ids] = math_utils.quat_apply(self._asset.data.root_quat_w[env_ids], forward_body)
            self._task_started[env_ids] = True

        delta_pos = self._asset.data.root_pos_w - self._start_pos
        delta_forward = torch.abs(torch.sum(delta_pos * self._start_forward_dir, dim=1))
        task_steps = env.episode_length_buf - self._task_start_step
        return self._task_started & task_phase_mask & (task_steps > min_steps) & (delta_forward < min_progress)


def active_termination_names(env: "ManagerBasedRLEnv", env_idx: int) -> str:
    names: list[str] = []
    for term_name in env.termination_manager.active_terms:
        if env.termination_manager.get_term(term_name)[env_idx].item():
            names.append(term_name)
    return "|".join(names)

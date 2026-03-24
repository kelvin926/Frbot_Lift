# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg

from Frbot_Lift.tasks.manager_based.frbot_lift import mdp as frbot_mdp
from Frbot_Lift.tasks.manager_based.frbot_lift.mdp.robot_frame import robot_tilt_angle_from_gravity

from . import bootstrap
from .reference_body import (
    REFERENCE_BODY_CFG,
    reference_body_ang_vel_robot,
    reference_body_lin_vel_robot,
    reference_body_projected_gravity_robot,
)
from .mast_top import (
    mast_top_ang_vel_robot,
    mast_top_relative_lin_vel_robot,
    mast_top_relative_pos_robot_deviation,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _task_mask(env: "ManagerBasedRLEnv") -> torch.Tensor:
    return bootstrap.task_phase_mask_f(env)


def task_phase_alive(env: "ManagerBasedRLEnv") -> torch.Tensor:
    return frbot_mdp.is_alive(env) * _task_mask(env)


class task_phase_terminated_term(ManagerTermBase):
    """Penalize selected termination terms only once the environment has entered the task phase."""

    def __init__(self, cfg: RewardTermCfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        term_keys = cfg.params.get("term_keys", ".*")
        self._term_names = env.termination_manager.find_terms(term_keys)

    def __call__(self, env: "ManagerBasedRLEnv", term_keys: str | list[str] = ".*") -> torch.Tensor:
        reset_buf = torch.zeros(env.num_envs, device=env.device)
        for term in self._term_names:
            reset_buf += env.termination_manager.get_term(term)
        return (reset_buf * (~env.termination_manager.time_outs)).float() * _task_mask(env)


def task_phase_flat_orientation_l2(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = REFERENCE_BODY_CFG,
) -> torch.Tensor:
    gravity_robot = reference_body_projected_gravity_robot(env, asset_cfg=asset_cfg)
    return torch.sum(torch.square(gravity_robot[:, :2]), dim=1) * _task_mask(env)


def task_phase_tilt_margin_l2(
    env: "ManagerBasedRLEnv",
    threshold: float,
    asset_cfg: SceneEntityCfg = REFERENCE_BODY_CFG,
) -> torch.Tensor:
    gravity_robot = reference_body_projected_gravity_robot(env, asset_cfg=asset_cfg)
    tilt_angle = robot_tilt_angle_from_gravity(gravity_robot)
    return torch.square((tilt_angle - threshold).clip(min=0.0)) * _task_mask(env)


def task_phase_ang_vel_xy_l2(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = REFERENCE_BODY_CFG,
) -> torch.Tensor:
    ang_vel_robot = reference_body_ang_vel_robot(env, asset_cfg=asset_cfg)
    return torch.sum(torch.square(ang_vel_robot[:, :2]), dim=1) * _task_mask(env)


def task_phase_lin_vel_z_l2(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = REFERENCE_BODY_CFG,
) -> torch.Tensor:
    lin_vel_robot = reference_body_lin_vel_robot(env, asset_cfg=asset_cfg)
    return torch.square(lin_vel_robot[:, 2]) * _task_mask(env)


def task_phase_forward_speed_abs_exp(
    env: "ManagerBasedRLEnv",
    target: float,
    std: float,
    asset_cfg: SceneEntityCfg = REFERENCE_BODY_CFG,
) -> torch.Tensor:
    lin_vel_robot = reference_body_lin_vel_robot(env, asset_cfg=asset_cfg)
    error = torch.square(torch.abs(lin_vel_robot[:, 0]) - target)
    return torch.exp(-error / (std**2)) * _task_mask(env)


def task_phase_forward_speed_exp(
    env: "ManagerBasedRLEnv",
    target: float,
    std: float,
    asset_cfg: SceneEntityCfg = REFERENCE_BODY_CFG,
) -> torch.Tensor:
    lin_vel_robot = reference_body_lin_vel_robot(env, asset_cfg=asset_cfg)
    error = torch.square(lin_vel_robot[:, 0] - target)
    return torch.exp(-error / (std**2)) * _task_mask(env)


def task_phase_forward_speed_deficit_l1(
    env: "ManagerBasedRLEnv",
    target: float,
    asset_cfg: SceneEntityCfg = REFERENCE_BODY_CFG,
) -> torch.Tensor:
    lin_vel_robot = reference_body_lin_vel_robot(env, asset_cfg=asset_cfg)
    return (target - torch.abs(lin_vel_robot[:, 0])).clip(min=0.0) * _task_mask(env)


def task_phase_forward_speed_deficit_signed_l1(
    env: "ManagerBasedRLEnv",
    target: float,
    asset_cfg: SceneEntityCfg = REFERENCE_BODY_CFG,
) -> torch.Tensor:
    lin_vel_robot = reference_body_lin_vel_robot(env, asset_cfg=asset_cfg)
    return (target - lin_vel_robot[:, 0]).clip(min=0.0) * _task_mask(env)


def task_phase_backward_speed_l1(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = REFERENCE_BODY_CFG,
) -> torch.Tensor:
    lin_vel_robot = reference_body_lin_vel_robot(env, asset_cfg=asset_cfg)
    return (-lin_vel_robot[:, 0]).clip(min=0.0) * _task_mask(env)


def task_phase_joint_deviation_l1(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    return frbot_mdp.joint_deviation_l1(env, asset_cfg=asset_cfg) * _task_mask(env)


def task_phase_joint_pos_limits(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    return frbot_mdp.joint_pos_limits(env, asset_cfg=asset_cfg) * _task_mask(env)


def task_phase_action_rate_l2(env: "ManagerBasedRLEnv") -> torch.Tensor:
    return frbot_mdp.action_rate_l2(env) * _task_mask(env)


def task_phase_lateral_velocity_l2(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = REFERENCE_BODY_CFG,
) -> torch.Tensor:
    lin_vel_robot = reference_body_lin_vel_robot(env, asset_cfg=asset_cfg)
    return torch.square(lin_vel_robot[:, 1]) * _task_mask(env)


def task_phase_mast_top_xy_displacement_l2(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_cfg: SceneEntityCfg = REFERENCE_BODY_CFG,
) -> torch.Tensor:
    rel_pos_robot = mast_top_relative_pos_robot_deviation(env, asset_cfg=asset_cfg, reference_cfg=reference_cfg)
    return torch.sum(torch.square(rel_pos_robot[:, :2]), dim=1) * _task_mask(env)


def task_phase_mast_top_xy_velocity_l2(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_cfg: SceneEntityCfg = REFERENCE_BODY_CFG,
) -> torch.Tensor:
    rel_vel_robot = mast_top_relative_lin_vel_robot(env, asset_cfg=asset_cfg, reference_cfg=reference_cfg)
    return torch.sum(torch.square(rel_vel_robot[:, :2]), dim=1) * _task_mask(env)


def task_phase_mast_top_vertical_velocity_l2(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_cfg: SceneEntityCfg = REFERENCE_BODY_CFG,
) -> torch.Tensor:
    rel_vel_robot = mast_top_relative_lin_vel_robot(env, asset_cfg=asset_cfg, reference_cfg=reference_cfg)
    return torch.square(rel_vel_robot[:, 2]) * _task_mask(env)


def task_phase_mast_top_angular_velocity_l2(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_cfg: SceneEntityCfg = REFERENCE_BODY_CFG,
) -> torch.Tensor:
    ang_vel_robot = mast_top_ang_vel_robot(env, asset_cfg=asset_cfg, reference_cfg=reference_cfg)
    return torch.sum(torch.square(ang_vel_robot), dim=1) * _task_mask(env)

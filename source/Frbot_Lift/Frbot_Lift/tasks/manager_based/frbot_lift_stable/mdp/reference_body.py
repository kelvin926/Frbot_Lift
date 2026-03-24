# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from Frbot_Lift.assets.frbot_lift_asset import REFERENCE_BODY_NAME_EXPR
from Frbot_Lift.tasks.manager_based.frbot_lift.mdp.robot_frame import body_to_robot_frame

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


REFERENCE_BODY_CFG = SceneEntityCfg("robot", body_names=REFERENCE_BODY_NAME_EXPR)


def _reference_body_id(asset: Articulation, asset_cfg: SceneEntityCfg) -> int:
    body_ids = getattr(asset_cfg, "body_ids", None)
    if body_ids is None or isinstance(body_ids, slice):
        body_names = getattr(asset_cfg, "body_names", None) or REFERENCE_BODY_NAME_EXPR
        body_ids, _ = asset.find_bodies(body_names, preserve_order=True)
    if len(body_ids) != 1:
        raise RuntimeError(f"Expected exactly one reference body, got {body_ids}.")
    return int(body_ids[0])


def reference_body_pos_w(
    env: "ManagerBasedEnv",
    asset_cfg: SceneEntityCfg = REFERENCE_BODY_CFG,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_id = _reference_body_id(asset, asset_cfg)
    return asset.data.body_pos_w[:, body_id]


def reference_body_quat_w(
    env: "ManagerBasedEnv",
    asset_cfg: SceneEntityCfg = REFERENCE_BODY_CFG,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_id = _reference_body_id(asset, asset_cfg)
    return asset.data.body_quat_w[:, body_id]


def reference_body_height(
    env: "ManagerBasedEnv",
    asset_cfg: SceneEntityCfg = REFERENCE_BODY_CFG,
) -> torch.Tensor:
    return reference_body_pos_w(env, asset_cfg=asset_cfg)[:, 2] - env.scene.env_origins[:, 2]


def reference_body_lin_vel_w(
    env: "ManagerBasedEnv",
    asset_cfg: SceneEntityCfg = REFERENCE_BODY_CFG,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_id = _reference_body_id(asset, asset_cfg)
    return asset.data.body_lin_vel_w[:, body_id]


def reference_body_ang_vel_w(
    env: "ManagerBasedEnv",
    asset_cfg: SceneEntityCfg = REFERENCE_BODY_CFG,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_id = _reference_body_id(asset, asset_cfg)
    return asset.data.body_ang_vel_w[:, body_id]


def reference_body_lin_vel_robot(
    env: "ManagerBasedEnv",
    asset_cfg: SceneEntityCfg = REFERENCE_BODY_CFG,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_id = _reference_body_id(asset, asset_cfg)
    body_quat_w = asset.data.body_quat_w[:, body_id]
    body_lin_vel_w = asset.data.body_lin_vel_w[:, body_id]
    body_lin_vel_b = math_utils.quat_apply_inverse(body_quat_w, body_lin_vel_w)
    return body_to_robot_frame(body_lin_vel_b)


def reference_body_ang_vel_robot(
    env: "ManagerBasedEnv",
    asset_cfg: SceneEntityCfg = REFERENCE_BODY_CFG,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_id = _reference_body_id(asset, asset_cfg)
    body_quat_w = asset.data.body_quat_w[:, body_id]
    body_ang_vel_w = asset.data.body_ang_vel_w[:, body_id]
    body_ang_vel_b = math_utils.quat_apply_inverse(body_quat_w, body_ang_vel_w)
    return body_to_robot_frame(body_ang_vel_b)


def reference_body_projected_gravity_robot(
    env: "ManagerBasedEnv",
    asset_cfg: SceneEntityCfg = REFERENCE_BODY_CFG,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_id = _reference_body_id(asset, asset_cfg)
    body_quat_w = asset.data.body_quat_w[:, body_id]
    gravity_b = math_utils.quat_apply_inverse(body_quat_w, asset.data.GRAVITY_VEC_W)
    return body_to_robot_frame(gravity_b)

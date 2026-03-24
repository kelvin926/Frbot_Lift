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

from Frbot_Lift.tasks.manager_based.frbot_lift.mdp.robot_frame import body_to_robot_frame

from .reference_body import (
    REFERENCE_BODY_CFG,
    _reference_body_id,
    reference_body_ang_vel_w,
    reference_body_pos_w,
    reference_body_quat_w,
    reference_body_lin_vel_w,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


_CACHE_ATTR = "_frbot_mast_top_cache"
_MIN_HEIGHT_ABOVE_BASE_M = 0.35


def _ensure_mast_cache(
    env: "ManagerBasedEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_cfg: SceneEntityCfg = REFERENCE_BODY_CFG,
) -> tuple[Articulation, dict[str, torch.Tensor]]:
    asset: Articulation = env.scene[asset_cfg.name]
    cache = getattr(asset, _CACHE_ATTR, None)
    if cache is not None:
        return asset, cache

    candidate_ids = []
    for body_id, body_name in enumerate(asset.body_names):
        if body_name == "Group":
            continue
        candidate_ids.append(body_id)

    if not candidate_ids:
        raise RuntimeError("Failed to identify any mast-top candidate bodies.")

    candidate_ids_t = torch.tensor(candidate_ids, device=env.device, dtype=torch.long)
    reference_body_id = _reference_body_id(asset, reference_cfg)

    ref_pos_w0 = asset.data.body_pos_w[0, reference_body_id].unsqueeze(0).repeat(len(candidate_ids), 1)
    ref_quat_w0 = asset.data.body_quat_w[0, reference_body_id].unsqueeze(0).repeat(len(candidate_ids), 1)
    body_pos_w0 = asset.data.body_pos_w[0, candidate_ids_t]
    rel_pos_w0 = body_pos_w0 - ref_pos_w0
    rel_pos_b0 = math_utils.quat_apply_inverse(ref_quat_w0, rel_pos_w0)
    nominal_rel_pos_robot = body_to_robot_frame(rel_pos_b0)

    cache = {
        "candidate_ids": candidate_ids_t,
        "nominal_rel_pos_robot": nominal_rel_pos_robot,
    }
    setattr(asset, _CACHE_ATTR, cache)
    return asset, cache


def mast_top_body_ids(
    env: "ManagerBasedEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_cfg: SceneEntityCfg = REFERENCE_BODY_CFG,
    min_height_above_base: float = _MIN_HEIGHT_ABOVE_BASE_M,
) -> torch.Tensor:
    _, selected_body_ids, _ = _mast_top_selection(
        env,
        asset_cfg=asset_cfg,
        reference_cfg=reference_cfg,
        min_height_above_base=min_height_above_base,
    )
    return selected_body_ids


def _mast_top_selection(
    env: "ManagerBasedEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_cfg: SceneEntityCfg = REFERENCE_BODY_CFG,
    min_height_above_base: float = _MIN_HEIGHT_ABOVE_BASE_M,
) -> tuple[Articulation, torch.Tensor, torch.Tensor]:
    asset, cache = _ensure_mast_cache(env, asset_cfg=asset_cfg, reference_cfg=reference_cfg)
    candidate_ids = cache["candidate_ids"]
    candidate_z = asset.data.body_pos_w[:, candidate_ids, 2]
    base_z = reference_body_pos_w(env, asset_cfg=reference_cfg)[:, 2].unsqueeze(-1)
    high_mask = candidate_z > (base_z + min_height_above_base)
    masked_candidate_z = torch.where(high_mask, candidate_z, torch.full_like(candidate_z, float("-inf")))
    candidate_slots = torch.where(
        torch.any(high_mask, dim=1),
        torch.argmax(masked_candidate_z, dim=1),
        torch.argmax(candidate_z, dim=1),
    )
    selected_body_ids = candidate_ids[candidate_slots]
    return asset, selected_body_ids, candidate_slots


def mast_top_pos_w(
    env: "ManagerBasedEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_cfg: SceneEntityCfg = REFERENCE_BODY_CFG,
) -> torch.Tensor:
    asset, selected_body_ids, _ = _mast_top_selection(env, asset_cfg=asset_cfg, reference_cfg=reference_cfg)
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    return asset.data.body_pos_w[env_ids, selected_body_ids]


def mast_top_lin_vel_w(
    env: "ManagerBasedEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_cfg: SceneEntityCfg = REFERENCE_BODY_CFG,
) -> torch.Tensor:
    asset, selected_body_ids, _ = _mast_top_selection(env, asset_cfg=asset_cfg, reference_cfg=reference_cfg)
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    return asset.data.body_lin_vel_w[env_ids, selected_body_ids]


def mast_top_ang_vel_w(
    env: "ManagerBasedEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_cfg: SceneEntityCfg = REFERENCE_BODY_CFG,
) -> torch.Tensor:
    asset, selected_body_ids, _ = _mast_top_selection(env, asset_cfg=asset_cfg, reference_cfg=reference_cfg)
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    return asset.data.body_ang_vel_w[env_ids, selected_body_ids]


def mast_top_relative_pos_robot(
    env: "ManagerBasedEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_cfg: SceneEntityCfg = REFERENCE_BODY_CFG,
) -> torch.Tensor:
    mast_pos_w = mast_top_pos_w(env, asset_cfg=asset_cfg, reference_cfg=reference_cfg)
    ref_pos_w = reference_body_pos_w(env, asset_cfg=reference_cfg)
    ref_quat_w = reference_body_quat_w(env, asset_cfg=reference_cfg)
    rel_pos_w = mast_pos_w - ref_pos_w
    rel_pos_b = math_utils.quat_apply_inverse(ref_quat_w, rel_pos_w)
    return body_to_robot_frame(rel_pos_b)


def mast_top_relative_pos_robot_deviation(
    env: "ManagerBasedEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_cfg: SceneEntityCfg = REFERENCE_BODY_CFG,
) -> torch.Tensor:
    _, _, candidate_slots = _mast_top_selection(env, asset_cfg=asset_cfg, reference_cfg=reference_cfg)
    asset, cache = _ensure_mast_cache(env, asset_cfg=asset_cfg, reference_cfg=reference_cfg)
    del asset
    nominal_rel_pos_robot = cache["nominal_rel_pos_robot"][candidate_slots]
    return mast_top_relative_pos_robot(env, asset_cfg=asset_cfg, reference_cfg=reference_cfg) - nominal_rel_pos_robot


def mast_top_relative_lin_vel_robot(
    env: "ManagerBasedEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_cfg: SceneEntityCfg = REFERENCE_BODY_CFG,
) -> torch.Tensor:
    mast_lin_vel_w = mast_top_lin_vel_w(env, asset_cfg=asset_cfg, reference_cfg=reference_cfg)
    ref_lin_vel_w = reference_body_lin_vel_w(env, asset_cfg=reference_cfg)
    ref_quat_w = reference_body_quat_w(env, asset_cfg=reference_cfg)
    rel_lin_vel_w = mast_lin_vel_w - ref_lin_vel_w
    rel_lin_vel_b = math_utils.quat_apply_inverse(ref_quat_w, rel_lin_vel_w)
    return body_to_robot_frame(rel_lin_vel_b)


def mast_top_vertical_velocity_robot(
    env: "ManagerBasedEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_cfg: SceneEntityCfg = REFERENCE_BODY_CFG,
) -> torch.Tensor:
    return mast_top_relative_lin_vel_robot(env, asset_cfg=asset_cfg, reference_cfg=reference_cfg)[:, 2:3]


def mast_top_ang_vel_robot(
    env: "ManagerBasedEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_cfg: SceneEntityCfg = REFERENCE_BODY_CFG,
) -> torch.Tensor:
    mast_ang_vel_w = mast_top_ang_vel_w(env, asset_cfg=asset_cfg, reference_cfg=reference_cfg)
    ref_quat_w = reference_body_quat_w(env, asset_cfg=reference_cfg)
    mast_ang_vel_b = math_utils.quat_apply_inverse(ref_quat_w, mast_ang_vel_w)
    return body_to_robot_frame(mast_ang_vel_b)

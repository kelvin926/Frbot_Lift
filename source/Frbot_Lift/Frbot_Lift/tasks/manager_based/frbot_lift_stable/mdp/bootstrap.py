# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from Frbot_Lift.assets.frbot_lift_asset import (
    BOOTSTRAP_DRIVE_RAMP_DURATION_S,
    BOOTSTRAP_HOLD_DURATION_S,
    BOOTSTRAP_LIFT_DURATION_S,
    BOOTSTRAP_SETTLE_DURATION_S,
)
from Frbot_Lift.tasks.manager_based.frbot_lift.mdp.robot_frame import (
    body_to_robot_frame,
    robot_roll_pitch_from_gravity,
    robot_tilt_angle_from_gravity,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


PHASE_SETTLE = 0
PHASE_LIFT_UP = 1
PHASE_HOLD_MAX = 2
PHASE_DRIVE_RAMP = 3
PHASE_TASK = 4

SETTLE_END_S = BOOTSTRAP_SETTLE_DURATION_S
LIFT_UP_END_S = SETTLE_END_S + BOOTSTRAP_LIFT_DURATION_S
HOLD_MAX_END_S = LIFT_UP_END_S + BOOTSTRAP_HOLD_DURATION_S
DRIVE_RAMP_END_S = HOLD_MAX_END_S + BOOTSTRAP_DRIVE_RAMP_DURATION_S

PHASE_NAMES = {
    PHASE_SETTLE: "settle",
    PHASE_LIFT_UP: "lift_up",
    PHASE_HOLD_MAX: "hold_max",
    PHASE_DRIVE_RAMP: "drive_ramp",
    PHASE_TASK: "task",
}


def step_dt(env: "ManagerBasedEnv") -> float:
    return float(getattr(env, "step_dt", env.cfg.sim.dt * env.cfg.decimation))


def elapsed_time_s(env: "ManagerBasedEnv") -> torch.Tensor:
    return env.episode_length_buf.to(dtype=torch.float32, device=env.device) * step_dt(env)


def phase_index(env: "ManagerBasedEnv") -> torch.Tensor:
    elapsed_s = elapsed_time_s(env)
    phase = torch.full((env.num_envs,), PHASE_SETTLE, device=env.device, dtype=torch.long)
    phase = torch.where(elapsed_s >= SETTLE_END_S, PHASE_LIFT_UP, phase)
    phase = torch.where(elapsed_s >= LIFT_UP_END_S, PHASE_HOLD_MAX, phase)
    phase = torch.where(elapsed_s >= HOLD_MAX_END_S, PHASE_DRIVE_RAMP, phase)
    phase = torch.where(elapsed_s >= DRIVE_RAMP_END_S, PHASE_TASK, phase)
    return phase


def phase_index_obs(env: "ManagerBasedEnv") -> torch.Tensor:
    return phase_index(env).to(dtype=torch.float32).unsqueeze(-1)


def phase_name_from_index(phase_idx: int) -> str:
    return PHASE_NAMES.get(int(phase_idx), f"unknown_{phase_idx}")


def _phase_progress(elapsed_s: torch.Tensor, start_s: float, duration_s: float) -> torch.Tensor:
    if duration_s <= 0.0:
        return (elapsed_s >= start_s).to(dtype=torch.float32)
    return torch.clamp((elapsed_s - start_s) / duration_s, min=0.0, max=1.0)


def lift_progress(env: "ManagerBasedEnv") -> torch.Tensor:
    elapsed_s = elapsed_time_s(env)
    return _phase_progress(elapsed_s, SETTLE_END_S, BOOTSTRAP_LIFT_DURATION_S)


def drive_progress(env: "ManagerBasedEnv") -> torch.Tensor:
    elapsed_s = elapsed_time_s(env)
    return _phase_progress(elapsed_s, HOLD_MAX_END_S, BOOTSTRAP_DRIVE_RAMP_DURATION_S)


def task_phase_mask(env: "ManagerBasedEnv") -> torch.Tensor:
    return phase_index(env) >= PHASE_TASK


def task_phase_mask_f(env: "ManagerBasedEnv") -> torch.Tensor:
    return task_phase_mask(env).to(dtype=torch.float32)


def suspension_enabled_mask(env: "ManagerBasedEnv") -> torch.Tensor:
    return phase_index(env) >= PHASE_DRIVE_RAMP


def bootstrap_phase_mask(env: "ManagerBasedEnv") -> torch.Tensor:
    return phase_index(env) < PHASE_TASK


def root_height_relative(
    env: "ManagerBasedEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]


def robot_tilt_angle(
    env: "ManagerBasedEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    projected_gravity_robot = body_to_robot_frame(asset.data.projected_gravity_b)
    return robot_tilt_angle_from_gravity(projected_gravity_robot)


def robot_roll_pitch_deg(
    env: "ManagerBasedEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> tuple[torch.Tensor, torch.Tensor]:
    asset: Articulation = env.scene[asset_cfg.name]
    projected_gravity_robot = body_to_robot_frame(asset.data.projected_gravity_b)
    roll, pitch = robot_roll_pitch_from_gravity(projected_gravity_robot)
    return torch.rad2deg(roll), torch.rad2deg(pitch)

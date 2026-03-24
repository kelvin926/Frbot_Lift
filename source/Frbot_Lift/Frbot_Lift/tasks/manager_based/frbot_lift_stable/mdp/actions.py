# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs.mdp.actions.actions_cfg import EMAJointPositionToLimitsActionCfg as BaseEMAJointPositionToLimitsActionCfg
from isaaclab.envs.mdp.actions.joint_actions_to_limits import EMAJointPositionToLimitsAction as BaseEMAJointPositionToLimitsAction
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass

from Frbot_Lift.tasks.manager_based.frbot_lift.mdp.robot_frame import body_to_robot_frame, robot_to_body_frame

from . import bootstrap


class BootstrapWindowedSuspensionAction(BaseEMAJointPositionToLimitsAction):
    """Suspension action that stays at the midpoint during bootstrap and only opens a narrow RL window afterwards."""

    def __init__(self, cfg: ActionTermCfg, env):
        super().__init__(cfg, env)
        self._midpoint = torch.full((self.num_envs, self.action_dim), float(cfg.midpoint), device=self.device)
        self._action_window = torch.full((self.num_envs, self.action_dim), float(cfg.action_window), device=self.device)

    def reset(self, env_ids=None) -> None:
        if env_ids is None or isinstance(env_ids, slice):
            self._raw_actions[:] = 0.0
            self._prev_applied_actions[:] = self._midpoint
            self._processed_actions[:] = self._midpoint
            return

        self._raw_actions[env_ids] = 0.0
        self._prev_applied_actions[env_ids] = self._midpoint[env_ids]
        self._processed_actions[env_ids] = self._midpoint[env_ids]

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions

        centered_targets = self._midpoint + actions.clamp(-1.0, 1.0) * self._action_window
        enabled_mask = bootstrap.suspension_enabled_mask(self._env).unsqueeze(-1)
        commanded_targets = torch.where(enabled_mask, centered_targets, self._midpoint)

        ema_actions = self._alpha * commanded_targets
        ema_actions += (1.0 - self._alpha) * self._prev_applied_actions
        self._processed_actions[:] = torch.clamp(
            ema_actions,
            self._asset.data.soft_joint_pos_limits[:, self._joint_ids, 0],
            self._asset.data.soft_joint_pos_limits[:, self._joint_ids, 1],
        )
        self._prev_applied_actions[:] = self._processed_actions


@configclass
class BootstrapWindowedSuspensionActionCfg(BaseEMAJointPositionToLimitsActionCfg):
    """Configuration for phase-gated suspension control around a neutral midpoint."""

    class_type: type[ActionTerm] = BootstrapWindowedSuspensionAction
    midpoint: float = 0.0
    action_window: float = 0.0


class BootstrapDrivetrainAction(ActionTerm):
    """Auxiliary zero-dim term that scripts steer, lift, and wheel targets through the bootstrap phases."""

    _asset: Articulation

    def __init__(self, cfg: ActionTermCfg, env):
        super().__init__(cfg, env)
        self._export_IO_descriptor = False
        self._raw_actions = torch.zeros((self.num_envs, 0), device=self.device)
        self._processed_actions = self._raw_actions

        self._target_velocity_robot = torch.tensor(
            getattr(cfg, "target_velocity_robot", (0.0, 0.0, 0.0)),
            device=self.device,
            dtype=torch.float32,
        ).unsqueeze(0).repeat(self.num_envs, 1)

        self._steer_joint_ids = None
        self._steer_targets = None
        steer_targets = getattr(cfg, "steer_position_targets", None) or {}
        if steer_targets:
            joint_names = list(steer_targets.keys())
            self._steer_joint_ids, _ = self._asset.find_joints(joint_names, preserve_order=True)
            values = torch.tensor(list(steer_targets.values()), device=self.device, dtype=torch.float32)
            self._steer_targets = values.unsqueeze(0).repeat(self.num_envs, 1)

        self._lift_joint_ids = None
        self._lift_start_targets = None
        self._lift_max_targets = None
        lift_start_targets = getattr(cfg, "lift_start_targets", None) or {}
        lift_max_targets = getattr(cfg, "lift_max_targets", None) or {}
        if lift_start_targets and lift_max_targets:
            joint_names = list(lift_max_targets.keys())
            self._lift_joint_ids, _ = self._asset.find_joints(joint_names, preserve_order=True)
            start_values = torch.tensor(
                [lift_start_targets[name] for name in joint_names], device=self.device, dtype=torch.float32
            )
            max_values = torch.tensor(
                [lift_max_targets[name] for name in joint_names], device=self.device, dtype=torch.float32
            )
            self._lift_start_targets = start_values.unsqueeze(0).repeat(self.num_envs, 1)
            self._lift_max_targets = max_values.unsqueeze(0).repeat(self.num_envs, 1)

        self._wheel_joint_ids = None
        self._wheel_velocity_targets = None
        wheel_velocity_targets = getattr(cfg, "wheel_velocity_targets", None) or {}
        if wheel_velocity_targets:
            joint_names = list(wheel_velocity_targets.keys())
            self._wheel_joint_ids, _ = self._asset.find_joints(joint_names, preserve_order=True)
            values = torch.tensor(list(wheel_velocity_targets.values()), device=self.device, dtype=torch.float32)
            self._wheel_velocity_targets = values.unsqueeze(0).repeat(self.num_envs, 1)

    @property
    def action_dim(self) -> int:
        return 0

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions = actions
        self._processed_actions = actions

    def reset(self, env_ids=None) -> None:
        if env_ids is None or isinstance(env_ids, slice):
            self._raw_actions[:] = 0.0
            self._processed_actions[:] = 0.0
            return
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0

    def apply_actions(self):
        if self._steer_targets is not None:
            self._asset.set_joint_position_target(self._steer_targets, joint_ids=self._steer_joint_ids)
        if self._lift_start_targets is not None and self._lift_max_targets is not None:
            lift_alpha = bootstrap.lift_progress(self._env).unsqueeze(-1)
            lift_targets = self._lift_start_targets + (self._lift_max_targets - self._lift_start_targets) * lift_alpha
            self._asset.set_joint_position_target(lift_targets, joint_ids=self._lift_joint_ids)
        if self._wheel_velocity_targets is not None:
            drive_alpha = bootstrap.drive_progress(self._env).unsqueeze(-1)
            self._asset.set_joint_velocity_target(self._wheel_velocity_targets * drive_alpha, joint_ids=self._wheel_joint_ids)

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "target_velocity_visualizer"):
                target_cfg = GREEN_ARROW_X_MARKER_CFG.copy()
                target_cfg.prim_path = "/Visuals/Actions/bootstrap_target_velocity"
                target_cfg.markers["arrow"].scale = (0.35, 0.12, 0.12)
                self.target_velocity_visualizer = VisualizationMarkers(target_cfg)

                current_cfg = BLUE_ARROW_X_MARKER_CFG.copy()
                current_cfg.prim_path = "/Visuals/Actions/bootstrap_current_velocity"
                current_cfg.markers["arrow"].scale = (0.35, 0.12, 0.12)
                self.current_velocity_visualizer = VisualizationMarkers(current_cfg)
            self.target_velocity_visualizer.set_visibility(True)
            self.current_velocity_visualizer.set_visibility(True)
        else:
            if hasattr(self, "target_velocity_visualizer"):
                self.target_velocity_visualizer.set_visibility(False)
                self.current_velocity_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self._asset.is_initialized:
            return

        body_pos_w = self._asset.data.body_pos_w
        marker_pos_w = self._asset.data.root_pos_w.clone()
        marker_pos_w[:, 2] = torch.max(body_pos_w[..., 2], dim=1).values + 0.35

        drive_alpha = bootstrap.drive_progress(self._env).unsqueeze(-1)
        current_velocity_robot = body_to_robot_frame(self._asset.data.root_lin_vel_b)
        current_scale, current_quat_w = self._resolve_velocity_to_arrow(current_velocity_robot)
        target_velocity_robot = self._target_velocity_robot * drive_alpha
        target_scale, target_quat_w = self._resolve_velocity_to_arrow(target_velocity_robot)

        self.target_velocity_visualizer.visualize(marker_pos_w, target_quat_w, target_scale)
        self.current_velocity_visualizer.visualize(marker_pos_w, current_quat_w, current_scale)

    def _resolve_velocity_to_arrow(self, velocity_robot: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        default_scale = torch.tensor(self.target_velocity_visualizer.cfg.markers["arrow"].scale, device=self.device)
        arrow_scale = default_scale.unsqueeze(0).repeat(velocity_robot.shape[0], 1)

        speed_xy = torch.linalg.norm(velocity_robot[:, :2], dim=1)
        arrow_scale[:, 0] *= torch.clamp(speed_xy * 8.0, min=0.35)

        velocity_body = robot_to_body_frame(
            torch.cat([velocity_robot[:, :2], torch.zeros_like(speed_xy).unsqueeze(-1)], dim=1)
        )
        velocity_world = math_utils.quat_apply(self._asset.data.root_quat_w, velocity_body)
        heading_angle = torch.atan2(velocity_world[:, 1], velocity_world[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        return arrow_scale, arrow_quat


@configclass
class BootstrapDrivetrainActionCfg(ActionTermCfg):
    """Configuration for scripted bootstrap steering, lift, and wheel targets."""

    class_type: type[ActionTerm] = BootstrapDrivetrainAction
    steer_position_targets: dict[str, float] | None = None
    lift_start_targets: dict[str, float] | None = None
    lift_max_targets: dict[str, float] | None = None
    wheel_velocity_targets: dict[str, float] | None = None
    target_velocity_robot: tuple[float, float, float] = (0.0, 0.0, 0.0)

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass

from .robot_frame import body_to_robot_frame, robot_to_body_frame


class FixedJointTargetsAction(ActionTerm):
    """Auxiliary zero-dim action term that keeps drivetrain joints at fixed targets."""

    _asset: Articulation

    def __init__(self, cfg: ActionTermCfg, env):
        super().__init__(cfg, env)
        self._export_IO_descriptor = False
        self._raw_actions = torch.zeros((self.num_envs, 0), device=self.device)
        self._processed_actions = self._raw_actions
        self._asset: Articulation
        self._target_velocity_robot = torch.tensor(
            getattr(cfg, "target_velocity_robot", (0.0, 0.0, 0.0)),
            device=self.device,
            dtype=torch.float32,
        ).unsqueeze(0).repeat(self.num_envs, 1)
        self._ramp_duration_s = float(getattr(cfg, "ramp_duration_s", 0.0))

        position_targets = getattr(cfg, "position_targets", None) or {}
        velocity_targets = getattr(cfg, "velocity_targets", None) or {}

        self._position_joint_ids = None
        self._position_targets = None
        if position_targets:
            joint_names = list(position_targets.keys())
            self._position_joint_ids, _ = self._asset.find_joints(joint_names, preserve_order=True)
            values = torch.tensor(list(position_targets.values()), device=self.device, dtype=torch.float32)
            self._position_targets = values.unsqueeze(0).repeat(self.num_envs, 1)

        self._velocity_joint_ids = None
        self._velocity_targets = None
        if velocity_targets:
            joint_names = list(velocity_targets.keys())
            self._velocity_joint_ids, _ = self._asset.find_joints(joint_names, preserve_order=True)
            values = torch.tensor(list(velocity_targets.values()), device=self.device, dtype=torch.float32)
            self._velocity_targets = values.unsqueeze(0).repeat(self.num_envs, 1)

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

    def apply_actions(self):
        if self._position_targets is not None:
            self._asset.set_joint_position_target(self._position_targets, joint_ids=self._position_joint_ids)
        if self._velocity_targets is not None:
            ramp = self._velocity_ramp_scale()
            self._asset.set_joint_velocity_target(self._velocity_targets * ramp, joint_ids=self._velocity_joint_ids)

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "target_velocity_visualizer"):
                target_cfg = GREEN_ARROW_X_MARKER_CFG.copy()
                target_cfg.prim_path = "/Visuals/Actions/drivetrain_target_velocity"
                target_cfg.markers["arrow"].scale = (0.35, 0.12, 0.12)
                self.target_velocity_visualizer = VisualizationMarkers(target_cfg)

                current_cfg = BLUE_ARROW_X_MARKER_CFG.copy()
                current_cfg.prim_path = "/Visuals/Actions/drivetrain_current_velocity"
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

        current_velocity_robot = body_to_robot_frame(self._asset.data.root_lin_vel_b)
        current_scale, current_quat_w = self._resolve_velocity_to_arrow(current_velocity_robot)
        target_velocity_robot = self._target_velocity_robot * self._velocity_ramp_scale()
        target_scale, target_quat_w = self._resolve_velocity_to_arrow(target_velocity_robot)

        self.target_velocity_visualizer.visualize(marker_pos_w, target_quat_w, target_scale)
        self.current_velocity_visualizer.visualize(marker_pos_w, current_quat_w, current_scale)

    def _resolve_velocity_to_arrow(self, velocity_robot: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        default_scale = torch.tensor(self.target_velocity_visualizer.cfg.markers["arrow"].scale, device=self.device)
        arrow_scale = default_scale.unsqueeze(0).repeat(velocity_robot.shape[0], 1)

        speed_xy = torch.linalg.norm(velocity_robot[:, :2], dim=1)
        arrow_scale[:, 0] *= torch.clamp(speed_xy * 4.0, min=0.35)

        velocity_body = robot_to_body_frame(torch.cat([velocity_robot[:, :2], torch.zeros_like(speed_xy).unsqueeze(-1)], dim=1))
        velocity_world = math_utils.quat_apply(self._asset.data.root_quat_w, velocity_body)
        heading_angle = torch.atan2(velocity_world[:, 1], velocity_world[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        return arrow_scale, arrow_quat

    def _velocity_ramp_scale(self) -> torch.Tensor:
        if self._ramp_duration_s <= 0.0:
            return torch.ones((self.num_envs, 1), device=self.device)

        step_dt = float(getattr(self._env, "step_dt", self._env.cfg.sim.dt * self._env.cfg.decimation))
        elapsed_s = self._env.episode_length_buf.to(dtype=torch.float32, device=self.device).unsqueeze(-1) * step_dt
        return torch.clamp(elapsed_s / self._ramp_duration_s, min=0.0, max=1.0)


@configclass
class FixedJointTargetsActionCfg(ActionTermCfg):
    """Configuration for fixed drivetrain targets."""

    class_type: type[ActionTerm] = FixedJointTargetsAction
    position_targets: dict[str, float] | None = None
    velocity_targets: dict[str, float] | None = None
    target_velocity_robot: tuple[float, float, float] = (0.0, 0.0, 0.0)
    ramp_duration_s: float = 0.0

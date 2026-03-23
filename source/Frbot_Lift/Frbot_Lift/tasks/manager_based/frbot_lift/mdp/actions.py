# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass


class FixedJointTargetsAction(ActionTerm):
    """Auxiliary zero-dim action term that keeps drivetrain joints at fixed targets."""

    _asset: Articulation

    def __init__(self, cfg: ActionTermCfg, env):
        super().__init__(cfg, env)
        self._export_IO_descriptor = False
        self._raw_actions = torch.zeros((self.num_envs, 0), device=self.device)
        self._processed_actions = self._raw_actions

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
            self._asset.set_joint_velocity_target(self._velocity_targets, joint_ids=self._velocity_joint_ids)


@configclass
class FixedJointTargetsActionCfg(ActionTermCfg):
    """Configuration for fixed drivetrain targets."""

    class_type: type[ActionTerm] = FixedJointTargetsAction
    position_targets: dict[str, float] | None = None
    velocity_targets: dict[str, float] | None = None

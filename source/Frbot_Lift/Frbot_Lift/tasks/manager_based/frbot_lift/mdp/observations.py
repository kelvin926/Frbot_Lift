# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def constant_forward_command(env: "ManagerBasedEnv", value: float) -> torch.Tensor:
    """Return a constant scalar command used as context for the policy."""

    return torch.full((env.num_envs, 1), value, device=env.device, dtype=torch.float32)

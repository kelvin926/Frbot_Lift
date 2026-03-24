# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch


def body_to_robot_frame(vector_b: torch.Tensor) -> torch.Tensor:
    """Map the imported Y-up body frame into a conventional robot frame (x forward, y lateral, z up)."""

    return torch.stack((vector_b[..., 2], vector_b[..., 0], vector_b[..., 1]), dim=-1)


def robot_to_body_frame(vector_robot: torch.Tensor) -> torch.Tensor:
    """Map vectors from the robot-aligned frame back into the imported raw body frame."""

    return torch.stack((vector_robot[..., 1], vector_robot[..., 2], vector_robot[..., 0]), dim=-1)


def robot_tilt_angle_from_gravity(projected_gravity_robot: torch.Tensor) -> torch.Tensor:
    """Compute upright tilt from a gravity vector expressed in the robot-aligned frame."""

    return torch.atan2(
        torch.linalg.vector_norm(projected_gravity_robot[..., :2], dim=-1),
        (-projected_gravity_robot[..., 2]).clip(min=1.0e-6),
    )


def robot_roll_pitch_from_gravity(projected_gravity_robot: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate roll/pitch from gravity in the robot-aligned frame."""

    roll = torch.atan2(projected_gravity_robot[..., 1], (-projected_gravity_robot[..., 2]).clip(min=1.0e-6))
    pitch = torch.atan2(
        -projected_gravity_robot[..., 0],
        torch.linalg.vector_norm(projected_gravity_robot[..., 1:], dim=-1).clip(min=1.0e-6),
    )
    return roll, pitch

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg


def _resolve_project_root() -> Path:
    """Prefer an ASCII-safe root passed from the launcher script on Windows."""

    env_root = os.environ.get("FRBOT_LIFT_ROOT")
    if env_root:
        return Path(env_root)

    alias_root = Path("R:/Frbot_Lift")
    if alias_root.exists():
        return alias_root

    return Path(__file__).resolve().parents[4]


PROJECT_ROOT = _resolve_project_root()
ASSET_PATH = Path(os.environ.get("FRBOT_LIFT_ASSET_PATH", str(PROJECT_ROOT / "assets" / "robots" / "lift_amr_new.usd")))

ARTICULATION_ROOT_PRIM_PATH = "/AMR/tn__amrassy_rev10_mDo28f3/Group"
BASE_ATTACH_PRIM_PATH = ARTICULATION_ROOT_PRIM_PATH

SUSPENSION_JOINTS = ["FL_suspenstion", "FR_suspenstion", "RL_suspenstion", "RR_suspenstion"]
STEER_JOINTS = ["FL_steer", "FR_steer", "RL_steer", "RR_steer"]
WHEEL_JOINTS = ["FL_Wheel", "FR_Wheel", "RL_Wheel", "RR_Wheel"]
LIFT_JOINTS = [
    "Link_1to7_2_joint",
    "Link_2to3_2_joint",
    "Link_3to4_2_joint",
    "Link_4to5_2_joint",
    "Link_5to6_2_joint",
    "Link_6to7_2_joint",
]

SUSPENSION_MIDPOINT_RAD = math.radians(-9.0)
WHEEL_TARGET_RAD_S = -math.radians(314.0)
# Use the observed baseline straight-drive speed as the nominal tracking target.
FORWARD_VELOCITY_TARGET_MPS = 0.16

STEER_POSITION_TARGETS_RAD = {
    "FL_steer": math.pi,
    "FR_steer": math.pi,
    "RL_steer": 0.0,
    "RR_steer": 0.0,
}
WHEEL_VELOCITY_TARGETS_RAD_S = {joint_name: WHEEL_TARGET_RAD_S for joint_name in WHEEL_JOINTS}
LIFT_POSITION_TARGETS_M = {
    "Link_1to7_2_joint": 5.4,
    "Link_2to3_2_joint": 0.9,
    "Link_3to4_2_joint": 0.9,
    "Link_4to5_2_joint": -0.9,
    "Link_5to6_2_joint": -0.9,
    "Link_6to7_2_joint": -0.9,
}
DEFAULT_JOINT_POSITIONS = {
    **STEER_POSITION_TARGETS_RAD,
    **LIFT_POSITION_TARGETS_M,
    **{joint_name: SUSPENSION_MIDPOINT_RAD for joint_name in SUSPENSION_JOINTS},
}


FRBOT_LIFT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(ASSET_PATH),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=2,
        ),
    ),
    articulation_root_prim_path=ARTICULATION_ROOT_PRIM_PATH,
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos=DEFAULT_JOINT_POSITIONS,
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=1.0,
    actuators={
        "steer": ImplicitActuatorCfg(
            joint_names_expr=STEER_JOINTS,
            effort_limit_sim=3000.0,
            velocity_limit_sim=20.0,
            stiffness=8000.0,
            damping=400.0,
        ),
        "lift": ImplicitActuatorCfg(
            joint_names_expr=LIFT_JOINTS,
            effort_limit_sim=20000.0,
            velocity_limit_sim=10.0,
            stiffness=40000.0,
            damping=4000.0,
        ),
        "suspension": ImplicitActuatorCfg(
            joint_names_expr=SUSPENSION_JOINTS,
            effort_limit_sim=12000.0,
            velocity_limit_sim=10.0,
            stiffness=30000.0,
            damping=2500.0,
        ),
        "wheel": ImplicitActuatorCfg(
            joint_names_expr=WHEEL_JOINTS,
            effort_limit_sim=20000.0,
            velocity_limit_sim=40.0,
            stiffness=0.0,
            damping=4000.0,
        )
    },
)

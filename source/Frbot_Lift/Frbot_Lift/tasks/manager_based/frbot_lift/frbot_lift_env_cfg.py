# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.utils import configclass

from Frbot_Lift.assets.frbot_lift_asset import (
    BASE_ATTACH_PRIM_PATH,
    BASE_INIT_HEIGHT_M,
    FORWARD_VELOCITY_TARGET_MPS,
    FRBOT_LIFT_CFG,
    LIFT_POSITION_TARGETS_M,
    STEER_POSITION_TARGETS_RAD,
    SUSPENSION_JOINTS,
    WHEEL_VELOCITY_TARGETS_RAD_S,
)

from . import mdp


LEVELING_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=True,
    size=(6.0, 6.0),
    border_width=1.0,
    num_rows=5,
    num_cols=5,
    horizontal_scale=0.05,
    vertical_scale=0.001,
    slope_threshold=0.6,
    use_cache=False,
    sub_terrains={
        "plane": terrain_gen.MeshPlaneTerrainCfg(proportion=0.25),
        "rough_low": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.25,
            noise_range=(0.003, 0.008),
            noise_step=0.001,
            border_width=0.2,
        ),
        "rough_mid": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.25,
            noise_range=(0.008, 0.015),
            noise_step=0.001,
            border_width=0.2,
        ),
        "rough_mid_dense": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.25,
            noise_range=(0.006, 0.012),
            noise_step=0.001,
            border_width=0.2,
        ),
    },
)


@configclass
class FrbotLiftSceneCfg(InteractiveSceneCfg):
    """Scene configuration for suspension leveling."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=LEVELING_TERRAINS_CFG,
        max_init_terrain_level=0,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.1,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.36, 0.34, 0.31)),
        debug_vis=False,
    )
    robot: ArticulationCfg = FRBOT_LIFT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    height_scanner = RayCasterCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot{BASE_ATTACH_PRIM_PATH}",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 1.0, 0.0)),
        ray_alignment="world",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.15, size=(2.0, 1.4)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.95, 0.95, 0.95), intensity=1500.0),
    )


@configclass
class ActionsCfg:
    """Action specification."""

    suspension = mdp.EMAJointPositionToLimitsActionCfg(
        asset_name="robot",
        joint_names=SUSPENSION_JOINTS,
        preserve_order=True,
        alpha=0.7,
    )
    drivetrain = mdp.FixedJointTargetsActionCfg(
        asset_name="robot",
        position_targets={**STEER_POSITION_TARGETS_RAD, **LIFT_POSITION_TARGETS_M},
        velocity_targets=WHEEL_VELOCITY_TARGETS_RAD_S,
        target_velocity_robot=(FORWARD_VELOCITY_TARGET_MPS, 0.0, 0.0),
        debug_vis=True,
    )


@configclass
class ObservationsCfg:
    """Observation specification."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Policy observations."""

        base_lin_vel = ObsTerm(func=mdp.robot_base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.robot_base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.robot_projected_gravity)
        suspension_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=SUSPENSION_JOINTS, preserve_order=True)},
        )
        suspension_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=SUSPENSION_JOINTS, preserve_order=True)},
        )
        previous_actions = ObsTerm(func=mdp.last_action, params={"action_name": "suspension"})
        forward_command = ObsTerm(func=mdp.constant_forward_command, params={"value": FORWARD_VELOCITY_TARGET_MPS})
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": 0.0},
            clip=(-2.0, 2.0),
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Reset events."""

    reset_root = EventTerm(
        func=mdp.reset_root_state_uniform_robot_frame,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.10, 0.10),
                "y": (-0.10, 0.10),
                "z": (0.0, 0.0),
                "roll": (-0.03, 0.03),
                "pitch": (-0.03, 0.03),
                "yaw": (0.0, 0.0),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    reset_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms."""

    alive = RewTerm(func=mdp.is_alive, weight=0.2)
    terminating = RewTerm(
        func=mdp.is_terminated_term,
        weight=-10.0,
        params={"term_keys": ["bad_orientation", "terrain_out_of_bounds", "stalled_progress"]},
    )
    flat_orientation = RewTerm(func=mdp.flat_orientation_robot_l2, weight=-24.0)
    tilt_margin = RewTerm(
        func=mdp.tilt_margin_l2,
        weight=-60.0,
        params={"threshold": math.radians(6.0)},
    )
    angular_velocity = RewTerm(func=mdp.robot_ang_vel_xy_l2, weight=-2.0)
    vertical_velocity = RewTerm(func=mdp.robot_lin_vel_z_l2, weight=-2.0)
    lateral_velocity = RewTerm(func=mdp.robot_lateral_velocity_l2, weight=-1.0)
    forward_velocity = RewTerm(
        func=mdp.track_forward_speed_abs_exp,
        weight=8.0,
        params={"target": FORWARD_VELOCITY_TARGET_MPS, "std": 0.08},
    )
    forward_speed = RewTerm(func=mdp.forward_speed_abs, weight=2.5)
    forward_speed_deficit = RewTerm(
        func=mdp.forward_speed_deficit_l1,
        weight=-4.0,
        params={"target": FORWARD_VELOCITY_TARGET_MPS * 0.8},
    )
    suspension_centering = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=SUSPENSION_JOINTS, preserve_order=True)},
    )
    suspension_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=SUSPENSION_JOINTS, preserve_order=True)},
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.03)
    action_magnitude = RewTerm(func=mdp.action_l2, weight=-0.005)


@configclass
class TerminationsCfg:
    """Termination terms."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    bad_orientation = DoneTerm(func=mdp.bad_orientation_robot_frame, params={"limit_angle": math.radians(15.0)})
    terrain_out_of_bounds = DoneTerm(func=mdp.terrain_out_of_bounds, params={"distance_limit": 2.4})
    stalled_progress = DoneTerm(func=mdp.stalled_progress, params={"min_progress": 0.03, "min_steps": 120})


@configclass
class CurriculumCfg:
    """Curriculum terms."""

    terrain_levels = CurrTerm(
        func=mdp.terrain_levels_progress,
        params={"progress_threshold": 0.20, "regression_threshold": 0.04},
    )


@configclass
class FrbotLiftEnvCfg(ManagerBasedRLEnvCfg):
    """Training environment."""

    scene: FrbotLiftSceneCfg = FrbotLiftSceneCfg(num_envs=1024, env_spacing=6.0, replicate_physics=True)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self) -> None:
        self.decimation = 4
        self.episode_length_s = 10.0
        self.viewer.eye = (5.5, 0.0, 3.0)
        self.viewer.lookat = (0.0, 0.0, 0.6)
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        # The long lift and dense wheel contacts can overflow the default GPU collision stack in GUI training.
        self.sim.physx.gpu_collision_stack_size = 2**28


@configclass
class FrbotLiftPlayEnvCfg(FrbotLiftEnvCfg):
    """Playback and video environment."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 8
        self.scene.env_spacing = 6.0
        self.observations.policy.enable_corruption = False
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = False
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
        self.curriculum = None

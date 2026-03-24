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
from isaaclab.terrains import FlatPatchSamplingCfg, TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.utils import configclass

from Frbot_Lift.assets.frbot_lift_asset import (
    BASE_ATTACH_PRIM_PATH,
    FORWARD_VELOCITY_TARGET_MPS,
    FRBOT_LIFT_STABLE_CFG,
    REFERENCE_BODY_NAME_EXPR,
    STABLE_BASE_INIT_HEIGHT_M,
    SUSPENSION_JOINTS,
    SUSPENSION_MIDPOINT_RAD,
    WHEEL_VELOCITY_TARGETS_RAD_S,
)

from . import mdp


PLANE_INIT_POS_FLAT_PATCH_CFG = {
    "init_pos": FlatPatchSamplingCfg(
        num_patches=32,
        patch_radius=[0.35, 0.25],
        x_range=(-1.9, -1.6),
        y_range=(-0.4, 0.4),
        z_range=(-0.05, 0.05),
        max_height_diff=0.001,
    )
}


ROUGH_INIT_POS_FLAT_PATCH_CFG = {
    "init_pos": FlatPatchSamplingCfg(
        num_patches=32,
        patch_radius=[0.35, 0.25],
        x_range=(-1.9, -1.6),
        y_range=(-0.4, 0.4),
        z_range=(-0.05, 0.05),
        max_height_diff=0.001,
    )
}


MILD_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=True,
    size=(8.0, 8.0),
    border_width=0.25,
    num_rows=3,
    num_cols=4,
    horizontal_scale=0.05,
    vertical_scale=0.001,
    slope_threshold=0.6,
    use_cache=False,
    sub_terrains={
        "plane": terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.82,
            flat_patch_sampling=PLANE_INIT_POS_FLAT_PATCH_CFG,
        ),
        "rough_low": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.18,
            noise_range=(0.0015, 0.004),
            noise_step=0.001,
            downsampled_scale=0.30,
            border_width=1.5,
            flat_patch_sampling=ROUGH_INIT_POS_FLAT_PATCH_CFG,
        ),
    },
)


@configclass
class StableFlatSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        env_spacing=6.0,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.1,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.36, 0.34, 0.31)),
        debug_vis=False,
    )
    robot: ArticulationCfg = FRBOT_LIFT_STABLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
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
class StableAdaptiveSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=MILD_ROUGH_TERRAINS_CFG,
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
    robot: ArticulationCfg = FRBOT_LIFT_STABLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
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
class StableFlatActionsCfg:
    suspension = mdp.BootstrapWindowedSuspensionActionCfg(
        asset_name="robot",
        joint_names=SUSPENSION_JOINTS,
        preserve_order=True,
        scale=1.0,
        rescale_to_limits=False,
        alpha=0.05,
        midpoint=SUSPENSION_MIDPOINT_RAD,
        action_window=math.radians(0.2),
    )
    drivetrain = mdp.BootstrapDrivetrainActionCfg(
        asset_name="robot",
        steer_position_targets=None,
        lift_start_targets=None,
        lift_max_targets=None,
        wheel_velocity_targets=WHEEL_VELOCITY_TARGETS_RAD_S,
        target_velocity_robot=(FORWARD_VELOCITY_TARGET_MPS, 0.0, 0.0),
        debug_vis=False,
    )


@configclass
class StableAdaptiveActionsCfg(StableFlatActionsCfg):
    suspension = mdp.BootstrapWindowedSuspensionActionCfg(
        asset_name="robot",
        joint_names=SUSPENSION_JOINTS,
        preserve_order=True,
        scale=1.0,
        rescale_to_limits=False,
        alpha=0.08,
        midpoint=SUSPENSION_MIDPOINT_RAD,
        action_window=math.radians(0.6),
    )


@configclass
class StableObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(
            func=mdp.reference_body_lin_vel_robot,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=REFERENCE_BODY_NAME_EXPR)},
        )
        base_ang_vel = ObsTerm(
            func=mdp.reference_body_ang_vel_robot,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=REFERENCE_BODY_NAME_EXPR)},
        )
        projected_gravity = ObsTerm(
            func=mdp.reference_body_projected_gravity_robot,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=REFERENCE_BODY_NAME_EXPR)},
        )
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
        phase = ObsTerm(func=mdp.phase_index_obs)
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
class StableFlatEventCfg:
    reset_root = EventTerm(
        func=mdp.reset_root_state_uniform_robot_frame_height,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
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
            "root_height": STABLE_BASE_INIT_HEIGHT_M,
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
class StableAdaptiveEventCfg(StableFlatEventCfg):
    startup_plane_terrain_types = EventTerm(
        func=mdp.assign_terrain_types,
        mode="startup",
        params={"terrain_types": (0, 1)},
    )
    reset_root = EventTerm(
        func=mdp.reset_root_state_from_terrain_robot_frame,
        mode="reset",
        params={
            "pose_range": {
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
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
            "root_height": STABLE_BASE_INIT_HEIGHT_M,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


@configclass
class StableRewardsCfg:
    alive = RewTerm(func=mdp.task_phase_alive, weight=0.2)
    terminating = RewTerm(
        func=mdp.task_phase_terminated_term,
        weight=-4.0,
        params={
            "term_keys": [
                "catastrophic_bad_orientation",
                "catastrophic_root_height",
                "task_bad_orientation",
                "task_excessive_motion",
                "terrain_out_of_bounds",
                "stalled_progress",
            ]
        },
    )
    flat_orientation = RewTerm(
        func=mdp.task_phase_flat_orientation_l2,
        weight=-20.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=REFERENCE_BODY_NAME_EXPR)},
    )
    tilt_margin = RewTerm(
        func=mdp.task_phase_tilt_margin_l2,
        weight=-45.0,
        params={
            "threshold": math.radians(6.0),
            "asset_cfg": SceneEntityCfg("robot", body_names=REFERENCE_BODY_NAME_EXPR),
        },
    )
    angular_velocity = RewTerm(
        func=mdp.task_phase_ang_vel_xy_l2,
        weight=-1.5,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=REFERENCE_BODY_NAME_EXPR)},
    )
    vertical_velocity = RewTerm(
        func=mdp.task_phase_lin_vel_z_l2,
        weight=-0.8,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=REFERENCE_BODY_NAME_EXPR)},
    )
    lateral_velocity = RewTerm(
        func=mdp.task_phase_lateral_velocity_l2,
        weight=-0.4,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=REFERENCE_BODY_NAME_EXPR)},
    )
    forward_velocity = RewTerm(
        func=mdp.task_phase_forward_speed_abs_exp,
        weight=18.0,
        params={
            "target": FORWARD_VELOCITY_TARGET_MPS,
            "std": 0.08,
            "asset_cfg": SceneEntityCfg("robot", body_names=REFERENCE_BODY_NAME_EXPR),
        },
    )
    forward_speed_deficit = RewTerm(
        func=mdp.task_phase_forward_speed_deficit_signed_l1,
        weight=-3.0,
        params={
            "target": FORWARD_VELOCITY_TARGET_MPS,
            "asset_cfg": SceneEntityCfg("robot", body_names=REFERENCE_BODY_NAME_EXPR),
        },
    )
    backward_speed = RewTerm(
        func=mdp.task_phase_backward_speed_l1,
        weight=-8.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=REFERENCE_BODY_NAME_EXPR)},
    )
    suspension_centering = RewTerm(
        func=mdp.task_phase_joint_deviation_l1,
        weight=-0.12,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=SUSPENSION_JOINTS, preserve_order=True)},
    )
    suspension_limits = RewTerm(
        func=mdp.task_phase_joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=SUSPENSION_JOINTS, preserve_order=True)},
    )
    action_rate = RewTerm(func=mdp.task_phase_action_rate_l2, weight=-0.02)


@configclass
class StableTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    catastrophic_bad_orientation = DoneTerm(
        func=mdp.catastrophic_bad_orientation_robot_frame,
        params={
            "limit_angle": math.radians(35.0),
            "asset_cfg": SceneEntityCfg("robot", body_names=REFERENCE_BODY_NAME_EXPR),
        },
    )
    catastrophic_root_height = DoneTerm(
        func=mdp.catastrophic_articulation_root_height,
        params={
            "min_height": -0.30,
            "max_height": 1.50,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    task_bad_orientation = DoneTerm(
        func=mdp.task_phase_bad_orientation_robot_frame,
        params={
            "limit_angle": math.radians(10.0),
            "asset_cfg": SceneEntityCfg("robot", body_names=REFERENCE_BODY_NAME_EXPR),
        },
    )
    task_excessive_motion = DoneTerm(
        func=mdp.task_phase_excessive_motion,
        params={
            "linear_speed_limit": 1.6,
            "angular_speed_limit": 4.0,
            "asset_cfg": SceneEntityCfg("robot", body_names=REFERENCE_BODY_NAME_EXPR),
        },
    )
    terrain_out_of_bounds = DoneTerm(
        func=mdp.task_phase_terrain_out_of_bounds,
        params={
            "distance_limit": 3.6,
            "asset_cfg": SceneEntityCfg("robot", body_names=REFERENCE_BODY_NAME_EXPR),
        },
    )
    stalled_progress = DoneTerm(
        func=mdp.phase_aware_stalled_progress,
        params={
            "min_progress": 0.015,
            "min_steps": 200,
            "asset_cfg": SceneEntityCfg("robot", body_names=REFERENCE_BODY_NAME_EXPR),
        },
    )


@configclass
class StableAdaptiveCurriculumCfg:
    terrain_levels = CurrTerm(
        func=mdp.phase_aware_terrain_levels_progress,
        params={
            "progress_threshold": 0.08,
            "regression_threshold": 0.02,
            "plane_terrain_types": (0, 1),
            "rough_terrain_types": (2,),
        },
    )


@configclass
class FrbotLiftStableFlatEnvCfg(ManagerBasedRLEnvCfg):
    scene: StableFlatSceneCfg = StableFlatSceneCfg(num_envs=1024, env_spacing=8.0, replicate_physics=True)
    observations: StableObservationsCfg = StableObservationsCfg()
    actions: StableFlatActionsCfg = StableFlatActionsCfg()
    events: StableFlatEventCfg = StableFlatEventCfg()
    rewards: StableRewardsCfg = StableRewardsCfg()
    terminations: StableTerminationsCfg = StableTerminationsCfg()
    curriculum = None

    def __post_init__(self) -> None:
        self.decimation = 4
        self.episode_length_s = 14.0
        self.viewer.eye = (5.5, 0.0, 3.0)
        self.viewer.lookat = (0.0, 0.0, 0.6)
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physx.gpu_collision_stack_size = 2**28
        self.sim.physx.enable_external_forces_every_iteration = True
        self.sim.physx.solve_articulation_contact_last = True
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt


@configclass
class FrbotLiftStableAdaptiveEnvCfg(ManagerBasedRLEnvCfg):
    scene: StableAdaptiveSceneCfg = StableAdaptiveSceneCfg(num_envs=1024, env_spacing=8.0, replicate_physics=True)
    observations: StableObservationsCfg = StableObservationsCfg()
    actions: StableAdaptiveActionsCfg = StableAdaptiveActionsCfg()
    events: StableAdaptiveEventCfg = StableAdaptiveEventCfg()
    rewards: StableRewardsCfg = StableRewardsCfg()
    terminations: StableTerminationsCfg = StableTerminationsCfg()
    curriculum: StableAdaptiveCurriculumCfg = StableAdaptiveCurriculumCfg()

    def __post_init__(self) -> None:
        self.decimation = 4
        self.episode_length_s = 14.0
        self.viewer.eye = (5.5, 0.0, 3.0)
        self.viewer.lookat = (0.0, 0.0, 0.6)
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physx.gpu_collision_stack_size = 2**28
        self.sim.physx.enable_external_forces_every_iteration = True
        self.sim.physx.solve_articulation_contact_last = True
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt


@configclass
class FrbotLiftStablePlayEnvCfg(FrbotLiftStableAdaptiveEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 8
        self.scene.env_spacing = 8.0
        self.actions.drivetrain.debug_vis = True
        self.observations.policy.enable_corruption = False
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = False
            self.scene.terrain.terrain_generator.num_rows = 4
            self.scene.terrain.terrain_generator.num_cols = 4
        self.curriculum = None

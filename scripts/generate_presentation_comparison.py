"""Render presentation-only baseline vs stabilized comparison videos.

This script does not use RL checkpoints. It creates two scripted demos:

* Baseline: straight motion with only a tiny residual mast sway.
* Stabilized: perfectly level horizontal translation with zero visible sway.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Generate presentation-only comparison videos.")
parser.add_argument(
    "--mode",
    choices=("baseline_uncorrected", "scripted_stabilized"),
    required=True,
    help="Presentation mode to render.",
)
parser.add_argument("--task", type=str, default="Inventory-Leveling-StableFlat-v0", help="Isaac Lab task to launch.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of robots to spawn.")
parser.add_argument("--video_length", type=int, default=500, help="Number of environment steps to record.")
parser.add_argument("--speed_scale", type=float, default=10.0, help="Presentation speed multiplier.")
parser.add_argument("--fps", type=int, default=50, help="Video frames per second.")
parser.add_argument("--width", type=int, default=1280, help="Render width passed to Isaac Sim.")
parser.add_argument("--height", type=int, default=720, help="Render height passed to Isaac Sim.")
parser.add_argument("--hide_label", action="store_true", default=False, help="Render without the corner label.")
parser.add_argument(
    "--output_dir",
    type=str,
    default=None,
    help="Optional output directory. Defaults to results/2026-03-25/proposal_demo under the project root.",
)
parser.add_argument("--seed", type=int, default=42, help="Seed used for deterministic layout.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric for debugging.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import imageio.v2 as imageio
import isaaclab.sim as sim_utils
import isaaclab.sim.utils.prims as prim_utils
import numpy as np
import torch
import carb
import omni.kit.app
import omni.kit.commands
import omni.usd
from PIL import Image, ImageDraw, ImageFont
from pxr import UsdGeom

import Frbot_Lift.tasks  # noqa: F401
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import SPHERE_MARKER_CFG
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_from_euler_xyz, quat_mul
from isaaclab_tasks.utils import parse_env_cfg

from Frbot_Lift.assets.frbot_lift_asset import (
    MAST_WHEEL_VELOCITY_TARGETS_RAD_S,
    STABLE_BASE_INIT_HEIGHT_M,
    SUSPENSION_JOINTS,
    SUSPENSION_MIDPOINT_RAD,
    WHEEL_JOINTS,
)
from Frbot_Lift.tasks.manager_based.frbot_lift.mdp.robot_frame import robot_to_body_frame


OUTPUT_FILENAMES = {
    "baseline_uncorrected": "proposal_baseline.mp4",
    "scripted_stabilized": "proposal_stabilized.mp4",
}

MODE_METADATA = {
    "baseline_uncorrected": {
        "label": "Baseline",
        "roll_amp_rad": math.radians(4.6),
        "roll_omega": 0.22,
        "pitch_amp_rad": 0.0,
        "pitch_omega": 0.22,
        "mast_sway_m": 0.34,
        "mast_sway_omega": 0.22,
        "arrow_roll_amp_rad": 0.0,
        "arrow_pitch_amp_rad": 0.0,
        "arrow_height_offset_m": 0.45,
        "arrow_stack_gap_m": 0.00,
    },
    "scripted_stabilized": {
        "label": "Stabilized",
        "roll_amp_rad": 0.0,
        "roll_omega": 0.28,
        "pitch_amp_rad": 0.0,
        "pitch_omega": 0.22,
        "mast_sway_m": 0.0,
        "mast_sway_omega": 0.28,
        "arrow_roll_amp_rad": 0.0,
        "arrow_pitch_amp_rad": 0.0,
        "arrow_height_offset_m": 0.45,
        "arrow_stack_gap_m": 0.00,
    },
}

GROUND_COLOR = (0.95, 0.95, 0.94)
DOME_LIGHT_COLOR = (0.98, 0.98, 0.98)
TRANSLATION_SPEED_MPS_AT_10X = 0.78
WHEEL_SPIN_RAD_S_AT_10X = 18.0
FORWARD_LOCAL_AXIS = (0.0, 0.0, -1.0)
ARROW_SCALE = (1.0, 1.0, 1.0)
CAMERA_TARGET_HEIGHT_M = 5.6
PRESENTATION_CAMERA_EYE = (52.0, -33.0, 10.0)
CAMERA_BACKOFF_FACTOR = 0.46
CAMERA_SIDE_FACTOR = 0.0
CAMERA_HEIGHT_FACTOR = 0.24
CAMERA_MIN_BACKOFF_M = 16.0
CAMERA_MIN_HEIGHT_M = 8.5
PRE_CAPTURE_WARMUP_FRAMES = 12

TRACE_LOG_PATH: Path | None = None


def resolve_project_root() -> Path:
    env_root = os.environ.get("FRBOT_LIFT_ROOT")
    if env_root:
        return Path(env_root)
    alias_root = Path("R:/Frbot_Lift")
    if alias_root.exists():
        return alias_root
    return Path(__file__).resolve().parents[1]


def resolve_output_dir(project_root: Path) -> Path:
    if args_cli.output_dir:
        return Path(args_cli.output_dir)
    return project_root / "results" / "2026-03-25" / "proposal_demo"


def trace(message: str):
    print(message, flush=True)
    if TRACE_LOG_PATH is not None:
        TRACE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with TRACE_LOG_PATH.open("a", encoding="utf-8") as trace_file:
            trace_file.write(message + "\n")


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    font_candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/segoeuib.ttf"),
    ]
    for font_path in font_candidates:
        if font_path.exists():
            return ImageFont.truetype(str(font_path), size=size)
    return ImageFont.load_default()


def add_label(frame: np.ndarray, label: str) -> np.ndarray:
    if frame.shape[-1] == 4:
        frame = frame[:, :, :3]
    image = Image.fromarray(frame.astype(np.uint8)).convert("RGBA")
    draw = ImageDraw.Draw(image, "RGBA")
    font = load_font(46)
    box = (28, 24, 360, 104)
    draw.rounded_rectangle(box, radius=18, fill=(10, 10, 10, 170))
    draw.text((52, 39), label, font=font, fill=(255, 255, 255, 255))
    return np.asarray(image.convert("RGB"))


def resize_frame(frame: np.ndarray) -> np.ndarray:
    target_size = (args_cli.width, args_cli.height)
    source_size = (frame.shape[1], frame.shape[0])
    if source_size == target_size:
        return frame
    image = Image.fromarray(frame.astype(np.uint8)).convert("RGB")
    image = image.resize(target_size, Image.Resampling.LANCZOS)
    return np.asarray(image)


def restyle_floor(frame: np.ndarray) -> np.ndarray:
    rgb = frame[:, :, :3].astype(np.float32).copy()
    floor_mask = (
        (rgb[:, :, 2] > rgb[:, :, 1] + 8.0)
        & (rgb[:, :, 2] > rgb[:, :, 0] + 16.0)
        & (rgb.mean(axis=-1) > 110.0)
    )
    off_white = np.array([240.0, 240.0, 238.0], dtype=np.float32)
    rgb[floor_mask] = rgb[floor_mask] * 0.08 + off_white * 0.92
    return rgb.clip(0, 255).astype(np.uint8)


def configure_env_cfg():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.seed = args_cli.seed
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.scene.env_spacing = 9.5
    env_cfg.scene.terrain.terrain_type = "plane"
    env_cfg.scene.terrain.terrain_generator = None
    env_cfg.scene.terrain.max_init_terrain_level = None
    env_cfg.scene.terrain.debug_vis = False
    env_cfg.scene.terrain.visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=GROUND_COLOR)
    if hasattr(env_cfg.scene, "dome_light") and env_cfg.scene.dome_light is not None:
        env_cfg.scene.dome_light = None
    env_cfg.episode_length_s = max(env_cfg.episode_length_s, 1000.0)
    if hasattr(env_cfg.terminations, "catastrophic_bad_orientation"):
        env_cfg.terminations.catastrophic_bad_orientation.params["limit_angle"] = math.radians(89.0)
    if hasattr(env_cfg.terminations, "catastrophic_root_height"):
        env_cfg.terminations.catastrophic_root_height.params["min_height"] = -10.0
        env_cfg.terminations.catastrophic_root_height.params["max_height"] = 10.0
    if hasattr(env_cfg.terminations, "task_bad_orientation"):
        env_cfg.terminations.task_bad_orientation.params["limit_angle"] = math.radians(89.0)
    if hasattr(env_cfg.terminations, "task_excessive_motion"):
        env_cfg.terminations.task_excessive_motion.params["linear_speed_limit"] = 100.0
        env_cfg.terminations.task_excessive_motion.params["angular_speed_limit"] = 100.0
    if hasattr(env_cfg.terminations, "terrain_out_of_bounds"):
        env_cfg.terminations.terrain_out_of_bounds.params["distance_limit"] = 1_000_000.0
    if hasattr(env_cfg.terminations, "stalled_progress"):
        env_cfg.terminations.stalled_progress.params["min_progress"] = -1.0
        env_cfg.terminations.stalled_progress.params["min_steps"] = 1_000_000
    return env_cfg


def compute_focus_origins(env_origins: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    num_envs = env_origins.shape[0]
    grid_size = int(round(math.sqrt(num_envs)))
    if grid_size * grid_size == num_envs and grid_size >= 5:
        origins_grid = env_origins.reshape(grid_size, grid_size, 3)
        center_start = max(grid_size // 2 - 2, 0)
        center_end = min(center_start + 4, grid_size)
        center_origins = origins_grid[center_start:center_end, center_start:center_end].reshape(-1, 3)
    else:
        center_origins = env_origins
    center = center_origins.mean(dim=0)
    span = center_origins.max(dim=0).values - center_origins.min(dim=0).values
    return center, span


def configure_camera(base_env, translation_distance_m: float):
    env_origins = base_env.scene.env_origins.detach().clone()
    center, _ = compute_focus_origins(env_origins)
    robot = base_env.scene["robot"]
    base_quat = robot.data.root_link_pose_w[:, 3:7].clone()
    forward_axis = torch.tensor(FORWARD_LOCAL_AXIS, device=base_env.device, dtype=torch.float32).repeat(
        base_env.num_envs, 1
    )
    forward_world = quat_apply(base_quat, forward_axis)
    forward_mean = forward_world.mean(dim=0)
    forward_xy = torch.tensor([forward_mean[0], forward_mean[1], 0.0], device=base_env.device)
    forward_xy = forward_xy / torch.linalg.vector_norm(forward_xy).clip(min=1.0e-6)
    target_vec = forward_xy * (translation_distance_m * 0.55)
    eye = PRESENTATION_CAMERA_EYE
    target = (
        center[0].item() + target_vec[0].item(),
        center[1].item() + target_vec[1].item(),
        CAMERA_TARGET_HEIGHT_M,
    )
    base_env.sim.set_camera_view(eye=eye, target=target)


def create_arrow_visualizers() -> tuple[VisualizationMarkers, VisualizationMarkers]:
    target_cfg = SPHERE_MARKER_CFG.replace(prim_path="/Visuals/Proposal/target_markers")
    current_cfg = SPHERE_MARKER_CFG.replace(prim_path="/Visuals/Proposal/current_markers")
    target_cfg.markers["sphere"].radius = 0.16
    current_cfg.markers["sphere"].radius = 0.16
    target_cfg.markers["sphere"].visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.12, 0.92, 0.20))
    current_cfg.markers["sphere"].visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.16, 0.36, 0.96))
    return VisualizationMarkers(target_cfg), VisualizationMarkers(current_cfg)


def spawn_presentation_floor(base_env):
    stage = prim_utils.get_current_stage()
    hidden_prefix = "/World/ground"
    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        if prim_path == hidden_prefix or prim_path.startswith(hidden_prefix + "/"):
            imageable = UsdGeom.Imageable(prim)
            if imageable:
                imageable.MakeInvisible()
    env_origins = base_env.scene.env_origins.detach().clone()
    center = env_origins.mean(dim=0)
    span = env_origins.max(dim=0).values - env_origins.min(dim=0).values
    floor_extent = max(120.0, float(max(span[0].item(), span[1].item()) + 40.0))
    floor_cfg = sim_utils.CuboidCfg(
        size=(floor_extent, floor_extent, 0.04),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.96, 0.96, 0.95)),
    )
    floor_cfg.func(
        "/World/PresentationFloor",
        floor_cfg,
        translation=(center[0].item(), center[1].item(), -0.02),
    )


def configure_default_light_rig():
    """Force the viewport lighting mode to Camera Rigs > Default."""

    settings = carb.settings.get_settings()
    settings.set("/exts/omni.kit.viewport.menubar.lighting/defaultRig", "Default")
    settings.set("/persistent/exts/omni.kit.viewport.menubar.lighting/autoLightRig/enabled", True)
    settings.set("/exts/omni.kit.viewport.menubar.lighting/autoLightRig/enableWithoutMenu", True)
    settings.set("/rtx/useViewLightingMode", False)
    try:
        ext_manager = omni.kit.app.get_app().get_extension_manager()
        ext_manager.set_extension_enabled_immediate("omni.kit.viewport.menubar.lighting", True)
        result = omni.kit.commands.execute(
            "SetLightingMenuModeCommand",
            lighting_mode="Default",
            usd_context_name="",
        )
        trace(f"[proposal] forced lighting rig: {result}")
    except Exception as exc:
        trace(f"[proposal] command path failed, trying direct module fallback: {exc}")
        try:
            from omni.kit.viewport.menubar.lighting.actions import _set_lighting_mode

            result = _set_lighting_mode("Default", usd_context=omni.usd.get_context(""))
            trace(f"[proposal] forced lighting rig via fallback: {result}")
        except Exception as fallback_exc:
            trace(f"[proposal] failed to force lighting rig: {fallback_exc}")


def make_zero_actions(env, device: torch.device) -> torch.Tensor:
    return torch.zeros(env.action_space.shape, device=device)


def compute_mast_offset_body(robot, base_root_pose: torch.Tensor) -> torch.Tensor:
    """Estimate the top marker anchor from the actual highest articulation body in the spawned asset."""

    body_pos_w = robot.data.body_pos_w
    candidate_ids = [
        body_id
        for body_id, body_name in enumerate(robot.body_names)
        if body_name != "Group" and "Wheel" not in body_name and "suspenstion" not in body_name
    ]
    if not candidate_ids:
        candidate_ids = list(range(body_pos_w.shape[1]))
    candidate_ids_t = torch.tensor(candidate_ids, device=body_pos_w.device, dtype=torch.long)
    env_ids = torch.arange(body_pos_w.shape[0], device=body_pos_w.device, dtype=torch.long)
    candidate_z = body_pos_w[:, candidate_ids_t, 2]
    top_slots = torch.argmax(candidate_z, dim=1)
    top_ids = candidate_ids_t[top_slots]
    top_pos_w = body_pos_w[env_ids, top_ids]
    return quat_apply_inverse(base_root_pose[:, 3:7], top_pos_w - base_root_pose[:, :3])


def quat_from_robot_roll_pitch(roll_robot: torch.Tensor, pitch_robot: torch.Tensor) -> torch.Tensor:
    """Convert robot-frame roll/pitch into the imported raw body frame.

    Robot frame is x-forward, y-lateral, z-up. The imported asset uses a corrected Y-up body frame where:
    body x -> robot lateral, body y -> robot up, body z -> robot rear.
    """

    zero = torch.zeros_like(roll_robot)
    # Rotation about robot lateral maps to body-x. Rotation about robot forward maps to negative body-z.
    body_euler_x = pitch_robot
    body_euler_z = -roll_robot
    return quat_from_euler_xyz(body_euler_x, zero, body_euler_z)


def make_env_variation(num_envs: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create deterministic per-env sway variations so the group does not move identically."""

    env_ids = torch.arange(num_envs, device=device, dtype=torch.float32)
    raw_a = torch.sin(env_ids * 12.9898 + 0.37) * 43758.5453
    raw_b = torch.sin(env_ids * 7.731 + 1.93) * 24634.6345
    raw_c = torch.sin(env_ids * 4.217 + 2.71) * 15317.1742
    frac_a = raw_a - torch.floor(raw_a)
    frac_b = raw_b - torch.floor(raw_b)
    frac_c = raw_c - torch.floor(raw_c)
    amp_scale = 0.72 + 0.58 * frac_a
    freq_scale = 0.82 + 0.42 * frac_b
    bias = (frac_c - 0.5) * 0.22
    return amp_scale, freq_scale, bias


def presentation_sway_signal(
    t: float,
    phases: torch.Tensor,
    amp_scale: torch.Tensor,
    freq_scale: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """Create a slow, irregular left-right sway signal for the presentation baseline."""

    time = torch.full_like(phases, t)
    slow = 0.58 * torch.sin((0.58 * freq_scale) * time + phases)
    medium = 0.30 * torch.sin((1.11 * (0.94 + 0.12 * freq_scale)) * time + 0.7 + phases * 1.6)
    weave = 0.18 * torch.sin((1.87 * (0.90 + 0.18 * freq_scale)) * time + 1.1 + phases * 2.2)
    envelope = 0.82 + 0.18 * torch.sin((0.21 * (0.95 + 0.10 * freq_scale)) * time + phases * 0.35)
    return (((slow + medium + weave) * envelope) * amp_scale + bias).clip(min=-1.35, max=1.35)


def compute_root_motion(
    mode: str,
    base_root_pose: torch.Tensor,
    base_quat: torch.Tensor,
    frame_idx: int,
    device: torch.device,
    amp_scale: torch.Tensor,
    freq_scale: torch.Tensor,
    bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    metadata = MODE_METADATA[mode]
    num_envs = base_root_pose.shape[0]
    t = frame_idx / float(args_cli.fps)
    speed_mps = TRANSLATION_SPEED_MPS_AT_10X * (args_cli.speed_scale / 10.0)
    phases = torch.arange(num_envs, device=device, dtype=torch.float32) * 0.13
    sway_signal = presentation_sway_signal(t, phases, amp_scale, freq_scale, bias)
    roll = metadata["roll_amp_rad"] * sway_signal
    pitch = metadata["pitch_amp_rad"] * torch.sin(metadata["pitch_omega"] * t + 0.4 + phases * 0.7)
    if mode == "scripted_stabilized":
        roll = torch.zeros_like(roll)
        pitch = torch.zeros_like(pitch)
    tilt_quat = quat_from_robot_roll_pitch(roll, pitch)
    forward_axis = torch.tensor(FORWARD_LOCAL_AXIS, device=device, dtype=torch.float32).repeat(num_envs, 1)
    forward_world = quat_apply(base_quat, forward_axis)
    translation = forward_world * (speed_mps * t)

    root_pose = base_root_pose.clone()
    root_pose[:, :3] = base_root_pose[:, :3] + translation
    root_pose[:, 2] = STABLE_BASE_INIT_HEIGHT_M
    root_pose[:, 3:7] = quat_mul(base_root_pose[:, 3:7], tilt_quat)

    root_velocity = torch.zeros((num_envs, 6), device=device)
    root_velocity[:, :3] = forward_world * speed_mps
    return root_pose, root_velocity


def compute_mast_tip_positions(
    mode: str,
    root_pose: torch.Tensor,
    base_quat: torch.Tensor,
    frame_idx: int,
    device: torch.device,
    mast_offset_body: torch.Tensor,
    amp_scale: torch.Tensor,
    freq_scale: torch.Tensor,
    bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    metadata = MODE_METADATA[mode]
    num_envs = root_pose.shape[0]
    t = frame_idx / float(args_cli.fps)
    phases = torch.arange(num_envs, device=device, dtype=torch.float32) * 0.13

    current_base_quat = root_pose[:, 3:7]
    up_axis_body = robot_to_body_frame(
        torch.tensor((0.0, 0.0, 1.0), device=device, dtype=torch.float32).repeat(num_envs, 1)
    )
    target_tip = root_pose[:, :3] + quat_apply(base_quat, mast_offset_body)
    target_tip = target_tip + quat_apply(base_quat, up_axis_body) * 0.16
    current_tip = root_pose[:, :3] + quat_apply(current_base_quat, mast_offset_body)
    current_tip = current_tip + quat_apply(current_base_quat, up_axis_body) * 0.16

    lateral_axis_body = robot_to_body_frame(
        torch.tensor((0.0, 1.0, 0.0), device=device, dtype=torch.float32).repeat(num_envs, 1)
    )
    lateral_world = quat_apply(base_quat, lateral_axis_body)
    lateral_world[:, 2] = 0.0
    lateral_world = lateral_world / torch.linalg.vector_norm(lateral_world, dim=-1, keepdim=True).clip(min=1.0e-6)
    mast_sway = metadata["mast_sway_m"] * presentation_sway_signal(t, phases, amp_scale, freq_scale, bias)
    current_tip = current_tip + lateral_world * mast_sway.unsqueeze(-1)
    return target_tip, current_tip, lateral_world


def compute_joint_state(
    mode: str,
    base_joint_pos: torch.Tensor,
    wheel_joint_ids: list[int],
    suspension_joint_ids: list[int],
    wheel_signs: torch.Tensor,
    frame_idx: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    joint_pos = base_joint_pos.clone()
    joint_vel = torch.zeros_like(joint_pos)
    t = frame_idx / float(args_cli.fps)
    wheel_speed_rad_s = WHEEL_SPIN_RAD_S_AT_10X * (args_cli.speed_scale / 10.0)
    wheel_angles = wheel_signs * wheel_speed_rad_s * t
    joint_pos[:, wheel_joint_ids] = base_joint_pos[:, wheel_joint_ids] + wheel_angles.unsqueeze(0)
    joint_vel[:, wheel_joint_ids] = wheel_signs.unsqueeze(0) * wheel_speed_rad_s
    joint_pos[:, suspension_joint_ids] = SUSPENSION_MIDPOINT_RAD
    if mode == "scripted_stabilized":
        micro_motion = math.radians(0.02) * math.sin(1.6 * t)
        joint_pos[:, suspension_joint_ids] = SUSPENSION_MIDPOINT_RAD + micro_motion
    return joint_pos, joint_vel


def compute_arrow_pose(
    mode: str,
    root_pose: torch.Tensor,
    base_quat: torch.Tensor,
    frame_idx: int,
    device: torch.device,
    amp_scale: torch.Tensor,
    freq_scale: torch.Tensor,
    bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    metadata = MODE_METADATA[mode]
    num_envs = root_pose.shape[0]
    t = frame_idx / float(args_cli.fps)
    phases = torch.arange(num_envs, device=device, dtype=torch.float32) * 0.13
    del amp_scale, freq_scale, bias

    target_arrow_pos = root_pose[:, :3].clone()
    target_arrow_pos[:, 2] += metadata["arrow_height_offset_m"] + metadata["arrow_stack_gap_m"]
    current_arrow_pos = root_pose[:, :3].clone()
    current_arrow_pos[:, 2] += metadata["arrow_height_offset_m"]

    forward_axis = torch.tensor(FORWARD_LOCAL_AXIS, device=device, dtype=torch.float32).repeat(num_envs, 1)
    forward_world = quat_apply(base_quat, forward_axis)
    heading = torch.atan2(forward_world[:, 1], forward_world[:, 0])
    target_arrow_quat = quat_from_euler_xyz(
        torch.zeros_like(heading), torch.zeros_like(heading), heading
    )

    arrow_roll = metadata["arrow_roll_amp_rad"] * torch.sin(1.15 * t + phases)
    arrow_pitch = metadata["arrow_pitch_amp_rad"] * torch.sin(0.95 * t + 0.3 + phases * 0.5)
    if mode == "scripted_stabilized":
        arrow_roll = torch.zeros_like(arrow_roll)
        arrow_pitch = torch.zeros_like(arrow_pitch)
    current_arrow_delta = quat_from_robot_roll_pitch(arrow_roll, arrow_pitch)
    current_arrow_quat = quat_mul(target_arrow_quat, current_arrow_delta)
    return target_arrow_pos, target_arrow_quat, current_arrow_pos, current_arrow_quat


def make_arrow_scales(num_envs: int, device: torch.device) -> torch.Tensor:
    return torch.tensor(ARROW_SCALE, device=device, dtype=torch.float32).repeat(num_envs, 1)


def render_mode(output_dir: Path):
    env_cfg = configure_env_cfg()
    trace(f"[proposal] creating env for mode={args_cli.mode}")
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    base_env = env.unwrapped
    robot = base_env.scene["robot"]
    device = base_env.device
    zero_actions = make_zero_actions(env, device)

    trace("[proposal] resetting env")
    env.reset()
    with torch.inference_mode():
        for _ in range(5):
            env.step(zero_actions)
    trace("[proposal] env warmup complete")
    spawn_presentation_floor(base_env)
    configure_default_light_rig()

    base_root_pose = robot.data.root_link_pose_w.clone()
    base_joint_pos = robot.data.joint_pos.clone()
    base_quat = base_root_pose[:, 3:7].clone()
    mast_offset_body = compute_mast_offset_body(robot, base_root_pose)
    amp_scale, freq_scale, bias = make_env_variation(base_env.num_envs, device)

    wheel_joint_ids, wheel_joint_names = robot.find_joints(WHEEL_JOINTS, preserve_order=True)
    suspension_joint_ids, _ = robot.find_joints(SUSPENSION_JOINTS, preserve_order=True)
    wheel_signs = torch.tensor(
        [1.0 if MAST_WHEEL_VELOCITY_TARGETS_RAD_S[name] >= 0.0 else -1.0 for name in wheel_joint_names],
        device=device,
        dtype=torch.float32,
    )

    translation_distance_m = TRANSLATION_SPEED_MPS_AT_10X * (args_cli.speed_scale / 10.0) * (
        args_cli.video_length / float(args_cli.fps)
    )
    configure_camera(base_env, translation_distance_m)

    target_arrows, current_arrows = create_arrow_visualizers()
    arrow_scales = make_arrow_scales(base_env.num_envs, device)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / OUTPUT_FILENAMES[args_cli.mode]
    if output_path.exists():
        output_path.unlink()

    trace(f"[proposal] writing video to {output_path}")
    writer = imageio.get_writer(str(output_path), fps=args_cli.fps, codec="libx264", quality=8)
    try:
        with torch.inference_mode():
            trace("[proposal] pre-capture render warmup")
            for _ in range(PRE_CAPTURE_WARMUP_FRAMES):
                env.step(zero_actions)
                root_pose, root_velocity = compute_root_motion(
                    args_cli.mode, base_root_pose, base_quat, 0, device, amp_scale, freq_scale, bias
                )
                joint_pos, joint_vel = compute_joint_state(
                    args_cli.mode,
                    base_joint_pos,
                    wheel_joint_ids,
                    suspension_joint_ids,
                    wheel_signs,
                    0,
                    device,
                )
                target_pos, target_quat, current_pos, current_quat = compute_arrow_pose(
                    args_cli.mode, root_pose, base_quat, 0, device, amp_scale, freq_scale, bias
                )
                target_tip, current_tip, _ = compute_mast_tip_positions(
                    args_cli.mode, root_pose, base_quat, 0, device, mast_offset_body, amp_scale, freq_scale, bias
                )
                robot.write_root_pose_to_sim(root_pose)
                robot.write_root_velocity_to_sim(root_velocity)
                robot.write_joint_state_to_sim(joint_pos, joint_vel)
                target_arrows.visualize(translations=target_tip, orientations=target_quat, scales=arrow_scales)
                current_arrows.visualize(translations=current_tip, orientations=current_quat, scales=arrow_scales)
                base_env.sim.render()
                env.render()
            for frame_idx in range(args_cli.video_length):
                if not simulation_app.is_running():
                    break
                if frame_idx % 10 == 0:
                    trace(f"[proposal] frame={frame_idx}")
                env.step(zero_actions)

                root_pose, root_velocity = compute_root_motion(
                    args_cli.mode, base_root_pose, base_quat, frame_idx, device, amp_scale, freq_scale, bias
                )
                joint_pos, joint_vel = compute_joint_state(
                    args_cli.mode,
                    base_joint_pos,
                    wheel_joint_ids,
                    suspension_joint_ids,
                    wheel_signs,
                    frame_idx,
                    device,
                )
                target_pos, target_quat, current_pos, current_quat = compute_arrow_pose(
                    args_cli.mode, root_pose, base_quat, frame_idx, device, amp_scale, freq_scale, bias
                )
                target_tip, current_tip, _ = compute_mast_tip_positions(
                    args_cli.mode, root_pose, base_quat, frame_idx, device, mast_offset_body, amp_scale, freq_scale, bias
                )

                robot.write_root_pose_to_sim(root_pose)
                robot.write_root_velocity_to_sim(root_velocity)
                robot.write_joint_state_to_sim(joint_pos, joint_vel)

                target_arrows.visualize(
                    translations=target_tip,
                    orientations=target_quat,
                    scales=arrow_scales,
                )
                current_arrows.visualize(
                    translations=current_tip,
                    orientations=current_quat,
                    scales=arrow_scales,
                )

                base_env.sim.render()
                frame = env.render()
                if frame is None:
                    continue
                processed_frame = restyle_floor(np.asarray(frame))
                if not args_cli.hide_label:
                    processed_frame = add_label(processed_frame, MODE_METADATA[args_cli.mode]["label"])
                processed_frame = resize_frame(processed_frame)
                writer.append_data(processed_frame)
    finally:
        trace("[proposal] closing writer and env")
        writer.close()
        env.close()


def main():
    global TRACE_LOG_PATH
    project_root = resolve_project_root()
    output_dir = resolve_output_dir(project_root)
    TRACE_LOG_PATH = project_root / "artifacts" / "launch_logs" / f"presentation_{args_cli.mode}.trace.log"
    if TRACE_LOG_PATH.exists():
        TRACE_LOG_PATH.unlink()
    render_mode(output_dir)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()

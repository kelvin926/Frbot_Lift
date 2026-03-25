# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import math
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--drive_speed_scale",
    type=float,
    default=1.0,
    help="Scale the scripted drivetrain wheel targets and forward command for play/eval.",
)
parser.add_argument(
    "--env_spacing",
    type=float,
    default=None,
    help="Override scene environment spacing for play/eval layouts.",
)
parser.add_argument(
    "--white_grid_floor",
    action="store_true",
    default=False,
    help="Tint the flat ground plane to a bright white grid for GUI play.",
)
parser.add_argument(
    "--warehouse_racks",
    action="store_true",
    default=False,
    help="Spawn lightweight warehouse rack proxies for play-only visualization.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

import isaaclab.sim as sim_utils
import isaaclab.sim.utils.prims as prim_utils
import isaaclab.terrains as terrain_gen
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import Frbot_Lift.tasks  # noqa: F401


TRAIN_TASK_OVERRIDES = {
    "Inventory-Leveling-Play-v0": "Inventory-Leveling-v0",
    "Inventory-Leveling-StablePlay-v0": "Inventory-Leveling-StableAdaptive-v0",
    "Inventory-Leveling-MastPlay-v0": "Inventory-Leveling-MastAdaptive-v0",
}


def resolve_train_task(task_name: str) -> str:
    return TRAIN_TASK_OVERRIDES.get(task_name, task_name.replace("-Play", "") if "-Play" in task_name else task_name)


def maybe_scale_play_drivetrain(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, speed_scale: float):
    """Scale scripted drivetrain targets for play-only runs without touching training configs."""

    if abs(speed_scale - 1.0) < 1e-6:
        return

    drivetrain = getattr(getattr(env_cfg, "actions", None), "drivetrain", None)
    if drivetrain is not None:
        wheel_velocity_targets = getattr(drivetrain, "wheel_velocity_targets", None)
        if wheel_velocity_targets:
            drivetrain.wheel_velocity_targets = {
                joint_name: float(target) * speed_scale for joint_name, target in wheel_velocity_targets.items()
            }

        target_velocity_robot = getattr(drivetrain, "target_velocity_robot", None)
        if target_velocity_robot is not None:
            drivetrain.target_velocity_robot = tuple(float(value) * speed_scale for value in target_velocity_robot)

    policy_obs = getattr(getattr(env_cfg, "observations", None), "policy", None)
    forward_command = getattr(policy_obs, "forward_command", None) if policy_obs is not None else None
    if forward_command is not None:
        forward_command.params["value"] = float(forward_command.params["value"]) * speed_scale

    rewards = getattr(env_cfg, "rewards", None)
    if rewards is not None:
        for reward_name in ("forward_velocity", "forward_speed_deficit"):
            reward_term = getattr(rewards, reward_name, None)
            if reward_term is not None and "target" in reward_term.params:
                reward_term.params["target"] = float(reward_term.params["target"]) * speed_scale

    terminations = getattr(env_cfg, "terminations", None)
    excessive_motion = getattr(terminations, "task_excessive_motion", None) if terminations is not None else None
    if excessive_motion is not None and "linear_speed_limit" in excessive_motion.params:
        excessive_motion.params["linear_speed_limit"] = max(
            float(excessive_motion.params["linear_speed_limit"]),
            float(excessive_motion.params["linear_speed_limit"]) * max(speed_scale, 1.0),
        )


def maybe_override_play_layout(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    env_spacing: float | None,
    white_grid_floor: bool,
):
    """Apply GUI-play-only scene layout overrides."""

    if env_spacing is not None:
        env_cfg.scene.env_spacing = env_spacing

    if white_grid_floor:
        terrain = getattr(env_cfg.scene, "terrain", None)
        if terrain is None:
            return

        spacing = float(env_cfg.scene.env_spacing)
        grid_cols = max(1, math.ceil(math.sqrt(int(env_cfg.scene.num_envs))))
        grid_rows = max(1, math.ceil(int(env_cfg.scene.num_envs) / grid_cols))
        plane_size = (max(64.0, (grid_cols + 1) * spacing), max(64.0, (grid_rows + 1) * spacing))

        terrain.terrain_type = "generator"
        terrain.use_terrain_origins = False
        terrain.env_spacing = spacing
        terrain.max_init_terrain_level = None
        terrain.terrain_generator = TerrainGeneratorCfg(
            curriculum=False,
            size=plane_size,
            border_width=0.0,
            num_rows=1,
            num_cols=1,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
            use_cache=False,
            sub_terrains={
                "plane": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0),
            },
        )
        terrain.visual_material = sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.96, 0.96, 0.96),
            roughness=0.95,
            metallic=0.0,
        )


def _spawn_visual_cuboid(
    prim_path: str,
    size: tuple[float, float, float],
    translation: tuple[float, float, float],
    color: tuple[float, float, float],
):
    cfg = sim_utils.CuboidCfg(
        size=size,
        collision_props=None,
        mass_props=None,
        rigid_props=None,
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=color,
            roughness=0.85,
            metallic=0.0,
        ),
    )
    cfg.func(prim_path, cfg, translation=translation)


def _spawn_proxy_rack(parent_prim_path: str, rack_name: str, center_xy: tuple[float, float], yaw_variant: int = 0):
    rack_width = 4.4 if yaw_variant == 0 else 1.6
    rack_depth = 1.4 if yaw_variant == 0 else 4.4
    rack_height = 4.8
    post_size = (0.12, 0.12, rack_height)
    beam_levels = (0.7, 2.1, 3.5)
    beam_thickness = 0.14

    cx, cy = center_xy
    base_path = f"{parent_prim_path}/{rack_name}"
    prim_utils.create_prim(base_path, prim_type="Xform", translation=(cx, cy, 0.0))

    post_dx = rack_width * 0.5 - 0.06
    post_dy = rack_depth * 0.5 - 0.06
    for post_index, (dx, dy) in enumerate(((-post_dx, -post_dy), (-post_dx, post_dy), (post_dx, -post_dy), (post_dx, post_dy))):
        _spawn_visual_cuboid(
            f"{base_path}/post_{post_index}",
            post_size,
            (cx + dx, cy + dy, rack_height * 0.5),
            (0.96, 0.52, 0.14),
        )

    for beam_index, z in enumerate(beam_levels):
        _spawn_visual_cuboid(
            f"{base_path}/beam_{beam_index}",
            (rack_width, rack_depth, beam_thickness),
            (cx, cy, z),
            (0.58, 0.60, 0.64),
        )

    box_specs = (
        ((-0.95, -0.22, 0.96), (0.85, 0.70, 0.46), (0.72, 0.49, 0.29)),
        ((0.10, 0.18, 0.98), (1.20, 0.82, 0.42), (0.77, 0.62, 0.39)),
        ((0.86, -0.12, 2.36), (0.82, 0.72, 0.40), (0.56, 0.66, 0.78)),
        ((-0.62, 0.20, 2.34), (0.74, 0.64, 0.38), (0.74, 0.39, 0.28)),
        ((0.00, -0.08, 3.74), (1.42, 0.90, 0.36), (0.86, 0.78, 0.62)),
    )
    for box_index, (offset, size, color) in enumerate(box_specs):
        ox, oy, oz = offset
        if yaw_variant == 1:
            ox, oy = oy, ox
            size = (size[1], size[0], size[2])
        _spawn_visual_cuboid(
            f"{base_path}/box_{box_index}",
            size,
            (cx + ox, cy + oy, oz),
            color,
        )


def maybe_spawn_warehouse_racks(env):
    decor_root = "/World/Decor/WarehouseBackdrop"
    if prim_utils.is_prim_path_valid(decor_root):
        prim_utils.delete_prim(decor_root)
    prim_utils.create_prim(decor_root, prim_type="Xform")

    env_origins = env.unwrapped.scene.env_origins.detach().cpu()
    if env_origins.numel() == 0:
        rack_centers = [(-18.0, -24.0), (-18.0, -8.0), (-18.0, 8.0), (-18.0, 24.0), (18.0, -24.0), (18.0, -8.0), (18.0, 8.0), (18.0, 24.0)]
    else:
        min_x = float(env_origins[:, 0].min().item())
        max_x = float(env_origins[:, 0].max().item())
        min_y = float(env_origins[:, 1].min().item())
        max_y = float(env_origins[:, 1].max().item())
        x_candidates = [min_x + 18.0, min_x + 42.0, max_x - 42.0, max_x - 18.0]
        y_candidates = [min_y + 16.0, min_y + 34.0, max_y - 34.0, max_y - 16.0]
        rack_centers = []
        for x in x_candidates:
            for y in y_candidates:
                rack_centers.append((x, y))

    for rack_index, center_xy in enumerate(rack_centers[:16]):
        _spawn_proxy_rack(decor_root, f"rack_{rack_index:02d}", center_xy, yaw_variant=rack_index % 2)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = resolve_train_task(task_name)

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    maybe_scale_play_drivetrain(env_cfg, args_cli.drive_speed_scale)
    maybe_override_play_layout(env_cfg, args_cli.env_spacing, args_cli.white_grid_floor)

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.warehouse_racks:
        maybe_spawn_warehouse_racks(env)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, dones, _ = env.step(actions)
            # reset recurrent states for episodes that have terminated
            policy_nn.reset(dones)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

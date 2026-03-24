import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Inspect reset pose for the Frbot Lift environment.")
parser.add_argument("--task", type=str, default="Inventory-Leveling-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--output", type=str, default="R:/Frbot_Lift/artifacts/inspect_reset_pose.json")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from pxr import UsdGeom

import Frbot_Lift.tasks  # noqa: F401


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    robot = env.unwrapped.scene["robot"]
    robot.update(env.unwrapped.physics_dt)
    stage = env.unwrapped.sim.stage
    bbox_cache = UsdGeom.BBoxCache(env.unwrapped.sim.current_time, ["default", "render", "proxy", "guide"])
    robot_prim = stage.GetPrimAtPath("/World/envs/env_0/Robot")
    robot_box = bbox_cache.ComputeWorldBound(robot_prim).ComputeAlignedBox()

    result = {
        "root_pos_w": robot.data.root_pos_w[0].cpu().tolist(),
        "root_quat_w": robot.data.root_quat_w[0].cpu().tolist(),
        "projected_gravity_b": robot.data.projected_gravity_b[0].cpu().tolist(),
        "robot_bbox_min_w": list(robot_box.GetMin()),
        "robot_bbox_max_w": list(robot_box.GetMax()),
        "body_names": robot.body_names,
        "body_pos_w": {name: robot.data.body_pos_w[0, idx].cpu().tolist() for idx, name in enumerate(robot.body_names)},
        "bodies": {},
    }

    interesting_names = [
        "Group",
        "NONE_9",
        "Mesh",
        "Mesh_001",
        "FL_tn__Drivemotorfixblock_tIBF",
        "FR_tn__Drivemotorfixblock_tIBF",
        "RL_tn__Drivemotorfixblock_tIBF",
        "RR_tn__Drivemotorfixblock_tIBF",
    ]
    for name in interesting_names:
        matches = [idx for idx, body_name in enumerate(robot.body_names) if body_name == name]
        for idx in matches:
            result["bodies"][f"{idx}:{name}"] = robot.data.body_pos_w[0, idx].cpu().tolist()

    output_path = Path(args_cli.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {output_path}", flush=True)

    env.close()


if __name__ == "__main__":
    with torch.inference_mode():
        main()
    simulation_app.close()

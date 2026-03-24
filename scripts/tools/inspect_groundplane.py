import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Inspect robot/world ground-plane prims in the Frbot Lift stage.")
parser.add_argument("--task", type=str, default="Inventory-Leveling-v0")
parser.add_argument("--num_envs", type=int, default=1)
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

    stage = env.unwrapped.sim.stage
    print("asset_path", env.unwrapped.scene["robot"].cfg.spawn.usd_path, flush=True)

    interesting_paths = []
    for prim in stage.Traverse():
        path_str = str(prim.GetPath())
        if "GroundPlane" in path_str or path_str == "/World/ground":
            interesting_paths.append(path_str)

    for path in interesting_paths:
        prim = stage.GetPrimAtPath(path)
        print("PATH", path, flush=True)
        print("  valid", prim.IsValid(), "type", prim.GetTypeName(), "active", prim.IsActive(), flush=True)
        if prim and prim.IsA(UsdGeom.Imageable):
            imageable = UsdGeom.Imageable(prim)
            print("  vis_attr", imageable.GetVisibilityAttr().Get(), flush=True)
            print("  computed_vis", imageable.ComputeVisibility(), flush=True)

    env.close()


if __name__ == "__main__":
    with torch.inference_mode():
        main()
    simulation_app.close()

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Dump articulation body names for a Frbot Lift task.")
parser.add_argument("--task", type=str, required=True, help="Task name.")
parser.add_argument("--artifact", type=str, required=True, help="Path to write the body-name list.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym

import Frbot_Lift.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1, use_fabric=not args_cli.disable_fabric)
    env = gym.make(args_cli.task, cfg=env_cfg)
    try:
        env.reset()
        robot = env.unwrapped.scene["robot"]
        lines = [f"{idx}: {name}" for idx, name in enumerate(robot.body_names)]
        Path(args_cli.artifact).write_text("\n".join(lines), encoding="utf-8")
        print("\n".join(lines))
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()

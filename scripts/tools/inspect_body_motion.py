from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Inspect body positions before/after a short rollout.")
parser.add_argument("--task", type=str, required=True, help="Task name.")
parser.add_argument("--artifact", type=str, required=True, help="Path to write JSON output.")
parser.add_argument("--steps", type=int, default=90, help="Number of zero-action steps.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import Frbot_Lift.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1, use_fabric=not args_cli.disable_fabric)
    env = gym.make(args_cli.task, cfg=env_cfg)
    try:
        env.reset()
        base_env = env.unwrapped
        robot = base_env.scene["robot"]
        body_names = robot.body_names
        start_body_pos = robot.data.body_pos_w[0].detach().cpu()
        actions = torch.zeros(env.action_space.shape, device=base_env.device)
        for _ in range(args_cli.steps):
            env.step(actions)
        end_body_pos = robot.data.body_pos_w[0].detach().cpu()
        payload = []
        for idx, name in enumerate(body_names):
            payload.append(
                {
                    "index": idx,
                    "name": name,
                    "start_pos": start_body_pos[idx].tolist(),
                    "end_pos": end_body_pos[idx].tolist(),
                    "delta_pos": (end_body_pos[idx] - start_body_pos[idx]).tolist(),
                }
            )
        Path(args_cli.artifact).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()

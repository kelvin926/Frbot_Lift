# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""List environments registered by the Frbot_Lift extension."""

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import gymnasium as gym
from prettytable import PrettyTable

import Frbot_Lift.tasks  # noqa: F401


def main():
    table = PrettyTable(["S. No.", "Task Name", "Entry Point", "Config"])
    table.title = "Available Environments in Frbot_Lift"
    table.align["Task Name"] = "l"
    table.align["Entry Point"] = "l"
    table.align["Config"] = "l"

    index = 0
    for task_spec in gym.registry.values():
        if task_spec.id.startswith("Inventory-Leveling"):
            table.add_row([index + 1, task_spec.id, task_spec.entry_point, task_spec.kwargs["env_cfg_entry_point"]])
            index += 1

    print(table)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()

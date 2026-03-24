# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents


gym.register(
    id="Inventory-Leveling-StableFlat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.frbot_lift_stable_env_cfg:FrbotLiftStableFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_flat_ppo_cfg:StableFlatPPORunnerCfg",
    },
)

gym.register(
    id="Inventory-Leveling-StableAdaptive-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.frbot_lift_stable_env_cfg:FrbotLiftStableAdaptiveEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_adaptive_ppo_cfg:StableAdaptivePPORunnerCfg",
    },
)

gym.register(
    id="Inventory-Leveling-StablePlay-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.frbot_lift_stable_env_cfg:FrbotLiftStablePlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_adaptive_ppo_cfg:StableAdaptivePPORunnerCfg",
    },
)

gym.register(
    id="Inventory-Leveling-WheelOnlyPlay-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.frbot_lift_stable_env_cfg:FrbotLiftWheelOnlyPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_adaptive_ppo_cfg:StableAdaptivePPORunnerCfg",
    },
)

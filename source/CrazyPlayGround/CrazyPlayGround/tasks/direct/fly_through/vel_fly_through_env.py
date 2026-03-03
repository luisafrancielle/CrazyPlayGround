# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Velocity-controlled fly-through environment (re-export from fly_through_env)."""

from .fly_through_env import FlyThroughEnv, VelFlyThroughEnvCfg

__all__ = ["FlyThroughEnv", "VelFlyThroughEnvCfg"]

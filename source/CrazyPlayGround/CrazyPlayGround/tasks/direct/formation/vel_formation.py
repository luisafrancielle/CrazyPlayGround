# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Velocity-controlled formation environment (re-export from formation_env)."""

from .formation_env import FormationEnv, VelFormationEnvCfg

__all__ = ["FormationEnv", "VelFormationEnvCfg"]

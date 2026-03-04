# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Velocity-controlled drone racing environment (re-export from drone_racing)."""

from .drone_racing import DroneRacingEnv, VelDroneRacingEnvCfg

__all__ = ["DroneRacingEnv", "VelDroneRacingEnvCfg"]

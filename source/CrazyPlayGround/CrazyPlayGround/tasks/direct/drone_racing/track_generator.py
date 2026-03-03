# Copyright (c) 2025, Kousheek Chakraborty
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This project uses the IsaacLab framework (https://github.com/isaac-sim/IsaacLab),
# which is licensed under the BSD-3-Clause License.

import pathlib as _pathlib

import torch
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils

# Absolute path to the gate USD asset, resolved relative to this file.
_GATE_USD = str(_pathlib.Path(__file__).resolve().parent / "assets" / "gate" / "gate.usd")


def spawn_track(track_config: dict) -> None:
    """Spawn all race gates as static kinematic prims under /World/Track/.

    Gates are placed once in the global world (not per-environment), so all
    parallel envs share a single physical track.

    Args:
        track_config: dict mapping gate_id (str) to a sub-dict with:
            - "pos": (x, y, z) world-frame position
            - "yaw": float yaw rotation around z-axis [rad], aligns the gate
                     opening with the intended travel direction
    """
    for gate_id, gate_cfg in track_config.items():
        prim_path = f"/World/Track/Gate_{gate_id}"
        cfg = sim_utils.UsdFileCfg(
            usd_path=_GATE_USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            scale=(1.0, 1.0, 1.0),
        )
        pos = tuple(gate_cfg["pos"])
        yaw = float(gate_cfg["yaw"])
        orient = math_utils.quat_from_euler_xyz(
            torch.tensor(0.0), torch.tensor(0.0), torch.tensor(yaw)
        ).tolist()
        cfg.func(prim_path, cfg, translation=pos, orientation=tuple(orient))

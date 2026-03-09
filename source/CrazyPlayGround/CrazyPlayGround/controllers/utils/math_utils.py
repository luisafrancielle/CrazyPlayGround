"""
Minimal quaternion and rotation utilities.

Replaces the ``isaaclab.utils.math`` dependency so that ``drone_control``
has no hard dependency on Isaac Lab.

Convention
----------
Quaternions are stored as **[w, x, y, z]** (scalar-first), matching the
Isaac Lab / Isaac Gym convention.  All operations are batched along the
leading dimensions.
"""

from __future__ import annotations

import torch


# ---------------------------------------------------------------------------
# Quaternion operations
# ---------------------------------------------------------------------------

def quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate *vec* by *quat* (active rotation).

    Uses the Rodrigues-like formula:
        v' = v + 2w (q_xyz × v) + 2 (q_xyz × (q_xyz × v))

    Args:
        quat: [..., 4]  in [w, x, y, z] order (need not be unit).
        vec:  [..., 3]

    Returns:
        Rotated vector of shape [..., 3].
    """
    w   = quat[..., 0:1]
    xyz = quat[..., 1:]
    t   = 2.0 * torch.linalg.cross(xyz, vec, dim=-1)
    return vec + w * t + torch.linalg.cross(xyz, t, dim=-1)


def quat_apply_inverse(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate *vec* by the **inverse** (conjugate) of *quat*.

    Equivalent to transforming from world frame to body frame when *quat*
    represents the body-to-world rotation.

    Args:
        quat: [..., 4]  in [w, x, y, z] order.
        vec:  [..., 3]

    Returns:
        Rotated vector of shape [..., 3].
    """
    q_inv = quat.clone()
    q_inv[..., 1:] = -q_inv[..., 1:]      # conjugate  →  q^{-1} for unit quat
    return quat_apply(q_inv, vec)


def euler_xyz_from_quat(
    quat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert unit quaternion **[w, x, y, z]** to intrinsic XYZ Euler angles.

    Returns three tensors *(roll, pitch, yaw)* in radians, each of the same
    leading shape as ``quat[..., 0]``.

    Args:
        quat: [..., 4]

    Returns:
        (roll, pitch, yaw) each of shape [...].
    """
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]

    roll  = torch.atan2(2.0 * (w * x + y * z),  1.0 - 2.0 * (x * x + y * y))
    pitch = torch.asin(torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw   = torch.atan2(2.0 * (w * z + x * y),  1.0 - 2.0 * (y * y + z * z))

    return roll, pitch, yaw


# ---------------------------------------------------------------------------
# Rotation matrix
# ---------------------------------------------------------------------------

def matrix_from_quat(quat: torch.Tensor) -> torch.Tensor:
    """Build a rotation matrix from unit quaternion **[w, x, y, z]**.

    Args:
        quat: [..., 4]

    Returns:
        Rotation matrix of shape [..., 3, 3].
    """
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    R = torch.stack([
        1.0 - 2.0 * (yy + zz),   2.0 * (xy - wz),          2.0 * (xz + wy),
              2.0 * (xy + wz),   1.0 - 2.0 * (xx + zz),     2.0 * (yz - wx),
              2.0 * (xz - wy),         2.0 * (yz + wx),   1.0 - 2.0 * (xx + yy),
    ], dim=-1).reshape(*quat.shape[:-1], 3, 3)
    return R


# ---------------------------------------------------------------------------
# Vector helpers
# ---------------------------------------------------------------------------

def normalize(vec: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Normalise *vec* along its last dimension."""
    return vec / (torch.linalg.norm(vec, dim=-1, keepdim=True).clamp_min(eps))


def expand_to(tensor: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Prepend size-1 dims until *tensor* has the same rank as *reference*, then expand."""
    while tensor.dim() < reference.dim():
        tensor = tensor.unsqueeze(0)
    return tensor.expand(reference.shape)

"""
Discrete-time PID controller, vectorized over N environments.

A single class covers both use cases in this package:

* **Single-axis** (used by ``cascade.py``): gains are scalars, state is ``[N]``.
* **Multi-axis** (used by ``cascade_pid.py``): gains are length-3 tensors,
  state is ``[N, 3]`` — one instance handles all three axes at once.

Key features
────────────
* Derivative filtering via Tustin bilinear low-pass (``tau > 0``) or raw
  backward difference (``tau = 0``).
* Derivative-on-measurement: pass ``measurement_dot`` to ``forward()`` to
  avoid derivative kick on setpoint steps.
* Feed-forward term (``kff``).
* Symmetric integral clamping (``integral_limit``), applied before the
  output saturation back-calculation anti-windup.
* Asymmetric output saturation (``limit_up``, ``limit_down``).
* Partial-environment reset via index tensor.

Backward compatibility
──────────────────────
The original positional signature ::

    PID_Vectorized(num_envs, device, kp=…, ki=…, kd=…,
                   limit_up=…, limit_down=…, tau=…)

is fully preserved.
"""

from __future__ import annotations

from typing import Optional, Union

import torch
import torch.nn as nn


class PID_Vectorized(nn.Module):
    """
    Discrete-time PID controller vectorized over N environments.

    Parameters
    ----------
    num_envs : int, optional
        Number of parallel environments.  When given, state buffers are
        pre-allocated as ``register_buffer`` entries (useful for
        ``nn.Module`` features such as ``.to(device)`` and
        ``state_dict()``).  When *None*, buffers are lazily allocated on
        the first ``forward()`` call (multi-axis mode).
    device : torch.device or str
        Target device.
    kp, ki, kd : float or array-like or Tensor
        Proportional, integral, derivative gains.  Scalars for single-axis;
        length-*axes* tensors for multi-axis (e.g. ``kp=[2., 2., 2.]``).
    kff : float or array-like or Tensor
        Feed-forward gain applied as ``kff * feedforward`` when a
        *feedforward* signal is passed to ``forward()``.  Default: 0.
    tau : float
        Derivative low-pass time constant [s].  Set to 0 to use a raw
        backward-difference derivative (no filtering).  Default: 0.01.
    limit_up, limit_down : float
        Symmetric or asymmetric output saturation bounds.
        Default: ±∞ (no saturation).
    integral_limit : float or array-like or Tensor, optional
        Per-axis symmetric cap applied to the integral *term* (``ki·∫e``)
        before the output saturation anti-windup.  *None* → no cap.
    """

    def __init__(
        self,
        num_envs: Optional[int] = None,
        device: Union[torch.device, str] = "cpu",
        *,
        kp: Union[float, list, torch.Tensor] = 1.0,
        ki: Union[float, list, torch.Tensor] = 0.0,
        kd: Union[float, list, torch.Tensor] = 0.0,
        kff: Union[float, list, torch.Tensor] = 0.0,
        tau: float = 0.01,
        limit_up: float = float("inf"),
        limit_down: float = float("-inf"),
        integral_limit: Optional[Union[float, list, torch.Tensor]] = None,
    ) -> None:
        super().__init__()

        dev = torch.device(device) if isinstance(device, str) else device
        self._device   = dev
        self._num_envs = num_envs

        def _g(v: Union[float, list, torch.Tensor]) -> torch.Tensor:
            return torch.as_tensor(v, dtype=torch.float32, device=dev)

        self.kp  = _g(kp)
        self.ki  = _g(ki)
        self.kd  = _g(kd)
        self.kff = _g(kff)
        self.tau = float(tau)

        self.limit_up   = float(limit_up)
        self.limit_down = float(limit_down)
        self.integral_limit: Optional[torch.Tensor] = (
            _g(integral_limit) if integral_limit is not None else None
        )

        # Pre-allocate buffers when num_envs is given (single-axis / nn.Module mode)
        if num_envs is not None:
            shape = (num_envs,) if self.kp.dim() == 0 else (num_envs, *self.kp.shape)
            zeros = torch.zeros(shape, dtype=torch.float32, device=dev)
            self.register_buffer("integrator",     zeros.clone())
            self.register_buffer("differentiator", zeros.clone())
            self.register_buffer("error_d1",       zeros.clone())
            self.register_buffer("u",              zeros.clone())
            self.register_buffer("u_unsat",        zeros.clone())
        else:
            # Lazy: will be created on first forward() call
            self.integrator:     Optional[torch.Tensor] = None
            self.differentiator: Optional[torch.Tensor] = None
            self.error_d1:       Optional[torch.Tensor] = None
            self.u:              Optional[torch.Tensor] = None
            self.u_unsat:        Optional[torch.Tensor] = None

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _ensure_state(self, error: torch.Tensor) -> None:
        """Lazily allocate state buffers matching ``error`` shape and device."""
        if self.integrator is None or self.integrator.shape != error.shape:
            z = torch.zeros_like(error)
            self.integrator     = z.clone()
            self.differentiator = z.clone()
            self.error_d1       = z.clone()
            self.u              = z.clone()
            self.u_unsat        = z.clone()

    def _ki_mask(self) -> torch.Tensor:
        """Boolean mask where ki is non-zero (enables integrator)."""
        return self.ki.abs() > 1e-12

    # ── Reset ────────────────────────────────────────────────────────────────

    def reset(self, env_ids: Optional[torch.Tensor] = None) -> None:
        """
        Reset integrator and derivative state.

        Parameters
        ----------
        env_ids : LongTensor, optional
            Indices of environments to reset.  *None* resets all.
        """
        if self.integrator is None:
            return

        if env_ids is None:
            if self._num_envs is not None:
                env_ids = torch.arange(self._num_envs, device=self._device)
            else:
                for buf in (self.integrator, self.differentiator,
                            self.error_d1, self.u, self.u_unsat):
                    buf.zero_()
                return

        for buf in (self.integrator, self.differentiator,
                    self.error_d1, self.u, self.u_unsat):
            buf[env_ids] = 0.0

    # ── Forward pass ─────────────────────────────────────────────────────────

    def forward(
        self,
        error: torch.Tensor,
        Ts: float,
        feedforward: Optional[torch.Tensor] = None,
        measurement_dot: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        One discrete PID step.

        Parameters
        ----------
        error : Tensor ``[N]`` or ``[N, axes]``
            Tracking error.
        Ts : float
            Timestep [s].
        feedforward : Tensor, optional
            Feed-forward signal; added as ``kff * feedforward``.
        measurement_dot : Tensor, optional
            Time-derivative of the *measurement* (not the error).  When
            provided the derivative term is computed as ``-kd·ṁeas`` to
            avoid a derivative kick on setpoint changes.

        Returns
        -------
        Tensor
            Saturated PID output, same shape as *error*.
        """
        self._ensure_state(error)

        ki_mask = self._ki_mask()  # scalar bool or [axes] bool

        # ── Integrator (trapezoidal rule) ────────────────────────────────────
        i_update = self.integrator + Ts / 2.0 * (error + self.error_d1)
        self.integrator = torch.where(ki_mask, i_update, self.integrator)

        # ── Integral term clamping (direct cap before output anti-windup) ────
        i_term = self.ki * self.integrator
        if self.integral_limit is not None:
            i_term = torch.clamp(i_term, -self.integral_limit, self.integral_limit)
            # Back-calculate integral state so it stays consistent
            ki_safe = torch.where(ki_mask, self.ki, torch.ones_like(self.ki))
            self.integrator = torch.where(ki_mask, i_term / ki_safe, self.integrator)

        # ── Derivative ───────────────────────────────────────────────────────
        if measurement_dot is not None:
            # Derivative on measurement: no kick on setpoint step
            d_term = -self.kd * measurement_dot
        elif self.tau > 0.0:
            # Tustin bilinear low-pass filter
            a = (2.0 * self.tau - Ts) / (2.0 * self.tau + Ts)
            b = 2.0 / (2.0 * self.tau + Ts)
            self.differentiator = a * self.differentiator + b * (error - self.error_d1)
            d_term = self.kd * self.differentiator
        else:
            # Raw backward difference (no filter)
            d_term = self.kd * (error - self.error_d1) / Ts

        self.error_d1 = error.clone()

        # ── PID sum ──────────────────────────────────────────────────────────
        u = self.kp * error + i_term + d_term
        if feedforward is not None:
            u = u + self.kff * feedforward
        self.u_unsat = u

        # ── Output saturation ────────────────────────────────────────────────
        u_clamped = torch.clamp(u, self.limit_down, self.limit_up)

        # ── Anti-windup: back-calculation from output saturation ─────────────
        sat_err = u_clamped - self.u_unsat
        ki_safe = torch.where(ki_mask, self.ki, torch.ones_like(self.ki))
        self.integrator = torch.where(
            ki_mask,
            self.integrator + sat_err / ki_safe,
            self.integrator,
        )

        self.u = u_clamped
        return u_clamped

    # ── Compatibility helper ─────────────────────────────────────────────────

    def apply_external_saturation_and_antiwindup(
        self, u_saturated: torch.Tensor
    ) -> None:
        """
        Apply anti-windup correction from an *external* saturation stage.

        Used by ``PosController_Vectorized`` when roll/pitch magnitude is
        clamped directionally after the velocity PID has already run.
        """
        ki_nz = (self.ki.item() != 0.0)
        if ki_nz:
            correction = (u_saturated - self.u_unsat) / self.ki
            self.integrator = self.integrator + correction
        self.u = u_saturated

"""Pure policy-action adapter for the NavAccSimPilot simulation trial."""
from __future__ import annotations

from typing import Dict, Sequence, Tuple
import math

import torch
from torch import Tensor

from .acceleration_to_rpyt import AccelerationToRPYT, AccelerationToRPYTConfig


class NavAccPolicyAdapter:
    """Map an unbounded Gaussian latent to acceleration and legacy API RPYT.

    The positive acceleration scale is an experimental simulation value.  This
    class owns no state, so training and evaluation can share the exact mapping.
    """

    def __init__(
        self,
        acceleration_scale: Sequence[float],
        converter: AccelerationToRPYT | None = None,
    ) -> None:
        if len(acceleration_scale) != 3:
            raise ValueError("acceleration_scale must contain xyz")
        self.acceleration_scale = tuple(float(value) for value in acceleration_scale)
        if not all(torch.isfinite(torch.tensor(self.acceleration_scale))):
            raise ValueError("acceleration_scale must be finite")
        if any(value <= 0.0 for value in self.acceleration_scale):
            raise ValueError("acceleration_scale entries must be positive")
        self.converter = converter or AccelerationToRPYT(
            AccelerationToRPYTConfig.cf21bl_simulation_reference()
        )
        self._validate_scale_envelope()

    def _validate_scale_envelope(self) -> None:
        """Prove every tanh-bounded command stays in the converter envelope."""
        sx, sy, sz = self.acceleration_scale
        cfg = self.converter.config
        min_vertical = cfg.gravity_m_s2 - sz
        if min_vertical <= 0.0:
            raise ValueError("acceleration scale permits non-positive specific force z")
        worst_xy = math.hypot(sx, sy)
        worst_tilt = math.atan2(worst_xy, min_vertical)
        if worst_tilt > math.radians(cfg.max_tilt_deg):
            raise ValueError("acceleration scale exceeds configured maximum tilt")

        min_force = cfg.mass_kg * min_vertical
        max_force = cfg.mass_kg * math.hypot(worst_xy, cfg.gravity_m_s2 + sz)

        def raw_pwm(total_force: float) -> float:
            per_motor = total_force / cfg.motors
            normalized = (
                -cfg.pwm_to_thrust_b_N
                + math.sqrt(
                    cfg.pwm_to_thrust_b_N ** 2
                    + 4.0 * cfg.pwm_to_thrust_a_N * per_motor
                )
            ) / (2.0 * cfg.pwm_to_thrust_a_N)
            return normalized * 65535.0

        if raw_pwm(min_force) < cfg.commander_thrust_min:
            raise ValueError("acceleration scale falls below commander thrust minimum")
        if raw_pwm(max_force) > cfg.commander_thrust_max:
            raise ValueError("acceleration scale exceeds commander thrust maximum")

    def nominal_acceleration(self, z_policy: Tensor) -> Tensor:
        """Map the retained raw latent to the nominal world acceleration."""
        if z_policy.ndim != 3 or z_policy.shape[-2:] != (1, 3):
            raise ValueError(
                f"z_policy must have shape [B,1,3], got {tuple(z_policy.shape)}"
            )
        if not z_policy.is_floating_point():
            raise TypeError("z_policy must be floating point")
        if not torch.isfinite(z_policy).all():
            raise ValueError("z_policy contains non-finite values")
        scale = z_policy.new_tensor(self.acceleration_scale)
        return scale * torch.tanh(z_policy)

    def __call__(
        self,
        z_policy: Tensor,
        psi_hold_W: Tensor,
    ) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
        if z_policy.ndim != 3 or z_policy.shape[-2:] != (1, 3):
            raise ValueError(
                f"z_policy must have shape [B,1,3], got {tuple(z_policy.shape)}"
            )
        if not z_policy.is_floating_point():
            raise TypeError("z_policy must be floating point")
        if psi_hold_W.ndim == 2 and psi_hold_W.shape[-1] == 1:
            psi_hold_W = psi_hold_W.unsqueeze(-1)
        if psi_hold_W.ndim != 3 or psi_hold_W.shape[-2:] != (1, 1):
            raise ValueError("psi_hold_W must have shape [B,1] or [B,1,1]")
        if psi_hold_W.shape[0] != z_policy.shape[0]:
            raise ValueError("z_policy and psi_hold_W batch sizes differ")
        if not torch.isfinite(z_policy).all():
            raise ValueError("z_policy contains non-finite values")
        if not torch.isfinite(psi_hold_W).all():
            raise ValueError("psi_hold_W contains non-finite values")

        a_nom = self.nominal_acceleration(z_policy)
        rpyt_api, diagnostics = self.converter(a_nom, psi_hold_W)
        if not diagnostics["valid"].all():
            raise ValueError("acceleration-to-RPYT rejected a policy command")
        if not torch.isfinite(rpyt_api).all():
            raise ValueError("acceleration-to-RPYT produced non-finite RPYT")
        return a_nom, rpyt_api, diagnostics

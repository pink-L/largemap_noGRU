"""Pure Torch conversion from world net acceleration to Crazyflie API RPYT.

The default factory is an exploratory simulation reference.  Its thrust curve,
mass, signs, and resulting command range are UNVALIDATED on a target vehicle.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Tuple

import torch
from torch import Tensor


FIRMWARE_REVISION = "2025.02"
FIRMWARE_COMMIT = "16c5630707e92830dcd16455a57b5560e5838735"
FIRMWARE_PLATFORM = "cf21bl"
CALIBRATION_STATUS = "UNVALIDATED"


def _wrap_radians(angle: Tensor) -> Tensor:
    """Match the firmware's closed [-pi, pi] wrapping at the endpoints."""
    wrapped = torch.remainder(angle + math.pi, 2.0 * math.pi) - math.pi
    return torch.where((wrapped == -math.pi) & (angle > 0), -wrapped, wrapped)


@dataclass(frozen=True)
class AccelerationToRPYTConfig:
    """Explicit parameters for the acceleration-to-RPYT boundary."""

    mass_kg: float
    gravity_m_s2: float
    max_tilt_deg: float
    min_force_norm_N: float
    motors: int
    pwm_to_thrust_a_N: float
    pwm_to_thrust_b_N: float
    commander_thrust_min: int
    commander_thrust_max: int
    firmware_revision: str
    firmware_commit: str
    firmware_platform: str
    calibration_status: str
    # Explicit opt-in escape for experiments whose command validity contract
    # checks finite/heading/upward-force/thrust but evaluates tilt post-flight.
    # Historical factories retain True.
    check_tilt: bool = True

    @classmethod
    def cf21bl_simulation_reference(cls) -> "AccelerationToRPYTConfig":
        """Return the pinned firmware/local asset reference, still UNVALIDATED."""
        return cls(
            mass_kg=0.042,
            gravity_m_s2=9.81,
            max_tilt_deg=35.0,
            min_force_norm_N=1.0e-6,
            motors=4,
            # power_distribution_quadrotor.c at the pinned commit.
            pwm_to_thrust_a_N=0.091492681,
            pwm_to_thrust_b_N=0.067673604,
            # crtp_commander_rpyt.c at the pinned commit.
            commander_thrust_min=1000,
            commander_thrust_max=60000,
            firmware_revision=FIRMWARE_REVISION,
            firmware_commit=FIRMWARE_COMMIT,
            firmware_platform=FIRMWARE_PLATFORM,
            calibration_status=CALIBRATION_STATUS,
        )

    def __post_init__(self) -> None:
        positive = {
            "mass_kg": self.mass_kg,
            "gravity_m_s2": self.gravity_m_s2,
            "min_force_norm_N": self.min_force_norm_N,
            "motors": self.motors,
            "pwm_to_thrust_a_N": self.pwm_to_thrust_a_N,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.pwm_to_thrust_b_N < 0:
            raise ValueError("pwm_to_thrust_b_N must be non-negative")
        if not 0 < self.max_tilt_deg < 90:
            raise ValueError("max_tilt_deg must lie in (0, 90)")
        if not 0 <= self.commander_thrust_min <= self.commander_thrust_max <= 65535:
            raise ValueError("invalid commander thrust limits")


class AccelerationToRPYT:
    """Build yaw-consistent attitude and an integer-valued API thrust command."""

    def __init__(self, config: AccelerationToRPYTConfig) -> None:
        self.config = config

    def force_to_commander(self, total_force_N: Tensor) -> Tensor:
        """Invert the pinned cf21bl per-motor force curve and round to uint16 value."""
        total_force_N = torch.as_tensor(total_force_N)
        per_motor_force = total_force_N.clamp_min(0.0) / float(self.config.motors)
        a = self.config.pwm_to_thrust_a_N
        b = self.config.pwm_to_thrust_b_N
        normalized_pwm = (-b + torch.sqrt(b * b + 4.0 * a * per_motor_force)) / (2.0 * a)
        raw = normalized_pwm * 65535.0
        return torch.round(raw).clamp(
            float(self.config.commander_thrust_min),
            float(self.config.commander_thrust_max),
        )

    def __call__(
        self,
        a_test_W: Tensor,
        psi_hold_W: Tensor,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Convert ``[B,1,3]`` net acceleration and held yaw to API RPYT.

        ``psi_hold_W`` is in radians and may be ``[B,1]`` or ``[B,1,1]``.
        The output is ``[roll_deg, pitch_deg, 0_deg_s, thrust_uint16_value]``.
        """
        if a_test_W.ndim != 3 or a_test_W.shape[-2:] != (1, 3):
            raise ValueError(f"a_test_W must have shape [B,1,3], got {tuple(a_test_W.shape)}")
        if not a_test_W.is_floating_point():
            raise TypeError("a_test_W must be floating point")
        if psi_hold_W.ndim == 2 and psi_hold_W.shape[-1] == 1:
            psi_hold_W = psi_hold_W.unsqueeze(-1)
        if psi_hold_W.ndim != 3 or psi_hold_W.shape[-2:] != (1, 1):
            raise ValueError(
                f"psi_hold_W must have shape [B,1] or [B,1,1], got {tuple(psi_hold_W.shape)}"
            )
        if psi_hold_W.shape[0] != a_test_W.shape[0]:
            raise ValueError("a_test_W and psi_hold_W batch sizes differ")
        psi_hold_W = psi_hold_W.to(device=a_test_W.device, dtype=a_test_W.dtype)

        gravity = a_test_W.new_tensor([0.0, 0.0, self.config.gravity_m_s2])
        specific_force_W = a_test_W + gravity
        specific_force_norm = torch.linalg.vector_norm(specific_force_W, dim=-1, keepdim=True)
        desired_force_N = specific_force_norm * self.config.mass_kg
        finite = torch.isfinite(a_test_W).all(dim=-1, keepdim=True) & torch.isfinite(
            psi_hold_W
        )
        nondegenerate_force = desired_force_N > self.config.min_force_norm_N

        fallback_b3 = torch.zeros_like(specific_force_W)
        fallback_b3[..., 2] = 1.0
        b3 = specific_force_W / specific_force_norm.clamp_min(
            self.config.min_force_norm_N / self.config.mass_kg
        )
        b3 = torch.where(nondegenerate_force.expand_as(b3), b3, fallback_b3)

        psi = _wrap_radians(psi_hold_W)
        zero = torch.zeros_like(psi)
        b2c = torch.cat((-torch.sin(psi), torch.cos(psi), zero), dim=-1)
        b1_raw = torch.linalg.cross(b2c, b3, dim=-1)
        b1_norm = torch.linalg.vector_norm(b1_raw, dim=-1, keepdim=True)
        nondegenerate_heading = b1_norm > 1.0e-7
        fallback_b1 = torch.zeros_like(b1_raw)
        fallback_b1[..., 0] = 1.0
        b1 = b1_raw / b1_norm.clamp_min(1.0e-7)
        b1 = torch.where(nondegenerate_heading.expand_as(b1), b1, fallback_b1)
        b2 = torch.linalg.cross(b3, b1, dim=-1)
        rotation_WB = torch.stack((b1, b2, b3), dim=-1)

        roll_rad = torch.atan2(rotation_WB[..., 2, 1], rotation_WB[..., 2, 2])
        pitch_rad = torch.asin((-rotation_WB[..., 2, 0]).clamp(-1.0, 1.0))
        desired_yaw = torch.atan2(rotation_WB[..., 1, 0], rotation_WB[..., 0, 0])
        tilt_rad = torch.acos(b3[..., 2:3].clamp(-1.0, 1.0))
        within_tilt = (tilt_rad <= math.radians(self.config.max_tilt_deg) + 1.0e-6
                       if self.config.check_tilt else torch.ones_like(tilt_rad, dtype=torch.bool))
        valid = finite & nondegenerate_force & nondegenerate_heading & (b3[..., 2:3] > 0) & within_tilt

        thrust_unclipped = self.force_to_commander(desired_force_N)
        # force_to_commander is already clipped; this independent predicate is telemetry.
        per_motor_force = desired_force_N / float(self.config.motors)
        a = self.config.pwm_to_thrust_a_N
        b = self.config.pwm_to_thrust_b_N
        raw_pwm = ((-b + torch.sqrt((b * b + 4.0 * a * per_motor_force).clamp_min(0.0))) / (2.0 * a)) * 65535.0
        thrust_saturated = (raw_pwm < self.config.commander_thrust_min) | (
            raw_pwm > self.config.commander_thrust_max
        )

        rad_to_deg = 180.0 / math.pi
        yaw_rate = torch.zeros_like(roll_rad)
        rpyt_api = torch.stack(
            (roll_rad * rad_to_deg, pitch_rad * rad_to_deg, yaw_rate, thrust_unclipped.squeeze(-1)),
            dim=-1,
        )
        diagnostics = {
            "valid": valid,
            "specific_force_W_m_s2": specific_force_W,
            "desired_force_N": desired_force_N,
            "desired_body_z_W": b3,
            "desired_rotation_WB": rotation_WB,
            "desired_rpy_rad": torch.stack((roll_rad, pitch_rad, desired_yaw), dim=-1),
            "desired_yaw_W_rad": desired_yaw.unsqueeze(-1),
            "tilt_rad": tilt_rad,
            "commander_thrust_raw": raw_pwm,
            "thrust_saturated": thrust_saturated,
        }
        return rpyt_api.to(dtype=a_test_W.dtype), diagnostics

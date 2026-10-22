"""Pure Torch model of the Crazyflie 2025.02 legacy RPYT PID path.

This module follows the pinned source-level arithmetic where practical.  It is
an UNVALIDATED simulation reference and has not been matched to firmware logs
or a physical Crazyflie.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Optional, Sequence, Tuple

import torch
from torch import Tensor


FIRMWARE_REVISION = "2025.02"
FIRMWARE_COMMIT = "16c5630707e92830dcd16455a57b5560e5838735"
FIRMWARE_PLATFORM = "cf21bl"
CALIBRATION_STATUS = "UNVALIDATED"


def wrap_degrees(angle: Tensor) -> Tensor:
    """Wrap like controller_pid.c/pid.c, retaining signed +/-180 endpoints."""
    wrapped = torch.remainder(angle + 180.0, 360.0) - 180.0
    return torch.where((wrapped == -180.0) & (angle > 0), -wrapped, wrapped)


def quaternion_to_euler_deg(quat_wxyz: Tensor) -> Tensor:
    """Convert scalar-first body-to-world quaternion to ZYX Euler degrees."""
    if quat_wxyz.shape[-1] != 4:
        raise ValueError("quat_wxyz last dimension must be 4")
    quat_wxyz = quat_wxyz / torch.linalg.vector_norm(
        quat_wxyz, dim=-1, keepdim=True
    ).clamp_min(1.0e-12)
    w, x, y, z = quat_wxyz.unbind(dim=-1)
    roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = torch.asin((2.0 * (w * y - z * x)).clamp(-1.0, 1.0))
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return torch.stack((roll, pitch, yaw), dim=-1) * (180.0 / math.pi)


def _trunc_int(value: Tensor) -> Tensor:
    """Represent C casts to signed integers while retaining a Torch float tensor."""
    return torch.trunc(value)


def _signed_int16(value: Tensor) -> Tensor:
    # Firmware intentionally excludes INT16_MIN because it may negate the result.
    return _trunc_int(value.clamp(-32767.0, 32767.0))


def legacy_mixer(
    thrust: Tensor,
    control_rpy: Tensor,
    *,
    idle_thrust: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Reproduce legacy X mixer integer arithmetic and common upper cap.

    ``control_rpy`` is the firmware ``control.roll/pitch/yaw`` value, including
    the yaw sign flip already performed by ``controller_pid.c``.
    """
    if thrust.shape[-1] != 1 or control_rpy.shape[-1] != 3:
        raise ValueError("thrust/control_rpy last dimensions must be 1 and 3")
    if thrust.shape[:-1] != control_rpy.shape[:-1]:
        raise ValueError("thrust/control_rpy leading shapes differ")
    roll, pitch, yaw = control_rpy.unbind(dim=-1)
    # C uses int16_t r = control->roll / 2.0f, which truncates toward zero.
    r = _trunc_int(roll / 2.0)
    p = _trunc_int(pitch / 2.0)
    t = thrust.squeeze(-1)
    uncapped = _trunc_int(
        torch.stack(
            (
                t - r + p + yaw,
                t - r - p - yaw,
                t + r - p + yaw,
                t + r + p - yaw,
            ),
            dim=-1,
        )
    )
    highest = uncapped.max(dim=-1, keepdim=True).values.clamp_min(0.0)
    reduction = (highest - 65535.0).clamp_min(0.0)
    capped = (uncapped - reduction).clamp_min(float(idle_thrust))
    return uncapped, capped, reduction > 0


@dataclass(frozen=True)
class PIDAxisConfig:
    kp: float
    ki: float
    kd: float
    kff: float
    integral_limit: float


@dataclass(frozen=True)
class CrazyflieFirmwarePIDConfig:
    """Pinned controller constants plus explicit simulation boundary mappings."""

    dt_s: float
    attitude_pid: Tuple[PIDAxisConfig, PIDAxisConfig, PIDAxisConfig]
    rate_pid: Tuple[PIDAxisConfig, PIDAxisConfig, PIDAxisConfig]
    api_to_firmware_attitude_sign: Tuple[float, float, float]
    simulation_to_firmware_attitude_sign: Tuple[float, float, float]
    api_yaw_rate_to_firmware_sign: float
    gyro_body_to_firmware_sign: Tuple[float, float, float]
    yaw_max_delta_deg: float
    commander_thrust_min: int
    commander_thrust_max: int
    idle_thrust: int
    pwm_to_thrust_a_N: float
    pwm_to_thrust_b_N: float
    simulation_max_thrust_per_rotor_N: float
    simulation_motor_from_firmware: Tuple[int, int, int, int]
    apply_motor_dynamics: bool
    motor_time_constant_s: float
    firmware_revision: str
    firmware_commit: str
    firmware_platform: str
    calibration_status: str

    @classmethod
    def cf21bl_2025_02(cls) -> "CrazyflieFirmwarePIDConfig":
        """Return defaults traced to the pinned official cf21bl source."""
        return cls(
            dt_s=0.002,
            attitude_pid=(
                PIDAxisConfig(6.0, 3.0, 0.0, 0.0, 20.0),
                PIDAxisConfig(6.0, 3.0, 0.0, 0.0, 20.0),
                PIDAxisConfig(6.0, 1.0, 0.35, 0.0, 360.0),
            ),
            rate_pid=(
                PIDAxisConfig(200.0, 400.0, 2.5, 0.0, 33.3),
                PIDAxisConfig(200.0, 400.0, 2.5, 0.0, 33.3),
                PIDAxisConfig(120.0, 16.7, 0.0, 0.0, 166.7),
            ),
            # cflib send_setpoint packs negative API pitch; legacy receive uses it.
            api_to_firmware_attitude_sign=(1.0, -1.0, 1.0),
            # Match standard ZYX simulation pitch to the firmware/cflib sign chain.
            simulation_to_firmware_attitude_sign=(1.0, -1.0, 1.0),
            # crtp_commander_rpyt.c: legacy rate input is inverted.
            api_yaw_rate_to_firmware_sign=-1.0,
            # controller_pid.c calls the pitch rate PID with -sensors->gyro.y.
            gyro_body_to_firmware_sign=(1.0, -1.0, 1.0),
            yaw_max_delta_deg=0.0,
            commander_thrust_min=1000,
            commander_thrust_max=60000,
            idle_thrust=7000,
            pwm_to_thrust_a_N=0.091492681,
            pwm_to_thrust_b_N=0.067673604,
            # Local crazyflie.yaml max force. This cross-model mapping is UNVALIDATED.
            simulation_max_thrust_per_rotor_N=0.289,
            # Local rotor_0..3 positions are (+x,+y),(-x,+y),(-x,-y),(+x,-y).
            # Firmware legacy-mixer M1..M4 must be mirrored into that prim order.
            simulation_motor_from_firmware=(3, 2, 1, 0),
            # MultirotorBase owns its 0.05 s lag; enable only without that downstream lag.
            apply_motor_dynamics=False,
            motor_time_constant_s=0.05,
            firmware_revision=FIRMWARE_REVISION,
            firmware_commit=FIRMWARE_COMMIT,
            firmware_platform=FIRMWARE_PLATFORM,
            calibration_status=CALIBRATION_STATUS,
        )

    def __post_init__(self) -> None:
        if self.dt_s <= 0 or self.motor_time_constant_s <= 0:
            raise ValueError("dt_s and motor_time_constant_s must be positive")
        if len(self.attitude_pid) != 3 or len(self.rate_pid) != 3:
            raise ValueError("attitude_pid and rate_pid must each contain three axes")
        for gains in (*self.attitude_pid, *self.rate_pid):
            if gains.integral_limit < 0:
                raise ValueError("PID integral limits must be non-negative")
        for signs in (
            self.api_to_firmware_attitude_sign,
            self.simulation_to_firmware_attitude_sign,
            self.gyro_body_to_firmware_sign,
        ):
            if len(signs) != 3 or any(abs(value) != 1.0 for value in signs):
                raise ValueError("axis mappings must contain three +/-1 signs")
        if abs(self.api_yaw_rate_to_firmware_sign) != 1.0:
            raise ValueError("api_yaw_rate_to_firmware_sign must be +/-1")
        if not 0 <= self.commander_thrust_min <= self.commander_thrust_max <= 65535:
            raise ValueError("invalid commander thrust limits")
        if not 0 <= self.idle_thrust <= 65535:
            raise ValueError("idle_thrust must be a uint16 value")
        if self.pwm_to_thrust_a_N <= 0 or self.pwm_to_thrust_b_N < 0:
            raise ValueError("invalid PWM-to-thrust curve")
        if self.simulation_max_thrust_per_rotor_N <= 0:
            raise ValueError("simulation_max_thrust_per_rotor_N must be positive")
        if tuple(sorted(self.simulation_motor_from_firmware)) != (0, 1, 2, 3):
            raise ValueError("simulation_motor_from_firmware must be a permutation of 0..3")


class CrazyflieFirmwarePID:
    """Stateful batched replica of the legacy attitude/rate PID and mixer core."""

    _STATE_NAMES: Sequence[str] = (
        "firmware_yaw_reference_deg",
        "attitude_integral",
        "attitude_previous_measured_deg",
        "rate_integral",
        "rate_previous_measured_deg_s",
        "motor_force_fraction",
    )

    def __init__(self, config: CrazyflieFirmwarePIDConfig) -> None:
        self.config = config
        self._initialized = False

    def _firmware_attitude_deg(self, quat_wxyz: Tensor) -> Tensor:
        euler = quaternion_to_euler_deg(quat_wxyz)
        signs = euler.new_tensor(self.config.simulation_to_firmware_attitude_sign)
        return euler * signs

    def _allocate(self, reference: Tensor) -> None:
        leading = reference.shape[:-1]
        if len(leading) != 2 or leading[-1] != 1:
            raise ValueError("controller tensors must have leading shape [B,1]")
        vector = reference.new_zeros((*leading, 3))
        self.firmware_yaw_reference_deg = reference.new_zeros((*leading, 1))
        self.attitude_integral = vector.clone()
        self.attitude_previous_measured_deg = vector.clone()
        self.rate_integral = vector.clone()
        self.rate_previous_measured_deg_s = vector.clone()
        self.motor_force_fraction = reference.new_zeros((*leading, 4))
        self._initialized = True

    @staticmethod
    def _yaw_tensor(measured_yaw_deg: Tensor, reference: Tensor) -> Tensor:
        yaw = torch.as_tensor(measured_yaw_deg, device=reference.device, dtype=reference.dtype)
        if yaw.ndim == 2 and yaw.shape[-1] == 1:
            yaw = yaw.unsqueeze(-1)
        if yaw.ndim != 3 or yaw.shape[-2:] != (1, 1):
            raise ValueError("measured_yaw_deg must have shape [B,1] or [B,1,1]")
        return wrap_degrees(yaw)

    def reset(
        self,
        env_ids: Optional[Tensor] = None,
        measured_yaw_deg: Optional[Tensor] = None,
        attitude_quat_wxyz: Optional[Tensor] = None,
        motor_state: Optional[Tensor] = None,
    ) -> None:
        """Reset all controller state, globally or for selected batch indices.

        ``measured_yaw_deg`` is already in firmware yaw convention. Prefer
        ``attitude_quat_wxyz`` when resetting from the simulator world attitude.
        """
        if measured_yaw_deg is None and attitude_quat_wxyz is None:
            if not self._initialized:
                raise ValueError("first reset requires measured_yaw_deg or attitude_quat_wxyz")
            reference = self.firmware_yaw_reference_deg
        elif attitude_quat_wxyz is not None:
            reference = torch.as_tensor(attitude_quat_wxyz)
            if reference.ndim != 3 or reference.shape[-2:] != (1, 4):
                raise ValueError("attitude_quat_wxyz must have shape [B,1,4]")
        else:
            reference = torch.as_tensor(measured_yaw_deg)
            if not reference.is_floating_point():
                reference = reference.float()

        if not self._initialized:
            self._allocate(reference)

        if env_ids is None:
            ids = torch.arange(
                self.firmware_yaw_reference_deg.shape[0],
                device=self.firmware_yaw_reference_deg.device,
            )
        else:
            ids = torch.as_tensor(env_ids, device=self.firmware_yaw_reference_deg.device, dtype=torch.long)

        if attitude_quat_wxyz is not None:
            actual = self._firmware_attitude_deg(
                torch.as_tensor(
                    attitude_quat_wxyz,
                    device=self.firmware_yaw_reference_deg.device,
                    dtype=self.firmware_yaw_reference_deg.dtype,
                )
            )
            if actual.shape[0] == self.firmware_yaw_reference_deg.shape[0]:
                actual = actual[ids]
            elif actual.shape[0] != ids.numel():
                raise ValueError("reset attitude batch must be full batch or match env_ids")
            yaw = wrap_degrees(actual[..., 2:3])
        else:
            yaw = self._yaw_tensor(measured_yaw_deg, self.firmware_yaw_reference_deg)
            if yaw.shape[0] == self.firmware_yaw_reference_deg.shape[0]:
                yaw = yaw[ids]
            elif yaw.shape[0] != ids.numel():
                raise ValueError("reset yaw batch must be full batch or match env_ids")
            actual = torch.zeros(
                (ids.numel(), 1, 3),
                device=yaw.device,
                dtype=yaw.dtype,
            )
            actual[..., 2:3] = yaw

        self.firmware_yaw_reference_deg[ids] = yaw
        self.attitude_integral[ids] = 0
        self.attitude_previous_measured_deg[ids] = actual
        self.rate_integral[ids] = 0
        self.rate_previous_measured_deg_s[ids] = 0
        if motor_state is None:
            self.motor_force_fraction[ids] = 0
        else:
            motors = torch.as_tensor(
                motor_state,
                device=self.motor_force_fraction.device,
                dtype=self.motor_force_fraction.dtype,
            )
            if motors.ndim != 3 or motors.shape[-2:] != (1, 4):
                raise ValueError("motor_state must have shape [B,1,4]")
            if motors.shape[0] == self.motor_force_fraction.shape[0]:
                motors = motors[ids]
            elif motors.shape[0] != ids.numel():
                raise ValueError("motor_state batch must be full batch or match env_ids")
            self.motor_force_fraction[ids] = motors.clamp(0.0, 1.0)

    def _pid_update(
        self,
        desired: Tensor,
        measured: Tensor,
        integral: Tensor,
        previous_measured: Tensor,
        gains: Tuple[PIDAxisConfig, PIDAxisConfig, PIDAxisConfig],
        *,
        yaw_angle: bool,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        gain_p = measured.new_tensor([axis.kp for axis in gains])
        gain_i = measured.new_tensor([axis.ki for axis in gains])
        gain_d = measured.new_tensor([axis.kd for axis in gains])
        gain_ff = measured.new_tensor([axis.kff for axis in gains])
        limits = measured.new_tensor([axis.integral_limit for axis in gains])

        error = desired - measured
        delta = -(measured - previous_measured)
        if yaw_angle:
            error = torch.cat((error[..., :2], wrap_degrees(error[..., 2:3])), dim=-1)
            delta = torch.cat((delta[..., :2], wrap_degrees(delta[..., 2:3])), dim=-1)
        derivative = delta / self.config.dt_s
        derivative = torch.where(torch.isfinite(derivative), derivative, torch.zeros_like(derivative))
        integral = integral + error * self.config.dt_s
        integral = torch.maximum(torch.minimum(integral, limits), -limits)
        out_p = gain_p * error
        out_i = gain_i * integral
        out_d = gain_d * derivative
        output = out_p + out_i + out_d + gain_ff * desired
        return output, error, integral, out_p, out_i, out_d

    def step(
        self,
        rpyt_api: Tensor,
        quat_wxyz: Tensor,
        gyro_body_rad_s: Tensor,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Run one 2 ms update for batched ``[B,1,*]`` tensors."""
        if not self._initialized:
            raise RuntimeError("reset() must be called before step()")
        for name, tensor, width in (
            ("rpyt_api", rpyt_api, 4),
            ("quat_wxyz", quat_wxyz, 4),
            ("gyro_body_rad_s", gyro_body_rad_s, 3),
        ):
            if tensor.ndim != 3 or tensor.shape[-2:] != (1, width):
                raise ValueError(f"{name} must have shape [B,1,{width}]")
            if tensor.shape[0] != self.firmware_yaw_reference_deg.shape[0]:
                raise ValueError(f"{name} batch differs from reset state")
        if not (rpyt_api.is_floating_point() and quat_wxyz.is_floating_point() and gyro_body_rad_s.is_floating_point()):
            raise TypeError("controller inputs must be floating point")
        if rpyt_api.device != self.firmware_yaw_reference_deg.device:
            raise ValueError("controller state and input devices differ")

        dtype = self.firmware_yaw_reference_deg.dtype
        rpyt_api = rpyt_api.to(dtype=dtype)
        quat_wxyz = quat_wxyz.to(dtype=dtype)
        gyro_body_rad_s = gyro_body_rad_s.to(dtype=dtype)
        actual_attitude = self._firmware_attitude_deg(quat_wxyz)

        attitude_sign = rpyt_api.new_tensor(self.config.api_to_firmware_attitude_sign)
        desired_roll_pitch = rpyt_api[..., :2] * attitude_sign[:2]
        firmware_yaw_rate = (
            rpyt_api[..., 2:3] * self.config.api_yaw_rate_to_firmware_sign
        )
        self.firmware_yaw_reference_deg = wrap_degrees(
            self.firmware_yaw_reference_deg + firmware_yaw_rate * self.config.dt_s
        )
        if self.config.yaw_max_delta_deg != 0.0:
            delta = wrap_degrees(
                self.firmware_yaw_reference_deg - actual_attitude[..., 2:3]
            ).clamp(-self.config.yaw_max_delta_deg, self.config.yaw_max_delta_deg)
            self.firmware_yaw_reference_deg = wrap_degrees(actual_attitude[..., 2:3] + delta)
        desired_attitude = torch.cat(
            (desired_roll_pitch, self.firmware_yaw_reference_deg), dim=-1
        )

        (
            rate_setpoint,
            attitude_error,
            self.attitude_integral,
            attitude_p,
            attitude_i,
            attitude_d,
        ) = self._pid_update(
            desired_attitude,
            actual_attitude,
            self.attitude_integral,
            self.attitude_previous_measured_deg,
            self.config.attitude_pid,
            yaw_angle=True,
        )
        self.attitude_previous_measured_deg = actual_attitude

        gyro_sign = gyro_body_rad_s.new_tensor(self.config.gyro_body_to_firmware_sign)
        actual_rate = gyro_body_rad_s * (180.0 / math.pi) * gyro_sign
        (
            rate_output_float,
            rate_error,
            self.rate_integral,
            rate_p,
            rate_i,
            rate_d,
        ) = self._pid_update(
            rate_setpoint,
            actual_rate,
            self.rate_integral,
            self.rate_previous_measured_deg_s,
            self.config.rate_pid,
            yaw_angle=False,
        )
        self.rate_previous_measured_deg_s = actual_rate

        actuator_output = _signed_int16(rate_output_float)
        # controller_pid.c negates yaw after retrieving the PID actuator output.
        control_rpy = torch.cat(
            (actuator_output[..., :2], -actuator_output[..., 2:3]), dim=-1
        )
        raw_thrust = torch.round(rpyt_api[..., 3:4]).clamp(0.0, 65535.0)
        decoded_thrust = torch.where(
            raw_thrust < self.config.commander_thrust_min,
            torch.zeros_like(raw_thrust),
            raw_thrust.clamp_max(float(self.config.commander_thrust_max)),
        )
        zero_thrust = decoded_thrust == 0
        control_rpy = torch.where(zero_thrust.expand_as(control_rpy), torch.zeros_like(control_rpy), control_rpy)

        motor_uncapped, motor_pwm, motor_saturated = legacy_mixer(
            decoded_thrust,
            control_rpy,
            idle_thrust=self.config.idle_thrust,
        )
        motor_pwm = torch.where(zero_thrust.expand_as(motor_pwm), torch.zeros_like(motor_pwm), motor_pwm)
        permutation = torch.as_tensor(
            self.config.simulation_motor_from_firmware,
            device=motor_pwm.device,
            dtype=torch.long,
        )
        simulation_motor_pwm = motor_pwm.index_select(-1, permutation)
        normalized_pwm = simulation_motor_pwm / 65535.0
        motor_force_N = (
            self.config.pwm_to_thrust_a_N * normalized_pwm.square()
            + self.config.pwm_to_thrust_b_N * normalized_pwm
        )
        target_force_fraction = (
            motor_force_N / self.config.simulation_max_thrust_per_rotor_N
        ).clamp(0.0, 1.0)
        if self.config.apply_motor_dynamics:
            alpha = min(self.config.dt_s / self.config.motor_time_constant_s, 1.0)
            self.motor_force_fraction = self.motor_force_fraction + alpha * (
                target_force_fraction - self.motor_force_fraction
            )
            applied_force_fraction = self.motor_force_fraction
        else:
            self.motor_force_fraction = target_force_fraction
            applied_force_fraction = target_force_fraction
        rotor_command = 2.0 * applied_force_fraction - 1.0

        if zero_thrust.any():
            mask3 = zero_thrust.expand_as(self.attitude_integral)
            mask4 = zero_thrust.expand_as(self.motor_force_fraction)
            self.firmware_yaw_reference_deg = torch.where(
                zero_thrust,
                wrap_degrees(actual_attitude[..., 2:3]),
                self.firmware_yaw_reference_deg,
            )
            self.attitude_integral = torch.where(mask3, torch.zeros_like(self.attitude_integral), self.attitude_integral)
            self.rate_integral = torch.where(mask3, torch.zeros_like(self.rate_integral), self.rate_integral)
            self.attitude_previous_measured_deg = torch.where(
                mask3, actual_attitude, self.attitude_previous_measured_deg
            )
            self.rate_previous_measured_deg_s = torch.where(
                mask3, torch.zeros_like(self.rate_previous_measured_deg_s), self.rate_previous_measured_deg_s
            )
            self.motor_force_fraction = torch.where(
                mask4, torch.zeros_like(self.motor_force_fraction), self.motor_force_fraction
            )
            rotor_command = torch.where(
                zero_thrust.expand_as(rotor_command),
                torch.full_like(rotor_command, -1.0),
                rotor_command,
            )

        diagnostics = {
            "firmware_yaw_reference_deg": self.firmware_yaw_reference_deg.clone(),
            "firmware_attitude_actual_deg": actual_attitude,
            "firmware_attitude_setpoint_deg": desired_attitude,
            "firmware_rate_actual_deg_s": actual_rate,
            "rate_setpoint_deg_s": rate_setpoint,
            "attitude_error_deg": attitude_error,
            "rate_error_deg_s": rate_error,
            "attitude_integral": self.attitude_integral.clone(),
            "rate_integral": self.rate_integral.clone(),
            "attitude_pid_p": attitude_p,
            "attitude_pid_i": attitude_i,
            "attitude_pid_d": attitude_d,
            "rate_pid_p": rate_p,
            "rate_pid_i": rate_i,
            "rate_pid_d": rate_d,
            "rate_pid_output_float": rate_output_float,
            "actuator_output_int16": actuator_output,
            "control_rpy_post_yaw_sign": control_rpy,
            "decoded_thrust": decoded_thrust,
            "zero_thrust_reset": zero_thrust,
            "motor_pwm_uncapped": motor_uncapped,
            "motor_pwm": motor_pwm,
            "simulation_motor_pwm": simulation_motor_pwm,
            "motor_saturated": motor_saturated,
            "motor_force_fraction_target": target_force_fraction,
            "motor_force_fraction_applied": applied_force_fraction,
        }
        return rotor_command.clamp(-1.0, 1.0), diagnostics

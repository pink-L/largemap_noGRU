#!/usr/bin/env python3
"""Reward and true-termination contract for dense-planar NavACC training."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class PlanarRewardConfig:
    mode: str = "legacy_pbrs_v2"
    velocity_rate: float = 1.0
    control_dt_s: float = .01
    discount_time_constant_s: float = 60.0
    gae_time_constant_s: float = 1.0
    scene_max_distance_m: float = math.sqrt(50.0 ** 2 * 2 + 4.5 ** 2)
    goal_radius_m: float = .2
    arrival_speed_m_s: float = .2
    arrival_dwell_s: float = .5
    deadline_s: float = 120.0
    potential_scale: float = 10.0
    time_rate: float = .05
    effort_rate: float = .02
    execution_smooth_rate: float = 0.0
    goal_brake_rate: float = .2
    goal_brake_acceleration_m_s2: float = .8
    goal_brake_latency_s: float = .2
    far_speed_allowance_m_s: float = 1.8
    lidar_risk_rate: float = .1
    height_rate: float = 8.0
    height_tolerance_m: float = .2
    arrival_bonus: float = 80.0
    failure_cost: float = 80.0
    # Opt-in candidate terms (overnight 2026-10-12).  Defaults preserve the
    # exact legacy/joint reward: with pb_mode="none" and oob_shaping_rate=0.
    # the two added terms below are identically zero.
    pb_mode: str = "none"
    oob_shaping_rate: float = 0.0
    oob_xy_limit_m: float = 24.0
    near_goal_pbrs_rate: float = 0.0
    near_goal_pbrs_max_d_m: float = 2.5
    near_goal_pbrs_inner_m: float = 0.5
    # Corr-AB 2026-10-14: coefficient of the existing unbounded CBF correction
    # penalty r_corr = c*(exp(-||a_safe_raw-a_nom||^2)-1).  Default .03 keeps
    # every legacy/joint phase bit-identical; only the corr-ab fork phases set
    # another value and the fork validator pins the allowed values.
    cbf_correction_coeff: float = .03
    failure_discount_compensation: bool = True
    hold_rate: float = 1.0
    hold_speed_reference_m_s: float = .5

    def __post_init__(self) -> None:
        if self.mode not in ("legacy_pbrs_v2", "zero_curriculum_v1", "hold48_v1", "hold36_continuous_v1", "hold30_continuous_v1", "hold50_continuous_v1", "obstacle_course_continuous_v1"):
            raise ValueError("unsupported reward mode")
        if self.pb_mode not in ("none", "oob_xy_shaping", "near_goal_pbrs"):
            raise ValueError("unsupported pb_mode")
        for name, value in self.__dict__.items():
            if name in ("mode", "pb_mode", "failure_discount_compensation"): continue
            if type(value) not in (int, float) or not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        positive = ("control_dt_s", "discount_time_constant_s",
                    "gae_time_constant_s", "scene_max_distance_m",
                    "goal_radius_m", "arrival_speed_m_s", "arrival_dwell_s",
                    "deadline_s", "potential_scale", "arrival_bonus",
                    "failure_cost", "goal_brake_acceleration_m_s2",
                    "goal_brake_latency_s", "far_speed_allowance_m_s",
                    "oob_xy_limit_m", "near_goal_pbrs_max_d_m",
                    "near_goal_pbrs_inner_m", "cbf_correction_coeff")
        if any(float(getattr(self, name)) <= 0 for name in positive):
            raise ValueError("positive reward constants must be positive")
        if any(float(getattr(self, name)) < 0 for name in
               ("velocity_rate", "time_rate", "effort_rate", "goal_brake_rate",
                "lidar_risk_rate", "height_rate", "height_tolerance_m",
                "oob_shaping_rate", "near_goal_pbrs_rate")):
            raise ValueError("reward rates and tolerances must be nonnegative")
        if type(self.failure_discount_compensation)!=bool or (self.mode in ("hold48_v1","hold36_continuous_v1", "hold30_continuous_v1", "hold50_continuous_v1", "obstacle_course_continuous_v1") and self.failure_discount_compensation) or (self.mode not in ("hold48_v1","hold36_continuous_v1", "hold30_continuous_v1", "hold50_continuous_v1", "obstacle_course_continuous_v1") and not self.failure_discount_compensation):
            raise ValueError("failure compensation must match episode mode")
        if self.execution_smooth_rate < 0 or (self.execution_smooth_rate and self.effort_rate != 0):
            raise ValueError("execution smooth replaces nominal effort; rates cannot both be active")
        if self.mode=="obstacle_course_continuous_v1" and self.deadline_s not in (32.,36.,40.,45.,50.):
            raise ValueError("course deadline must be40/45/50")
        if self.hold_rate<0 or self.hold_speed_reference_m_s<=0:
            raise ValueError("invalid hold constants")
        if self.mode in ("hold48_v1","hold36_continuous_v1", "hold30_continuous_v1", "hold50_continuous_v1", "obstacle_course_continuous_v1") and (self.deadline_s!=(self.deadline_s if self.mode=="obstacle_course_continuous_v1" and self.deadline_s in (32.,36.,40.,45.,50.) else 50. if self.mode=="hold50_continuous_v1" else 30. if self.mode=="hold30_continuous_v1" else 36. if self.mode=="hold36_continuous_v1" else 48.) or self.goal_radius_m!=.5 or self.pb_mode!="none" or self.oob_shaping_rate!=0 or self.near_goal_pbrs_rate!=0):
            raise ValueError("hold48 requires explicit48s/.5m/noPBRS config")
        if self.near_goal_pbrs_inner_m >= self.near_goal_pbrs_max_d_m:
            raise ValueError("near-goal PBRS inner radius must be below max distance")

    @property
    def gamma(self) -> float:
        # This float32 value is the authoritative scalar used by reward and GAE.
        return float(torch.tensor(math.exp(
            -self.control_dt_s / self.discount_time_constant_s),
            dtype=torch.float32).item())

    @property
    def gae_lambda(self) -> float:
        return float(torch.tensor(math.exp(
            -self.control_dt_s / self.gae_time_constant_s),
            dtype=torch.float32).item())

    @property
    def dwell_steps(self) -> int:
        value = self.arrival_dwell_s / self.control_dt_s
        if abs(value - round(value)) > 1e-9:
            raise ValueError("arrival dwell must be an integer control-step count")
        return int(round(value))


@dataclass(frozen=True)
class PlanarRewardStep:
    reward: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    arrival: torch.Tensor
    failure: torch.Tensor
    timeout: torch.Tensor
    dwell_steps: torch.Tensor
    potential_before: torch.Tensor
    potential_after: torch.Tensor
    shaping: torch.Tensor
    running_cost: torch.Tensor
    terminal_reward: torch.Tensor
    failure_discount_exponent: torch.Tensor
    time_cost: torch.Tensor
    effort_cost: torch.Tensor
    brake_cost: torch.Tensor
    lidar_cost: torch.Tensor
    height_cost: torch.Tensor
    oob_cost: torch.Tensor
    near_goal_pbrs_cost: torch.Tensor
    near_goal_phi_before: torch.Tensor
    near_goal_phi_after_effective: torch.Tensor
    effective_nonfinite: torch.Tensor | None = None
    execution_smooth_cost: torch.Tensor | None = None
    first_reach: torch.Tensor | None = None
    hold_reward: torch.Tensor | None = None
    raw_in_goal: torch.Tensor | None = None
    ever_reached: torch.Tensor | None = None


def _potential(position: torch.Tensor, goal: torch.Tensor,
               config: PlanarRewardConfig) -> torch.Tensor:
    distance = torch.linalg.vector_norm(goal - position, dim=-1)
    goal_ball_distance = (distance - config.goal_radius_m).clamp_min(0)
    return -config.potential_scale * (
        goal_ball_distance / config.scene_max_distance_m).clamp(0, 1)


@torch.no_grad()
def planar_reward_step(
    position_before: torch.Tensor,
    position_after: torch.Tensor,
    velocity_after: torch.Tensor,
    goal: torch.Tensor,
    nominal_acceleration: torch.Tensor,
    static_lidar: torch.Tensor,
    start_height_m: torch.Tensor,
    goal_height_m: torch.Tensor,
    previous_dwell_steps: torch.Tensor,
    completed_elapsed_steps: torch.Tensor,
    *,
    collision_static: torch.Tensor,
    collision_dynamic: torch.Tensor,
    crash: torch.Tensor,
    out_of_bounds: torch.Tensor,
    nonfinite: torch.Tensor,
    cbf_abort: torch.Tensor,
    config: PlanarRewardConfig = PlanarRewardConfig(),
    cbf_reward: torch.Tensor | None = None,
    previous_ever_reached: torch.Tensor | None = None,
    executed_acceleration: torch.Tensor | None = None,
    previous_executed_acceleration: torch.Tensor | None = None,
    previous_execution_valid: torch.Tensor | None = None,
) -> PlanarRewardStep:
    """Compute one reward transition and true termination.

    ``completed_elapsed_steps`` includes this transition.  Therefore the first
    transition has value one and its terminal discount exponent is zero.
    Collector cuts are external trajectory boundaries and are not represented
    as termination here.
    """
    if position_before.ndim != 2 or position_before.shape[-1] != 3:
        raise ValueError("position tensors must be [E,3]")
    e = position_before.shape[0]
    float_shapes = {
        "position_after": (position_after, (e, 3)),
        "velocity_after": (velocity_after, (e, 3)),
        "goal": (goal, (e, 3)),
        "nominal_acceleration": (nominal_acceleration, (e, 3)),
        "start_height_m": (start_height_m, (e,)),
        "goal_height_m": (goal_height_m, (e,)),
    }
    for name, (value, shape) in float_shapes.items():
        if value.shape != shape or value.dtype != position_before.dtype \
                or value.device != position_before.device:
            raise ValueError(f"{name} must match {shape}, dtype and device")
    if static_lidar.shape[0] != e or not static_lidar.is_floating_point():
        raise ValueError("static_lidar must have leading environment dimension")
    if previous_dwell_steps.shape != (e,) or completed_elapsed_steps.shape != (e,):
        raise ValueError("dwell and elapsed steps must be [E]")
    masks = (collision_static, collision_dynamic, crash, out_of_bounds,
             nonfinite, cbf_abort)
    if any(value.shape != (e,) or value.dtype != torch.bool for value in masks):
        raise ValueError("failure masks must be bool [E]")
    if bool((completed_elapsed_steps < 1).any()):
        raise ValueError("completed_elapsed_steps includes this transition and must be >=1")

    finite_before_goal = torch.isfinite(position_before).all(-1) \
        & torch.isfinite(goal).all(-1)
    finite_after_goal = torch.isfinite(position_after).all(-1) \
        & torch.isfinite(goal).all(-1)
    finite_velocity = torch.isfinite(velocity_after).all(-1)
    finite_nominal = torch.isfinite(nominal_acceleration).all(-1)
    finite_heights = torch.isfinite(start_height_m) \
        & torch.isfinite(goal_height_m)
    finite_lidar = torch.isfinite(static_lidar.reshape(e, -1)).all(-1)
    finite_transition = finite_before_goal & finite_after_goal \
        & finite_velocity & finite_nominal & finite_heights & finite_lidar
    execution_smooth_cost = torch.zeros(e, dtype=position_before.dtype, device=position_before.device)
    if config.execution_smooth_rate:
        for name, value in (("executed_acceleration",executed_acceleration),("previous_executed_acceleration",previous_executed_acceleration)):
            if value is None or value.shape != (e,3) or value.dtype != position_before.dtype or value.device != position_before.device:
                raise ValueError(f"{name} must match [E,3], dtype/device")
        if previous_execution_valid is None or previous_execution_valid.shape != (e,) or previous_execution_valid.dtype != torch.bool or previous_execution_valid.device != position_before.device:
            raise ValueError("previous_execution_valid must be bool [E] on matching device")
        valid = previous_execution_valid
        finite_exec = torch.isfinite(executed_acceleration).all(-1)
        finite_prev = torch.isfinite(previous_executed_acceleration).all(-1)
        delta = torch.where(valid[:,None],executed_acceleration-previous_executed_acceleration,torch.zeros_like(executed_acceleration))
        raw_cost = config.control_dt_s * config.execution_smooth_rate * delta.square().sum(-1)
        finite_cost = torch.isfinite(raw_cost)
        finite_transition = finite_transition & finite_exec & (~valid | finite_prev) & finite_cost
        # An invalid request/difference is a fatal transition, never a zero-cost success.
        execution_smooth_cost = torch.where(finite_cost, raw_cost, torch.zeros_like(raw_cost))
    nonfinite = nonfinite | ~finite_transition
    before_safe = torch.where(finite_before_goal[:, None], position_before,
                              torch.zeros_like(position_before))
    before_goal_safe = torch.where(finite_before_goal[:, None], goal,
                                   torch.zeros_like(goal))
    after_safe = torch.where(finite_after_goal[:, None], position_after,
                             torch.zeros_like(position_after))
    after_goal_safe = torch.where(finite_after_goal[:, None], goal,
                                  torch.zeros_like(goal))
    velocity_safe = torch.where(finite_velocity[:, None], velocity_after,
                                torch.zeros_like(velocity_after))
    nominal_safe = torch.where(finite_nominal[:, None], nominal_acceleration,
                               torch.zeros_like(nominal_acceleration))
    start_height_safe = torch.where(
        finite_heights, start_height_m, torch.zeros_like(start_height_m))
    goal_height_safe = torch.where(
        finite_heights, goal_height_m, torch.zeros_like(goal_height_m))
    lidar_safe = torch.where(
        finite_lidar.reshape((e,) + (1,) * (static_lidar.ndim - 1)),
        static_lidar, torch.zeros_like(static_lidar))
    fatal_failure = (collision_static | collision_dynamic | crash
                     | out_of_bounds | nonfinite | cbf_abort)
    distance = torch.linalg.vector_norm(after_goal_safe - after_safe, dim=-1)
    speed = torch.linalg.vector_norm(velocity_safe, dim=-1)
    in_goal = (distance <= config.goal_radius_m) \
        & (speed <= config.arrival_speed_m_s)
    dwell = torch.where(in_goal, previous_dwell_steps + 1,
                        torch.zeros_like(previous_dwell_steps))
    arrival_raw = (distance < config.goal_radius_m) if config.mode == "zero_curriculum_v1" else dwell >= config.dwell_steps
    at_deadline = completed_elapsed_steps.to(position_before.dtype) \
        * config.control_dt_s >= config.deadline_s - 1e-9
    # The endpoint is inclusive: a valid dwell completed on the 120 s step is
    # arrival.  Fatal physical/control failures still take priority.
    timeout = at_deadline & ~arrival_raw & ~fatal_failure
    failure = fatal_failure | timeout
    arrival = arrival_raw & ~failure
    terminated = failure | arrival
    first_reach = hold_reward = raw_in_goal = ever_reached = None
    truncated = torch.zeros_like(terminated)
    if config.mode in ('hold48_v1','hold36_continuous_v1','hold30_continuous_v1','hold50_continuous_v1','obstacle_course_continuous_v1'):
        if previous_ever_reached is None or previous_ever_reached.shape != (e,) or previous_ever_reached.dtype != torch.bool:
            raise ValueError('hold48 requires bool previous ever_reached [E]')
        raw_in_goal = finite_after_goal & finite_velocity & (distance < config.goal_radius_m)
        first_reach = raw_in_goal & ~previous_ever_reached & ~fatal_failure
        hold_eligible = raw_in_goal & previous_ever_reached & ~fatal_failure
        ever_reached = previous_ever_reached | first_reach
        hold_reward = config.control_dt_s * config.hold_rate * hold_eligible.to(speed.dtype) * torch.exp(-(speed/config.hold_speed_reference_m_s).square())
        timeout = torch.zeros_like(fatal_failure)
        failure = terminated = fatal_failure
        truncated = at_deadline & ~fatal_failure
        arrival = raw_in_goal & ~fatal_failure & (terminated | truncated)

    before_phi = _potential(before_safe, before_goal_safe, config)
    after_phi_raw = _potential(after_safe, after_goal_safe, config)
    after_phi = torch.where(terminated, torch.zeros_like(after_phi_raw),
                            after_phi_raw)
    gamma = torch.tensor(config.gamma, dtype=position_before.dtype,
                         device=position_before.device)
    shaping = gamma * after_phi - before_phi
    if config.mode in ("zero_curriculum_v1", "hold48_v1", "hold36_continuous_v1", "hold30_continuous_v1", "hold50_continuous_v1", "obstacle_course_continuous_v1"):
        direction = (after_goal_safe-after_safe) / distance.clamp_min(1e-6)[:, None]
        shaping = config.control_dt_s * config.velocity_rate * (velocity_safe*direction).sum(-1)
        before_phi = torch.zeros_like(before_phi); after_phi = torch.zeros_like(after_phi)
    low = torch.minimum(start_height_safe, goal_height_safe) \
        - config.height_tolerance_m
    high = torch.maximum(start_height_safe, goal_height_safe) \
        + config.height_tolerance_m
    z = after_safe[:, 2]
    height_error2 = (low - z).clamp_min(0).square() \
        + (z - high).clamp_min(0).square()
    goal_ball_gap = (distance - config.goal_radius_m).clamp_min(0)
    allowed_speed = (torch.sqrt(
        2 * config.goal_brake_acceleration_m_s2 * goal_ball_gap)
        - config.goal_brake_acceleration_m_s2 * config.goal_brake_latency_s) \
        .clamp(config.arrival_speed_m_s, config.far_speed_allowance_m_s)
    excess_speed = (speed - allowed_speed).clamp_min(0)
    lidar_risk = lidar_safe.reshape(e, -1).clamp_min(0).mean(-1)
    rates = (config.time_rate
             + (torch.zeros_like(distance) if config.execution_smooth_rate else config.effort_rate * nominal_safe.square().sum(-1))
             + (config.goal_brake_rate if config.mode == "legacy_pbrs_v2" else 0.) * excess_speed.square()
             + config.lidar_risk_rate * lidar_risk
             + config.height_rate * height_error2)
    running_cost = config.control_dt_s * rates
    if config.execution_smooth_rate:
        running_cost = running_cost + execution_smooth_cost
    # --- Opt-in candidate terms (default off => exactly zero) ---
    # B: penalty for XY displacement beyond oob_xy_limit_m, discouraging the
    # observed policy fly-away while leaving the hard OOB check untouched.
    oob_cost = torch.zeros_like(rates)
    if config.pb_mode == "oob_xy_shaping":
        xy = after_safe[:, :2]
        excess_xy = (xy.abs() - config.oob_xy_limit_m).clamp_min(0)
        oob_cost = config.control_dt_s * config.oob_shaping_rate \
            * excess_xy.square().sum(-1)
    # C: near-goal potential-based shaping. Phi(d)=rate*(max_d-clamp(d-inner,0,max_d)),
    # Phi=0 for d>=inner+max_d (>=3m by default) and maximal near the goal ball.
    # All real terminals (arrival and failure) flush next-effective Phi to 0.
    near_goal_cost = torch.zeros_like(rates)
    phi_before = torch.zeros_like(rates)
    phi_after_effective = torch.zeros_like(rates)
    if config.pb_mode == "near_goal_pbrs":
        # Phi(d) = rate * (max_d - clamp(d - inner, 0, max_d)).  With the
        # defaults (inner=.5 m, max_d=2.5 m) this is Phi=12.5 inside the goal
        # ball, decreasing linearly to Phi=0 for every d >= inner + max_d = 3 m.
        def _phi(d):
            return config.near_goal_pbrs_rate * (
                config.near_goal_pbrs_max_d_m
                - (d - config.near_goal_pbrs_inner_m).clamp(
                    0, config.near_goal_pbrs_max_d_m))
        phi_before = _phi(torch.linalg.vector_norm(
            before_goal_safe - before_safe, dim=-1))
        phi_after_raw = _phi(distance)
        phi_after_effective = torch.where(
            terminated, torch.zeros_like(phi_after_raw), phi_after_raw)
        # PBRS: r_shape = gamma*Phi(next_effective) - Phi(current).  Approach to
        # the goal gives a positive term; a terminal flush of Phi(next_effective)
        # to 0 makes the term negative at arrival/failure by up to
        # -rate*max_d (e.g. -12.5), by construction.
        near_goal_cost = gamma * phi_after_effective - phi_before
    exponent = (completed_elapsed_steps - 1).clamp_min(0)
    failure_payment = -config.failure_cost / torch.pow(gamma, exponent)
    terminal_reward = torch.where(
        failure, failure_payment,
        torch.where(arrival, torch.full_like(rates, config.arrival_bonus),
                    torch.zeros_like(rates)))
    if config.mode in ('hold48_v1','hold36_continuous_v1','hold30_continuous_v1','hold50_continuous_v1','obstacle_course_continuous_v1'):
        terminal_reward = torch.where(failure, torch.full_like(rates,-config.failure_cost), torch.zeros_like(rates)) + config.arrival_bonus * first_reach.to(rates.dtype)
    reward = shaping - running_cost + terminal_reward - oob_cost + near_goal_cost
    if config.mode in ('hold48_v1','hold36_continuous_v1','hold30_continuous_v1','hold50_continuous_v1','obstacle_course_continuous_v1'):reward = reward + hold_reward
    if config.mode in ("zero_curriculum_v1", "hold48_v1", "hold36_continuous_v1", "hold30_continuous_v1", "hold50_continuous_v1", "obstacle_course_continuous_v1"):
        if cbf_reward is None or cbf_reward.shape != (e,) or not torch.isfinite(cbf_reward).all():
            raise ValueError("zero curriculum requires finite actual CBF reward [E]")
        reward = reward + cbf_reward
    return PlanarRewardStep(
        reward=reward, terminated=terminated,
        truncated=truncated, arrival=arrival,
        failure=failure, timeout=timeout, dwell_steps=dwell,
        potential_before=before_phi, potential_after=after_phi,
        shaping=shaping, running_cost=running_cost,
        terminal_reward=terminal_reward,
        failure_discount_exponent=exponent,
        time_cost=torch.full_like(rates,config.control_dt_s*config.time_rate),
        effort_cost=(torch.zeros_like(distance) if config.execution_smooth_rate else config.control_dt_s*config.effort_rate*nominal_safe.square().sum(-1)),
        brake_cost=config.control_dt_s*(config.goal_brake_rate if config.mode == "legacy_pbrs_v2" else 0.)*excess_speed.square(),
        lidar_cost=config.control_dt_s*config.lidar_risk_rate*lidar_risk,
        height_cost=config.control_dt_s*config.height_rate*height_error2,
        oob_cost=oob_cost,
        near_goal_pbrs_cost=near_goal_cost,
        near_goal_phi_before=phi_before,
        near_goal_phi_after_effective=phi_after_effective,
        effective_nonfinite=nonfinite,
        execution_smooth_cost=execution_smooth_cost,
        first_reach=first_reach,hold_reward=hold_reward,raw_in_goal=raw_in_goal,ever_reached=ever_reached)


__all__ = ["PlanarRewardConfig", "PlanarRewardStep", "planar_reward_step"]


def summarize_execution_smooth_batch(batch):
    """Observed NEXT transition scalars; pure reductions, no new samples/RNG."""
    result={}
    for name in ('execution_smooth_cost','execution_smooth_delta2','execution_smooth_valid','nominal_effort_cost','reward_reconstruction_error'):
        value=batch.get(('next','info','planar_'+name))
        if value is None or not bool(torch.isfinite(value).all()):
            raise RuntimeError('missing/nonfinite smooth rollout diagnostic '+name)
        result[name]={'sum':float(value.sum()),'max':float(value.max()),'nonzero':int((value!=0).sum()),'count':int(value.numel()),'first_transition_sum':float(value[:,0].sum()),'first_transition_nonzero':int((value[:,0]!=0).sum())}
    if result['nominal_effort_cost']['nonzero'] or result['reward_reconstruction_error']['max']>1e-5:
        raise RuntimeError('executed smooth reward decomposition failed')
    return result

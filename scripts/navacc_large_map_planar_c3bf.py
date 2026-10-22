#!/usr/bin/env python3
"""Pure Torch planar top-one C3BF action filter.

The policy action is the primary nominal acceleration.  Dynamic bodies are
eligible by XY centre distance only; optional static circles join the same
candidate table.  At most one positively violated lag-mapped row is projected.
Passing that selected row is deliberately not an all-obstacle certificate.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class PlanarC3BFConfig:
    normal_h_recovery_mode: str = "strict_abort"
    eligibility_distance_m: float = 3.0
    uav_radius_m: float = .10
    safety_margin_m: float = .05
    gamma_per_s: float = 1.2
    hocbf_lambda_per_s: float = 1.2
    minimum_relative_speed_m_s: float = 1e-4
    minimum_rho2_m2: float = 1e-4
    minimum_q_norm_m: float = 1e-8
    row_tolerance_m2_s2: float = 1e-6
    dt_s: float = .01
    motor_lag_tau_s: float = .05
    xy_axis_limit_m_s2: float = .8
    z_axis_limit_m_s2: float = .5
    total_limit_m_s2: float = 1.0

    def __post_init__(self) -> None:
        if self.normal_h_recovery_mode not in ("strict_abort", "project_feasible"):
            raise ValueError("normal_h_recovery_mode must be strict_abort or project_feasible")
        for name, value in self.__dict__.items():
            if name == "normal_h_recovery_mode":
                continue
            if type(value) not in (int, float) or not math.isfinite(float(value)) \
                    or float(value) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.z_axis_limit_m_s2 >= self.total_limit_m_s2:
            raise ValueError("z axis limit must leave a nonzero XY disk")


@dataclass(frozen=True)
class PlanarC3BFResult:
    acceleration: torch.Tensor
    nominal_acceleration: torch.Tensor
    cbf_abort: torch.Tensor
    domain_valid: torch.Tensor
    initial_g_safe: torch.Tensor
    initial_psi1_safe: torch.Tensor
    initial_c3bf_h_safe: torch.Tensor
    normal_h_recovery_attempted: torch.Tensor
    normal_h_recovery_accepted: torch.Tensor
    selected_row_feasible: torch.Tensor
    selected_row_pass: torch.Tensor
    action_bounds_pass: torch.Tensor
    selected_entity_id: torch.Tensor
    selected_is_static: torch.Tensor
    active_count: torch.Tensor
    selected_count: torch.Tensor
    normalized_nominal_violation: torch.Tensor
    selected_raw_residual: torch.Tensor
    selected_post_residual: torch.Tensor
    eligible_post_violation_count: torch.Tensor
    max_normalized_post_violation: torch.Tensor
    branch: torch.Tensor
    claim_all_rows_certified: torch.Tensor


def _validate_float(name: str, value: torch.Tensor, shape: tuple[int, ...]) -> None:
    if value.shape != shape or not value.is_floating_point():
        raise ValueError(f"{name} must be floating point {shape}, got {tuple(value.shape)}")


def bound_primary_acceleration(raw: torch.Tensor,
                               config: PlanarC3BFConfig) -> torch.Tensor:
    """Bound a finite primary action by axis and total acceleration limits."""
    if raw.ndim != 2 or raw.shape[-1] != 3 or not raw.is_floating_point():
        raise ValueError("raw action must be floating point [E,3]")
    if not bool(torch.isfinite(raw).all()):
        raise ValueError("raw primary action is non-finite")
    result = raw.clone()
    xy_norm = torch.linalg.vector_norm(result[:, :2], dim=-1)
    xy_scale = torch.minimum(
        torch.ones_like(xy_norm),
        torch.full_like(xy_norm, config.xy_axis_limit_m_s2)
        / xy_norm.clamp_min(torch.finfo(result.dtype).tiny))
    result[:, :2] *= xy_scale[:, None]
    result[:, 2].clamp_(-config.z_axis_limit_m_s2,
                        config.z_axis_limit_m_s2)
    norm = torch.linalg.vector_norm(result, dim=-1)
    scale = torch.minimum(torch.ones_like(norm),
                          torch.full_like(norm, config.total_limit_m_s2)
                          / norm.clamp_min(torch.finfo(result.dtype).tiny))
    return result * scale[:, None]


def _nearest_halfplane_disk(nominal_xy: torch.Tensor, q_xy: torch.Tensor,
                            rhs: torch.Tensor, radius: torch.Tensor,
                            tolerance: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Nearest point to nominal in ``q*x<=rhs`` intersected with an XY disk."""
    q2 = q_xy.square().sum(-1); qnorm = torch.sqrt(q2)
    tiny = torch.finfo(nominal_xy.dtype).eps
    degenerate = qnorm <= tiny
    already = (q_xy * nominal_xy).sum(-1) <= rhs + tolerance
    feasible = (~degenerate) & (rhs >= -radius * qnorm - tolerance)
    feasible |= degenerate & (rhs >= -tolerance)
    projected = nominal_xy - (((q_xy * nominal_xy).sum(-1) - rhs)
                              / q2.clamp_min(tiny))[:, None] * q_xy
    projected_inside = torch.linalg.vector_norm(projected, dim=-1) <= radius + tolerance

    # When orthogonal projection lies outside the disk, the closest feasible
    # point is one of the two line-circle intersections.  Keep nominal stable
    # by choosing the smaller squared distance; the minus branch wins ties.
    unit = q_xy / qnorm.clamp_min(tiny)[:, None]
    tangent = torch.stack((-unit[:, 1], unit[:, 0]), -1)
    along = rhs / qnorm.clamp_min(tiny)
    span = (radius.square() - along.square()).clamp_min(0).sqrt()
    base = along[:, None] * unit
    minus, plus = base - span[:, None] * tangent, base + span[:, None] * tangent
    use_plus = ((plus - nominal_xy).square().sum(-1)
                < (minus - nominal_xy).square().sum(-1))
    intersection = torch.where(use_plus[:, None], plus, minus)
    corrected = torch.where(projected_inside[:, None], projected, intersection)
    output = torch.where(already[:, None], nominal_xy, corrected)
    output = torch.where(degenerate[:, None], nominal_xy, output)
    output = torch.where(feasible[:, None], output,
                         torch.full_like(output, float("nan")))
    return output, feasible


@torch.no_grad()
def filter_planar_top1(
    position: torch.Tensor,
    velocity: torch.Tensor,
    measured_acceleration: torch.Tensor,
    nominal_acceleration: torch.Tensor,
    dynamic_position: torch.Tensor,
    dynamic_velocity: torch.Tensor,
    dynamic_radius: torch.Tensor,
    *,
    static_position: torch.Tensor | None = None,
    static_radius: torch.Tensor | None = None,
    static_mode: str = "lidar144",
    config: PlanarC3BFConfig = PlanarC3BFConfig(),
) -> PlanarC3BFResult:
    """Filter one batch using one globally selected lag-mapped planar row.

    Dynamic truth is shared: positions/velocities are ``[N,3]``.  Static C3BF
    mode accepts conservative XY circles ``[S,2]`` and stationary velocity.
    Unknown positions fail closed.  An invalid velocity outside the 3 m active
    set is ignored; the same value inside the active set invalidates the lane.
    """
    if static_mode not in ("lidar144", "static_c3bf"):
        raise ValueError("static_mode must be lidar144 or static_c3bf")
    if position.ndim != 2 or position.shape[-1] != 3:
        raise ValueError("position must be [E,3]")
    e = position.shape[0]; n = dynamic_position.shape[0]
    for name, value in (("position", position), ("velocity", velocity),
                        ("measured_acceleration", measured_acceleration),
                        ("nominal_acceleration", nominal_acceleration)):
        _validate_float(name, value, (e, 3))
    _validate_float("dynamic_position", dynamic_position, (n, 3))
    _validate_float("dynamic_velocity", dynamic_velocity, (n, 3))
    _validate_float("dynamic_radius", dynamic_radius, (n,))
    tensors = (velocity, measured_acceleration, nominal_acceleration,
               dynamic_position, dynamic_velocity, dynamic_radius)
    if any(value.device != position.device or value.dtype != position.dtype
           for value in tensors):
        raise ValueError("all C3BF tensors must share device and dtype")
    if not bool(torch.isfinite(position).all()) \
            or not bool(torch.isfinite(measured_acceleration).all()) \
            or not bool(torch.isfinite(nominal_acceleration).all()) \
            or not bool(torch.isfinite(dynamic_radius).all()) \
            or bool((dynamic_radius <= 0).any()):
        raise ValueError("C3BF position, radius or acceleration input is invalid")

    dynamic_position_known = torch.isfinite(dynamic_position).all(-1)
    body_position = torch.where(
        dynamic_position_known[:, None], dynamic_position,
        torch.zeros_like(dynamic_position))[:, :2]
    body_velocity = dynamic_velocity[:, :2]
    body_radius = dynamic_radius
    body_id = torch.arange(n, device=position.device, dtype=torch.long)
    is_static = torch.zeros(n, device=position.device, dtype=torch.bool)
    if static_mode == "static_c3bf":
        if static_position is None or static_radius is None:
            raise ValueError("static_c3bf requires static positions and radii")
        s = static_position.shape[0]
        _validate_float("static_position", static_position, (s, 2))
        _validate_float("static_radius", static_radius, (s,))
        if (static_position.device != position.device
                or static_position.dtype != position.dtype
                or static_radius.device != position.device
                or static_radius.dtype != position.dtype
                or not bool(torch.isfinite(static_position).all())
                or not bool(torch.isfinite(static_radius).all())
                or bool((static_radius <= 0).any())):
            raise ValueError("static C3BF geometry is invalid")
        body_position = torch.cat((body_position, static_position), 0)
        body_velocity = torch.cat((body_velocity,
                                   torch.zeros_like(static_position)), 0)
        body_radius = torch.cat((body_radius, static_radius), 0)
        body_id = torch.cat((body_id, torch.arange(
            n, n + s, device=position.device, dtype=torch.long)), 0)
        is_static = torch.cat((is_static, torch.ones(
            s, device=position.device, dtype=torch.bool)), 0)
        position_known = torch.cat((dynamic_position_known, torch.ones(
            s, device=position.device, dtype=torch.bool)), 0)
    else:
        position_known = dynamic_position_known

    p = body_position[None] - position[:, None, :2]
    w = body_velocity[None] - velocity[:, None, :2]
    distance = torch.linalg.vector_norm(p, dim=-1)
    # An unknown position cannot be proven outside the 3 m eligibility set.
    # It is made active and invalid for every lane so the result fails closed.
    active = (distance <= config.eligibility_distance_m) | ~position_known[None]
    finite_velocity = torch.isfinite(body_velocity).all(-1)[None].expand(e, -1)
    lane_domain = ((~active) | (finite_velocity & position_known[None])).all(-1) \
        & torch.isfinite(velocity).all(-1)
    safe_w = torch.where(torch.isfinite(w), w, torch.zeros_like(w))
    speed = torch.linalg.vector_norm(safe_w, dim=-1)
    radius = (body_radius[None] + config.uav_radius_m
              + config.safety_margin_m)
    rho2 = p.square().sum(-1) - radius.square()
    low = speed <= config.minimum_relative_speed_m_s
    normal_valid = (rho2 > config.minimum_rho2_m2)
    rho = rho2.clamp_min(0).sqrt()
    q_normal = p + rho[..., None] * safe_w \
        / speed.clamp_min(config.minimum_relative_speed_m_s)[..., None]
    h_normal = (p * safe_w).sum(-1) + rho * speed
    b_normal = (speed / rho.clamp_min(torch.finfo(position.dtype).tiny)
                + config.gamma_per_s) * h_normal

    g = rho2
    psi1 = 2 * (p * safe_w).sum(-1) + config.hocbf_lambda_per_s * g
    q_low = p
    b_low = (safe_w.square().sum(-1)
             + 2 * config.hocbf_lambda_per_s * (p * safe_w).sum(-1)
             + .5 * config.hocbf_lambda_per_s ** 2 * g)
    q = torch.where(low[..., None], q_low, q_normal)
    b = torch.where(low, b_low, b_normal)
    h = torch.where(low, psi1, h_normal)
    qnorm = torch.linalg.vector_norm(q, dim=-1)
    branch_valid = torch.where(low, g >= 0,
                               normal_valid & (qnorm >= config.minimum_q_norm_m))
    row_valid = active & finite_velocity & position_known[None] & branch_valid \
        & torch.isfinite(q).all(-1) & torch.isfinite(b) & torch.isfinite(h)
    lane_domain &= ((~active) | row_valid).all(-1)
    low_active = active & low
    normal_active = active & ~low
    initial_g_safe = ((~low_active) | (g >= 0)).all(-1)
    initial_psi_safe = ((~low_active) | (psi1 >= 0)).all(-1)
    initial_h_safe = ((~normal_active) | (h_normal >= 0)).all(-1)

    alpha = math.exp(-config.dt_s / config.motor_lag_tau_s)
    q_lag = (1 - alpha) * q
    b_lag = b - alpha * (q * measured_acceleration[:, None, :2]).sum(-1)
    residual = ((q_lag * nominal_acceleration[:, None, :2]).sum(-1) - b_lag)
    score = residual.clamp_min(0) / torch.linalg.vector_norm(
        q_lag, dim=-1).clamp_min(torch.finfo(position.dtype).tiny)
    score = torch.where(row_valid, score, torch.full_like(score, -torch.inf))
    maximum = score.max(-1).values if score.shape[1] else torch.full(
        (e,), -torch.inf, device=position.device, dtype=position.dtype)
    recovery_enabled = config.normal_h_recovery_mode == "project_feasible"
    base_initial_safe = initial_g_safe & initial_psi_safe
    recovery_attempted = (base_initial_safe & lane_domain & ~initial_h_safe
                          if recovery_enabled else torch.zeros_like(lane_domain))
    selection_initial_safe = base_initial_safe & (
        initial_h_safe | recovery_attempted)
    has_positive = lane_domain & selection_initial_safe \
        & torch.isfinite(maximum) & (maximum > 0)
    if score.shape[1]:
        tie = row_valid & (score == maximum[:, None]) & has_positive[:, None]
        sentinel = torch.full_like(body_id[None].expand(e, -1),
                                   torch.iinfo(torch.long).max)
        selected_id = torch.where(tie, body_id[None], sentinel).min(-1).values
        selected_slot = (tie & (body_id[None] == selected_id[:, None])).long().argmax(-1)
        gather = selected_slot[:, None]
        selected_q = q_lag.gather(1, gather[..., None].expand(-1, 1, 2)).squeeze(1)
        selected_b = b_lag.gather(1, gather).squeeze(1)
        selected_residual = residual.gather(1, gather).squeeze(1)
        selected_static = is_static[selected_slot]
    else:
        selected_id = torch.full((e,), -1, device=position.device, dtype=torch.long)
        selected_q = torch.zeros(e, 2, device=position.device, dtype=position.dtype)
        selected_b = torch.zeros(e, device=position.device, dtype=position.dtype)
        selected_residual = torch.zeros_like(selected_b)
        selected_static = torch.zeros(e, device=position.device, dtype=torch.bool)
    selected_id = torch.where(has_positive, selected_id,
                              torch.full_like(selected_id, -1))
    selected_static &= has_positive

    uz = nominal_acceleration[:, 2]
    xy_radius = torch.minimum(
        torch.full_like(uz, config.xy_axis_limit_m_s2),
        (config.total_limit_m_s2 ** 2 - uz.square()).clamp_min(0).sqrt())
    projected_xy, feasible = _nearest_halfplane_disk(
        nominal_acceleration[:, :2], selected_q, selected_b, xy_radius,
        config.row_tolerance_m2_s2)
    feasible = lane_domain & selection_initial_safe & ((~has_positive) | feasible)
    candidate_xy = torch.where(has_positive[:, None], projected_xy,
                               nominal_acceleration[:, :2])
    candidate = torch.cat((candidate_xy, uz[:, None]), -1)
    selected_after = (selected_q * candidate_xy).sum(-1) - selected_b
    bounds_pass = (torch.linalg.vector_norm(candidate_xy, dim=-1)
                   <= xy_radius + config.row_tolerance_m2_s2) \
        & (uz.abs() <= config.z_axis_limit_m_s2 + config.row_tolerance_m2_s2) \
        & (torch.linalg.vector_norm(candidate, dim=-1)
           <= config.total_limit_m_s2 + config.row_tolerance_m2_s2)
    selected_pass = lane_domain & selection_initial_safe & (
        (~has_positive) | (selected_after <= config.row_tolerance_m2_s2))
    # ``accepted`` covers both a projected recovery and the idempotent case in
    # which the bounded nominal already satisfies every eligible lag row.  The
    # latter has selected_count==0; raw initial_h_safe remains false and this
    # result still makes no all-row certificate claim.
    recovery_accepted = recovery_attempted & feasible & selected_pass \
        & bounds_pass & torch.isfinite(candidate).all(-1)
    abort = (~lane_domain) | (~initial_g_safe) | (~initial_psi_safe) \
        | ((~initial_h_safe) & (~recovery_accepted)) \
        | (~feasible) | (~selected_pass) | (~bounds_pass) \
        | (~torch.isfinite(candidate).all(-1))
    first_actual = alpha * measured_acceleration[:, :2] \
        + (1 - alpha) * candidate_xy
    post_residual = (q * first_actual[:, None]).sum(-1) - b
    post_normalized = post_residual.clamp_min(0) \
        / qnorm.clamp_min(torch.finfo(position.dtype).tiny)
    post_violated = row_valid & (
        post_residual > config.row_tolerance_m2_s2)
    post_count = post_violated.sum(-1)
    post_max = (torch.where(row_valid, post_normalized,
                           torch.full_like(post_normalized, -torch.inf)).max(-1).values
                if row_valid.shape[1] else torch.zeros(e,device=position.device,dtype=position.dtype))
    post_max = torch.where(row_valid.any(-1), post_max,
                           torch.zeros_like(post_max))
    post_valid = lane_domain & selection_initial_safe & feasible \
        & torch.isfinite(candidate).all(-1)
    post_count = torch.where(post_valid, post_count,
                             torch.full_like(post_count, -1))
    post_max = torch.where(post_valid, post_max,
                           torch.full_like(post_max, float("nan")))
    acceleration = torch.where(abort[:, None],
                               torch.full_like(candidate, float("nan")), candidate)
    return PlanarC3BFResult(
        acceleration=acceleration, nominal_acceleration=nominal_acceleration,
        cbf_abort=abort, domain_valid=lane_domain,
        initial_g_safe=initial_g_safe, initial_psi1_safe=initial_psi_safe,
        initial_c3bf_h_safe=initial_h_safe,
        normal_h_recovery_attempted=recovery_attempted,
        normal_h_recovery_accepted=recovery_accepted,
        selected_row_feasible=feasible, selected_row_pass=selected_pass,
        action_bounds_pass=bounds_pass,
        selected_entity_id=selected_id, selected_is_static=selected_static,
        active_count=active.sum(-1), selected_count=has_positive.long(),
        normalized_nominal_violation=torch.where(
            has_positive, maximum, torch.zeros_like(maximum)),
        selected_raw_residual=torch.where(
            has_positive, selected_residual, torch.zeros_like(selected_residual)),
        selected_post_residual=torch.where(
            has_positive & post_valid, selected_after,
            torch.where(post_valid, torch.zeros_like(selected_after),
                        torch.full_like(selected_after, float("nan")))),
        eligible_post_violation_count=post_count,
        max_normalized_post_violation=post_max,
        branch=torch.where(low, torch.ones_like(low, dtype=torch.long),
                           torch.zeros_like(low, dtype=torch.long)),
        claim_all_rows_certified=torch.zeros(e, device=position.device,
                                              dtype=torch.bool))


__all__ = ["PlanarC3BFConfig", "PlanarC3BFResult",
           "bound_primary_acceleration", "filter_planar_top1"]

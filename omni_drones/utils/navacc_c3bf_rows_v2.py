"""Versioned, simulator-independent C3BF row construction for NavAcc.

The module keeps inactive slots distinct from invalid active estimates.  Every
active slot must be valid for ``domain_valid`` to hold; an invalid slot is never
dropped from the constraint set.  All tensors stay on their input device and
the implementation has no per-environment Python loop or host synchronization.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math

import torch


VERSION = "navacc_c3bf_rows_v2"


class RowReason(IntEnum):
    INACTIVE = 0
    VALID = 1
    INVALID_ESTIMATE = 2
    STALE_ESTIMATE = 3
    NONFINITE_STATE = 4
    INSIDE_INFLATED_RADIUS = 5
    RHO_BOUNDARY = 6
    NEAR_RHO = 7
    LOW_RELATIVE_SPEED = 8
    DEGENERATE_Q = 9


@dataclass(frozen=True)
class C3BFRowConfig:
    """Parameters shared by all v2 solver arms (SI units)."""

    inflated_radius_m: float = 0.5
    gamma_per_s: float = 1.0
    epsilon_obs_m_s2: float = 0.0
    epsilon_track_m_s2: float = 0.0
    min_speed_m_s: float = 1.0e-4
    min_rho2_m2: float = 1.0e-4
    min_q_m: float = 0.01
    stale_after_s: float = 0.25

    def __post_init__(self) -> None:
        for name in ("inflated_radius_m", "gamma_per_s", "min_speed_m_s",
                     "min_rho2_m2", "min_q_m", "stale_after_s"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("epsilon_obs_m_s2", "epsilon_track_m_s2"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")


@dataclass(frozen=True)
class C3BFRows:
    version: str
    h: torch.Tensor                 # [E,N], NaN outside valid rows
    q: torch.Tensor                 # [E,N,2], NaN outside valid rows
    b: torch.Tensor                 # [E,N], NaN outside valid rows
    row_valid: torch.Tensor         # bool [E,N]
    reason: torch.Tensor            # int64 [E,N], RowReason
    domain_valid: torch.Tensor      # bool [E]
    active: torch.Tensor            # bool [E,N]
    entity_id: torch.Tensor         # int64 [E,N], stable scene IDs
    rho2: torch.Tensor              # [E,N], diagnostic
    speed: torch.Tensor             # [E,N], diagnostic
    q_norm: torch.Tensor            # [E,N], diagnostic
    gamma_per_s: float


def _validate_common(q_like: torch.Tensor, active: torch.Tensor,
                     entity_id: torch.Tensor) -> None:
    if q_like.ndim != 3 or q_like.shape[-1] != 2:
        raise ValueError("row vectors must have shape [E,N,2]")
    if not q_like.is_floating_point():
        raise TypeError("row tensors must be floating point")
    if active.shape != q_like.shape[:2] or active.dtype != torch.bool:
        raise ValueError("active must be bool [E,N]")
    if entity_id.shape != q_like.shape[:2] or entity_id.dtype not in (torch.int32, torch.int64):
        raise ValueError("entity_id must be an integer [E,N] tensor")
    if active.device != q_like.device or entity_id.device != q_like.device:
        raise ValueError("row tensors must share one device")


def _default_entity_id(e: int, n: int, device: torch.device) -> torch.Tensor:
    return torch.arange(n, dtype=torch.long, device=device).expand(e, n)


@torch.no_grad()
def build_c3bf_rows_v2(
    p_xy: torch.Tensor,
    w_xy: torch.Tensor,
    a_obs_hat_xy: torch.Tensor,
    active: torch.Tensor,
    estimate_valid: torch.Tensor,
    age_s: torch.Tensor,
    config: C3BFRowConfig = C3BFRowConfig(),
    *,
    entity_id: torch.Tensor | None = None,
    inflated_radius_m: torch.Tensor | None = None,
) -> C3BFRows:
    """Construct ``q.u_xy <= b`` rows from one decision-time snapshot.

    ``p = p_obs-p_uav`` and ``w = v_obs-v_uav``.  Invalid active rows make the
    entire environment domain invalid and their mathematical outputs are NaN.
    """
    if p_xy.ndim != 3 or p_xy.shape[-1] != 2:
        raise ValueError("p_xy must have shape [E,N,2]")
    if w_xy.shape != p_xy.shape or a_obs_hat_xy.shape != p_xy.shape:
        raise ValueError("p_xy, w_xy, and a_obs_hat_xy must have equal shapes")
    e, n = p_xy.shape[:2]
    if active.shape != (e, n) or estimate_valid.shape != (e, n) or age_s.shape != (e, n):
        raise ValueError("slot metadata must have shape [E,N]")
    if active.dtype != torch.bool or estimate_valid.dtype != torch.bool:
        raise TypeError("active and estimate_valid must be bool")
    if not p_xy.is_floating_point() or w_xy.dtype != p_xy.dtype:
        raise TypeError("p_xy and w_xy must share a floating dtype")
    if a_obs_hat_xy.dtype != p_xy.dtype or age_s.dtype != p_xy.dtype:
        raise TypeError("state, acceleration estimate, and age must share a floating dtype")
    if any(x.device != p_xy.device for x in
           (w_xy, a_obs_hat_xy, active, estimate_valid, age_s)):
        raise ValueError("row-builder inputs must share one device")
    ids = (_default_entity_id(e, n, p_xy.device) if entity_id is None else entity_id)
    _validate_common(p_xy, active, ids)

    finite_state = (torch.isfinite(p_xy).all(-1) & torch.isfinite(w_xy).all(-1)
                    & torch.isfinite(a_obs_hat_xy).all(-1) & torch.isfinite(age_s))
    safe_p = torch.where(torch.isfinite(p_xy), p_xy, torch.zeros_like(p_xy))
    safe_w = torch.where(torch.isfinite(w_xy), w_xy, torch.zeros_like(w_xy))
    safe_a_obs = torch.where(torch.isfinite(a_obs_hat_xy), a_obs_hat_xy,
                             torch.zeros_like(a_obs_hat_xy))
    speed = torch.linalg.vector_norm(safe_w, dim=-1)
    radius = float(config.inflated_radius_m)
    if inflated_radius_m is not None:
        if inflated_radius_m.shape != (e, n) or inflated_radius_m.device != p_xy.device or inflated_radius_m.dtype != p_xy.dtype or not torch.isfinite(inflated_radius_m).all() or (inflated_radius_m <= 0).any():
            raise ValueError("inflated radius must be positive finite [E,N] matching rows")
        radius = inflated_radius_m
    rho2 = safe_p.square().sum(-1) - radius ** 2
    safe_speed = torch.where(speed > config.min_speed_m_s, speed, torch.ones_like(speed))
    safe_rho = torch.sqrt(torch.clamp_min(rho2, 0.0))
    q_calc = safe_p + safe_rho.unsqueeze(-1) * safe_w / safe_speed.unsqueeze(-1)
    q_norm = torch.linalg.vector_norm(q_calc, dim=-1)

    reason = torch.full((e, n), int(RowReason.VALID), dtype=torch.long, device=p_xy.device)
    reason = torch.where(q_norm <= config.min_q_m,
                         torch.full_like(reason, int(RowReason.DEGENERATE_Q)), reason)
    reason = torch.where(speed <= config.min_speed_m_s,
                         torch.full_like(reason, int(RowReason.LOW_RELATIVE_SPEED)), reason)
    reason = torch.where((rho2 > 0.0) & (rho2 <= config.min_rho2_m2),
                         torch.full_like(reason, int(RowReason.NEAR_RHO)), reason)
    reason = torch.where(rho2 == 0.0,
                         torch.full_like(reason, int(RowReason.RHO_BOUNDARY)), reason)
    reason = torch.where(rho2 < 0.0,
                         torch.full_like(reason, int(RowReason.INSIDE_INFLATED_RADIUS)), reason)
    reason = torch.where(~finite_state,
                         torch.full_like(reason, int(RowReason.NONFINITE_STATE)), reason)
    reason = torch.where(age_s > config.stale_after_s,
                         torch.full_like(reason, int(RowReason.STALE_ESTIMATE)), reason)
    reason = torch.where((age_s < 0.0) | ~estimate_valid,
                         torch.full_like(reason, int(RowReason.INVALID_ESTIMATE)), reason)
    reason = torch.where(~active,
                         torch.full_like(reason, int(RowReason.INACTIVE)), reason)
    row_valid = active & (reason == int(RowReason.VALID))
    domain_valid = ((~active) | row_valid).all(-1)

    pw = (safe_p * safe_w).sum(-1)
    h_calc = pw + safe_rho * speed
    uncertainty = float(config.epsilon_obs_m_s2 + config.epsilon_track_m_s2)
    b_calc = ((speed / torch.where(safe_rho > 0.0, safe_rho, torch.ones_like(safe_rho))
               + float(config.gamma_per_s)) * h_calc
              + (q_calc * safe_a_obs).sum(-1) - q_norm * uncertainty)
    nan_scalar = torch.full_like(h_calc, float("nan"))
    h = torch.where(row_valid, h_calc, nan_scalar)
    q = torch.where(row_valid.unsqueeze(-1), q_calc, torch.full_like(q_calc, float("nan")))
    b = torch.where(row_valid, b_calc, nan_scalar)
    return C3BFRows(
        version=VERSION, h=h, q=q, b=b, row_valid=row_valid, reason=reason,
        domain_valid=domain_valid, active=active, entity_id=ids.to(torch.long),
        rho2=rho2, speed=speed, q_norm=q_norm, gamma_per_s=float(config.gamma_per_s))


@torch.no_grad()
def build_halfplane_rows_v2(
    q: torch.Tensor,
    b: torch.Tensor,
    *,
    h: torch.Tensor | None = None,
    active: torch.Tensor | None = None,
    entity_id: torch.Tensor | None = None,
    gamma_per_s: float = 1.0,
    min_q_m: float = 0.01,
) -> C3BFRows:
    """Build the shared row container from explicit half-planes for tests/replay."""
    if q.ndim != 3 or q.shape[-1] != 2 or b.shape != q.shape[:2]:
        raise ValueError("expected q [E,N,2] and b [E,N]")
    if not q.is_floating_point() or b.dtype != q.dtype:
        raise TypeError("q and b must share a floating dtype")
    e, n = q.shape[:2]
    active_t = (torch.ones((e, n), dtype=torch.bool, device=q.device)
                if active is None else active)
    ids = (_default_entity_id(e, n, q.device) if entity_id is None else entity_id)
    _validate_common(q, active_t, ids)
    if b.device != q.device:
        raise ValueError("q and b must share one device")
    h_t = torch.zeros_like(b) if h is None else h
    if h_t.shape != b.shape or h_t.dtype != q.dtype or h_t.device != q.device:
        raise ValueError("h must match b shape, dtype, and device")
    if not math.isfinite(float(gamma_per_s)) or float(gamma_per_s) <= 0.0:
        raise ValueError("gamma_per_s must be finite and positive")
    if not math.isfinite(float(min_q_m)) or float(min_q_m) <= 0.0:
        raise ValueError("min_q_m must be finite and positive")

    q_norm = torch.linalg.vector_norm(torch.where(torch.isfinite(q), q, torch.zeros_like(q)), dim=-1)
    finite = torch.isfinite(q).all(-1) & torch.isfinite(b) & torch.isfinite(h_t)
    reason = torch.full((e, n), int(RowReason.VALID), dtype=torch.long, device=q.device)
    reason = torch.where(q_norm <= min_q_m,
                         torch.full_like(reason, int(RowReason.DEGENERATE_Q)), reason)
    reason = torch.where(~finite,
                         torch.full_like(reason, int(RowReason.NONFINITE_STATE)), reason)
    reason = torch.where(~active_t,
                         torch.full_like(reason, int(RowReason.INACTIVE)), reason)
    row_valid = active_t & (reason == int(RowReason.VALID))
    domain_valid = ((~active_t) | row_valid).all(-1)
    nan_h = torch.full_like(h_t, float("nan"))
    return C3BFRows(
        version=VERSION, h=torch.where(row_valid, h_t, nan_h),
        q=torch.where(row_valid.unsqueeze(-1), q, torch.full_like(q, float("nan"))),
        b=torch.where(row_valid, b, torch.full_like(b, float("nan"))),
        row_valid=row_valid, reason=reason, domain_valid=domain_valid,
        active=active_t, entity_id=ids.to(torch.long),
        rho2=torch.full_like(b, float("nan")), speed=torch.full_like(b, float("nan")),
        q_norm=q_norm, gamma_per_s=float(gamma_per_s))

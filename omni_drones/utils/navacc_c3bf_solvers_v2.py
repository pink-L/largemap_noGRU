"""Pure Torch NavAcc C3BF v2 action map and analytic solver candidates."""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math
from typing import Sequence

import torch

try:  # Supports both package import and the repository's direct-file tests.
    from .navacc_c3bf_rows_v2 import C3BFRows
except ImportError:  # pragma: no cover - exercised by direct-file tests
    from navacc_c3bf_rows_v2 import C3BFRows


NOMINAL_ACTION_VERSION = "navacc_nominal_ball_v2"
SOLVER_VERSION = "navacc_c3bf_solvers_v2"
SOLVER_METHODS = (
    "top1_violation_closed_form",
    "softmin_closed_form",
    "all_halfplanes_candidates",
)


class SolverStatus(IntEnum):
    SUCCESS = 0
    INVALID_NOMINAL = 1
    INVALID_DOMAIN = 2
    DEGENERATE_METHOD = 3
    NO_VERIFIED_CANDIDATE = 4
    NUMERICAL_FAILURE = 5


@dataclass(frozen=True)
class NominalActionV2:
    version: str
    z_policy: torch.Tensor
    u_raw: torch.Tensor
    u_nom: torch.Tensor
    finite: torch.Tensor
    limited: torch.Tensor


@dataclass(frozen=True)
class SolverConfigV2:
    row_tolerance_m2_s2: float = 1.0e-6
    softmin_beta: float = 1.0
    aggregate_q_min_m: float = 1.0e-8
    parallel_relative_tolerance: float = 1.0e-7
    candidate_chunk_size: int = 64
    row_chunk_size: int = 64
    top1_projection_float64: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.top1_projection_float64, bool):
            raise ValueError("top1_projection_float64 must be boolean")
        for name in ("row_tolerance_m2_s2", "softmin_beta", "aggregate_q_min_m",
                     "parallel_relative_tolerance"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("candidate_chunk_size", "row_chunk_size"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class SolverResultV2:
    version: str
    method: str
    u_filtered: torch.Tensor
    correction: torch.Tensor
    objective: torch.Tensor
    output_defined: torch.Tensor
    method_constraint_pass: torch.Tensor
    all_rows_pass: torch.Tensor
    finite_output: torch.Tensor
    status: torch.Tensor
    selected_id: torch.Tensor
    raw_residual_max: torch.Tensor
    normalized_residual_max: torch.Tensor
    candidate_allocated: torch.Tensor
    candidate_generated: torch.Tensor
    candidate_usable: torch.Tensor
    projection_rounds: torch.Tensor
    native_margin_nominal: torch.Tensor
    aggregate_h: torch.Tensor
    aggregate_q: torch.Tensor
    aggregate_b: torch.Tensor


def _as_scale(scale: float | Sequence[float] | torch.Tensor,
              reference: torch.Tensor) -> torch.Tensor:
    if not isinstance(scale, torch.Tensor):
        values = (float(scale),) if isinstance(scale, (int, float)) else tuple(float(x) for x in scale)
        if len(values) not in (1, 3) or any(not math.isfinite(x) or x <= 0.0 for x in values):
            raise ValueError("scale must be finite and positive")
    scale_t = torch.as_tensor(scale, dtype=reference.dtype, device=reference.device)
    if scale_t.ndim == 0:
        return scale_t
    if scale_t.shape != (3,):
        raise ValueError("scale must be a scalar or length-three value")
    return scale_t


@torch.no_grad()
def map_policy_action_v2(
    z_policy: torch.Tensor,
    *,
    scale: float | Sequence[float] | torch.Tensor = 2.0,
    max_norm: float = 2.0,
) -> NominalActionV2:
    """Map latent policy action to a finite, three-dimensional nominal ball.

    ``z_policy`` is retained verbatim for PPO log-probability bookkeeping.
    A non-finite latent produces an undefined all-NaN nominal row; it is never
    silently repaired into a normal action.
    """
    if z_policy.ndim != 2 or z_policy.shape[-1] != 3 or not z_policy.is_floating_point():
        raise ValueError("z_policy must be floating point [E,3]")
    if not math.isfinite(float(max_norm)) or float(max_norm) <= 0.0:
        raise ValueError("max_norm must be finite and positive")
    scale_t = _as_scale(scale, z_policy)
    u_raw = torch.tanh(z_policy) * scale_t
    finite = torch.isfinite(z_policy).all(-1) & torch.isfinite(u_raw).all(-1)
    safe_raw = torch.where(finite.unsqueeze(-1), u_raw, torch.zeros_like(u_raw))
    raw_norm = torch.linalg.vector_norm(safe_raw, dim=-1, keepdim=True)
    factor = torch.clamp(float(max_norm) / raw_norm.clamp_min(torch.finfo(z_policy.dtype).tiny), max=1.0)
    projected = safe_raw * factor
    u_nom = torch.where(finite.unsqueeze(-1), projected,
                        torch.full_like(projected, float("nan")))
    limited = finite & (raw_norm.squeeze(-1) > float(max_norm))
    return NominalActionV2(NOMINAL_ACTION_VERSION, z_policy, u_raw, u_nom, finite, limited)


def _validate_inputs(u_nom: torch.Tensor, rows: C3BFRows) -> tuple[int, int]:
    if u_nom.ndim != 2 or u_nom.shape[-1] != 3 or not u_nom.is_floating_point():
        raise ValueError("u_nom must be floating point [E,3]")
    e = u_nom.shape[0]
    if rows.q.ndim != 3 or rows.q.shape[0] != e or rows.q.shape[-1] != 2:
        raise ValueError("rows.q must be [E,N,2]")
    n = rows.q.shape[1]
    expected = (e, n)
    if (rows.h.shape != expected or rows.b.shape != expected
            or rows.row_valid.shape != expected or rows.reason.shape != expected
            or rows.active.shape != expected or rows.entity_id.shape != expected
            or rows.domain_valid.shape != (e,)):
        raise ValueError("row container shapes are inconsistent")
    if rows.q.dtype != u_nom.dtype or rows.h.dtype != u_nom.dtype or rows.b.dtype != u_nom.dtype:
        raise TypeError("u_nom and row values must share a floating dtype")
    if any(x.device != u_nom.device for x in
           (rows.q, rows.h, rows.b, rows.row_valid, rows.reason, rows.active,
            rows.entity_id, rows.domain_valid)):
        raise ValueError("u_nom and rows must share one device")
    return e, n


def _safe_rows(rows: C3BFRows) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q = torch.where(rows.row_valid.unsqueeze(-1), rows.q, torch.zeros_like(rows.q))
    b = torch.where(rows.row_valid, rows.b, torch.zeros_like(rows.b))
    h = torch.where(rows.row_valid, rows.h, torch.zeros_like(rows.h))
    return q, b, h


def _residual_metrics(candidate: torch.Tensor, rows: C3BFRows,
                      output_defined: torch.Tensor, tolerance: float
                      ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    e, n = rows.active.shape
    if n == 0:
        zero = torch.zeros(e, dtype=candidate.dtype, device=candidate.device)
        passes = output_defined & rows.domain_valid
        return zero, zero, passes
    q, b, _ = _safe_rows(rows)
    residual = (q * candidate[:, None, :2]).sum(-1) - b
    neg_inf = torch.full_like(residual, -float("inf"))
    raw_max = torch.where(rows.active, residual, neg_inf).max(-1).values
    has_active = rows.active.any(-1)
    raw_max = torch.where(has_active, raw_max, torch.zeros_like(raw_max))
    q_norm = torch.linalg.vector_norm(q, dim=-1)
    normalized = torch.where(
        rows.active,
        residual.clamp_min(0.0) / q_norm.clamp_min(torch.finfo(candidate.dtype).tiny),
        torch.zeros_like(residual),
    ).max(-1).values
    known = output_defined & rows.domain_valid & torch.isfinite(candidate).all(-1)
    nan = torch.full_like(raw_max, float("nan"))
    raw_max = torch.where(known, raw_max, nan)
    normalized = torch.where(known, normalized, nan)
    passes = known & (raw_max <= float(tolerance))
    return raw_max, normalized, passes


def _common_nominal_margin(u_nom: torch.Tensor, rows: C3BFRows,
                           nominal_finite: torch.Tensor) -> torch.Tensor:
    e, n = rows.active.shape
    if n == 0:
        return torch.zeros(e, dtype=u_nom.dtype, device=u_nom.device)
    q, b, _ = _safe_rows(rows)
    margin = b - (q * u_nom[:, None, :2]).sum(-1)
    inf = torch.full_like(margin, float("inf"))
    minimum = torch.where(rows.active, margin, inf).min(-1).values
    minimum = torch.where(rows.active.any(-1), minimum, torch.zeros_like(minimum))
    valid = rows.domain_valid & nominal_finite
    return torch.where(valid, minimum.clamp_max(0.0),
                       torch.full_like(minimum, float("nan")))


def _top1(u_nom: torch.Tensor, rows: C3BFRows, config: SolverConfigV2,
          base_valid: torch.Tensor):
    e, n = rows.active.shape
    selected_id = torch.full((e,), -1, dtype=torch.long, device=u_nom.device)
    if n == 0:
        return (u_nom, base_valid, base_valid, selected_id,
                torch.zeros(e, dtype=torch.long, device=u_nom.device),
                torch.zeros(e, dtype=u_nom.dtype, device=u_nom.device),
                torch.full((e,), float("nan"), dtype=u_nom.dtype, device=u_nom.device),
                torch.full((e, 2), float("nan"), dtype=u_nom.dtype, device=u_nom.device),
                torch.full((e,), float("nan"), dtype=u_nom.dtype, device=u_nom.device))
    q, b, _ = _safe_rows(rows)
    residual = (q * u_nom[:, None, :2]).sum(-1) - b
    q_norm = torch.linalg.vector_norm(q, dim=-1)
    score = residual.clamp_min(0.0) / q_norm.clamp_min(torch.finfo(u_nom.dtype).tiny)
    score = torch.where(rows.active, score, torch.full_like(score, -float("inf")))
    max_score = score.max(-1).values
    has_violation = base_valid & rows.active.any(-1) & (max_score > 0.0)
    tied = rows.active & (score == max_score.unsqueeze(-1)) & has_violation.unsqueeze(-1)
    id_sentinel = torch.full_like(rows.entity_id, torch.iinfo(rows.entity_id.dtype).max)
    tie_id = torch.where(tied, rows.entity_id, id_sentinel).min(-1).values
    chosen = tied & (rows.entity_id == tie_id.unsqueeze(-1))
    chosen_slot = chosen.to(torch.int64).argmax(-1)
    gather_q = q.gather(1, chosen_slot[:, None, None].expand(-1, 1, 2)).squeeze(1)
    gather_residual = residual.gather(1, chosen_slot[:, None]).squeeze(1).clamp_min(0.0)
    denom = gather_q.square().sum(-1).clamp_min(torch.finfo(u_nom.dtype).tiny)
    correction_xy = gather_residual.unsqueeze(-1) * gather_q / denom.unsqueeze(-1)
    if config.top1_projection_float64:
        # Selection/score/tie-breaking above stay in the original row dtype.
        # Only the selected closed-form projection uses double precision.
        # Verification below deliberately uses the final representable dtype.
        q64 = gather_q.to(torch.float64)
        b64 = b.gather(1, chosen_slot[:, None]).squeeze(1).to(torch.float64)
        nominal64 = u_nom[:, :2].to(torch.float64)
        residual64 = ((q64 * nominal64).sum(-1) - b64).clamp_min(0.0)
        denom64 = q64.square().sum(-1).clamp_min(torch.finfo(torch.float64).tiny)
        projected64 = nominal64 - residual64[:, None] * q64 / denom64[:, None]
        candidate_xy = torch.where(has_violation[:, None],
                                   projected64.to(u_nom.dtype), u_nom[:, :2])
    else:
        candidate_xy = u_nom[:, :2] - torch.where(
            has_violation.unsqueeze(-1), correction_xy, torch.zeros_like(correction_xy))
    candidate = torch.cat((candidate_xy, u_nom[:, 2:3]), dim=-1)
    selected_id = torch.where(has_violation, tie_id, selected_id)
    selected_after = ((gather_q * candidate_xy).sum(-1)
                      - b.gather(1, chosen_slot[:, None]).squeeze(1))
    method_pass = base_valid & (~has_violation | (selected_after <= config.row_tolerance_m2_s2))
    native = torch.where(
        base_valid, torch.where(has_violation, -gather_residual, torch.zeros_like(gather_residual)),
        torch.full_like(gather_residual, float("nan")))
    rounds = (base_valid & has_violation).to(torch.long)
    nan = torch.full((e,), float("nan"), dtype=u_nom.dtype, device=u_nom.device)
    nan_q = torch.full((e, 2), float("nan"), dtype=u_nom.dtype, device=u_nom.device)
    return candidate, base_valid, method_pass, selected_id, rounds, native, nan, nan_q, nan


def _softmin(u_nom: torch.Tensor, rows: C3BFRows, config: SolverConfigV2,
             base_valid: torch.Tensor):
    e, n = rows.active.shape
    selected_id = torch.full((e,), -1, dtype=torch.long, device=u_nom.device)
    if n == 0:
        nan = torch.full((e,), float("nan"), dtype=u_nom.dtype, device=u_nom.device)
        nan_q = torch.full((e, 2), float("nan"), dtype=u_nom.dtype, device=u_nom.device)
        return (u_nom, base_valid, base_valid, selected_id,
                torch.zeros(e, dtype=torch.long, device=u_nom.device),
                torch.zeros(e, dtype=u_nom.dtype, device=u_nom.device), nan, nan_q, nan)
    q, b, h = _safe_rows(rows)
    has_active = rows.active.any(-1)
    usable_rows = rows.row_valid
    has_usable = usable_rows.any(-1)
    logits = torch.where(usable_rows, -float(config.softmin_beta) * h,
                         torch.full_like(h, -float("inf")))
    logits = torch.where(has_usable.unsqueeze(-1), logits, torch.zeros_like(logits))
    log_sum = torch.logsumexp(logits, dim=-1)
    weights = torch.softmax(logits, dim=-1)
    weights = torch.where(usable_rows & has_usable.unsqueeze(-1), weights,
                          torch.zeros_like(weights))
    aggregate_h = -log_sum / float(config.softmin_beta)
    aggregate_known = base_valid & has_active
    aggregate_h = torch.where(aggregate_known, aggregate_h,
                              torch.full_like(aggregate_h, float("nan")))
    aggregate_q = (weights.unsqueeze(-1) * q).sum(1)
    aggregate_b = (weights * (b - float(rows.gamma_per_s) * h)).sum(-1)
    aggregate_b = aggregate_b + float(rows.gamma_per_s) * torch.where(
        aggregate_known, aggregate_h, torch.zeros_like(aggregate_h))
    aggregate_q = torch.where(aggregate_known.unsqueeze(-1), aggregate_q,
                              torch.full_like(aggregate_q, float("nan")))
    aggregate_b = torch.where(aggregate_known, aggregate_b,
                              torch.full_like(aggregate_b, float("nan")))

    safe_q = torch.where(has_active.unsqueeze(-1), aggregate_q, torch.zeros_like(aggregate_q))
    safe_b = torch.where(has_active, aggregate_b, torch.zeros_like(aggregate_b))
    residual = (safe_q * u_nom[:, :2]).sum(-1) - safe_b
    q_norm = torch.linalg.vector_norm(safe_q, dim=-1)
    needs_projection = has_active & (residual > config.row_tolerance_m2_s2)
    nondegenerate = q_norm > config.aggregate_q_min_m
    # The feasibility test for a zero-normal half-plane is exact: 0 <= B.
    # A negative B is a method degeneration even when its magnitude falls
    # below the ordinary row acceptance tolerance.
    degenerate_infeasible = has_active & ~nondegenerate & (safe_b < 0.0)
    method_defined = (base_valid & ~degenerate_infeasible
                      & (~needs_projection | nondegenerate))
    denom = q_norm.square().clamp_min(torch.finfo(u_nom.dtype).tiny)
    correction_xy = residual.clamp_min(0.0).unsqueeze(-1) * safe_q / denom.unsqueeze(-1)
    candidate_xy = u_nom[:, :2] - torch.where(
        (needs_projection & nondegenerate).unsqueeze(-1), correction_xy,
        torch.zeros_like(correction_xy))
    candidate = torch.cat((candidate_xy, u_nom[:, 2:3]), dim=-1)
    residual_after = (safe_q * candidate_xy).sum(-1) - safe_b
    method_pass = method_defined & (~has_active | (residual_after <= config.row_tolerance_m2_s2))
    native = torch.where(
        base_valid,
        torch.where(has_active, (-residual).clamp_max(0.0), torch.zeros_like(residual)),
        torch.full_like(residual, float("nan")))
    rounds = (method_defined & needs_projection).to(torch.long)
    return (candidate, method_defined, method_pass, selected_id, rounds, native,
            aggregate_h, aggregate_q, aggregate_b)


def _candidate_feasible(candidate: torch.Tensor, candidate_valid: torch.Tensor,
                        rows: C3BFRows, config: SolverConfigV2) -> torch.Tensor:
    """Validate [E,C,2] candidates in row chunks, never materializing [E,K,N]."""
    feasible = candidate_valid & torch.isfinite(candidate).all(-1)
    q, b, _ = _safe_rows(rows)
    n = rows.active.shape[1]
    for start in range(0, n, config.row_chunk_size):
        stop = min(start + config.row_chunk_size, n)
        residual = torch.einsum("eci,eri->ecr", candidate, q[:, start:stop]) - b[:, None, start:stop]
        row_ok = (~rows.active[:, None, start:stop]
                  | (residual <= config.row_tolerance_m2_s2)).all(-1)
        feasible = feasible & row_ok
    return feasible


def _all_candidates(u_nom: torch.Tensor, rows: C3BFRows, config: SolverConfigV2,
                    base_valid: torch.Tensor):
    e, n = rows.active.shape
    q, b, _ = _safe_rows(rows)
    pair_index = torch.triu_indices(n, n, offset=1, device=u_nom.device)
    pair_count = pair_index.shape[1]
    allocated_value = 1 + n + pair_count
    allocated = torch.full((e,), allocated_value, dtype=torch.long, device=u_nom.device)
    generated = torch.zeros(e, dtype=torch.long, device=u_nom.device)
    usable = torch.zeros(e, dtype=torch.long, device=u_nom.device)
    best_distance = torch.full((e,), float("inf"), dtype=u_nom.dtype, device=u_nom.device)
    best_xy = torch.zeros(e, 2, dtype=u_nom.dtype, device=u_nom.device)

    def consider(candidate: torch.Tensor, valid: torch.Tensor) -> None:
        nonlocal generated, usable, best_distance, best_xy
        valid = valid & base_valid.unsqueeze(-1)
        feasible = _candidate_feasible(candidate, valid, rows, config)
        generated = generated + valid.sum(-1)
        usable = usable + feasible.sum(-1)
        distance = (candidate - u_nom[:, None, :2]).square().sum(-1)
        distance = torch.where(feasible, distance, torch.full_like(distance, float("inf")))
        chunk_best, index = distance.min(-1)
        chunk_xy = candidate.gather(1, index[:, None, None].expand(-1, 1, 2)).squeeze(1)
        update = chunk_best < best_distance
        best_distance = torch.where(update, chunk_best, best_distance)
        best_xy = torch.where(update.unsqueeze(-1), chunk_xy, best_xy)

    consider(u_nom[:, None, :2], torch.ones(e, 1, dtype=torch.bool, device=u_nom.device))
    for start in range(0, n, config.candidate_chunk_size):
        stop = min(start + config.candidate_chunk_size, n)
        q_chunk = q[:, start:stop]
        b_chunk = b[:, start:stop]
        residual = torch.einsum("eci,ei->ec", q_chunk, u_nom[:, :2]) - b_chunk
        denom = q_chunk.square().sum(-1).clamp_min(torch.finfo(u_nom.dtype).tiny)
        candidate = u_nom[:, None, :2] - residual.unsqueeze(-1) * q_chunk / denom.unsqueeze(-1)
        consider(candidate, rows.row_valid[:, start:stop])
    for start in range(0, pair_count, config.candidate_chunk_size):
        stop = min(start + config.candidate_chunk_size, pair_count)
        i = pair_index[0, start:stop]
        j = pair_index[1, start:stop]
        q1, q2 = q[:, i], q[:, j]
        b1, b2 = b[:, i], b[:, j]
        determinant = q1[..., 0] * q2[..., 1] - q1[..., 1] * q2[..., 0]
        scale = torch.linalg.vector_norm(q1, dim=-1) * torch.linalg.vector_norm(q2, dim=-1)
        nonparallel = determinant.abs() > float(config.parallel_relative_tolerance) * scale
        pair_valid = rows.row_valid[:, i] & rows.row_valid[:, j] & nonparallel
        safe_det = torch.where(nonparallel, determinant, torch.ones_like(determinant))
        x = (b1 * q2[..., 1] - q1[..., 1] * b2) / safe_det
        y = (q1[..., 0] * b2 - b1 * q2[..., 0]) / safe_det
        consider(torch.stack((x, y), dim=-1), pair_valid)

    found = torch.isfinite(best_distance) & base_valid
    candidate = torch.cat((best_xy, u_nom[:, 2:3]), dim=-1)
    candidate = torch.where(found.unsqueeze(-1), candidate,
                            torch.full_like(candidate, float("nan")))
    native = _common_nominal_margin(u_nom, rows, torch.isfinite(u_nom).all(-1))
    selected_id = torch.full((e,), -1, dtype=torch.long, device=u_nom.device)
    rounds = torch.zeros(e, dtype=torch.long, device=u_nom.device)
    nan = torch.full((e,), float("nan"), dtype=u_nom.dtype, device=u_nom.device)
    nan_q = torch.full((e, 2), float("nan"), dtype=u_nom.dtype, device=u_nom.device)
    return (candidate, found, found, selected_id, rounds, native, nan, nan_q, nan,
            allocated, generated, usable)


@torch.no_grad()
def solve_c3bf_v2(
    method: str,
    u_nom: torch.Tensor,
    rows: C3BFRows,
    *,
    entity_id: torch.Tensor | None = None,
    config: SolverConfigV2 = SolverConfigV2(),
) -> SolverResultV2:
    """Run one named analytic arm without any output acceleration bound."""
    if method not in SOLVER_METHODS:
        raise ValueError(f"unknown C3BF v2 method {method!r}")
    e, _ = _validate_inputs(u_nom, rows)
    if entity_id is not None:
        if entity_id.shape != rows.entity_id.shape or entity_id.device != u_nom.device:
            raise ValueError("entity_id override must match rows [E,N] on the same device")
        rows = C3BFRows(**{**rows.__dict__, "entity_id": entity_id.to(torch.long)})
    nominal_finite = torch.isfinite(u_nom).all(-1)
    base_valid = nominal_finite & rows.domain_valid

    zero_count = torch.zeros(e, dtype=torch.long, device=u_nom.device)
    if method == "top1_violation_closed_form":
        values = _top1(u_nom, rows, config, base_valid)
        candidate, output_defined, method_pass, selected, rounds, native, agg_h, agg_q, agg_b = values
        allocated = generated = usable = zero_count
    elif method == "softmin_closed_form":
        values = _softmin(u_nom, rows, config, base_valid)
        candidate, output_defined, method_pass, selected, rounds, native, agg_h, agg_q, agg_b = values
        allocated = generated = usable = zero_count
    else:
        values = _all_candidates(u_nom, rows, config, base_valid)
        (candidate, output_defined, method_pass, selected, rounds, native,
         agg_h, agg_q, agg_b, allocated, generated, usable) = values

    proposed_defined = output_defined
    candidate_finite = torch.isfinite(candidate).all(-1)
    finite_output = proposed_defined & candidate_finite
    output_defined = proposed_defined & candidate_finite
    method_pass = method_pass & output_defined
    raw_max, normalized_max, all_pass = _residual_metrics(
        candidate, rows, output_defined, config.row_tolerance_m2_s2)
    if method == "all_halfplanes_candidates":
        # For A3 the method constraint is the complete original row set, so a
        # cast/roundoff miss found by the independent final check is a method
        # verification failure rather than a successful candidate search.
        method_pass = method_pass & all_pass
    correction = candidate - u_nom
    correction = torch.where(output_defined.unsqueeze(-1), correction,
                             torch.full_like(correction, float("nan")))
    objective = correction.square().sum(-1)
    objective = torch.where(output_defined, objective,
                            torch.full_like(objective, float("nan")))
    u_filtered = torch.where(output_defined.unsqueeze(-1), candidate,
                             torch.full_like(candidate, float("nan")))

    status = torch.full((e,), int(SolverStatus.SUCCESS), dtype=torch.long, device=u_nom.device)
    if method == "softmin_closed_form":
        status = torch.where(base_valid & ~output_defined,
                             torch.full_like(status, int(SolverStatus.DEGENERATE_METHOD)), status)
    elif method == "all_halfplanes_candidates":
        status = torch.where(base_valid & ~output_defined,
                             torch.full_like(status, int(SolverStatus.NO_VERIFIED_CANDIDATE)), status)
    numerical = ((proposed_defined & ~candidate_finite)
                 | (output_defined & ~method_pass))
    status = torch.where(numerical,
                         torch.full_like(status, int(SolverStatus.NUMERICAL_FAILURE)), status)
    status = torch.where(~rows.domain_valid,
                         torch.full_like(status, int(SolverStatus.INVALID_DOMAIN)), status)
    status = torch.where(~nominal_finite,
                         torch.full_like(status, int(SolverStatus.INVALID_NOMINAL)), status)

    return SolverResultV2(
        version=SOLVER_VERSION, method=method, u_filtered=u_filtered,
        correction=correction, objective=objective, output_defined=output_defined,
        method_constraint_pass=method_pass, all_rows_pass=all_pass,
        finite_output=finite_output, status=status, selected_id=selected,
        raw_residual_max=raw_max, normalized_residual_max=normalized_max,
        candidate_allocated=allocated, candidate_generated=generated,
        candidate_usable=usable, projection_rounds=rounds,
        native_margin_nominal=native, aggregate_h=agg_h,
        aggregate_q=agg_q, aggregate_b=agg_b)

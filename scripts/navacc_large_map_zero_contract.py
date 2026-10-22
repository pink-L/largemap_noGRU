"""First zero-initialized NavACC curriculum: raw top1 and physical-time reward.

The old lag/disk planar filter and PBRS remain separate historical modes.
Dynamic truth is piecewise constant: obstacle acceleration is zero between
velocity discontinuities, not a continuous acceleration certificate.
"""
from dataclasses import dataclass
import torch
from omni_drones.utils.navacc_c3bf_rows_v2 import C3BFRowConfig, build_c3bf_rows_v2
from omni_drones.utils.navacc_c3bf_solvers_v2 import SolverConfigV2, solve_c3bf_v2

@dataclass(frozen=True)
class ZeroCandidate:
    nominal: torch.Tensor
    candidate: torch.Tensor
    verified: torch.Tensor
    domain_valid: torch.Tensor
    raw_violation: torch.Tensor
    violation_defined: torch.Tensor
    active_count: torch.Tensor
    selected_id: torch.Tensor
    selected_count: torch.Tensor
    post_violation_count: torch.Tensor
    q: torch.Tensor
    b: torch.Tensor
    row_valid: torch.Tensor
    initial_h_safe: torch.Tensor
    method_pass: torch.Tensor
    bounds_pass: torch.Tensor

@torch.no_grad()
def zero_top1(position, velocity, nominal, dynamic_position, dynamic_velocity,
              *, body_radius=None, inflated_radius_m=.4, gamma_per_s=1., min_q_m=.01):
    """Reuse original v2 all-row builder and one orthogonal projection.

    No lag mapping, disk intersection, post-filter clipping, height filtering,
    or initial-h>=0 rejection. Bounds are execution checks, not a new solver.
    """
    p = dynamic_position[None, :, :2] - position[:, None, :2]
    w = dynamic_velocity[None, :, :2] - velocity[:, None, :2]
    active = (torch.linalg.vector_norm(p, dim=-1) <= 3.) | ~torch.isfinite(p).all(-1)
    rows = build_c3bf_rows_v2(p, w, torch.zeros_like(p), active,
            torch.isfinite(p).all(-1) & torch.isfinite(w).all(-1),
            torch.zeros_like(active, dtype=p.dtype),
            C3BFRowConfig(inflated_radius_m=inflated_radius_m,
                          gamma_per_s=gamma_per_s, min_q_m=min_q_m),
            inflated_radius_m=(None if body_radius is None else
                (body_radius[None]+.1+.05).expand(p.shape[:2])))
    result = solve_c3bf_v2('top1_violation_closed_form', nominal, rows,
                         config=SolverConfigV2())
    finite = torch.isfinite(nominal).all(-1)
    defined = rows.domain_valid & finite
    margin = rows.b - (rows.q * nominal[:, None, :2]).sum(-1)
    if margin.shape[1]:
        raw = torch.where(rows.active, margin, torch.full_like(margin, torch.inf)).min(-1).values.clamp_max(0)
        post = (rows.q * result.u_filtered[:, None, :2]).sum(-1) - rows.b
        post_count = (rows.row_valid & (post > 1e-6)).sum(-1)
    else:
        raw = torch.zeros_like(position[:, 0]); post_count = torch.zeros_like(raw, dtype=torch.long)
    defined = defined & torch.isfinite(raw)
    bounds = (torch.linalg.vector_norm(result.u_filtered[:, :2], dim=-1) <= .8+1e-6) & (result.u_filtered[:, 2].abs() <= .5+1e-6) & (torch.linalg.vector_norm(result.u_filtered, dim=-1) <= 1.+1e-6)
    verified = defined & result.output_defined & result.method_constraint_pass & result.finite_output & bounds
    return ZeroCandidate(nominal, result.u_filtered, verified, rows.domain_valid,
        torch.where(defined, raw, torch.zeros_like(raw)), defined,
        active.sum(-1), result.selected_id, result.projection_rounds,
        torch.where(defined & result.finite_output, post_count, torch.full_like(post_count,-1)),
        rows.q,rows.b,rows.row_valid,((~rows.active)|(rows.h>=0)).all(-1),
        result.method_constraint_pass,bounds)

@torch.no_grad()
def zero_cbf_reward(raw_violation, violation_defined, executed, nominal, execution_defined):
    violation_defined = violation_defined & torch.isfinite(raw_violation)
    execution_defined = execution_defined & torch.isfinite(executed).all(-1) & torch.isfinite(nominal).all(-1)
    violation = torch.where(violation_defined, .1*raw_violation, torch.zeros_like(raw_violation))
    correction = torch.where(execution_defined,
        .03*(torch.exp(-(executed-nominal).square().sum(-1))-1),
        torch.zeros_like(raw_violation))
    return violation, correction

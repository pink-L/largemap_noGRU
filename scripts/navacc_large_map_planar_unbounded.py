"""Explicit raw-top1 candidate execution without nominal acceleration bounds.

This mode is a command protocol, not a continuous physical safety certificate.
The original converter clips commander thrust; requested acceleration therefore
must never be reported as achieved physical acceleration.
"""
from dataclasses import dataclass
import torch
from navacc_large_map_zero_contract import ZeroCandidate
from omni_drones.utils.navacc_c3bf_rows_v2 import C3BFRowConfig, build_c3bf_rows_v2
from omni_drones.utils.navacc_c3bf_solvers_v2 import SolverConfigV2, solve_c3bf_v2

PROTOCOL = 'nominal_continue_unbounded_raw_top1_v1'
PROFILE = 'planar_velocity5_height8_lidar0_unbounded_joint_rawtop1_v1'
REWARD_SCHEMA = 'raw_margin_and_valid_original_candidate_correction_v1'
ACTION_SEMANTICS = 'primary_acceleration_unbounded_raw_top1_rpyt_satallowed_v1'
NUMERIC_SCHEMA = 'top1_selected_float64_cast_verify_float32_tol1e4_v1'
ROW_TOLERANCE_M2_S2 = 1.0e-4
RAW_POST_DIAGNOSTIC_TOLERANCE_M2_S2 = 1.0e-6
CBF_CORRECTION_COEFF_DEFAULT = .03
RADIUS_SCHEMA = 'obstacle_equivalent_radius_plus_uav_radius_no_extra_margin_v1'
UAV_SAFETY_RADIUS_M = 0.1
EXTRA_SAFETY_MARGIN_M = 0.0
# The bounded contract is the *explicit* legacy opt-in written by the zero-stage
# branch.  It must be named in the resolved config: an absent protocol key is an
# error, never a silent fallback to this branch.
LEGACY_BOUNDED_PROTOCOL = 'nominal_continue_raw_top1_v1'
# Execution-contract keys a stage may declare; `apply_protocol_config` writes them
# from the constants above so config, stage spec and validation cannot drift.
PROTOCOL_CONFIG_KEYS = {
    ("task", "large_map", "execution_protocol"): PROTOCOL,
    ("task", "large_map", "c3bf_numeric_schema"): NUMERIC_SCHEMA,
    ("task", "large_map", "c3bf_selected_projection_dtype"): "float64",
    ("task", "large_map", "c3bf_output_verification_dtype"): "float32",
    ("task", "large_map", "c3bf_row_tolerance_m2_s2"): ROW_TOLERANCE_M2_S2,
    ("task", "large_map", "c3bf_raw_post_diagnostic_tolerance_m2_s2"): RAW_POST_DIAGNOSTIC_TOLERANCE_M2_S2,
    ("task", "large_map", "c3bf_radius_schema"): RADIUS_SCHEMA,
    ("task", "large_map", "c3bf_uav_safety_radius_m"): UAV_SAFETY_RADIUS_M,
    ("task", "large_map", "c3bf_extra_safety_margin_m"): EXTRA_SAFETY_MARGIN_M,
    ("task", "large_map", "converter_tilt_validity_check"): False,
    ("task", "reward", "cbf_schema"): REWARD_SCHEMA,
    ("task", "action", "semantics"): ACTION_SEMANTICS,
}


def _get_nested(document, path):
    node = document
    for component in path:
        if node is None:
            return None
        node = node.get(component) if isinstance(node, dict) else getattr(node, component, None)
    return node


def _set_nested(document, path, value):
    node = document
    for component in path[:-1]:
        node = node[component] if isinstance(node, dict) else getattr(node, component)
    if isinstance(node, dict):
        node[path[-1]] = value
    else:
        setattr(node, path[-1], value)


def protocol_of(cfg):
    """Read the declared execution protocol from a config of any mapping type."""
    value = _get_nested(cfg, ("task", "large_map", "execution_protocol"))
    return "" if value is None else str(value)


def apply_protocol_config(cfg):
    """Write the complete unbounded execution/reward/action contract into cfg."""
    for path, value in PROTOCOL_CONFIG_KEYS.items():
        _set_nested(cfg, path, value)
    return {'.'.join(path): value for path, value in PROTOCOL_CONFIG_KEYS.items()}


def validate_protocol_config(cfg):
    """Return the contract keys; raise if any of them is missing or differs."""
    observed, mismatched = {}, {}
    for path, value in PROTOCOL_CONFIG_KEYS.items():
        current = _get_nested(cfg, path)
        observed['.'.join(path)] = current
        if current != value:
            mismatched['.'.join(path)] = {'expected': value, 'observed': current}
    if mismatched:
        raise ValueError(f'unbounded execution contract is incomplete or differs: {mismatched}')
    return observed

@dataclass(frozen=True)
class UnboundedCandidate(ZeroCandidate):
    output_defined: torch.Tensor
    finite_output: torch.Tensor
    candidate_defined: torch.Tensor

@dataclass(frozen=True)
class UnboundedExecution:
    command: UnboundedCandidate
    commanded_acceleration: torch.Tensor
    rpyt: torch.Tensor
    filtered: torch.Tensor
    fallback: torch.Tensor
    executable: torch.Tensor
    candidate_conversion_valid: torch.Tensor
    nominal_conversion_valid: torch.Tensor
    candidate_diagnostics: dict
    nominal_diagnostics: dict
    violation_reward: torch.Tensor
    correction_reward: torch.Tensor
    correction_defined: torch.Tensor

@torch.no_grad()
def unbounded_top1(position, velocity, nominal, body_position, body_velocity,
                   *, body_radius, inflated_radius_m=.4,
                   gamma_per_s=1., min_q_m=.01, nominal_bounds=None):
    """Original all-active rows and selected-row float64 closed projection.

    Domain, output definition, method and finite predicates stay independent.
    Nominal bounds remain diagnostic only for the projected candidate.
    """
    p=body_position[None,:,:2]-position[:,None,:2]
    w=body_velocity[None,:,:2]-velocity[:,None,:2]
    active=(torch.linalg.vector_norm(p,dim=-1)<=3.)|~torch.isfinite(p).all(-1)
    rows=build_c3bf_rows_v2(p,w,torch.zeros_like(p),active,
        torch.isfinite(p).all(-1)&torch.isfinite(w).all(-1),
        torch.zeros_like(active,dtype=p.dtype),
        C3BFRowConfig(inflated_radius_m=inflated_radius_m,
                      gamma_per_s=gamma_per_s,min_q_m=min_q_m),
        inflated_radius_m=(body_radius[None]+UAV_SAFETY_RADIUS_M
                           +EXTRA_SAFETY_MARGIN_M).expand(p.shape[:2]))
    result=solve_c3bf_v2('top1_violation_closed_form',nominal,rows,
                         config=SolverConfigV2(row_tolerance_m2_s2=ROW_TOLERANCE_M2_S2,
                                               top1_projection_float64=True))
    defined=rows.domain_valid & torch.isfinite(nominal).all(-1)
    if p.shape[1]:
        margin=rows.b-(rows.q*nominal[:,None,:2]).sum(-1)
        raw=torch.where(rows.active,margin,torch.full_like(margin,torch.inf)).min(-1).values.clamp_max(0)
        post=(rows.q*result.u_filtered[:,None,:2]).sum(-1)-rows.b
        count=(rows.row_valid & (post>RAW_POST_DIAGNOSTIC_TOLERANCE_M2_S2)).sum(-1)
    else:
        raw=torch.zeros_like(position[:,0]);count=torch.zeros_like(raw,dtype=torch.long)
    defined &= torch.isfinite(raw)
    if nominal_bounds is None:
        from navacc_large_map_planar_c3bf import PlanarC3BFConfig
        nominal_bounds=PlanarC3BFConfig()
    bounds=(result.u_filtered[:,:2].norm(dim=-1)<=nominal_bounds.xy_axis_limit_m_s2+1e-6)&(result.u_filtered[:,2].abs()<=nominal_bounds.z_axis_limit_m_s2+1e-6)&(result.u_filtered.norm(dim=-1)<=nominal_bounds.total_limit_m_s2+1e-6)
    candidate_defined=defined & result.output_defined & result.method_constraint_pass & result.finite_output & torch.isfinite(result.u_filtered).all(-1)
    return UnboundedCandidate(nominal,result.u_filtered,candidate_defined,
        rows.domain_valid,torch.where(defined,raw,torch.zeros_like(raw)),defined,
        active.sum(-1),result.selected_id,result.projection_rounds,
        torch.where(defined & result.finite_output,count,torch.full_like(count,-1)),
        rows.q,rows.b,rows.row_valid,((~rows.active)|(rows.h>=0)).all(-1),
        result.method_constraint_pass,bounds,result.output_defined,
        result.finite_output,candidate_defined)

@torch.no_grad()
def unbounded_cbf_reward(command, c_corr=CBF_CORRECTION_COEFF_DEFAULT):
    """Reward original mathematically valid candidate, even converter fallback.

    ``c_corr`` is the single-variable Corr-AB coefficient (default .03 keeps
    every legacy phase bit-identical).  A huge finite squared correction may
    become +inf; exp(-inf)=0 is the intended bounded reward limit, not an
    undefined candidate.
    """
    correction_defined=(command.candidate_defined & command.domain_valid
        & command.output_defined & command.finite_output & command.method_pass
        & torch.isfinite(command.candidate).all(-1)
        & torch.isfinite(command.nominal).all(-1))
    safe_delta=torch.where(correction_defined[:,None],
        command.candidate-command.nominal,torch.zeros_like(command.nominal))
    rc=torch.where(correction_defined,c_corr*(torch.exp(-safe_delta.square().sum(-1))-1),
                   torch.zeros_like(command.raw_violation))
    rv=torch.where(correction_defined & command.violation_defined,
                   .1*command.raw_violation,torch.zeros_like(command.raw_violation))
    return rv,rc,correction_defined

@torch.no_grad()
def execute_unbounded(command, converter, yaw, c_corr=CBF_CORRECTION_COEFF_DEFAULT):
    # nan_to_num only protects converter arithmetic. It never validates input.
    rpyt_c,dc=converter(torch.nan_to_num(command.candidate)[:,None],yaw)
    rpyt_n,dn=converter(command.nominal[:,None],yaw)
    cv=dc['valid'].reshape(-1)&torch.isfinite(rpyt_c).all((-1,-2))
    nv=dn['valid'].reshape(-1)&torch.isfinite(rpyt_n).all((-1,-2))
    valid_candidate=(command.candidate_defined & command.domain_valid
        & command.output_defined & command.finite_output & command.method_pass
        & torch.isfinite(command.candidate).all(-1)
        & torch.isfinite(command.nominal).all(-1))
    filtered=valid_candidate & cv
    fallback=~filtered & nv
    executable=filtered|fallback
    requested=torch.where(filtered[:,None],command.candidate,
        torch.where(fallback[:,None],command.nominal,torch.zeros_like(command.nominal)))
    rpyt=torch.where(filtered[:,None,None],rpyt_c,
        torch.where(fallback[:,None,None],rpyt_n,torch.zeros_like(rpyt_n)))
    rv,rc,defined=unbounded_cbf_reward(command,c_corr)
    return UnboundedExecution(command,requested,rpyt,filtered,fallback,executable,
        cv,nv,dc,dn,rv,rc,defined)

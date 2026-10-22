#!/usr/bin/env python3
"""TorchRL action transform for primary-PPO dense-planar NavACC control."""
from __future__ import annotations

import torch
from torchrl.envs.transforms import Compose, InitTracker, Transform, TransformedEnv

from omni_drones.utils.torchrl.compat import (
    DiscreteTensorSpec, UnboundedContinuousTensorSpec)


DIAGNOSTIC_BOOL_KEYS = (
    "planar_cbf_candidate_defined", "planar_cbf_candidate_output_defined",
    "planar_cbf_candidate_finite", "planar_cbf_candidate_conversion_valid",
    "planar_cbf_candidate_bounds_required", "planar_cbf_candidate_thrust_saturated",
    "planar_cbf_fallback", "planar_cbf_unrecoverable",
    "planar_cbf_violation_defined", "planar_cbf_correction_defined",
    "planar_cbf_abort", "planar_cbf_domain_valid",
    "planar_cbf_initial_g_safe", "planar_cbf_initial_psi1_safe",
    "planar_cbf_initial_h_safe",
    "planar_cbf_normal_h_recovery_attempted",
    "planar_cbf_normal_h_recovery_accepted",
    "planar_cbf_selected_row_feasible", "planar_cbf_selected_row_pass",
    "planar_cbf_action_bounds_pass",
    "planar_cbf_selected_is_static", "planar_cbf_all_rows_certified",
)
DIAGNOSTIC_SCALAR_KEYS = (
    "planar_candidate_commander_thrust_raw", "planar_candidate_desired_force_N",
    "planar_reward_cbf_violation", "planar_reward_cbf_correction",
    "planar_cbf_normalized_nominal_violation",
    "planar_cbf_selected_post_residual",
    "planar_cbf_max_normalized_post_violation",
)
DIAGNOSTIC_INTEGER_KEYS = (
    "planar_execution_source", "planar_candidate_reject_reason",
    "planar_cbf_selected_entity_id", "planar_cbf_active_count",
    "planar_cbf_selected_count", "planar_cbf_post_violation_count",
)


def _publish(base_env, td, values) -> None:
    for key, value in values.items():
        persistent = base_env.info.get(key, None)
        if persistent is None or persistent.shape != value.shape \
                or persistent.dtype != value.dtype or persistent.device != value.device:
            raise RuntimeError(f"planar training diagnostic mismatch for {key}")
        persistent.copy_(value); td.set(("info", key), value.clone())


class PlanarPrimaryActionTransform(Transform):
    """Preserve PPO sample, map it to the primary nominal, then apply top1 C3BF."""

    def __init__(self, base_env):
        super().__init__([], in_keys_inv=[("agents", "action"),
                                          ("info", "drone_state"),
                                          ("info", "yaw_hold_W")])
        self.base_env = base_env; self.backend = base_env.planar_backend
        self.previous_velocity = torch.zeros(
            base_env.num_envs, 3, device=base_env.device)
        self.velocity_valid = torch.zeros(
            base_env.num_envs, device=base_env.device, dtype=torch.bool)
        self.last_command = None; self.last_measured_acceleration = None

    def transform_input_spec(self, input_spec):
        spec = input_spec["full_action_spec", "agents", "action"]
        input_spec["full_action_spec", "agents", "action"] = \
            UnboundedContinuousTensorSpec(
                spec.shape[:-1] + (3,), device=spec.device)
        return input_spec

    @torch.no_grad()
    def _inv_call(self, td):
        z = td["agents", "action"].clone()
        expected = (self.base_env.num_envs, 1, 3)
        if z.shape != expected or not bool(torch.isfinite(z).all()):
            raise ValueError(f"finite PPO latent must have shape {expected}")
        prior_z = td.get(("agents", "z_policy"), None)
        if prior_z is not None and not torch.equal(prior_z, z):
            raise ValueError("z_policy differs from sampled PPO action")
        td.set(("agents", "z_policy"), z.clone())
        log_prob = td.get("sample_log_prob", None)
        if log_prob is None or not bool(torch.isfinite(log_prob).all()):
            raise ValueError("finite sample_log_prob is required")
        retained_log_prob = log_prob.clone()
        state = td["info", "drone_state"].reshape(self.base_env.num_envs, -1)
        if state.shape[1] < 10 or state.device != z.device:
            raise ValueError("native drone state must be device-local and at least 10-wide")
        velocity = state[:, 7:10]
        init = td.get("is_init", None)
        init_mask = (torch.zeros(self.base_env.num_envs, device=z.device,
                                 dtype=torch.bool) if init is None else
                     init.reshape(self.base_env.num_envs, -1).any(-1))
        measured = torch.where(
            (self.velocity_valid & ~init_mask)[:, None],
            (velocity - self.previous_velocity) / .01, torch.zeros_like(velocity))
        self.last_measured_acceleration = measured.detach().clone()
        primary = torch.tanh(z[:, 0]) * torch.tensor(
            [self.backend.c3bf_config.xy_axis_limit_m_s2,
             self.backend.c3bf_config.xy_axis_limit_m_s2,
             self.backend.c3bf_config.z_axis_limit_m_s2],
            device=z.device, dtype=z.dtype)
        dynamic_position, dynamic_velocity = self.base_env._shared_dynamic_truth()
        if getattr(getattr(self.base_env,"_planar_reward_cfg",None),"mode",None) in ("zero_curriculum_v1","hold48_v1","hold30_continuous_v1", "hold50_continuous_v1", "obstacle_course_continuous_v1"):
            from types import SimpleNamespace
            from navacc_large_map_planar_c3bf import bound_primary_acceleration
            bounded=bound_primary_acceleration(primary,self.backend.c3bf_config)
            if getattr(self.base_env,'_goal_frame',None) is not None:
                from navacc_goal_frame import goal_to_world
                if not bool(self.base_env._goal_frame.valid.all()):raise ValueError('action before valid episode GoalFrame')
                bounded=goal_to_world(bounded,self.base_env._goal_frame.theta)
            command = SimpleNamespace(nominal_acceleration=bounded)
        else:
            command = self.backend.command(
            state[:, :3], velocity, measured, primary,
            dynamic_position=dynamic_position,
            dynamic_velocity=dynamic_velocity)
        if getattr(getattr(self.base_env,"_planar_reward_cfg",None),"mode",None) in ("zero_curriculum_v1","hold48_v1","hold30_continuous_v1", "hold50_continuous_v1", "obstacle_course_continuous_v1"):
            self._zero_execution(td, state, velocity, command, dynamic_position, dynamic_velocity)
            self.previous_velocity.copy_(velocity); self.velocity_valid.fill_(True)
            if not torch.equal(td["agents", "z_policy"],z) or not torch.equal(td["sample_log_prob"],retained_log_prob):
                raise RuntimeError("zero execution changed PPO probability data")
            return td
        self.previous_velocity.copy_(velocity); self.velocity_valid.fill_(True)
        self.last_command = command
        acceleration = command.acceleration.unsqueeze(1)
        rpyt, diagnostics = self.base_env.acceleration_to_rpyt(
            acceleration, td["info", "yaw_hold_W"])
        valid = diagnostics["valid"].reshape(self.base_env.num_envs) \
            & torch.isfinite(rpyt).all((-1, -2))
        if not bool(valid.all()) or bool(diagnostics["thrust_saturated"].any()):
            raise RuntimeError("planar acceleration-to-RPYT conversion failed")
        from navacc_large_map_planar_obs20 import publish_selected_request
        publish_selected_request(self.base_env, acceleration[:,0])
        td.set(("info", "a_nom"), acceleration)
        td.set(("info", "rpyt_tilt_rad"), diagnostics["tilt_rad"])
        td.set(("info", "rpyt_thrust_saturated"),
               diagnostics["thrust_saturated"])
        values = {
            "planar_policy_nominal_acceleration":
                command.nominal_acceleration.unsqueeze(1),
            "planar_cbf_abort": command.cbf_abort[:, None],
            "planar_cbf_domain_valid": command.domain_valid[:, None],
            "planar_cbf_initial_g_safe": command.initial_g_safe[:, None],
            "planar_cbf_initial_psi1_safe": command.initial_psi1_safe[:, None],
            "planar_cbf_initial_h_safe": command.initial_c3bf_h_safe[:, None],
            "planar_cbf_normal_h_recovery_attempted":
                command.normal_h_recovery_attempted[:, None],
            "planar_cbf_normal_h_recovery_accepted":
                command.normal_h_recovery_accepted[:, None],
            "planar_cbf_selected_row_feasible": command.selected_row_feasible[:, None],
            "planar_cbf_selected_row_pass": command.selected_row_pass[:, None],
            "planar_cbf_action_bounds_pass": command.action_bounds_pass[:, None],
            "planar_cbf_selected_is_static": command.selected_is_static[:, None],
            "planar_cbf_all_rows_certified": command.claim_all_rows_certified[:, None],
            "planar_cbf_selected_entity_id": command.selected_entity_id[:, None],
            "planar_cbf_active_count": command.active_count[:, None],
            "planar_cbf_selected_count": command.selected_count[:, None],
            "planar_cbf_post_violation_count":
                command.eligible_post_violation_count[:, None],
            "planar_cbf_normalized_nominal_violation":
                command.normalized_nominal_violation[:, None],
            "planar_cbf_selected_post_residual":
                command.selected_post_residual[:, None],
            "planar_cbf_max_normalized_post_violation":
                command.max_normalized_post_violation[:, None],
        }
        _publish(self.base_env, td, values)
        td.set(("agents", "action"), rpyt)
        if not torch.equal(td["agents", "z_policy"], z) \
                or not torch.equal(td["sample_log_prob"], retained_log_prob):
            raise RuntimeError("planar transform modified PPO probability data")
        return td


    def _zero_execution(self, td, state, velocity, command, dynamic_position, dynamic_velocity):
        from navacc_large_map_planar_unbounded import LEGACY_BOUNDED_PROTOCOL, PROTOCOL, protocol_of
        cfg=getattr(self.base_env, "cfg", None)
        protocol=protocol_of(cfg)
        if protocol == PROTOCOL:
            return self._unbounded_execution(td, state, velocity, command,
                                             dynamic_position, dynamic_velocity)
        # The bounded raw-top1 contract is only reachable as an explicit legacy
        # opt-in.  An absent or unknown protocol key is a configuration error and
        # must never silently degrade to the old bounded execution branch.
        if protocol != LEGACY_BOUNDED_PROTOCOL:
            raise ValueError(
                "zero/hold48 execution requires an explicit execution protocol "
                f"({PROTOCOL!r} or the declared legacy {LEGACY_BOUNDED_PROTOCOL!r}); "
                f"resolved config carries {protocol!r}")
        from navacc_large_map_zero_contract import zero_top1, zero_cbf_reward
        positions, velocities = dynamic_position, dynamic_velocity
        radii = self.backend.dynamic_cbf_radius
        dynamic_count = dynamic_position.shape[0]
        joint = self.backend.static_mode == "static_c3bf"
        if joint:
            positions = torch.cat((positions, self.backend.static_centers), dim=0)
            velocities = torch.cat((velocities, torch.zeros_like(self.backend.static_centers)), dim=0)
            radii = torch.cat((radii, self.backend.static_cbf_radius), dim=0)
        c = zero_top1(state[:, :3], velocity, command.nominal_acceleration,
                     positions, velocities, body_radius=radii)
        selected_static = (c.selected_id >= dynamic_count) & (c.selected_count > 0) if joint else torch.zeros_like(c.domain_valid)
        yaw = td["info", "yaw_hold_W"]
        candidate = torch.nan_to_num(c.candidate)
        rpyt_c, diag_c = self.base_env.acceleration_to_rpyt(candidate[:, None], yaw)
        rpyt_n, diag_n = self.base_env.acceleration_to_rpyt(c.nominal[:, None], yaw)
        cv = diag_c["valid"].reshape(-1) & torch.isfinite(rpyt_c).all((-1,-2)) & ~diag_c["thrust_saturated"].reshape(-1)
        nv = diag_n["valid"].reshape(-1) & torch.isfinite(rpyt_n).all((-1,-2)) & ~diag_n["thrust_saturated"].reshape(-1)
        filtered = c.verified & cv; fallback = ~filtered & nv
        executable = filtered | fallback
        actual = torch.where(filtered[:, None], candidate, torch.where(fallback[:, None], c.nominal, torch.zeros_like(c.nominal)))
        rpyt = torch.where(filtered[:, None, None], rpyt_c, torch.where(fallback[:, None, None], rpyt_n, torch.zeros_like(rpyt_n)))
        rv, rc = zero_cbf_reward(c.raw_violation,c.violation_defined,actual,c.nominal,executable)
        from navacc_large_map_planar_obs20 import publish_selected_request
        publish_selected_request(self.base_env, actual, executable)
        td.set(("info", "a_nom"), actual[:, None])
        td.set(("info", "rpyt_tilt_rad"), torch.where(filtered[:,None,None],diag_c["tilt_rad"],diag_n["tilt_rad"]))
        td.set(("info", "rpyt_thrust_saturated"),torch.where(filtered[:,None,None],diag_c["thrust_saturated"],diag_n["thrust_saturated"]))
        post=(c.q*actual[:,None,:2]).sum(-1)-c.b
        post_count=(c.row_valid & (post>1e-6)).sum(-1)
        post_count=torch.where(c.domain_valid & executable,post_count,torch.full_like(post_count,-1))
        values = {
            "planar_policy_nominal_acceleration":c.nominal[:,None],
            "planar_cbf_initial_g_safe":c.domain_valid[:,None],
            "planar_cbf_initial_psi1_safe":c.domain_valid[:,None],
            "planar_cbf_initial_h_safe":c.initial_h_safe[:,None],
            "planar_cbf_selected_row_feasible":c.verified[:,None],
            "planar_cbf_selected_row_pass":c.method_pass[:,None],
            "planar_cbf_action_bounds_pass":c.bounds_pass[:,None],
            "planar_cbf_fallback":fallback[:,None], "planar_cbf_unrecoverable":(~executable)[:,None],
            "planar_cbf_violation_defined":c.violation_defined[:,None],
            "planar_cbf_correction_defined":executable[:,None],
            "planar_reward_cbf_violation":rv[:,None], "planar_reward_cbf_correction":rc[:,None],
            "planar_execution_source":torch.where(filtered,0,torch.where(fallback,1,2))[:,None],
            "planar_candidate_reject_reason":torch.where(filtered,0,torch.where(~c.domain_valid,2,torch.where(~c.verified,3,torch.where(~cv,5,0))))[:,None],
            "planar_cbf_abort":torch.zeros_like(executable)[:,None],
            "planar_cbf_domain_valid":c.domain_valid[:,None],
            "planar_cbf_selected_entity_id":c.selected_id[:,None],
            "planar_cbf_selected_is_static":selected_static[:,None],
            "planar_cbf_active_count":c.active_count[:,None],
            "planar_cbf_selected_count":c.selected_count[:,None],
            "planar_cbf_post_violation_count":post_count[:,None],
            "planar_cbf_all_rows_certified":torch.zeros_like(executable)[:,None],
        }
        _publish(self.base_env, td, values)
        td.set(("agents", "action"), rpyt)
        self.last_command = c

    def _unbounded_execution(self, td, state, velocity, command,
                             dynamic_position, dynamic_velocity):
        from navacc_large_map_planar_unbounded import unbounded_top1, execute_unbounded
        _c_corr=float(getattr(getattr(self.base_env,"_planar_reward_cfg",None),
                              "cbf_correction_coeff",0.03))
        positions, velocities = dynamic_position, dynamic_velocity
        radii = self.backend.dynamic_cbf_radius
        dynamic_count = dynamic_position.shape[0]
        joint = self.backend.static_mode == "static_c3bf"
        if joint:
            positions = torch.cat((positions, self.backend.static_centers), dim=0)
            velocities = torch.cat((velocities, torch.zeros_like(self.backend.static_centers)), dim=0)
            radii = torch.cat((radii, self.backend.static_cbf_radius), dim=0)
        c = unbounded_top1(state[:, :3], velocity, command.nominal_acceleration,
                          positions, velocities, body_radius=radii,
                          nominal_bounds=self.backend.c3bf_config)
        ex = execute_unbounded(c, self.base_env.acceleration_to_rpyt,
                               td["info", "yaw_hold_W"], _c_corr)
        dc, dn = ex.candidate_diagnostics, ex.nominal_diagnostics
        from navacc_large_map_planar_obs20 import publish_selected_request
        publish_selected_request(self.base_env, ex.commanded_acceleration, ex.filtered | ex.fallback)
        td.set(("info", "a_nom"), ex.commanded_acceleration[:, None])
        td.set(("info", "rpyt_tilt_rad"), torch.where(ex.filtered[:, None, None], dc["tilt_rad"], dn["tilt_rad"]))
        td.set(("info", "rpyt_thrust_saturated"), torch.where(ex.filtered[:, None, None], dc["thrust_saturated"], dn["thrust_saturated"]))
        post=(c.q*ex.commanded_acceleration[:,None,:2]).sum(-1)-c.b
        post_count=(c.row_valid & (post>1e-6)).sum(-1)
        post_count=torch.where(c.domain_valid & ex.executable,post_count,torch.full_like(post_count,-1))
        selected_static=(c.selected_id>=dynamic_count)&(c.selected_count>0) if joint else torch.zeros_like(c.domain_valid)
        reject=torch.where(ex.filtered,0,torch.where(~c.domain_valid,2,
            torch.where(~c.candidate_defined,3,torch.where(~ex.candidate_conversion_valid,5,0))))
        values={
            # Persist the already chosen converter flag to NEXT for this action.
            # The candidate diagnostic remains separately published below.
            "rpyt_thrust_saturated":td["info", "rpyt_thrust_saturated"],
            "planar_policy_nominal_acceleration":c.nominal[:,None],
            "planar_candidate_acceleration":c.candidate[:,None],
            "planar_commanded_acceleration":ex.commanded_acceleration[:,None],
            "planar_commanded_rpyt":ex.rpyt,
            "planar_cbf_initial_g_safe":c.domain_valid[:,None],
            "planar_cbf_initial_psi1_safe":c.domain_valid[:,None],
            "planar_cbf_initial_h_safe":c.initial_h_safe[:,None],
            "planar_cbf_selected_row_feasible":c.candidate_defined[:,None],
            "planar_cbf_selected_row_pass":c.method_pass[:,None],
            "planar_cbf_action_bounds_pass":c.bounds_pass[:,None],
            "planar_cbf_candidate_bounds_required":torch.zeros_like(c.domain_valid)[:,None],
            "planar_cbf_candidate_defined":c.candidate_defined[:,None],
            "planar_cbf_candidate_output_defined":c.output_defined[:,None],
            "planar_cbf_candidate_finite":c.finite_output[:,None],
            "planar_cbf_candidate_conversion_valid":ex.candidate_conversion_valid[:,None],
            "planar_cbf_candidate_thrust_saturated":dc["thrust_saturated"].reshape(-1,1),
            "planar_candidate_commander_thrust_raw":dc["commander_thrust_raw"].reshape(-1,1),
            "planar_candidate_desired_force_N":dc["desired_force_N"].reshape(-1,1),
            "planar_cbf_fallback":ex.fallback[:,None],
            "planar_cbf_unrecoverable":(~ex.executable)[:,None],
            "planar_cbf_violation_defined":c.violation_defined[:,None],
            "planar_cbf_correction_defined":ex.correction_defined[:,None],
            "planar_reward_cbf_violation":ex.violation_reward[:,None],
            "planar_reward_cbf_correction":ex.correction_reward[:,None],
            "planar_execution_source":torch.where(ex.filtered,0,torch.where(ex.fallback,1,2))[:,None],
            "planar_candidate_reject_reason":reject[:,None],
            "planar_cbf_abort":torch.zeros_like(c.domain_valid)[:,None],
            "planar_cbf_domain_valid":c.domain_valid[:,None],
            "planar_cbf_selected_entity_id":c.selected_id[:,None],
            "planar_cbf_selected_is_static":selected_static[:,None],
            "planar_cbf_active_count":c.active_count[:,None],
            "planar_cbf_selected_count":c.selected_count[:,None],
            "planar_cbf_post_violation_count":post_count[:,None],
            "planar_cbf_all_rows_certified":torch.zeros_like(c.domain_valid)[:,None],
        }
        _publish(self.base_env, td, values)
        td.set(("agents", "action"), ex.rpyt)
        self.last_command = c


def build_planar_training_env(base_env):
    e, d = base_env.num_envs, base_env.device
    boolean = lambda: DiscreteTensorSpec(2, (e, 1), dtype=torch.bool, device=d)
    scalar = lambda: UnboundedContinuousTensorSpec((e, 1), device=d)
    integer = lambda: UnboundedContinuousTensorSpec(
        (e, 1), device=d, dtype=torch.long)
    locked = base_env.is_spec_locked; base_env.set_spec_lock_(False)
    for key in DIAGNOSTIC_BOOL_KEYS:
        base_env.observation_spec["info", key] = boolean()
    for key in DIAGNOSTIC_SCALAR_KEYS:
        base_env.observation_spec["info", key] = scalar()
    for key in DIAGNOSTIC_INTEGER_KEYS:
        base_env.observation_spec["info", key] = integer()
    base_env.observation_spec[
        "info", "planar_policy_nominal_acceleration"] = \
        UnboundedContinuousTensorSpec((e, 1, 3), device=d)
    for key in ("planar_candidate_acceleration", "planar_commanded_acceleration"):
        base_env.observation_spec["info", key] = UnboundedContinuousTensorSpec((e,1,3),device=d)
    base_env.observation_spec["info", "planar_commanded_rpyt"] = UnboundedContinuousTensorSpec((e,1,4),device=d)
    base_env.set_spec_lock_(locked)
    base_env.info = base_env.observation_spec["info"].zero()
    transform = PlanarPrimaryActionTransform(base_env)
    return TransformedEnv(
        base_env, Compose(InitTracker(), transform)).train(), transform


__all__ = ["DIAGNOSTIC_BOOL_KEYS", "DIAGNOSTIC_INTEGER_KEYS",
           "DIAGNOSTIC_SCALAR_KEYS", "PlanarPrimaryActionTransform",
           "build_planar_training_env"]

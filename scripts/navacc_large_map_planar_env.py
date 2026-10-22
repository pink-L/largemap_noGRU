#!/usr/bin/env python3
"""Additive Isaac environment for dense-planar primary-PPO NavACC training."""
from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np
import torch
from tensordict import TensorDict

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from navacc_large_map_env import NavAccNavRLShared  # noqa: E402
from navacc_large_map_planar_backend import PlanarLargeMapBackend  # noqa: E402
from navacc_large_map_planar_c3bf import PlanarC3BFConfig  # noqa: E402
from navacc_large_map_planar_guidance import (  # noqa: E402
    GLOBAL_GUIDANCE_SCHEMA, GLOBAL_OBSERVATION_SCHEMA, GUIDANCE_SCHEMA,
    ROUTE_OBSERVATION_SCHEMA, PlanarRouteGuidance)
from navacc_large_map_planar_reward import (  # noqa: E402
    PlanarRewardConfig, planar_reward_step)
from navacc_large_map_planar_scene import (  # noqa: E402
    PlanarMissionSampler, PlanarSceneConfig, sample_free_planar_missions,
    validate_planar_scene)
from omni_drones.envs.single.nav_acc_sim_pilot import NavAccSimPilot  # noqa: E402
from omni_drones.utils.torch import quat_rotate_inverse  # noqa: E402
from omni_drones.utils.torchrl.compat import (  # noqa: E402
    DiscreteTensorSpec, UnboundedContinuousTensorSpec)

class NavAccNavRLSharedPlanar(NavAccNavRLShared):
    """Shared world with per-lane free missions and fixed-z bodies."""

    def _design_scene(self):
        paths=super()._design_scene()
        schema=self.cfg.task.large_map.get("mass_schema",None)
        if schema is not None:
            if schema != "total_articulation_mass_v1":
                raise ValueError("unsupported planar mass schema")
            from omni_drones.controllers.acceleration_to_rpyt import AccelerationToRPYTConfig
            target=float(self.cfg.task.large_map.get("total_mass_kg",float("nan")))
            converter_mass=AccelerationToRPYTConfig.cf21bl_simulation_reference().mass_kg
            if target != .042 or float(self.drone.params["mass"]) != target or converter_mass != target:
                raise ValueError("total mass schema requires configured/robot/converter mass exactly42g")
            self.drone.params["mass_semantics"]=schema
        return paths

    @staticmethod
    def _validate_frozen_config(cfg) -> None:
        raise RuntimeError("Use the explicit portable LargeMapNoGRU task")

    def _build_actor_routes(self):
        # Parent waypoint routing is unused in both modes.  Every serialized
        # mission still passes the complete cached A* + smoothing oracle; only
        # static_route_goal_v1 stores those routes for the planar guide.
        self._planar_sampler = PlanarMissionSampler(
            self.large_scene, self._planar_cfg)
        truth = self.large_scene.dynamic_state(0.)
        route_counts = []
        initial_routes = []
        exact_initial_fallback_count = 0
        for start, goal in zip(self.large_scene.starts, self.large_scene.goals):
            start_clear = self._planar_sampler.point_clearance(
                start, dynamic_positions=truth.position)
            goal_clear = self._planar_sampler.point_clearance(
                goal, dynamic_positions=truth.position)
            separation = float(np.linalg.norm(goal[:2] - start[:2]))
            route = self._planar_sampler.plan_route(start[:2], goal[:2])
            if not route:
                from navacc_large_map_planar_sampling3m import fixed_global_initial_route_witness
                route = fixed_global_initial_route_witness(
                    self._planar_sampler,start[:2],goal[:2],
                    protocol=self.cfg.task.large_map.get('evaluation_protocol',None),
                    guidance_mode=self._planar_guidance_mode)
                if not route and self.cfg.task.large_map.get('goal_curriculum_schema') in ('largemap_no_gru_v1',):
                    from navacc_goal_curriculum import route_witness as curriculum_route_witness
                    route=curriculum_route_witness(self._planar_sampler,start[:2],goal[:2],guidance_mode=self._planar_guidance_mode,cfg=self.cfg)
                if not route and (self.cfg.task.observation.get('mode')=='public21_goal_fixed_si_yaw144_dyn10_v1' or self.cfg.task.large_map.get('overnight_contract_marker') in ('overnight_20261016_v1','executed_smooth_hold30_20261017_v1')):
                    if self.cfg.task.observation.get('mode')=='public21_goal_fixed_si_yaw144_dyn10_v1' or self.cfg.task.large_map.get('overnight_contract_marker')=='executed_smooth_hold30_20261017_v1':
                        from navacc_hold30_transition import route_witness
                    else:
                        from navacc_overnight_hold36 import route_witness
                    route=route_witness(self._planar_sampler,start[:2],goal[:2],guidance_mode=self._planar_guidance_mode)
                exact_initial_fallback_count += int(bool(route))
            if (start_clear < self._planar_cfg.mission_clearance_m
                    or goal_clear < self._planar_cfg.mission_clearance_m
                    or separation < (min(3.,self._planar_cfg.mission_min_separation_m) if self._planar_cfg.easy_mission_probability else self._planar_cfg.mission_min_separation_m)
                    or not route):
                raise RuntimeError(
                    "serialized planar mission failed exact validation")
            route_counts.append(len(route))
            initial_routes.append(route)
        route_guidance = self._planar_guidance_mode == GUIDANCE_SCHEMA
        self._planar_initial_routes = (tuple(initial_routes)
                                       if route_guidance else ())
        self._planar_initial_route_validation = {
            "parent_actor_routes_consumed": False,
            "planar_guidance_routes_consumed": route_guidance,
            "initial_missions_checked": len(route_counts),
            "exact_initial_admission_witness_count": exact_initial_fallback_count,
            "route_waypoint_count_min": min(route_counts),
            "route_waypoint_count_max": max(route_counts),
        }
        return None

    def __init__(self, cfg, headless):
        from navacc_large_map_planar_obs20 import PUBLIC20_MODES
        self._planar_goal_mode=cfg.task.observation.get("mode")=="public21_goal_fixed_si_yaw144_dyn10_v1"
        self._planar_public20 = cfg.task.observation.get("mode", "legacy16") in PUBLIC20_MODES
        self._hold36 = cfg.task.large_map.get("episode_mode") in ("hold30_continuous_v1","hold50_continuous_v1", "obstacle_course_continuous_v1")
        self._hold48 = cfg.task.large_map.get("episode_mode") in ("hold48_v1","hold30_continuous_v1","hold50_continuous_v1", "obstacle_course_continuous_v1")
        self._planar_guidance_mode = str(cfg.task.observation.guidance_mode)
        self._planar_cfg = PlanarSceneConfig(
            mission_xy_half_extent_m=float(
                cfg.task.reset.mission_xy_half_extent_m),
            mission_start_z_m=float(cfg.task.reset.mission_start_z_m),
            mission_goal_z_m=float(cfg.task.reset.mission_goal_z_m),
            mission_clearance_m=float(cfg.task.reset.mission_clearance_m),
            mission_min_separation_m=float(
                cfg.task.reset.mission_min_separation_m),
            mission_max_separation_m=float(cfg.task.reset.get("mission_max_separation_m",60.)),
            easy_mission_probability=float(cfg.task.reset.get("easy_mission_probability",0.)),
            mission_sampling_schema=str(cfg.task.reset.get("mission_sampling_schema","interior_v1")))
        self._planar_reset_epoch = None
        from navacc_large_map_planar_sampling3m import evaluation_deadline_s
        self._planar_reward_cfg = PlanarRewardConfig(
            deadline_s=evaluation_deadline_s(cfg),
            mode=str(cfg.task.reward.get("mode","legacy_pbrs_v2")),
            velocity_rate=float(cfg.task.reward.get("velocity_rate",1.)),
            lidar_risk_rate=float(cfg.task.reward.get("lidar_risk_rate",.1)),
            goal_radius_m=float(cfg.task.reward.get("goal_radius_m",.2)),
            goal_brake_rate=float(cfg.task.reward.get("goal_brake_rate",.2)),
            control_dt_s=float(cfg.sim.dt) * int(cfg.sim.substeps),
            discount_time_constant_s=float(
                cfg.task.reward.discount_time_constant_s),
            gae_time_constant_s=float(cfg.task.reward.gae_time_constant_s),
            height_rate=float(cfg.task.reward.height_interval_rate_per_s),
            height_tolerance_m=float(
                cfg.task.reward.height_interval_tolerance_m),
            potential_scale=float(cfg.task.reward.potential_scale),
            arrival_bonus=float(cfg.task.reward.arrival_bonus_once),
            failure_cost=float(cfg.task.reward.failure_discounted_cost),
            pb_mode=str(cfg.task.reward.get("pb_mode","none")),
            oob_shaping_rate=float(cfg.task.reward.get("oob_shaping_rate",0.)),
            oob_xy_limit_m=float(cfg.task.reward.get("oob_xy_limit_m",24.)),
            near_goal_pbrs_rate=float(
                cfg.task.reward.get("near_goal_pbrs_rate",0.)),
            near_goal_pbrs_max_d_m=float(
                cfg.task.reward.get("near_goal_pbrs_max_d_m",2.5)),
            near_goal_pbrs_inner_m=float(
                cfg.task.reward.get("near_goal_pbrs_inner_m",.5)),
            time_rate=float(cfg.task.reward.get("time_rate",.05)),
            effort_rate=float(cfg.task.reward.get("effort_rate",.02)),
            execution_smooth_rate=float(cfg.task.reward.get("execution_smooth_rate",0.)),
            failure_discount_compensation=cfg.task.reward.get("failure_discount_compensation",True),
            hold_rate=float(cfg.task.reward.get("hold_rate",1.)),
            hold_speed_reference_m_s=float(cfg.task.reward.get("hold_speed_reference_m_s",.5)),
            cbf_correction_coeff=float(
                cfg.task.reward.get("cbf_correction_coeff",.03)))
        super().__init__(cfg, headless)
        from navacc_large_map_planar_unbounded import PROTOCOL
        if str(cfg.task.large_map.get("execution_protocol", "")) == PROTOCOL:
            from dataclasses import replace
            from omni_drones.controllers.acceleration_to_rpyt import AccelerationToRPYT
            self.acceleration_to_rpyt=AccelerationToRPYT(
                replace(self.acceleration_to_rpyt.config,check_tilt=False))
        validate_planar_scene(self.large_scene, config=self._planar_cfg)
        self.planar_backend = PlanarLargeMapBackend(
            self.large_scene, self.num_envs, device=self.device,
            dtype=torch.float32,
            ray_chunk=int(cfg.task.large_map.get("batched_ray_chunk", 32)),
            collision_chunk=int(cfg.task.large_map.get(
                "batched_collision_chunk", 64)),
            static_mode=str(cfg.task.large_map.static_safety_mode),
            observation_mode=(str(cfg.task.observation.mode) if self._planar_public20 else "legacy16"),
            actor_dynamic_radius_m=float(
                cfg.task.observation.actor_dynamic_radius_m),
            actor_dynamic_nearest_count=int(
                cfg.task.observation.actor_dynamic_nearest),
            actor_dynamic_threat_count=int(
                cfg.task.observation.actor_dynamic_cv_threat),
            actor_dynamic_min_relative_speed_m_s=float(
                cfg.task.observation.actor_dynamic_min_relative_speed_m_s),
            c3bf_config=PlanarC3BFConfig(
                eligibility_distance_m=float(
                    cfg.task.large_map.c3bf_xy_center_eligibility_m),
                safety_margin_m=float(cfg.task.large_map.c3bf_safety_margin_m),
                gamma_per_s=float(cfg.task.large_map.c3bf_gamma_per_s),
                hocbf_lambda_per_s=float(
                    cfg.task.large_map.c3bf_hocbf_lambda_per_s),
                minimum_relative_speed_m_s=float(
                    cfg.task.large_map.c3bf_minimum_relative_speed_m_s),
                row_tolerance_m2_s2=float(
                    cfg.task.large_map.c3bf_row_tolerance_m2_s2),
                dt_s=float(cfg.sim.dt) * int(cfg.sim.substeps),
                motor_lag_tau_s=float(cfg.task.large_map.motor_lag_tau_s),
                xy_axis_limit_m_s2=float(
                    cfg.task.action.acceleration_scale_m_s2[0]),
                z_axis_limit_m_s2=float(
                    cfg.task.action.acceleration_scale_m_s2[2]),
                total_limit_m_s2=float(
                    cfg.task.action.total_acceleration_max_m_s2),
                normal_h_recovery_mode=str(
                    cfg.task.large_map.normal_h_recovery_mode)))
        self._planar_guidance = None
        if self._planar_guidance_mode == GUIDANCE_SCHEMA:
            self._planar_guidance = PlanarRouteGuidance(
                self._planar_sampler, self.num_envs,
                lookahead_m=float(cfg.task.observation.route_lookahead_m))
            self._planar_guidance.set_routes(
                range(self.num_envs), self._planar_initial_routes)
        self._planar_last_route_failure = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device)
        self._planar_reset_epoch = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device)
        self._dynamic_clock_origin_s = float(
            cfg.task.large_map.get('dynamic_clock_origin_s', 0.) or 0.)
        self._planar_start_height = self.planar_backend.starts[:, 2].clone()
        self._planar_goal_height = self.planar_backend.goals[:, 2].clone()
        self._planar_arrival_dwell = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device)
        state = self.drone.get_state(env_frame=False).reshape(self.num_envs, -1)
        self._planar_previous_position = state[:, :3].clone()
        if self._planar_goal_mode:
            from navacc_goal_frame import GoalFrame,yaw_from_quaternion
            self._goal_frame=GoalFrame(self.num_envs,device=self.device,dtype=torch.float32)
            self._goal_frame.reset(torch.arange(self.num_envs,device=self.device),state[:,:3],self._world_goals[:,0],yaw_from_quaternion(state[:,3:7]),self._planar_reset_epoch)

        if self._planar_public20:
            self._planar_previous_selected_request = torch.zeros(self.num_envs,3,device=self.device)
            self._planar_previous_selected_request_valid = torch.zeros(self.num_envs,device=self.device,dtype=torch.bool)
        if self._hold48:
            if cfg.task.large_map.get('episode_mode')=='obstacle_course_continuous_v1':
                from navacc_obstacle_course_state import CourseState as Hold48State,CourseCohort as RegisteredCohort
            elif cfg.task.large_map.get('episode_mode')=='hold50_continuous_v1':
                from navacc_hold50_continuous import Hold50State as Hold48State, ContinuousCohort as RegisteredCohort
            elif getattr(self,'_hold36',False):
                from navacc_hold30_continuous import Hold30State as Hold48State, ContinuousCohort as RegisteredCohort
            else:
                from navacc_large_map_hold48 import Hold48State, RegisteredCohort
            self._hold_state=Hold48State(self.num_envs,self.device)
            self._hold_cohort=(RegisteredCohort(self.num_envs,int(cfg.task.large_map.cohort_target_size),context=__import__("omegaconf").OmegaConf.to_container(cfg.task.large_map.course_context,resolve=True)) if cfg.task.large_map.get("episode_mode")=="obstacle_course_continuous_v1" else RegisteredCohort(self.num_envs,int(cfg.task.large_map.cohort_target_size)))
            if getattr(self,'_hold36',False):self._hold36_fixed_completed=torch.zeros(self.num_envs,device=self.device,dtype=torch.bool)
        self._planar_last_static_lidar = torch.zeros(
            self.num_envs, 36, 4, device=self.device)
        from navacc_goal_curriculum import active as _curriculum_active,set_native_activation
        if _curriculum_active(cfg):set_native_activation(self,int(cfg.task.large_map.curriculum_stage))
        self.planar_native_contract = {
            "backend": "shared_dense_planar_primary_ppo_top1_v1",
            "observation_schema": self.planar_backend.actor_observation_schema,
            "actor_task_schema": str(cfg.task.observation.schema),
            "route_guidance_schema": self._planar_guidance_mode,
            "actor_dynamic_slot_policy": "near2_cvthreat2_v1",
            "dynamic_fixed_z": True, "dynamic_count": len(self.large_scene.dynamic_bodies),
            "static_count": len(self.large_scene.static_boxes), "static_mode": self.planar_backend.static_mode,
            "global_clock_reset_with_partial_episode": False,
            "c3bf_selected_rows_max": 1,
            "all_physical_bodies_collision_checked": True,
        }

    def _set_specs(self):
        super()._set_specs()
        # Shared physical truth stays [120,*].  Publishing the inherited
        # [E,120,8] diagnostic would copy the same map into every lane.
        try:
            del self.observation_spec["info", "large_dynamic_state"]
        except KeyError:
            pass
        e, d = self.num_envs, self.device
        scalar = lambda: UnboundedContinuousTensorSpec((e, 1), device=d)
        boolean = lambda: DiscreteTensorSpec(2, (e, 1), dtype=torch.bool, device=d)
        integer = lambda: UnboundedContinuousTensorSpec(
            (e, 1), device=d, dtype=torch.long)
        for key in ("planar_cbf_abort", "planar_cbf_domain_valid",
                    "planar_cbf_initial_g_safe", "planar_cbf_initial_psi1_safe",
                    "planar_cbf_initial_h_safe", "planar_state_nonfinite",
                    "planar_cbf_normal_h_recovery_attempted",
                    "planar_cbf_normal_h_recovery_accepted",
                    "planar_route_failure", "planar_route_replanned",
                    "planar_dynamic_truth_invalid",
                    "planar_cbf_selected_row_feasible",
                    "planar_cbf_selected_row_pass",
                    "planar_cbf_action_bounds_pass",
                    "planar_cbf_selected_is_static",
                    "planar_cbf_all_rows_certified", "planar_mission_forced_opposite",
                    "planar_mission_opposite", "planar_mission_short_component"):
            self.observation_spec["info", key] = boolean()
        for key in ("planar_cbf_selected_entity_id", "planar_cbf_active_count",
                    "planar_cbf_selected_count",
                    "planar_cbf_post_violation_count"):
            self.observation_spec["info", key] = integer()
        for key in ("planar_cbf_normalized_nominal_violation",
                    "planar_cbf_selected_post_residual",
                    "planar_cbf_max_normalized_post_violation",
                    "planar_start_height_m", "planar_goal_height_m"):
            self.observation_spec["info", key] = scalar()
        for kind in ('static','dynamic'):
            for suffix in ('clearance_3d_m','clearance_xy_m'):
                self.observation_spec['info',f'planar_{kind}_{suffix}']=scalar()
            self.observation_spec['info',f'planar_{kind}_encounter_center3m_count']=integer()
            self.observation_spec['info',f'planar_{kind}_clearance_defined']=boolean()
        self.observation_spec["info", "planar_route_progress_m"] = scalar()
        self.observation_spec["info", "planar_route_index"] = integer()
        self.observation_spec["info", "planar_actor_local_goal"] = \
            UnboundedContinuousTensorSpec((e, 3), device=d)
        self.observation_spec[
            "info", "planar_policy_nominal_acceleration"] = \
            UnboundedContinuousTensorSpec((e, 1, 3), device=d)
        self.observation_spec["info", "planar_actor_dynamic_indices"] = \
            UnboundedContinuousTensorSpec((e, 4), device=d, dtype=torch.long)
        for key in ("reward_time", "reward_effort", "reward_execution_smooth", "execution_smooth_delta2", "execution_smooth_valid_steps", "execution_smooth_first_steps", "reward_lidar", "reward_height", "reward_brake", "arrival_speed", "reward_velocity", "reward_running_cost", "reward_terminal", "reward_cbf_violation", "reward_cbf_correction", "cbf_abort", "nonfinite", "navigation_failure"):
            self.observation_spec["stats", key] = scalar()
        if self._planar_reward_cfg.execution_smooth_rate:
            for key in ('planar_execution_smooth_cost','planar_execution_smooth_delta2','planar_execution_smooth_valid','planar_nominal_effort_cost','planar_reward_reconstruction_error'):
                self.observation_spec['info',key]=scalar()
        if self._planar_public20:
            self.observation_spec["agents","observation","state"] = UnboundedContinuousTensorSpec((e,1,21 if self._hold48 else 20),device=d)
            self.observation_spec["info","planar_previous_selected_request_valid"] = boolean()
        if self._hold48:
            self.observation_spec['info','hold48_bootstrap_terminal']=boolean()
            for key in ('raw_in_goal_end','safe_reach_goal_end','ever_reached','first_reach','hold_time_s','first_reach_time_s','first_reach_speed','reward_hold','horizon_reached','deadline_unfinished'):
                self.observation_spec['stats',key]=scalar()
        if self._planar_goal_mode:
            self.observation_spec['agents','observation','dynamic_obstacle']=UnboundedContinuousTensorSpec((e,1,4,10),device=d)
        self.info = self.observation_spec["info"].zero()
        self.stats = self.observation_spec["stats"].zero()

    def _sync_dynamic_prims(self):
        origin = getattr(self, '_dynamic_clock_origin_s', 0.)
        phase_time = self._global_time_s - origin
        if phase_time < 0:
            # The parent constructor calls this before the phase clock is
            # restored; that call only needs the initial body positions.
            phase_time = 0.
        truth = self.large_scene.dynamic_state(phase_time)
        expected_z = np.asarray([
            body.initial_position[2] for body in self.large_scene.dynamic_bodies])
        if (not np.allclose(truth.position[:, 2], expected_z, atol=0, rtol=0)
                or not np.allclose(truth.velocity[:, 2], 0, atol=0, rtol=0)):
            raise RuntimeError("planar native dynamic truth changed z or vz")
        positions = torch.as_tensor(np.array(truth.position, copy=True),
                                    device=self.device, dtype=torch.float32)
        self._planar_dynamic_velocity = torch.as_tensor(
            np.array(truth.velocity, copy=True), device=self.device,
            dtype=torch.float32)
        rotations = torch.zeros(len(positions), 4, device=self.device)
        rotations[:, 0] = 1
        if self._dynamic_view is not None: self._dynamic_view.set_world_poses(positions, rotations)
        physical_velocity = torch.zeros(len(positions), 6, device=self.device,
                                        dtype=positions.dtype)
        physical_velocity[:, :3] = self._planar_dynamic_velocity
        if self._dynamic_view is not None: self._dynamic_view.set_velocities(physical_velocity)
        self._planar_dynamic_position = positions

    def _shared_dynamic_truth(self):
        if self._dynamic_view is None:
            self._planar_dynamic_truth_invalid = False
            return self._planar_dynamic_position, self._planar_dynamic_velocity
        positions, _ = self._dynamic_view.get_world_poses(clone=True)
        physical_velocity = self._dynamic_view.get_velocities(clone=True)
        n = len(self.large_scene.dynamic_bodies)
        if positions.shape != (n, 3) or physical_velocity.shape != (n, 6):
            raise RuntimeError("planar physical dynamic views changed shape")
        positions = positions.to(device=self.device, dtype=torch.float32)
        physical_velocity = physical_velocity.to(
            device=self.device, dtype=torch.float32)
        expected_position = self._planar_dynamic_position
        expected_velocity = self._planar_dynamic_velocity
        invalid = (~torch.isfinite(positions).all()
                   or ~torch.isfinite(physical_velocity).all()
                   or ~torch.isfinite(expected_position).all()
                   or ~torch.isfinite(expected_velocity).all()
                   or not torch.allclose(
                       positions, expected_position, atol=1e-4, rtol=0)
                   or not torch.allclose(
                       physical_velocity[:, :3], expected_velocity,
                       atol=1e-6, rtol=0)
                   or not torch.allclose(
                       positions[:, 2], expected_position[:, 2],
                       atol=1e-5, rtol=0)
                   or not torch.allclose(
                       physical_velocity[:, 2], torch.zeros_like(
                           physical_velocity[:, 2]), atol=1e-6, rtol=0))
        self._planar_dynamic_truth_invalid = bool(invalid)
        return positions, physical_velocity[:, :3]

    def _reset_goal_frame(self,ids,starts,goals):
        if getattr(self,'_goal_frame',None) is None:return
        from navacc_goal_frame import yaw_from_quaternion
        _,rotation=self.drone.get_world_poses(clone=True)
        yaw=yaw_from_quaternion(rotation.reshape(self.num_envs,4)[ids])
        self._goal_frame.reset(ids,starts,goals,yaw,self._planar_reset_epoch[ids])

    def _reset_idx(self, env_ids):
        ids = torch.as_tensor(env_ids, device=self.device,
                              dtype=torch.long).reshape(-1)
        super()._reset_idx(ids)
        if getattr(self,'_hold48',False) and hasattr(self,'_hold_state'):
            self._hold_state.reset(ids)
            # Mission epoch increments later, for both fixed and sampled reset.
            cohort_ids=ids[~self._hold36_fixed_completed[ids]] if getattr(self,'_hold36',False) and getattr(self,'_planar_fixed_eval',False) else ids
            self._hold_cohort.on_reset(cohort_ids.tolist(),(self._planar_reset_epoch[cohort_ids]+1).tolist(),**({"clock_s":float(self._global_time_s)} if getattr(self,'_hold36',False) else {}))
        if getattr(self,"_planar_public20",False) and hasattr(self,"_planar_previous_selected_request"):
            self._planar_previous_selected_request[ids] = 0
            self._planar_previous_selected_request_valid[ids] = False
        if self._planar_reset_epoch is None or not hasattr(self, "planar_backend"):
            return
        dynamic_position, _ = self._shared_dynamic_truth()
        if self._planar_dynamic_truth_invalid:
            raise RuntimeError(
                "cannot sample a reset mission from invalid physical dynamic truth")
        if getattr(self,"_planar_fixed_eval",False):
            starts_t = torch.as_tensor(self.large_scene.starts,device=self.device,dtype=torch.float32)[ids]
            goals_t = torch.as_tensor(self.large_scene.goals,device=self.device,dtype=torch.float32)[ids]
            all_starts=self.planar_backend.starts.clone();all_goals=self.planar_backend.goals.clone()
            all_starts[ids]=starts_t;all_goals[ids]=goals_t
            self.planar_backend.set_missions(all_starts,all_goals)
            self._world_goals[ids,0]=goals_t
            _,rotation=self.drone.get_world_poses(clone=True)
            self.drone.set_world_poses(starts_t[:,None],rotation[ids],ids)
            self.drone.set_velocities(torch.zeros_like(self.init_vels[ids]),ids)
            self._planar_start_height[ids]=starts_t[:,2];self._planar_goal_height[ids]=goals_t[:,2]
            self._planar_arrival_dwell[ids]=0;self._planar_previous_position[ids]=starts_t
            self._large_previous_positions[ids]=starts_t
            self.info['planar_mission_forced_opposite'][ids]=False
            self.info['planar_mission_short_component'][ids]=False
            opposite=((starts_t[:,0].abs()==24.) & (goals_t[:,0]==-starts_t[:,0])) | ((starts_t[:,1].abs()==24.) & (goals_t[:,1]==-starts_t[:,1]))
            self.info['planar_mission_opposite'][ids]=opposite[:,None]
            if self._hold48:self._planar_reset_epoch[ids]+=1
            self._reset_goal_frame(ids,starts_t,goals_t)
            return
        dynamic_position_cpu = dynamic_position.detach().cpu().numpy()
        # One deterministic stream per selected lane means partial resets never
        # perturb missions of other lanes or the shared obstacle clock.
        starts, goals, routes = [], [], []
        mixed_records=[]
        for env_id in ids.detach().cpu().tolist():
            seed = int(self.large_scene.seed) ^ 0x504C414E \
                ^ (env_id * 0x9E3779B1) ^ int(self._planar_reset_epoch[env_id])
            sampled = sample_free_planar_missions(
                self.large_scene, 1, seed=seed, config=self._planar_cfg,
                dynamic_positions=dynamic_position_cpu,
                sampler=self._planar_sampler,
                return_routes=self._planar_guidance is not None)
            if self._planar_guidance is not None:
                sampled_start, sampled_goal, _, sampled_routes = sampled
                routes.append(sampled_routes[0])
            else:
                sampled_start, sampled_goal, _ = sampled
            self.info['planar_mission_forced_opposite'][env_id]=bool(sampled[2].get('accepted_forced_components',[False])[0])
            component=bool(sampled[2].get('accepted_short_components',[False])[0])
            self.info['planar_mission_short_component'][env_id]=component
            if self._planar_cfg.mission_sampling_schema in ('navrl_edges_short_full50_xy_v1','navrl_edges_short_full30_xy_v1','navrl_edges_short_full20_xy_v1','navrl_edges_short_full15_xy_v1','navrl_edges_short_full10_xy_v1','navrl_edges_short_full05_xy_v1','navrl_edges_short_full00_xy_v1'):
                mixed_records.append(dict(env_id=env_id,episode_epoch=int(self._planar_reset_epoch[env_id])+1,short_component=component,xy_distance_m=float(np.linalg.norm(sampled_goal[0,:2]-sampled_start[0,:2])),pair_proposals_full_short=sampled[2]['component_pair_proposals_full_short'],rejections_full_short=sampled[2]['component_rejections_full_short']))
            if self.cfg.task.large_map.get('goal_curriculum_schema') in ('largemap_no_gru_v1',) and mixed_records:
                endpoint_clearance=min(self._planar_sampler.point_clearance(sampled_start[0],dynamic_positions=dynamic_position_cpu),self._planar_sampler.point_clearance(sampled_goal[0],dynamic_positions=dynamic_position_cpu))
                if endpoint_clearance<.3-1e-8:raise ValueError('actual dynamic-clock reset clearance failed')
                mixed_records[-1].update(actual_dynamic_clock_s=float(self._global_time_s),minimum_endpoint_clearance_m=float(endpoint_clearance))
            ss,gg=sampled_start[0],sampled_goal[0]
            opposite=bool((abs(ss[0])==24. and gg[0]==-ss[0]) or (abs(ss[1])==24. and gg[1]==-ss[1]))
            self.info['planar_mission_opposite'][env_id]=opposite
            starts.append(sampled_start[0]); goals.append(sampled_goal[0])
            self._planar_reset_epoch[env_id] += 1
        if mixed_records and getattr(self,'_mixed_sampling_receipt_path',None) is not None:
            import json
            record=dict(schema=('full_only_per_episode_reset_v1' if self._planar_cfg.mission_sampling_schema=='navrl_edges_short_full00_xy_v1' else 'mixed10_per_episode_reset_v1' if self._planar_cfg.mission_sampling_schema=='navrl_edges_short_full10_xy_v1' else 'mixed05_per_episode_reset_v1' if self._planar_cfg.mission_sampling_schema=='navrl_edges_short_full05_xy_v1' else 'mixed15_per_episode_reset_v1' if self._planar_cfg.mission_sampling_schema=='navrl_edges_short_full15_xy_v1' else 'mixed20_per_episode_reset_v1' if self._planar_cfg.mission_sampling_schema=='navrl_edges_short_full20_xy_v1' else 'mixed30_per_episode_reset_v1' if self._planar_cfg.mission_sampling_schema=='navrl_edges_short_full30_xy_v1' else 'mixed50_per_episode_reset_v1'),short_component_probability=(0. if self._planar_cfg.mission_sampling_schema=='navrl_edges_short_full00_xy_v1' else .1 if self._planar_cfg.mission_sampling_schema=='navrl_edges_short_full10_xy_v1' else .05 if self._planar_cfg.mission_sampling_schema=='navrl_edges_short_full05_xy_v1' else .15 if self._planar_cfg.mission_sampling_schema=='navrl_edges_short_full15_xy_v1' else .2 if self._planar_cfg.mission_sampling_schema=='navrl_edges_short_full20_xy_v1' else .3 if self._planar_cfg.mission_sampling_schema=='navrl_edges_short_full30_xy_v1' else .5),global_clock_s=float(self._global_time_s),component_selection='one independent Bernoulli per new episode, held across all rejection',count=len(mixed_records),short_component_count=sum(x['short_component'] for x in mixed_records),full_component_count=sum(not x['short_component'] for x in mixed_records),episodes=mixed_records)
            with Path(self._mixed_sampling_receipt_path).open('a') as log:log.write(json.dumps(record)+'\n')
        starts_t = torch.as_tensor(np.asarray(starts), device=self.device,
                                   dtype=torch.float32)
        goals_t = torch.as_tensor(np.asarray(goals), device=self.device,
                                  dtype=torch.float32)
        all_starts = self.planar_backend.starts.clone()
        all_goals = self.planar_backend.goals.clone()
        all_starts[ids] = starts_t; all_goals[ids] = goals_t
        self.planar_backend.set_missions(all_starts, all_goals)
        if self._planar_guidance is not None:
            self._planar_guidance.set_routes(ids.detach().cpu().tolist(), routes)
        self._planar_last_route_failure[ids] = False
        self._world_goals[ids, 0] = goals_t
        current_pos, current_rot = self.drone.get_world_poses(clone=True)
        self.drone.set_world_poses(starts_t[:, None], current_rot[ids], ids)
        self.drone.set_velocities(torch.zeros_like(self.init_vels[ids]), ids)
        target_rot = torch.zeros(len(ids), 4, device=self.device); target_rot[:, 0] = 1
        self.target_vis.set_world_poses(goals_t, target_rot, env_indices=ids)
        self._planar_start_height[ids] = starts_t[:, 2]
        self._planar_goal_height[ids] = goals_t[:, 2]
        self._planar_arrival_dwell[ids] = 0
        self._planar_previous_position[ids] = starts_t
        self._large_previous_positions[ids] = starts_t
        self._reset_goal_frame(ids,starts_t,goals_t)

    def _capture_large_collisions(self):
        state = self.drone.get_state(env_frame=False).reshape(self.num_envs, -1)
        dynamic_position, _ = self._shared_dynamic_truth()
        finite = torch.isfinite(state[:, :3]).all(-1)
        static = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        dynamic = torch.zeros_like(static)
        if bool(finite.any()):
            finite_static, finite_dynamic = self.planar_backend.collision_flags(
                state[finite, :3], torch.nan_to_num(
                    dynamic_position, nan=0., posinf=20., neginf=-20.))
            static[finite] = finite_static; dynamic[finite] = finite_dynamic
        self._large_static_latched |= static[:, None]
        self._large_dynamic_latched |= dynamic[:, None]
        self._large_previous_positions.copy_(state[:, :3])

    def _compute_state_and_obs(self):
        self.drone_state = self.drone.get_state(env_frame=False)
        raw_state = self.drone_state.reshape(self.num_envs, -1)
        state_nonfinite = ~torch.isfinite(raw_state).all(-1)
        state = torch.where(state_nonfinite[:, None], torch.zeros_like(raw_state),
                            raw_state)
        self.info["drone_state"].copy_(state[:, :13].reshape(
            self.num_envs, 1, 13))
        self.info["planar_state_nonfinite"].copy_(state_nonfinite[:, None])
        position, quat, velocity = state[:, :3], state[:, 3:7], state[:, 7:10]
        gyro = quat_rotate_inverse(quat, state[:, 10:13])
        dynamic_position, dynamic_velocity = self._shared_dynamic_truth()
        dynamic_truth_invalid = torch.full(
            (self.num_envs,), self._planar_dynamic_truth_invalid,
            device=self.device, dtype=torch.bool)
        if self._planar_cfg.mission_sampling_schema in ('navrl_edges_xy_v1','navrl_edges_opposite50_xy_v1'):
            from navacc_large_map_planar_encounters import obstacle_observations
            telemetry=obstacle_observations(self.planar_backend,position,dynamic_position)
            for key,value in telemetry.items():self.info[key].copy_(value)
        if self._planar_goal_mode and self._planar_dynamic_truth_invalid:raise ValueError('goal-frame cannot hide invalid physical obstacle truth')
        actor_dynamic_position = torch.nan_to_num(
            dynamic_position, nan=0., posinf=20., neginf=-20.)
        actor_dynamic_velocity = torch.nan_to_num(
            dynamic_velocity, nan=0., posinf=0., neginf=0.)
        if self._planar_guidance is not None:
            guidance = self._planar_guidance.local_goals(
                position, self._world_goals[:, 0])
            local_goal = guidance.local_goal
            route_failure = guidance.route_failure
            route_replanned = guidance.replanned
            route_index = guidance.route_index
            route_progress = guidance.route_progress_m
        else:
            local_goal = self._world_goals[:, 0]
            route_failure = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device)
            route_replanned = route_failure
            route_index = torch.full(
                (self.num_envs,), -1, dtype=torch.long, device=self.device)
            route_progress = torch.zeros(
                self.num_envs, dtype=position.dtype, device=self.device)
        self._planar_last_route_failure.copy_(route_failure)
        obs = self.planar_backend.actor_observation(
            position, velocity, quat, gyro,
            self.info["yaw_hold_W"].reshape(self.num_envs),
            self.progress_buf / self.max_episode_length,
            actor_dynamic_position, actor_dynamic_velocity,
            local_goal=(local_goal if self._planar_guidance is not None
                        else None),
            previous_selected_request=(self._planar_previous_selected_request if self._planar_public20 else None),
            ever_reached=(self._hold_state.ever if self._hold48 else None),
            goal_frame_theta=(self._goal_frame.theta if self._planar_goal_mode else None),
            history_valid=(self._planar_previous_selected_request_valid if self._planar_goal_mode else None))
        if self._planar_public20:
            self.info["planar_previous_selected_request_valid"].copy_(self._planar_previous_selected_request_valid[:,None])
        self._planar_last_static_lidar.copy_(obs["static_lidar"])
        self.info["large_nearest_dynamic_indices"].copy_(
            obs["nearest_dynamic_indices"])
        self.info["planar_actor_dynamic_indices"].copy_(
            obs["actor_dynamic_indices"])
        self.info["large_global_time_s"].fill_(self._global_time_s)
        self.info["large_collision_static"].copy_(self._large_static_latched)
        self.info["large_collision_dynamic"].copy_(self._large_dynamic_latched)
        self.info["planar_dynamic_truth_invalid"].copy_(
            dynamic_truth_invalid[:, None])
        self.info["planar_route_failure"].copy_(route_failure[:, None])
        self.info["planar_route_replanned"].copy_(route_replanned[:, None])
        self.info["planar_route_index"].copy_(route_index[:, None])
        self.info["planar_route_progress_m"].copy_(
            route_progress[:, None])
        self.info["planar_actor_local_goal"].copy_(local_goal)
        self.info["large_goal_distance_m"].copy_(torch.linalg.vector_norm(
            self._world_goals[:, 0] - position, dim=-1, keepdim=True))
        self.info["planar_start_height_m"].copy_(
            self._planar_start_height[:, None])
        self.info["planar_goal_height_m"].copy_(
            self._planar_goal_height[:, None])
        self._hold_next_state = obs["state"] if self._hold48 else None
        return TensorDict({
            "agents": {"observation": {
                "state": obs["state"].unsqueeze(1),
                "static_lidar": obs["static_lidar"].unsqueeze(1),
                "dynamic_obstacle": obs["dynamic_obstacle"].unsqueeze(1),
            }, "intrinsics": self.drone.intrinsics},
            "info": self.info.clone(), "stats": self.stats.clone(),
        }, self.batch_size)

    def _compute_reward_and_done(self):
        state = self.drone_state.reshape(self.num_envs, -1)
        position, velocity = state[:, :3], state[:, 7:10]
        safe_state = torch.nan_to_num(state, nan=0., posinf=0., neginf=0.)
        quat = safe_state[:, 3:7]
        tilt_cos = (1 - 2 * (quat[:, 1].square() + quat[:, 2].square())).clamp(-1, 1)
        crash = (position[:, 2] < .15) | (torch.acos(tilt_cos) > math.radians(40))
        oob = (position[:, :2].abs() > 25).any(-1) | (position[:, 2] > 4.5)
        nonfinite = ~torch.isfinite(state).all(-1)
        cbf_abort = self.info["planar_cbf_abort"].reshape(self.num_envs) \
            | self.info["planar_dynamic_truth_invalid"].reshape(self.num_envs)
        navigation_failure = self._planar_last_route_failure
        reward = planar_reward_step(
            self._planar_previous_position, position, velocity,
            self._world_goals[:, 0],
            self.info["planar_policy_nominal_acceleration"].reshape(
                self.num_envs, 3),
            (self._planar_last_static_lidar / 2. if self._planar_public20 else self._planar_last_static_lidar), self._planar_start_height,
            self._planar_goal_height, self._planar_arrival_dwell,
            self.progress_buf,
            collision_static=self._large_static_latched[:, 0],
            collision_dynamic=self._large_dynamic_latched[:, 0],
            crash=crash, out_of_bounds=oob, nonfinite=nonfinite,
            cbf_abort=(cbf_abort | navigation_failure),
            config=self._planar_reward_cfg,
            cbf_reward=(self.info["planar_reward_cbf_violation"]+self.info["planar_reward_cbf_correction"]).reshape(self.num_envs) if self._planar_reward_cfg.mode in ("zero_curriculum_v1","hold48_v1","hold30_continuous_v1", "hold50_continuous_v1", "obstacle_course_continuous_v1") else None,
            previous_ever_reached=(self._hold_state.ever if self._hold48 else None),
            executed_acceleration=self.current_a_nom.reshape(self.num_envs,3),
            previous_executed_acceleration=self.previous_a_nom.reshape(self.num_envs,3),
            previous_execution_valid=self.previous_a_nom_valid.reshape(self.num_envs))
        if self._planar_reward_cfg.execution_smooth_rate:
            nonfinite=reward.effective_nonfinite
            _valid=self.previous_a_nom_valid.reshape(self.num_envs)
            _delta2=reward.execution_smooth_cost/(self._planar_reward_cfg.control_dt_s*self._planar_reward_cfg.execution_smooth_rate)
            self.info['planar_execution_smooth_cost'].copy_(reward.execution_smooth_cost[:,None])
            self.info['planar_execution_smooth_delta2'].copy_(_delta2[:,None])
            self.info['planar_execution_smooth_valid'].copy_(_valid[:,None])
            self.info['planar_nominal_effort_cost'].copy_(reward.effort_cost[:,None])
            _rebuilt=reward.shaping-reward.running_cost+reward.terminal_reward-reward.oob_cost+reward.near_goal_pbrs_cost+reward.hold_reward+(self.info['planar_reward_cbf_violation']+self.info['planar_reward_cbf_correction']).reshape(self.num_envs)
            self.info['planar_reward_reconstruction_error'].copy_((reward.reward-_rebuilt).abs()[:,None])
            self.stats['execution_smooth_delta2']+=_delta2[:,None]
            self.stats['execution_smooth_valid_steps']+=_valid[:,None]
            self.stats['execution_smooth_first_steps']+=(~_valid)[:,None]
        if self._hold48:
            speed_actual=torch.linalg.vector_norm(velocity,dim=-1)
            self._hold_state.step(reward.raw_in_goal,reward.failure,speed_actual,self.progress_buf,
                self._planar_reward_cfg.control_dt_s,self._planar_reward_cfg.hold_rate,self._planar_reward_cfg.hold_speed_reference_m_s)
            # IsaacEnv builds NEXT observation before reward: mutate its shared
            # state tensor before _step returns, preserving same-transition flag.
            if self._hold_next_state is not None:self._hold_next_state[:,20].copy_(self._hold_state.ever)
            done_hold=reward.terminated | reward.truncated
            self.info['hold48_bootstrap_terminal'].copy_(done_hold[:,None])
            for key,value in {'raw_in_goal_end':reward.raw_in_goal,'safe_reach_goal_end':reward.raw_in_goal & ~reward.failure,
                'ever_reached':self._hold_state.ever,'first_reach':reward.first_reach,'hold_time_s':self._hold_state.hold_time,
                'first_reach_time_s':self._hold_state.first_time,'first_reach_speed':self._hold_state.first_speed}.items():
                self.stats[key].copy_(value[:,None])
            self.stats['horizon_reached'].copy_(reward.truncated[:,None])
            self.stats['deadline_unfinished'].copy_((reward.truncated & ~reward.raw_in_goal)[:,None])
            self.stats['reward_hold']+=reward.hold_reward[:,None]
        self._planar_arrival_dwell.copy_(reward.dwell_steps)
        self._large_arrived.copy_(reward.arrival[:, None])
        distance = torch.linalg.vector_norm(
            self._world_goals[:, 0] - position, dim=-1)
        speed = torch.linalg.vector_norm(velocity, dim=-1)
        self.stats["arrival_speed"].copy_(torch.where(reward.arrival,speed,torch.zeros_like(speed))[:,None])
        self.stats["reward_time"] += reward.time_cost[:,None]
        self.stats["reward_effort"] += reward.effort_cost[:,None]
        self.stats["reward_execution_smooth"] += reward.execution_smooth_cost[:,None]
        self.stats["reward_lidar"] += reward.lidar_cost[:,None]
        self.stats["reward_height"] += reward.height_cost[:,None]
        self.stats["reward_brake"] += reward.brake_cost[:,None]
        self.stats["reward_velocity"] += reward.shaping[:,None]
        self.stats["reward_running_cost"] += reward.running_cost[:,None]
        self.stats["reward_terminal"] += reward.terminal_reward[:,None]
        if self._planar_reward_cfg.mode in ("zero_curriculum_v1", "hold48_v1", "hold30_continuous_v1", "hold50_continuous_v1", "obstacle_course_continuous_v1"):
            self.stats["reward_cbf_violation"] += self.info["planar_reward_cbf_violation"]
            self.stats["reward_cbf_correction"] += self.info["planar_reward_cbf_correction"]
        self.stats["return"] += reward.reward[:, None]
        self.stats["episode_len"][:] = self.progress_buf[:, None]
        self.stats["pos_error"].copy_(distance[:, None])
        self.stats["speed"].copy_(speed[:, None])
        self.stats["crash"].copy_(crash[:, None].float())
        self.stats["oob"].copy_(oob[:, None].float())
        self.stats["arrival"].copy_(reward.arrival[:, None].float())
        self.stats["collision_static"].copy_(self._large_static_latched.float())
        self.stats["collision_dynamic"].copy_(self._large_dynamic_latched.float())
        self.stats["timeout"].copy_(reward.timeout[:, None].float())
        self.stats["cbf_abort"].copy_(cbf_abort[:, None].float())
        self.stats["nonfinite"].copy_(nonfinite[:, None].float())
        self.stats["navigation_failure"].copy_(
            navigation_failure[:, None].float())
        reason = torch.zeros_like(self.info["termination_reason"])
        reason = torch.where(reward.arrival[:, None], torch.full_like(reason, 5), reason)
        reason = torch.where(self._large_static_latched, torch.full_like(reason, 6), reason)
        reason = torch.where(self._large_dynamic_latched, torch.full_like(reason, 7), reason)
        reason = torch.where(crash[:, None], torch.ones_like(reason), reason)
        reason = torch.where(oob[:, None], torch.full_like(reason, 2), reason)
        reason = torch.where(nonfinite[:, None], torch.full_like(reason, 3), reason)
        reason = torch.where(cbf_abort[:, None], torch.full_like(reason, 8), reason)
        reason = torch.where(navigation_failure[:, None],
                             torch.full_like(reason, 9), reason)
        reason = torch.where(reward.timeout[:, None], torch.full_like(reason, 4), reason)
        if self._hold48:
            # Explicit new normal horizon labels: 10 inside, 11 outside. Both
            # are completed finite missions, never physical timeout penalties.
            reason=torch.where(reward.truncated[:,None],torch.where(reward.raw_in_goal[:,None],torch.full_like(reason,10),torch.full_like(reason,11)),reason)
        self.info["termination_reason"].copy_(reason)
        if self._hold48:
            cohort_done=done_hold
            if getattr(self,'_hold36',False) and getattr(self,'_planar_fixed_eval',False):
                cohort_done=done_hold & ~self._hold36_fixed_completed
                self._hold36_fixed_completed |= cohort_done
            self._hold_cohort.on_step(cohort_done,self._planar_reset_epoch,{
                'raw_in_goal_end':reward.raw_in_goal,'safe_reach_goal_end':reward.raw_in_goal & ~reward.failure,
                'ever_reached':self._hold_state.ever,'fatal':reward.failure,'first_reach_time_s':self._hold_state.first_time,
                'hold_time_s':self._hold_state.hold_time,'arrival_speed_first':self._hold_state.first_speed,
                'end_distance':(self._world_goals[:,0]-position).norm(dim=-1),'end_speed':speed_actual,
                'collision':(reason.reshape(-1)==6)|(reason.reshape(-1)==7),'crash':reason.reshape(-1)==1,
                'oob':reason.reshape(-1)==2,'deadline_unfinished':reward.truncated & ~reward.raw_in_goal,
                'horizon_reached':reward.truncated},**({'clock_s':float(self._global_time_s)} if getattr(self,'_hold36',False) else {}))

        self.previous_a_nom.copy_(self.current_a_nom)
        self.previous_a_nom_valid.fill_(True)
        self._planar_previous_position.copy_(torch.nan_to_num(
            position, nan=0., posinf=0., neginf=0.))
        done = (reward.terminated | reward.truncated)[:, None]
        return TensorDict({
            "agents": {"reward": reward.reward[:, None, None]},
            "done": done, "terminated": reward.terminated[:,None],
            "truncated": reward.truncated[:,None],
            "info": self.info.clone(), "stats": self.stats.clone(),
        }, self.batch_size)


__all__ = ["GLOBAL_GUIDANCE_SCHEMA", "GLOBAL_OBSERVATION_SCHEMA",
           "ROUTE_OBSERVATION_SCHEMA", "NavAccNavRLSharedPlanar"]


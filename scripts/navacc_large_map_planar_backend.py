#!/usr/bin/env python3
"""Device-resident dense-planar observation, collision and top-one C3BF backend."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import torch

from navacc_large_map import LargeMapScene
from navacc_large_map_batched import BatchedLargeMapBackend, UAV_RADIUS_M
from navacc_large_map_planar_c3bf import (
    PlanarC3BFConfig, PlanarC3BFResult, bound_primary_acceleration,
    filter_planar_top1,
)
from navacc_large_map_planar_scene import validate_planar_scene


ACTOR_OBSERVATION_SCHEMA = "near2_cvthreat2_v1"


def _stable_lexicographic_order(*fields: torch.Tensor) -> torch.Tensor:
    """Return row-wise order for fields listed most-to-least significant."""
    if not fields or any(field.ndim != 2 for field in fields):
        raise ValueError("ranking fields must be non-empty [E,N] tensors")
    order = torch.arange(fields[0].shape[1], device=fields[0].device) \
        .expand(fields[0].shape[0], -1)
    for field in reversed(fields):
        order = order.gather(1, torch.argsort(
            field.gather(1, order), dim=1, stable=True))
    return order


def _constant_velocity_entry_time(relative_position: torch.Tensor,
                                  relative_velocity: torch.Tensor,
                                  radius: torch.Tensor | float, *,
                                  minimum_speed_m_s: float) -> torch.Tensor:
    """Current-CV first entry time for a closed XY disc, or +inf.

    This is an actor-slot ranking heuristic.  It does not inspect the future
    velocity trace and is not a C3BF or collision certificate.
    """
    speed2 = relative_velocity.square().sum(-1)
    dot = (relative_position * relative_velocity).sum(-1)
    radius_t = torch.as_tensor(
        radius, device=relative_position.device,
        dtype=relative_position.dtype)
    clearance2 = relative_position.square().sum(-1) - radius_t.square()
    inside = clearance2 <= 0
    discriminant = dot.square() - speed2 * clearance2
    moving = speed2 > minimum_speed_m_s ** 2
    closing = dot < 0
    tangent_or_crossing = discriminant >= 0
    safe_speed2 = speed2.clamp_min(minimum_speed_m_s ** 2)
    root = (-dot - torch.sqrt(discriminant.clamp_min(0))) / safe_speed2
    finite_entry = moving & closing & tangent_or_crossing & (root >= 0)
    result = torch.full_like(speed2, float("inf"))
    result = torch.where(finite_entry, root, result)
    return torch.where(inside, torch.zeros_like(result), result)


def rank_planar_actor_dynamic_slots(relative_position_xy: torch.Tensor,
                                    relative_velocity_xy: torch.Tensor,
                                    body_radius_m: torch.Tensor,
                                    body_ids: torch.Tensor, *,
                                    actor_radius_m: float = 4.0,
                                    eligibility_radius_m: float = 3.0,
                                    uav_radius_m: float = .1,
                                    safety_margin_m: float = .05,
                                    minimum_speed_m_s: float = 1e-6,
                                    ) -> dict[str, torch.Tensor]:
    """Select two nearest and two current-CV threat slots without duplicates."""
    if (relative_position_xy.ndim != 3
            or relative_position_xy.shape[-1] != 2
            or relative_velocity_xy.shape != relative_position_xy.shape
            or body_radius_m.shape != (relative_position_xy.shape[1],)
            or body_ids.shape != (relative_position_xy.shape[1],)
            or not bool(torch.isfinite(relative_position_xy).all())
            or not bool(torch.isfinite(relative_velocity_xy).all())
            or not bool(torch.isfinite(body_radius_m).all())):
        raise ValueError("dynamic actor ranking inputs are invalid")
    if (not all(math.isfinite(value) and value > 0 for value in (
            actor_radius_m, eligibility_radius_m, uav_radius_m,
            safety_margin_m, minimum_speed_m_s))
            or actor_radius_m <= eligibility_radius_m):
        raise ValueError("dynamic actor ranking constants are invalid")
    e, n, _ = relative_position_xy.shape
    if n == 0:
        missing = torch.full((e, 4), -1, device=body_ids.device,
                             dtype=torch.long)
        return {"closest_indices": missing.clone(),
                "actor_indices": missing.clone(),
                "actor_body_ids": missing.clone()}
    distances = torch.linalg.vector_norm(relative_position_xy, dim=-1)
    ids = body_ids[None].expand(e, -1)
    nearest_order = _stable_lexicographic_order(distances, ids)
    closest = nearest_order[:, :min(4, n)]
    if closest.shape[1] < 4:
        closest = torch.cat((closest, torch.full(
            (e, 4 - closest.shape[1]), -1, device=closest.device,
            dtype=torch.long)), 1)

    candidate = distances <= actor_radius_m
    near = nearest_order[:, :min(2, n)]
    near_valid = candidate.gather(1, near)
    near = torch.where(near_valid, near, torch.full_like(near, -1))
    if near.shape[1] < 2:
        near = torch.cat((near, torch.full(
            (e, 2 - near.shape[1]), -1, device=near.device,
            dtype=torch.long)), 1)
    excluded_count = torch.zeros(e, n, device=near.device, dtype=torch.long)
    excluded_count.scatter_add_(
        1, near.clamp_min(0), (near >= 0).to(torch.long))
    available = candidate & (excluded_count == 0)

    risk_radius = body_radius_m[None] + uav_radius_m + safety_margin_m
    collision_ttc = _constant_velocity_entry_time(
        relative_position_xy, relative_velocity_xy, risk_radius,
        minimum_speed_m_s=minimum_speed_m_s)
    eligibility_ttc = _constant_velocity_entry_time(
        relative_position_xy, relative_velocity_xy, eligibility_radius_m,
        minimum_speed_m_s=minimum_speed_m_s)
    inside_risk = distances <= risk_radius
    category = torch.where(
        inside_risk, torch.zeros_like(distances, dtype=torch.long),
        torch.where(torch.isfinite(collision_ttc),
                    torch.ones_like(distances, dtype=torch.long),
                    torch.where(torch.isfinite(eligibility_ttc),
                                torch.full_like(distances, 2, dtype=torch.long),
                                torch.full_like(distances, 3, dtype=torch.long))))
    ranking_time = torch.where(
        inside_risk, torch.zeros_like(distances),
        torch.where(torch.isfinite(collision_ttc), collision_ttc,
                    torch.where(torch.isfinite(eligibility_ttc),
                                eligibility_ttc,
                                torch.full_like(distances, float("inf")))))
    category = torch.where(available, category,
                           torch.full_like(category, 4))
    threat_order = _stable_lexicographic_order(
        category, ranking_time, distances, ids)
    threat = threat_order[:, :min(2, n)]
    threat_valid = available.gather(1, threat)
    threat = torch.where(threat_valid, threat, torch.full_like(threat, -1))
    if threat.shape[1] < 2:
        threat = torch.cat((threat, torch.full(
            (e, 2 - threat.shape[1]), -1, device=threat.device,
            dtype=torch.long)), 1)
    actor = torch.cat((near, threat), 1)
    safe_actor = actor.clamp_min(0)
    actor_body_ids = body_ids[safe_actor]
    actor_body_ids = torch.where(
        actor >= 0, actor_body_ids, torch.full_like(actor_body_ids, -1))
    return {"closest_indices": closest, "actor_indices": actor,
            "actor_body_ids": actor_body_ids,
            "collision_ttc_s": collision_ttc,
            "eligibility_entry_ttc_s": eligibility_ttc,
            "risk_radius_m": risk_radius.expand(e, -1)}


@dataclass(frozen=True)
class PlanarCommand:
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
    selected_post_residual: torch.Tensor
    eligible_post_violation_count: torch.Tensor
    max_normalized_post_violation: torch.Tensor
    claim_all_rows_certified: torch.Tensor


class PlanarLargeMapBackend(BatchedLargeMapBackend):
    """Shared-map backend without the historical outer-route controller."""

    def __init__(self, scene: LargeMapScene, num_envs: int, *, device: Any,
                 dtype: torch.dtype = torch.float32, ray_chunk: int = 32,
                 collision_chunk: int = 64, static_mode: str = "lidar144",
                 c3bf_config: PlanarC3BFConfig = PlanarC3BFConfig(),
                 actor_dynamic_radius_m: float = 4.0,
                 actor_dynamic_nearest_count: int = 2,
                 actor_dynamic_threat_count: int = 2,
                 actor_dynamic_min_relative_speed_m_s: float = 1e-6,
                 observation_mode: str = "legacy16"):
        if type(num_envs) is not int or num_envs < 1:
            raise ValueError("num_envs must be positive")
        if static_mode not in ("lidar144", "static_c3bf"):
            raise ValueError("unsupported static safety mode")
        validate_planar_scene(scene)
        if len(scene.static_boxes) < 0 or len(scene.dynamic_bodies) < 0:
            raise ValueError("invalid scene counts")
        from navacc_large_map_planar_obs20 import PUBLIC20_MODES
        if observation_mode not in ("legacy16",) + PUBLIC20_MODES:
            raise ValueError("unsupported planar observation mode")
        self.observation_mode = observation_mode
        if observation_mode=="public21_goal_fixed_si_yaw144_dyn10_v1" and any(body.kind!="cylinder" or not math.isfinite(body.width_m) or not math.isfinite(body.height_m) or body.width_m<=0 or body.height_m<=0 for body in scene.dynamic_bodies):raise ValueError("goal dyn10 requires finite positive physical cylinder radius/height")
        self.scene = scene; self.num_envs = num_envs
        self.device = torch.device(device); self.dtype = dtype
        self.ray_chunk = int(ray_chunk); self.collision_chunk = int(collision_chunk)
        self.static_mode = static_mode; self.c3bf_config = c3bf_config
        if (actor_dynamic_radius_m != 4.0
                or actor_dynamic_nearest_count != 2
                or actor_dynamic_threat_count != 2
                or actor_dynamic_min_relative_speed_m_s != 1e-6):
            raise ValueError(
                "near2_cvthreat2_v1 actor ranking constants are fixed")
        self.actor_observation_schema = ACTOR_OBSERVATION_SCHEMA
        self.actor_dynamic_radius_m = float(actor_dynamic_radius_m)
        self.actor_dynamic_min_relative_speed_m_s = float(
            actor_dynamic_min_relative_speed_m_s)
        tensor = lambda value, dt=dtype: torch.as_tensor(
            np.array(value, copy=True), device=self.device, dtype=dt)
        self.static_centers = tensor([box.center for box in scene.static_boxes]).reshape(-1, 3)
        self.static_half = tensor([box.size * .5 for box in scene.static_boxes]).reshape(-1, 3)
        self.static_cylinder=tensor([box.kind=="cylinder" for box in scene.static_boxes],torch.bool)
        self.static_cbf_radius = torch.where(self.static_cylinder,self.static_half[:,0],
            torch.linalg.vector_norm(self.static_half[:, :2], dim=-1))
        self.dynamic_width = tensor([body.width_m for body in scene.dynamic_bodies]).reshape(-1)
        self.dynamic_height = tensor([body.height_m for body in scene.dynamic_bodies]).reshape(-1)
        self.dynamic_cube = torch.as_tensor(
            [body.kind == "cube" for body in scene.dynamic_bodies],
            device=self.device, dtype=torch.bool)
        self.dynamic_cbf_radius = torch.where(
            self.dynamic_cube, self.dynamic_width / math.sqrt(2.),
            self.dynamic_width * .5)
        dynamic_body_ids = [body.body_id for body in scene.dynamic_bodies]
        if len(set(dynamic_body_ids)) != len(dynamic_body_ids):
            raise ValueError("dynamic body IDs must be unique")
        self.dynamic_body_ids = torch.as_tensor(
            dynamic_body_ids,
            device=self.device, dtype=torch.long)
        self.starts = tensor(scene.starts); self.goals = tensor(scene.goals)
        if len(self.starts) != num_envs:
            raise ValueError("planar scene must contain one reset mission per environment")

        # Match the established 144-ray tensor implementation exactly.
        ray_dtype = torch.float64
        azimuth = torch.deg2rad(torch.arange(
            0, 360, 10, device=self.device, dtype=ray_dtype).repeat_interleave(4))
        elevation = torch.deg2rad(torch.tensor(
            [-10., 0., 10., 20.], device=self.device,
            dtype=ray_dtype).repeat(36))
        self.ray_directions = torch.stack((
            torch.cos(elevation) * torch.cos(azimuth),
            torch.cos(elevation) * torch.sin(azimuth), torch.sin(elevation)), -1)
        extent = scene.map_half_extent_m
        wall_centers = torch.tensor(
            [[extent, 0, 2.25], [-extent, 0, 2.25],
             [0, extent, 2.25], [0, -extent, 2.25]],
            device=self.device, dtype=torch.float64)
        wall_half = torch.tensor(
            [[.01, 30., 2.25], [.01, 30., 2.25],
             [30., .01, 2.25], [30., .01, 2.25]],
            device=self.device, dtype=torch.float64)
        exact_centers = torch.cat((torch.as_tensor(
            np.array([box.center for box in scene.static_boxes], dtype=np.float64),
            device=self.device).reshape(-1, 3), wall_centers), 0)
        exact_half = torch.cat((torch.as_tensor(
            np.array([box.size * .5 for box in scene.static_boxes], dtype=np.float64),
            device=self.device).reshape(-1, 3), wall_half), 0)
        self.ray_low = exact_centers - exact_half
        self.ray_high = exact_centers + exact_half
        self.ray_cylinder=torch.cat((self.static_cylinder,torch.zeros(4,device=self.device,dtype=torch.bool)))

    def set_missions(self, starts: torch.Tensor, goals: torch.Tensor) -> None:
        for name, value in (("starts", starts), ("goals", goals)):
            if (value.shape != (self.num_envs, 3) or value.device != self.device
                    or value.dtype != self.dtype or not bool(torch.isfinite(value).all())):
                raise ValueError(f"{name} must be finite [E,3] on backend device")
        self.starts.copy_(starts); self.goals.copy_(goals)

    def actor_observation(self, position: torch.Tensor, velocity: torch.Tensor,
                          quaternion_wxyz: torch.Tensor, gyro_body: torch.Tensor,
                          yaw_hold_rad: torch.Tensor, progress_fraction: torch.Tensor,
                          dynamic_position: torch.Tensor, dynamic_velocity: torch.Tensor,
                          local_goal: torch.Tensor | None = None,
                          previous_selected_request: torch.Tensor | None = None,
                          ever_reached: torch.Tensor | None = None,
                          goal_frame_theta: torch.Tensor | None = None,
                          history_valid: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        tensors = (position, velocity, quaternion_wxyz, gyro_body,
                   yaw_hold_rad, progress_fraction, dynamic_position,
                   dynamic_velocity)
        if any(not bool(torch.isfinite(value).all()) for value in tensors):
            raise ValueError("planar observation inputs must be finite")
        if local_goal is None:
            local_goal = self.goals
        delta = local_goal - position
        from navacc_large_map_planar_obs20 import PUBLIC20_MODES, GOAL_SCALE_MODE, HOLD_MODE
        new_mode = self.observation_mode in PUBLIC20_MODES
        if new_mode:
            from navacc_large_map_planar_obs20 import REF_DISTANCE_M
            local_goal = self.goals
            delta = (self.goals - position) * (3. / REF_DISTANCE_M)
        else:
            delta = delta * (3. / torch.linalg.vector_norm(
                delta, dim=-1, keepdim=True).clamp_min(3.))
        state = torch.cat((
            delta / 3., quaternion_wxyz, velocity / 3., gyro_body / 10.,
            torch.sin(yaw_hold_rad)[:, None], torch.cos(yaw_hold_rad)[:, None],
            progress_fraction.clamp(0, 1)[:, None]), -1)
        if new_mode:
            if previous_selected_request is None or previous_selected_request.shape != position.shape or not bool(torch.isfinite(previous_selected_request).all()):
                raise ValueError("obs20 needs finite [E,3] previous selected request")
            state = torch.cat((state, (self.goals-position).norm(dim=-1, keepdim=True)/REF_DISTANCE_M,
                previous_selected_request / position.new_tensor([.8,.8,.5])), -1)
            if self.observation_mode in (GOAL_SCALE_MODE, HOLD_MODE):
                true_delta = self.goals - position
                state[:, :2] = true_delta[:, :2] / true_delta[:, :2].norm(dim=-1, keepdim=True).clamp_min(1.)
                state[:, 2] = true_delta[:, 2] / 1.
                state[:, 16] = torch.log1p(true_delta.norm(dim=-1) / 1.)
            if self.observation_mode == HOLD_MODE:
                if ever_reached is None or ever_reached.shape != (self.num_envs,) or ever_reached.dtype != torch.bool:
                    raise ValueError('hold48 observation requires bool ever_reached [E]')
                state = torch.cat((state, ever_reached.to(state.dtype)[:,None]), -1)
            w,x,y,z = quaternion_wxyz.unbind(-1)
            yaw = torch.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
            lidar = self._static_lidar(position, yaw_rad=yaw)
        else:
            lidar = self._static_lidar(position)
        goal_mode = self.observation_mode == 'public21_goal_fixed_si_yaw144_dyn10_v1'
        if goal_mode:
            from navacc_goal_frame import encode_state21
            if goal_frame_theta is None:raise ValueError('goal-frame observation needs fixed episode frame')
            state=encode_state21(position_W=position,goal_W=self.goals,velocity_W=velocity,quaternion_WB=quaternion_wxyz,gyro_B=gyro_body,hold_yaw_W=yaw_hold_rad,progress=progress_fraction,max_episode_length=1.,history_request_W=previous_selected_request,ever_reached=ever_reached,theta=goal_frame_theta,history_valid=history_valid)
        rel = dynamic_position[None, :, :2] - position[:, None, :2]
        distances = torch.linalg.vector_norm(rel, dim=-1)
        nearest = torch.full((self.num_envs,4),-1,device=self.device,dtype=torch.long)
        available_n = min(4, distances.shape[1])
        nearest[:,:available_n] = torch.argsort(distances, dim=1, stable=True)[:,:available_n]
        relative_velocity = dynamic_velocity[None, :, :2] \
            - velocity[:, None, :2]
        ranking = rank_planar_actor_dynamic_slots(
            rel, relative_velocity, self.dynamic_cbf_radius,
            self.dynamic_body_ids,
            actor_radius_m=self.actor_dynamic_radius_m,
            eligibility_radius_m=self.c3bf_config.eligibility_distance_m,
            uav_radius_m=self.c3bf_config.uav_radius_m,
            safety_margin_m=self.c3bf_config.safety_margin_m,
            minimum_speed_m_s=self.actor_dynamic_min_relative_speed_m_s)
        actor_indices = ranking["actor_indices"]
        if not dynamic_position.shape[0]:
            return {"state":state,"static_lidar":lidar,
                "dynamic_obstacle":torch.zeros(self.num_envs,4,10 if goal_mode else 8,device=self.device,dtype=self.dtype),
                "nearest_dynamic_indices":nearest,"actor_dynamic_indices":ranking["actor_body_ids"],"local_goal":local_goal}
        safe_indices = actor_indices.clamp_min(0)
        gather2 = safe_indices[..., None].expand(-1, -1, 2)
        near_rel = rel.gather(1, gather2)
        near_vel = relative_velocity.gather(1, gather2)
        near_radius = self.dynamic_cbf_radius[safe_indices]
        available = actor_indices >= 0
        dynamic = torch.zeros(self.num_envs, 4, 8, device=self.device,
                              dtype=self.dtype)
        dynamic[..., :2] = near_rel / 3.; dynamic[..., 2:4] = near_vel / 3.
        dynamic[..., 4] = near_radius; dynamic[..., 5] = 1.; dynamic[..., 7] = 1.
        dynamic *= available[..., None]
        if goal_mode:
            from navacc_goal_frame import encode_dynamic10
            rel3=dynamic_position[None,:,:]-position[:,None,:]
            vel3=dynamic_velocity[None,:,:]-velocity[:,None,:]
            gather3=safe_indices[...,None].expand(-1,-1,3)
            dynamic=encode_dynamic10(rel3.gather(1,gather3),vel3.gather(1,gather3),near_radius,self.dynamic_height[safe_indices],available,goal_frame_theta)
        return {"state": state, "static_lidar": lidar,
                "dynamic_obstacle": dynamic,
                "nearest_dynamic_indices": nearest,
                "actor_dynamic_indices": ranking["actor_body_ids"],
                "local_goal": local_goal}

    @staticmethod
    def _bounded_brake(velocity: torch.Tensor, position: torch.Tensor,
                       goals: torch.Tensor,
                       config: PlanarC3BFConfig) -> torch.Tensor:
        raw = torch.stack((-2 * velocity[:, 0], -2 * velocity[:, 1],
                           1.5 * (goals[:, 2] - position[:, 2])
                           - 1.4 * velocity[:, 2]), -1)
        return bound_primary_acceleration(raw, config)

    def command(self, position: torch.Tensor, velocity: torch.Tensor,
                measured_acceleration: torch.Tensor,
                primary_policy_acceleration: torch.Tensor, *,
                dynamic_position: torch.Tensor,
                dynamic_velocity: torch.Tensor) -> PlanarCommand:
        nominal = bound_primary_acceleration(
            primary_policy_acceleration, self.c3bf_config)
        filtered: PlanarC3BFResult = filter_planar_top1(
            position, velocity, measured_acceleration, nominal,
            dynamic_position, dynamic_velocity, self.dynamic_cbf_radius,
            static_position=(self.static_centers[:, :2]
                             if self.static_mode == "static_c3bf" else None),
            static_radius=(self.static_cbf_radius
                           if self.static_mode == "static_c3bf" else None),
            static_mode=self.static_mode, config=self.c3bf_config)
        brake = self._bounded_brake(
            velocity, position, self.goals, self.c3bf_config)
        acceleration = torch.where(
            filtered.cbf_abort[:, None], brake, filtered.acceleration)
        if not bool(torch.isfinite(acceleration).all()):
            raise RuntimeError("planar bounded brake produced non-finite output")
        return PlanarCommand(
            acceleration=acceleration, nominal_acceleration=nominal,
            cbf_abort=filtered.cbf_abort,
            domain_valid=filtered.domain_valid,
            initial_g_safe=filtered.initial_g_safe,
            initial_psi1_safe=filtered.initial_psi1_safe,
            initial_c3bf_h_safe=filtered.initial_c3bf_h_safe,
            normal_h_recovery_attempted=filtered.normal_h_recovery_attempted,
            normal_h_recovery_accepted=filtered.normal_h_recovery_accepted,
            selected_row_feasible=filtered.selected_row_feasible,
            selected_row_pass=filtered.selected_row_pass,
            action_bounds_pass=filtered.action_bounds_pass,
            selected_entity_id=filtered.selected_entity_id,
            selected_is_static=filtered.selected_is_static,
            active_count=filtered.active_count,
            selected_count=filtered.selected_count,
            normalized_nominal_violation=filtered.normalized_nominal_violation,
            selected_post_residual=filtered.selected_post_residual,
            eligible_post_violation_count=filtered.eligible_post_violation_count,
            max_normalized_post_violation=filtered.max_normalized_post_violation,
            claim_all_rows_certified=filtered.claim_all_rows_certified)


__all__ = ["ACTOR_OBSERVATION_SCHEMA", "PlanarCommand",
           "PlanarLargeMapBackend", "rank_planar_actor_dynamic_slots"]

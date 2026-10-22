#!/usr/bin/env python3
"""Device-resident outer-ring backend for large batched NavACC execution.

The shared 350 static and 120 dynamic objects are stored once.  Environment
state is batched, while lidar and exact collision queries are chunked to bound
temporary memory.  This backend intentionally supports only the validated
outer-ring scope; interior A*, emergency MPC, and future motion-oracle queries
are refused.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import torch

from navacc_large_map import LargeMapScene


UAV_RADIUS_M = .10


def validate_batched_outer_config(config: Any) -> None:
    """Lock proof-critical config values used by the tensor backend."""
    expected = {
        "outer_geofence_m": 21.25,
        "outer_reference_m": 21.75,
        "outer_corner_position_tolerance_m": .10,
        "outer_corner_speed_tolerance_m_s": .10,
        "known_dynamic_center_half_extent_m": 20.0,
        "known_dynamic_max_half_width_m": .5,
    }
    for key, value in expected.items():
        actual = config.get(key, None)
        if actual is None or float(actual) != value:
            raise ValueError(f"batched outer proof requires {key}={value}")


@dataclass(frozen=True)
class BatchedCommand:
    acceleration: torch.Tensor
    nominal_acceleration: torch.Tensor
    policy_residual: torch.Tensor
    failure: torch.Tensor
    geometry_clear: torch.Tensor
    geometry_min_margin_m: torch.Tensor
    brake_tail_terminal_speed_m_s: torch.Tensor
    brake_tail_stop_complete: torch.Tensor
    waypoint: torch.Tensor
    local_goal: torch.Tensor
    terminal_hold_active: torch.Tensor
    terminal_hold_z_integral_accel_m_s2: torch.Tensor
    dynamic_rows_considered: torch.Tensor
    dynamic_rows_excluded: torch.Tensor
    known_motion_region_dependency: torch.Tensor


class BatchedLargeMapBackend:
    """Pure Torch observation, geometry and outer controller implementation."""

    def __init__(self, scene: LargeMapScene, num_envs: int, *, device: Any,
                 dtype: torch.dtype = torch.float32, ray_chunk: int = 32,
                 collision_chunk: int = 64):
        if num_envs < 1 or ray_chunk < 1 or collision_chunk < 1:
            raise ValueError("batch and chunk sizes must be positive")
        self.scene = scene; self.num_envs = int(num_envs)
        self.device = torch.device(device); self.dtype = dtype
        self.ray_chunk = int(ray_chunk); self.collision_chunk = int(collision_chunk)
        tensor = lambda x, dt=dtype: torch.as_tensor(
            np.array(x, copy=True), device=self.device, dtype=dt)
        self.static_centers = tensor([b.center for b in scene.static_boxes]).reshape(-1, 3)
        self.static_half = tensor([b.size * .5 for b in scene.static_boxes]).reshape(-1, 3)
        self.static_cylinder = tensor([b.kind=="cylinder" for b in scene.static_boxes],torch.bool)
        self.dynamic_width = tensor([b.width_m for b in scene.dynamic_bodies]).reshape(-1)
        self.dynamic_height = tensor([b.height_m for b in scene.dynamic_bodies]).reshape(-1)
        self.dynamic_cube = torch.as_tensor(
            [b.kind == "cube" for b in scene.dynamic_bodies],
            device=self.device, dtype=torch.bool)
        if (scene.map_half_extent_m != 25.0 or scene.obstacle_half_extent_m != 20.0
                or scene.dynamic_period_s != 2.0 or scene.deadline_s != 120.0):
            raise ValueError("batched proof requires map25/interior20/period2/deadline120")
        if len(self.static_centers):
            static_extent = (self.static_centers[:, :2].abs()
                             + self.static_half[:, :2] + UAV_RADIUS_M + .12).amax()
            self.static_exclusion_margin_m = float(21.25 - static_extent)
            if not math.isfinite(self.static_exclusion_margin_m) \
                    or self.static_exclusion_margin_m <= 0:
                raise ValueError("inflated static geometry is not inside outer geofence")
        else:
            self.static_exclusion_margin_m = math.inf
        if len(self.dynamic_width):
            max_half = float((self.dynamic_width * .5).amax())
            initial_xy = torch.as_tensor(
                np.array([b.initial_position[:2] for b in scene.dynamic_bodies]),
                device=self.device, dtype=self.dtype)
            if (max_half > .5 + 1e-7
                    or not bool((initial_xy.abs() <= 20.0 + 1e-7).all())
                    or 20.0 + max_half + UAV_RADIUS_M >= 21.25):
                raise ValueError("dynamic geometry exceeds outer reachable-region proof")
        starts, goals = scene.starts, scene.goals
        if (not np.allclose(starts[:, 1], 24.0)
                or not np.allclose(goals[:, 1], -24.0)
                or not np.allclose(starts[:, 2], 2.0)
                or not np.allclose(goals[:, 2], 2.0)
                or np.any(np.abs(starts[:, 0]) + UAV_RADIUS_M >= 25.0)
                or np.any(np.abs(goals[:, 0]) + UAV_RADIUS_M >= 25.0)):
            raise ValueError("batched outer backend requires opposing-edge missions")
        if not (20.0 + .5 + UAV_RADIUS_M < 21.25 < 21.75
                < scene.map_half_extent_m - UAV_RADIUS_M):
            raise ValueError("outer geofence/reference proof constants are inconsistent")
        if (len(scene.static_boxes) not in (0, 350)
                or len(scene.dynamic_bodies) not in (0, 120)):
            raise ValueError("batched protocol requires 350/120 or explicit zero-object tests")
        self.starts = tensor(scene.starts); self.goals = tensor(scene.goals)
        self.lane_index = torch.arange(num_envs, device=self.device) % scene.num_lanes
        self.waypoint = torch.ones(num_envs, device=self.device, dtype=torch.long)
        self.terminal_integral = torch.zeros(num_envs, device=self.device, dtype=dtype)
        side = torch.where(self.starts[:, 0] < 0, -1., 1.)
        self.routes = torch.stack((
            self.starts[:, :2],
            torch.stack((side * 21.75, torch.full_like(side, 21.75)), -1),
            torch.stack((side * 21.75, torch.full_like(side, -21.75)), -1),
            self.goals[:, :2]), 1)
        ray_dtype = torch.float64
        azimuth = torch.deg2rad(torch.arange(0, 360, 10, device=self.device,
                                             dtype=ray_dtype).repeat_interleave(4))
        elevation = torch.deg2rad(torch.tensor([-10., 0., 10., 20.],
                                               device=self.device, dtype=ray_dtype).repeat(36))
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
        # Preserve the v6 NumPy observation convention: native float32 origins
        # are promoted and intersect the source JSON geometry/rays in float64.
        # Only the shared [354,3]/[144,3] constants and one chunk temporary use
        # double precision; the final actor observation is cast back to dtype.
        exact_centers = torch.cat((torch.as_tensor(
            np.array([b.center for b in scene.static_boxes], dtype=np.float64),
            device=self.device).reshape(-1, 3), wall_centers.to(torch.float64)), 0)
        exact_half = torch.cat((torch.as_tensor(
            np.array([b.size * .5 for b in scene.static_boxes], dtype=np.float64),
            device=self.device).reshape(-1, 3), wall_half.to(torch.float64)), 0)
        exact_low, exact_high = exact_centers - exact_half, exact_centers + exact_half
        self.ray_low = exact_low; self.ray_high = exact_high
        self.ray_cylinder = torch.cat((self.static_cylinder,torch.zeros(4,device=self.device,dtype=torch.bool)))

    def reset(self, mask: torch.Tensor) -> None:
        mask = torch.as_tensor(mask, device=self.device, dtype=torch.bool).reshape(-1)
        if mask.shape != (self.num_envs,):
            raise ValueError("reset mask must have shape [E]")
        self.waypoint[mask] = 1; self.terminal_integral[mask] = 0

    def _lane(self, count: int) -> torch.Tensor:
        if count == self.num_envs: return self.lane_index
        return torch.arange(count, device=self.device) % self.scene.num_lanes

    def outer_local_goal(self, position: torch.Tensor, velocity: torch.Tensor,
                         *, update_waypoint: bool = True) -> torch.Tensor:
        position = position.to(device=self.device, dtype=self.dtype)
        velocity = velocity.to(device=self.device, dtype=self.dtype)
        e = position.shape[0]; lane = self._lane(e)
        waypoint = self.waypoint[:e].clone()
        route = self.routes[lane]
        target = route[torch.arange(e, device=self.device), waypoint]
        distance = torch.linalg.vector_norm(target - position[:, :2], dim=-1)
        speed = torch.linalg.vector_norm(velocity[:, :2], dim=-1)
        on_x = position[:, 0].abs() >= 21.25
        on_y = torch.where(waypoint == 1, position[:, 1] >= 21.25,
                           position[:, 1] <= -21.25)
        advance = ((waypoint == 1) | (waypoint == 2)) & (distance <= .10) \
            & (speed <= .10) & on_x & on_y
        waypoint = torch.where(advance, waypoint + 1, waypoint)
        if update_waypoint: self.waypoint[:e].copy_(waypoint)
        target = route[torch.arange(e, device=self.device), waypoint]
        delta = target - position[:, :2]
        norm = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
        local_xy = position[:, :2] + delta / norm.clamp_min(1e-12) * norm.clamp_max(5.)
        return torch.cat((local_xy, self.goals[lane, 2:3]), -1)

    def _static_lidar(self, origins: torch.Tensor, *, yaw_rad=None) -> torch.Tensor:
        output = []
        low, high = self.ray_low, self.ray_high
        for begin in range(0, len(origins), self.ray_chunk):
            origin = origins[begin:begin + self.ray_chunk].to(torch.float64)
            directions = self.ray_directions
            if yaw_rad is not None:
                yaw = yaw_rad[begin:begin + self.ray_chunk].to(torch.float64)
                c, sn = yaw.cos()[:, None], yaw.sin()[:, None]
                x, y, z = self.ray_directions.unbind(-1)
                directions = torch.stack((c*x-sn*y, sn*x+c*y, z.expand(len(yaw), -1)), -1)
            direction = directions[None, :, None, :] if directions.ndim == 2 else directions[:, :, None, :]
            o = origin[:, None, None, :]
            parallel = direction.abs() <= 1e-9
            safe_direction = torch.where(parallel, torch.ones_like(direction), direction)
            t0 = (low[None, None] - o) / safe_direction
            t1 = (high[None, None] - o) / safe_direction
            outside = (o < low[None, None]) | (o > high[None, None])
            invalid_parallel = (parallel & outside).any(-1)
            enter = torch.where(parallel, -torch.inf, torch.minimum(t0, t1)).amax(-1)
            leave = torch.where(parallel, torch.inf, torch.maximum(t0, t1)).amin(-1)
            hit = (~invalid_parallel) & (leave >= enter.clamp_min(0)) & (enter <= 4.)
            distances = torch.where(hit, enter.clamp_min(0), 4.)
            if bool(self.ray_cylinder.any()):
                idx=self.ray_cylinder;centers=(low[idx]+high[idx])*.5;half=(high[idx]-low[idx])*.5
                distances[:,:,idx]=self._ray_cylinder_distances(origin,directions,centers,half,4.)
            distance=distances.amin(-1)
            encoded = 2. * (1. - distance / 4.) if yaw_rad is None else 4. - distance.clamp(0., 4.)
            output.append(encoded.reshape(-1, 36, 4))
        return torch.cat(output, 0).to(self.dtype)

    def actor_observation(self, position: torch.Tensor, velocity: torch.Tensor,
                          quaternion_wxyz: torch.Tensor, gyro_body: torch.Tensor,
                          yaw_hold_rad: torch.Tensor, progress_fraction: torch.Tensor,
                          dynamic_position: torch.Tensor, dynamic_velocity: torch.Tensor,
                          local_goal: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        tensors = (position, velocity, quaternion_wxyz, gyro_body,
                   yaw_hold_rad, progress_fraction, dynamic_position, dynamic_velocity)
        if any(not bool(torch.isfinite(value).all()) for value in tensors):
            raise ValueError("batched observation inputs must be finite")
        e = position.shape[0]; lane = self._lane(e)
        if local_goal is None: local_goal = self.outer_local_goal(position, velocity)
        delta = local_goal - position
        delta = delta * (3. / torch.linalg.vector_norm(delta, dim=-1, keepdim=True).clamp_min(3.))
        state = torch.cat((delta / 3., quaternion_wxyz, velocity / 3., gyro_body / 10.,
                           torch.sin(yaw_hold_rad)[:, None], torch.cos(yaw_hold_rad)[:, None],
                           progress_fraction.clamp(0, 1)[:, None]), -1)
        rel = dynamic_position[None, :, :2] - position[:, None, :2]
        distances = torch.linalg.vector_norm(rel, dim=-1)
        nearest = torch.argsort(distances, dim=1, stable=True)[:, :4]
        gather2 = nearest[..., None].expand(-1, -1, 2)
        near_rel = rel.gather(1, gather2)
        near_vel = dynamic_velocity[None, :, :2].expand(e, -1, -1).gather(1, gather2) \
            - velocity[:, None, :2]
        radius = torch.where(self.dynamic_cube, self.dynamic_width / math.sqrt(2.),
                             self.dynamic_width * .5)
        near_radius = radius[nearest]
        available = distances.gather(1, nearest) <= 4.
        dynamic = torch.zeros(e, 4, 8, device=self.device, dtype=self.dtype)
        dynamic[..., :2] = near_rel / 3.; dynamic[..., 2:4] = near_vel / 3.
        dynamic[..., 4] = near_radius; dynamic[..., 5] = 1.; dynamic[..., 7] = 1.
        dynamic *= available[..., None]
        return {"state": state, "static_lidar": self._static_lidar(position),
                "dynamic_obstacle": dynamic,
                "nearest_dynamic_indices": nearest,
                "local_goal": local_goal, "lane_index": lane}

    @staticmethod
    def _ray_cylinder_distances(origins,directions,centers,half,max_range):
        oc=origins[:,None,None]-centers[None,None];d=directions[None,:,None] if directions.ndim == 2 else directions[:,:,None]
        aa=d[...,:2].square().sum(-1);bb=(oc[...,:2]*d[...,:2]).sum(-1)
        cc=oc[...,:2].square().sum(-1)-half[None,None,:,0].square()
        parallel=aa<=1e-18;safea=torch.where(parallel,torch.ones_like(aa),aa);disc=bb.square()-aa*cc
        root=disc.clamp_min(0).sqrt();lo=(-bb-root)/safea;hi=(-bb+root)/safea
        lo=torch.where(parallel,-torch.inf,lo);hi=torch.where(parallel,torch.inf,hi)
        radial_valid=torch.where(parallel,cc<=0.,disc>=0.)
        dz=d[...,2];pz=oc[...,2];parallelz=dz.abs()<=1e-9;safez=torch.where(parallelz,torch.ones_like(dz),dz)
        z0=(-half[None,None,:,2]-pz)/safez;z1=(half[None,None,:,2]-pz)/safez
        zlo=torch.where(parallelz,-torch.inf,torch.minimum(z0,z1));zhi=torch.where(parallelz,torch.inf,torch.maximum(z0,z1))
        zvalid=(~parallelz)|(pz.abs()<=half[None,None,:,2])
        enter=torch.maximum(lo,zlo);leave=torch.minimum(hi,zhi)
        hit=radial_valid&zvalid&(leave>=enter.clamp_min(0))&(enter<=max_range)
        return torch.where(hit,enter.clamp_min(0),max_range)

    def _static_sdf(self,points,inflation):
        result=self._box_sdf(points,self.static_centers,self.static_half,inflation)
        if bool(self.static_cylinder.any()):
            idx=self.static_cylinder;radial=(points[:,None,:2]-self.static_centers[None,:,:2]).norm(dim=-1)[:,idx]-self.static_half[idx,0][None]
            dz=(points[:,None,2]-self.static_centers[None,:,2]).abs()[:,idx]-self.static_half[idx,2][None]
            delta=torch.stack((radial,dz),-1)
            result[:,idx]=delta.clamp_min(0).norm(dim=-1)+delta.amax(-1).clamp_max(0)-inflation
        return result

    def _static_sdf_xy(self,points,inflation):
        delta=(points[:,None,:2]-self.static_centers[None,:,:2]).abs()-self.static_half[None,:,:2]
        result=delta.clamp_min(0).norm(dim=-1)+delta.amax(-1).clamp_max(0)-inflation
        if bool(self.static_cylinder.any()):
            idx=self.static_cylinder
            result[:,idx]=(points[:,None,:2]-self.static_centers[None,:,:2]).norm(dim=-1)[:,idx]-self.static_half[idx,0][None]-inflation
        return result

    @staticmethod
    def _box_sdf(points: torch.Tensor, centers: torch.Tensor,
                 half: torch.Tensor, inflation: float) -> torch.Tensor:
        delta = (points[:, None] - centers[None]).abs() - half[None]
        return torch.linalg.vector_norm(delta.clamp_min(0), dim=-1) \
            + delta.amax(-1).clamp_max(0) - inflation

    def _dynamic_sdf(self, points: torch.Tensor,
                     centers: torch.Tensor, inflation: float) -> torch.Tensor:
        result = torch.empty(len(points), len(centers), device=self.device, dtype=self.dtype)
        dz = (points[:, None, 2] - centers[None, :, 2]).abs() - self.dynamic_height[None] * .5
        if bool(self.dynamic_cube.any()):
            idx = self.dynamic_cube
            dxy = (points[:, None, :2] - centers[None, :, :2]).abs()[:, idx] \
                - self.dynamic_width[idx][None, :, None] * .5
            delta = torch.cat((dxy, dz[:, idx, None]), -1)
            result[:, idx] = torch.linalg.vector_norm(delta.clamp_min(0), dim=-1) \
                + delta.amax(-1).clamp_max(0) - inflation
        if bool((~self.dynamic_cube).any()):
            idx = ~self.dynamic_cube
            radial = torch.linalg.vector_norm(
                points[:, None, :2] - centers[None, :, :2], dim=-1)[:, idx] \
                - self.dynamic_width[idx][None] * .5
            delta = torch.stack((radial, dz[:, idx]), -1)
            result[:, idx] = torch.linalg.vector_norm(delta.clamp_min(0), dim=-1) \
                + delta.amax(-1).clamp_max(0) - inflation
        return result

    def collision_flags(self, position: torch.Tensor,
                        dynamic_position: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not bool(torch.isfinite(position).all()) or not bool(torch.isfinite(dynamic_position).all()):
            raise ValueError("collision inputs must be finite")
        static_parts, dynamic_parts = [], []
        for begin in range(0, len(position), self.collision_chunk):
            p = position[begin:begin + self.collision_chunk]
            static_parts.append((self._static_sdf(p,UAV_RADIUS_M) <= 0).any(-1)
                                if len(self.static_centers) else
                                torch.zeros(len(p), device=self.device, dtype=torch.bool))
            dynamic_parts.append((self._dynamic_sdf(p, dynamic_position,
                                                     UAV_RADIUS_M) <= 0).any(-1)
                                 if len(dynamic_position) else
                                 torch.zeros(len(p), device=self.device, dtype=torch.bool))
        return torch.cat(static_parts), torch.cat(dynamic_parts)

    @staticmethod
    def outer_swept_margin(previous: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
        endpoint = torch.minimum(previous[:, :2].abs().amax(-1),
                                 current[:, :2].abs().amax(-1)) - 21.25
        return endpoint - .5 * torch.linalg.vector_norm(
            current[:, :2] - previous[:, :2], dim=-1)

    @staticmethod
    def dynamic_region_status(dynamic_position: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        finite = torch.isfinite(dynamic_position).all()
        margin = (20. - dynamic_position[:, :2].abs().amax()) if finite else \
            torch.tensor(-torch.inf, device=dynamic_position.device,
                         dtype=dynamic_position.dtype)
        return margin, (~finite) | (margin < -1e-4)

    @staticmethod
    def _bounded(value: torch.Tensor) -> torch.Tensor:
        result = value.clone(); result[:, 2].clamp_(-.5, .5)
        radius = torch.minimum(torch.full_like(result[:, 2], .8),
                               (1. - result[:, 2].square()).clamp_min(0).sqrt())
        norm = torch.linalg.vector_norm(result[:, :2], dim=-1)
        scale = torch.minimum(torch.ones_like(norm), radius / norm.clamp_min(1e-12))
        result[:, :2] *= scale[:, None]
        return result

    def _stop_tail_gate(self, position: torch.Tensor, velocity: torch.Tensor,
                        measured: torch.Tensor, acceleration: torch.Tensor
                        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        speed0 = torch.linalg.vector_norm(velocity[:, :2], dim=-1)
        horizon = torch.maximum(torch.full_like(speed0, .75),
            (.20 + (speed0 + .16) / .8 + .30).clamp_max(2.5))
        p = position.clone(); v = velocity.clone(); actual = measured.clone()
        previous = p.clone(); clear = torch.ones(len(p), device=self.device, dtype=torch.bool)
        initial_wall = ((p[:, :2].abs() + UAV_RADIUS_M <= self.scene.map_half_extent_m).all(-1)
                        & (p[:, 2] >= UAV_RADIUS_M) & (p[:, 2] <= 4.5 - UAV_RADIUS_M))
        initial_outer = p[:, :2].abs().amax(-1) - 21.25
        clear &= initial_wall & (initial_outer >= 0)
        minimum = initial_outer.clone()
        previous_t = torch.zeros_like(speed0)
        for index in range(1, 52):
            target_t = torch.minimum(torch.full_like(speed0, index * .05), horizon)
            step = target_t - previous_t; active = step > 1e-12
            request = acceleration.clone()
            braking = previous_t >= .20
            unit = v[:, :2] / torch.linalg.vector_norm(v[:, :2], dim=-1, keepdim=True).clamp_min(1e-12)
            brake = torch.cat((-.8 * unit, (-1.4 * v[:, 2]).clamp(-.5, .5)[:, None]), -1)
            request = torch.where(braking[:, None], brake, request)
            alpha = torch.exp(-step / .05)
            next_actual = alpha[:, None] * actual + (1 - alpha)[:, None] * request
            next_v = v + next_actual * step[:, None]
            next_p = p + .5 * (v + next_v) * step[:, None]
            actual = torch.where(active[:, None], next_actual, actual)
            v = torch.where(active[:, None], next_v, v)
            p = torch.where(active[:, None], next_p, p)
            wall = ((p[:, :2].abs() + UAV_RADIUS_M <= self.scene.map_half_extent_m).all(-1)
                    & (p[:, 2] >= UAV_RADIUS_M) & (p[:, 2] <= 4.5 - UAV_RADIUS_M))
            swept = self.outer_swept_margin(previous, p)
            clear &= (~active) | (wall & (swept >= 0))
            minimum = torch.minimum(minimum, torch.where(active, swept,
                                                          torch.full_like(swept, torch.inf)))
            previous = torch.where(active[:, None], p, previous); previous_t = target_t
            if bool((previous_t >= horizon - 1e-12).all()): break
        terminal = torch.linalg.vector_norm(v[:, :2], dim=-1)
        clear &= terminal <= .05
        return clear, minimum, terminal

    def command(self, position: torch.Tensor, velocity: torch.Tensor,
                measured_acceleration: torch.Tensor, policy_acceleration: torch.Tensor,
                *, terminal_hold: torch.Tensor | None = None,
                reset_mask: torch.Tensor | None = None,
                dynamic_position: torch.Tensor | None = None) -> BatchedCommand:
        values = (position, velocity, measured_acceleration, policy_acceleration)
        if any(value.shape != (self.num_envs, 3) for value in values):
            raise ValueError("controller inputs must be [E,3]")
        if any(not bool(torch.isfinite(value).all()) for value in values):
            raise ValueError("controller inputs must be finite")
        if dynamic_position is not None:
            margin, violation = self.dynamic_region_status(dynamic_position)
            if bool(violation):
                raise RuntimeError(
                    f"known dynamic region invalid; margin={float(margin)}")
        if reset_mask is not None: self.reset(reset_mask)
        terminal = (torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
                    if terminal_hold is None else
                    torch.as_tensor(terminal_hold, device=self.device,
                                    dtype=torch.bool).reshape(self.num_envs))
        lane = self.lane_index; local = self.outer_local_goal(position, velocity)
        delta = local - position; goal_delta = self.goals[lane] - position
        waypoint_distance = torch.linalg.vector_norm(delta[:, :2], dim=-1)
        goal_distance = torch.linalg.vector_norm(goal_delta, dim=-1)
        route = self.routes[lane]; target = route[torch.arange(self.num_envs, device=self.device), self.waypoint]
        corner_distance = torch.linalg.vector_norm(target - position[:, :2], dim=-1)
        speed = torch.minimum(torch.full_like(goal_distance, .8), 2 * goal_distance)
        speed = torch.minimum(speed, 2 * waypoint_distance)
        speed = torch.minimum(speed, (1.6 * goal_distance).clamp_min(0).sqrt())
        speed = torch.minimum(speed, (1.6 * waypoint_distance).clamp_min(0).sqrt())
        nonfinal = self.waypoint < 3
        turn = (.01 + 1.6 * (corner_distance - .35).clamp_min(0)).sqrt()
        speed = torch.where(nonfinal, torch.minimum(speed, turn), speed)
        desired = delta / torch.linalg.vector_norm(delta, dim=-1, keepdim=True).clamp_min(1e-12) \
            * speed[:, None]
        base = torch.stack((2 * (desired[:, 0] - velocity[:, 0]),
                            2 * (desired[:, 1] - velocity[:, 1]),
                            1.5 * goal_delta[:, 2] - 1.4 * velocity[:, 2]), -1)
        residual = policy_acceleration.clone()
        residual *= torch.minimum(torch.ones(self.num_envs, device=self.device, dtype=self.dtype),
                                  .05 / torch.linalg.vector_norm(residual, dim=-1).clamp_min(1e-12))[:, None]
        residual *= torch.minimum(torch.minimum(goal_distance / .5,
                                                waypoint_distance / .5),
                                  torch.ones_like(goal_distance))[:, None]
        integral_candidate = (self.terminal_integral + .8 * goal_delta[:, 2] * .01).clamp(-.2, .2)
        terminal_base = torch.stack((1.5 * goal_delta[:, 0] - 2 * velocity[:, 0],
                                     1.5 * goal_delta[:, 1] - 2 * velocity[:, 1],
                                     1.5 * goal_delta[:, 2] - 1.4 * velocity[:, 2]
                                     + integral_candidate), -1)
        base = torch.where(terminal[:, None], terminal_base, base)
        residual = torch.where(terminal[:, None], torch.zeros_like(residual), residual)
        nominal = self._bounded(base + residual)
        clear, margin, terminal_speed = self._stop_tail_gate(
            position, velocity, measured_acceleration, nominal)
        brake = self._bounded(torch.stack((-2 * velocity[:, 0], -2 * velocity[:, 1],
                                           1.5 * goal_delta[:, 2] - 1.4 * velocity[:, 2]), -1))
        brake_clear, brake_margin, brake_terminal = self._stop_tail_gate(
            position, velocity, measured_acceleration, brake)
        acceleration = torch.where(clear[:, None], nominal, brake)
        selected_clear = torch.where(clear, clear, brake_clear)
        selected_margin = torch.where(clear, margin, brake_margin)
        selected_terminal = torch.where(clear, terminal_speed, brake_terminal)
        failure = ~clear
        commit = terminal & selected_clear & ~failure
        self.terminal_integral.copy_(torch.where(commit, integral_candidate,
                                                 self.terminal_integral))
        return BatchedCommand(
            acceleration, nominal, residual, failure, selected_clear, selected_margin,
            selected_terminal, selected_clear & (selected_terminal <= .05),
            self.waypoint.clone(), local, terminal,
            torch.where(terminal, self.terminal_integral,
                        torch.full_like(self.terminal_integral, torch.nan)),
            torch.zeros(self.num_envs, device=self.device, dtype=torch.long),
            torch.full((self.num_envs,), len(self.scene.dynamic_bodies),
                       device=self.device, dtype=torch.long),
            torch.ones(self.num_envs, device=self.device, dtype=torch.bool))


__all__ = ["BatchedCommand", "BatchedLargeMapBackend",
           "validate_batched_outer_config"]

#!/usr/bin/env python3
"""Device-batched cached static-route local-goal guidance for planar PPO."""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
import torch

from navacc_large_map_planar_scene import PlanarMissionSampler

GUIDANCE_SCHEMA = "static_route_goal_v1"
GLOBAL_GUIDANCE_SCHEMA = "global_goal_v1"
GLOBAL_OBSERVATION_SCHEMA = (
    "NavAcc-V4-NBSP-state16-static36x4-L4-dynamic4x8-"
    "near2_cvthreat2_v1")
ROUTE_OBSERVATION_SCHEMA = (
    "NavAcc-V4-NBSP-state16-static36x4-L4-dynamic4x8-"
    "near2_cvthreat2-routegoal-v1")

@dataclass(frozen=True)
class RouteGuidanceResult:
    local_goal: torch.Tensor
    route_index: torch.Tensor
    route_progress_m: torch.Tensor
    route_failure: torch.Tensor
    replanned: torch.Tensor

class PlanarRouteGuidance:
    """Per-lane cached routes and device-batched closed-segment visibility."""
    def __init__(self, sampler: PlanarMissionSampler, num_envs: int, *,
                 lookahead_m: float = 3.0):
        if type(num_envs) is not int or num_envs < 1:
            raise ValueError("num_envs must be positive")
        if type(lookahead_m) not in (int, float) or not math.isfinite(float(lookahead_m)) or lookahead_m <= 0:
            raise ValueError("lookahead_m must be finite and positive")
        self.sampler = sampler; self.num_envs = num_envs; self.lookahead_m = float(lookahead_m)
        self.routes: list[tuple[np.ndarray, ...] | None] = [None] * num_envs
        self.indices = np.zeros(num_envs, dtype=np.int64)
        self.failed = np.zeros(num_envs, dtype=bool)
        self._revision = 0; self._tensor_revision = -1; self._tensor_device = None

    def _sync_host_state(self):
        if self._tensor_revision == self._revision:
            self.indices[:] = self._index_tensor.detach().cpu().numpy()
            self.failed[:] = self._failed_tensor.detach().cpu().numpy()

    @staticmethod
    def _copy_route(route):
        route = tuple(np.asarray(p, dtype=np.float64).reshape(2).copy() for p in route)
        return route if len(route) >= 2 else None

    def set_routes(self, env_ids, routes) -> None:
        # Device state is copied only on an explicit reset/replan, never on the
        # 100 Hz observation path. This also preserves untouched lane cursors.
        self._sync_host_state()
        ids = [int(v) for v in env_ids]
        if len(ids) != len(routes) or len(set(ids)) != len(ids) or any(v < 0 or v >= self.num_envs for v in ids):
            raise ValueError("route lane selection is invalid")
        for lane, route in zip(ids, routes):
            copied = self._copy_route(route)
            if copied is None or any(not self.sampler.segment_static_clear(a, b) for a, b in zip(copied[:-1], copied[1:])):
                raise ValueError("route is empty or not statically valid")
            self.routes[lane] = copied; self.indices[lane] = 1; self.failed[lane] = False
        self._revision += 1

    def _buffers(self, device, dtype):
        if self._tensor_revision == self._revision and self._tensor_device == (device, dtype):
            return self._route, self._valid, self._cumulative
        if any(route is None for route in self.routes):
            raise RuntimeError("all route lanes must be initialized")
        width = max(len(route) for route in self.routes)
        points = np.zeros((self.num_envs, width, 2), dtype=np.float64)
        valid = np.zeros((self.num_envs, width), dtype=bool)
        cumulative = np.zeros((self.num_envs, width), dtype=np.float64)
        for lane, route in enumerate(self.routes):
            array = np.stack(route); count = len(route)
            points[lane, :count] = array; points[lane, count:] = array[-1]
            valid[lane, :count] = True
            if count > 1:
                cumulative[lane, 1:count] = np.cumsum(np.linalg.norm(np.diff(array, axis=0), axis=1))
                cumulative[lane, count:] = cumulative[lane, count - 1]
        self._route = torch.as_tensor(points, device=device, dtype=dtype)
        self._valid = torch.as_tensor(valid, device=device)
        self._cumulative = torch.as_tensor(cumulative, device=device, dtype=dtype)
        self._low = torch.as_tensor(
            self.sampler.route_low_xy, device=device, dtype=dtype)
        self._high = torch.as_tensor(
            self.sampler.route_high_xy, device=device, dtype=dtype)
        self._index_tensor = torch.as_tensor(
            self.indices, device=device, dtype=torch.long).clone()
        self._failed_tensor = torch.as_tensor(
            self.failed, device=device, dtype=torch.bool).clone()
        self._tensor_revision = self._revision; self._tensor_device = (device, dtype)
        return self._route, self._valid, self._cumulative

    def _visible(self, here, targets):
        # [E,K,350] closed-slab intersection. Tangency is blocked.
        low, high = self._low, self._high
        a = here[:, None, None, :]; d = (targets - here[:, None, :])[:, :, None, :]
        shape = (here.shape[0], targets.shape[1], low.shape[0])
        enter = torch.zeros(shape, device=here.device, dtype=here.dtype)
        leave = torch.ones_like(enter); possible = torch.ones_like(enter, dtype=torch.bool)
        eps = torch.finfo(here.dtype).eps
        for axis in range(2):
            parallel = d[..., axis].abs() <= eps
            inside = (low[None, None, :, axis] <= a[..., axis]) & (a[..., axis] <= high[None, None, :, axis])
            possible &= (~parallel) | inside
            safe = torch.where(parallel, torch.ones_like(d[..., axis]), d[..., axis])
            t0 = (low[None, None, :, axis] - a[..., axis]) / safe
            t1 = (high[None, None, :, axis] - a[..., axis]) / safe
            enter = torch.where(parallel, enter, torch.maximum(enter, torch.minimum(t0, t1)))
            leave = torch.where(parallel, leave, torch.minimum(leave, torch.maximum(t0, t1)))
        collision = possible & (enter <= leave)
        limit = self.sampler.scene.map_half_extent_m - self.sampler.route_inflation_m
        bounds = (here.abs().amax(-1) <= limit)[:, None] & (targets.abs().amax(-1) <= limit)
        return (~collision.any(-1)) & bounds

    def _replan(self, lanes, positions, goals):
        self._sync_host_state()
        replanned = np.zeros(self.num_envs, dtype=bool)
        changed_failure = False
        for lane in lanes:
            route = self.sampler.plan_route(positions[lane, :2], goals[lane, :2])
            if route:
                self.set_routes([lane], [route]); replanned[lane] = True
            else:
                self.failed[lane] = True; changed_failure = True
        if changed_failure:
            self._revision += 1
        return replanned

    def local_goals(self, position: torch.Tensor, global_goal: torch.Tensor) -> RouteGuidanceResult:
        if position.shape != (self.num_envs, 3) or global_goal.shape != position.shape or not bool(torch.isfinite(position).all()) or not bool(torch.isfinite(global_goal).all()):
            raise ValueError("route guidance requires finite [E,3] state")
        route, valid, cumulative = self._buffers(position.device, position.dtype)
        index = self._index_tensor
        slots = torch.arange(route.shape[1], device=position.device)[None]
        visible = self._visible(position[:, :2], route) & valid & (slots >= index[:, None])
        chosen = torch.where(visible, slots, torch.full_like(slots, -1)).max(-1).values
        replanned_np = np.zeros(self.num_envs, dtype=bool)
        missing = (chosen < 0).nonzero().reshape(-1).detach().cpu().tolist()
        if missing:
            replanned_np = self._replan(missing, position.detach().cpu().numpy(), global_goal.detach().cpu().numpy())
            route, valid, cumulative = self._buffers(position.device, position.dtype)
            index = self._index_tensor; slots = torch.arange(route.shape[1], device=position.device)[None]
            visible = self._visible(position[:, :2], route) & valid & (slots >= index[:, None])
            chosen = torch.where(visible, slots, torch.full_like(slots, -1)).max(-1).values
        failure = (chosen < 0) | self._failed_tensor
        safe = chosen.clamp_min(0); target = route[torch.arange(self.num_envs, device=position.device), safe]
        delta = target - position[:, :2]; distance = torch.linalg.vector_norm(delta, dim=-1)
        scale = torch.minimum(torch.ones_like(distance), torch.full_like(distance, self.lookahead_m) / distance.clamp_min(torch.finfo(position.dtype).tiny))
        local = global_goal.clone(); local[:, :2] = position[:, :2] + scale[:, None] * delta
        local = torch.where(failure[:, None], global_goal, local)
        new_index = torch.maximum(index, safe); self._index_tensor.copy_(new_index)
        progress = cumulative[torch.arange(self.num_envs, device=position.device), new_index]
        self._failed_tensor.copy_(failure)
        return RouteGuidanceResult(local, new_index, progress, failure,
            torch.as_tensor(replanned_np, device=position.device))

__all__ = ["GLOBAL_GUIDANCE_SCHEMA", "GLOBAL_OBSERVATION_SCHEMA",
           "GUIDANCE_SCHEMA", "ROUTE_OBSERVATION_SCHEMA",
           "PlanarRouteGuidance", "RouteGuidanceResult"]

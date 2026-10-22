#!/usr/bin/env python3
"""Additive fixed-height dynamic scene and free-mission sampler.

This module deliberately leaves the historical large-map generator untouched.
It preserves that generator's static geometry, dynamic initial positions and
XY velocity traces, while forcing every dynamic velocity's z component to zero.
Episode missions are sampled independently from the dense interior with exact
static/dynamic shape clearance checks.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from collections import deque
import heapq
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from navacc_large_map import (
    DynamicBody, EPS, GRID_RESOLUTION_M, LargeMapScene, UAV_RADIUS_M,
    generate_scene, save_scene, StaticBox,static_signed_clearances,_segment_disc_clear,
)


@dataclass(frozen=True)
class PlanarSceneConfig:
    mission_xy_half_extent_m: float = 20.0
    mission_start_z_m: float = 2.0
    mission_goal_z_m: float = 2.0
    mission_clearance_m: float = 0.30
    mission_min_separation_m: float = 20.0
    mission_max_separation_m: float = 60.0
    easy_mission_probability: float = 0.0
    mission_sampling_schema: str = "interior_v1"
    maximum_sampling_attempts: int = 200_000
    dynamic_z_tolerance_m: float = 1e-12

    def __post_init__(self) -> None:
        if self.mission_sampling_schema not in ('interior_v1','navrl_edges_xy_v1','navrl_edges_opposite50_xy_v1','navrl_edges_short1to3_xy_v1','navrl_edges_short_full50_xy_v1','navrl_edges_short_full30_xy_v1','navrl_edges_short_full20_xy_v1','navrl_edges_short_full15_xy_v1','navrl_edges_short_full10_xy_v1','navrl_edges_short_full05_xy_v1','navrl_edges_short_full00_xy_v1'):
            raise ValueError('unsupported mission sampling schema')
        if self.mission_sampling_schema in ('navrl_edges_xy_v1','navrl_edges_opposite50_xy_v1') and (self.mission_xy_half_extent_m!=24. or self.mission_start_z_m!=2. or self.mission_goal_z_m!=2. or self.mission_min_separation_m!=.5 or self.mission_max_separation_m!=70. or self.easy_mission_probability!=0.):
            raise ValueError('edgeXY schema requires ±24/Z2/min.5/max70/no easy mixture')
        if self.mission_sampling_schema=='navrl_edges_short1to3_xy_v1' and (self.mission_xy_half_extent_m!=24. or self.mission_start_z_m!=2. or self.mission_goal_z_m!=2. or self.mission_min_separation_m!=1. or self.mission_max_separation_m!=3. or self.easy_mission_probability!=0. or self.mission_clearance_m!=.3):
            raise ValueError('short edge schema requires explicit ±24/Z2/min1/max3/clearance.3/no mixture')
        if self.mission_sampling_schema in ('navrl_edges_short_full50_xy_v1','navrl_edges_short_full30_xy_v1','navrl_edges_short_full20_xy_v1','navrl_edges_short_full15_xy_v1','navrl_edges_short_full10_xy_v1','navrl_edges_short_full05_xy_v1','navrl_edges_short_full00_xy_v1') and (self.mission_xy_half_extent_m!=24. or self.mission_start_z_m!=2. or self.mission_goal_z_m!=2. or self.mission_min_separation_m!=.5 or self.mission_max_separation_m!=70. or self.easy_mission_probability!=0. or self.mission_clearance_m!=.3):
            raise ValueError('mixed50 schema requires explicit ±24/Z2/min.5/max70/clearance.3')
        numeric = (
            self.mission_xy_half_extent_m, self.mission_start_z_m,
            self.mission_goal_z_m,
            self.mission_clearance_m, self.mission_min_separation_m,
            self.mission_max_separation_m,self.easy_mission_probability,
            self.dynamic_z_tolerance_m,
        )
        if (not all(type(value) in (int, float) and math.isfinite(float(value))
                    for value in numeric)
                or not 0 < self.mission_xy_half_extent_m < 25.0
                or not UAV_RADIUS_M <= self.mission_start_z_m <= 4.5 - UAV_RADIUS_M
                or not UAV_RADIUS_M <= self.mission_goal_z_m <= 4.5 - UAV_RADIUS_M
                or self.mission_clearance_m < 0
                or self.mission_min_separation_m <= 0
                or self.mission_max_separation_m < self.mission_min_separation_m
                or not 0 <= self.easy_mission_probability <= 1
                or type(self.maximum_sampling_attempts) is not int
                or self.maximum_sampling_attempts <= 0
                or self.dynamic_z_tolerance_m < 0):
            raise ValueError("planar scene configuration is invalid")


def _point_box_clearance(point: np.ndarray, center: np.ndarray,
                         half_size: np.ndarray) -> float:
    delta = np.abs(point - center) - half_size
    return float(np.linalg.norm(np.maximum(delta, 0.0))
                 + min(float(delta.max()), 0.0) - UAV_RADIUS_M)


def _point_body_clearance(point: np.ndarray, body: DynamicBody,
                          center: np.ndarray | None = None) -> float:
    center = body.initial_position if center is None else np.asarray(center)
    dz = abs(float(point[2] - center[2])) - body.height_m * .5
    if body.kind == "cube":
        dxy = np.abs(point[:2] - center[:2]) - body.width_m * .5
        delta = np.array([dxy[0], dxy[1], dz])
    else:
        radial = float(np.linalg.norm(point[:2] - center[:2])
                       - body.width_m * .5)
        delta = np.array([radial, dz])
    return float(np.linalg.norm(np.maximum(delta, 0.0))
                 + min(float(delta.max()), 0.0) - UAV_RADIUS_M)


def mission_point_clearance(scene: LargeMapScene, point: np.ndarray, *,
                            dynamic_positions: np.ndarray | None = None) -> float:
    point = np.asarray(point, dtype=np.float64).reshape(3)
    wall = min(scene.map_half_extent_m - UAV_RADIUS_M - abs(float(point[0])),
               scene.map_half_extent_m - UAV_RADIUS_M - abs(float(point[1])),
               float(point[2]) - UAV_RADIUS_M,
               4.5 - UAV_RADIUS_M - float(point[2]))
    static = float(static_signed_clearances(point[None],scene.static_boxes,UAV_RADIUS_M).min()) if scene.static_boxes else math.inf
    if dynamic_positions is not None:
        dynamic_positions = np.asarray(dynamic_positions, dtype=np.float64)
        if (dynamic_positions.shape != (len(scene.dynamic_bodies), 3)
                or not np.isfinite(dynamic_positions).all()):
            raise ValueError("dynamic_positions must be finite [N,3]")
    dynamic = min((_point_body_clearance(
        point, body, None if dynamic_positions is None else dynamic_positions[index])
        for index, body in enumerate(scene.dynamic_bodies)), default=math.inf)
    return min(float(wall), float(static), float(dynamic))


def _segment_aabb_clear_exact(a: np.ndarray, b: np.ndarray, box,
                              inflation_m: float) -> bool:
    if box.kind=="cylinder":
        return _segment_disc_clear(a,b,box.center[:2],box.horizontal_radius_m+inflation_m)
    low = box.center[:2] - box.size[:2] * .5 - inflation_m
    high = box.center[:2] + box.size[:2] * .5 + inflation_m
    direction = b - a; enter, leave = 0.0, 1.0
    for axis in range(2):
        if abs(direction[axis]) <= EPS:
            if low[axis] <= a[axis] <= high[axis]:
                continue
            return True
        t0 = (low[axis] - a[axis]) / direction[axis]
        t1 = (high[axis] - a[axis]) / direction[axis]
        enter = max(enter, min(t0, t1)); leave = min(leave, max(t0, t1))
        if enter > leave:
            return True
    return False


class PlanarMissionSampler:
    """Scene-scoped exact-clearance and cached-grid mission sampler."""

    def __init__(self, scene: LargeMapScene,
                 config: PlanarSceneConfig = PlanarSceneConfig()):
        self.scene = scene; self.config = config
        self.static_center = np.asarray(
            [box.center for box in scene.static_boxes], dtype=np.float64).reshape(-1,3)
        self.static_half = np.asarray(
            [box.size * .5 for box in scene.static_boxes], dtype=np.float64).reshape(-1,3)
        self.static_cylinder=np.asarray([b.kind=="cylinder" for b in scene.static_boxes],dtype=bool)
        self.dynamic_initial = np.asarray(
            [body.initial_position for body in scene.dynamic_bodies],
            dtype=np.float64).reshape(-1,3)
        self.dynamic_width = np.asarray(
            [body.width_m for body in scene.dynamic_bodies], dtype=np.float64)
        self.dynamic_height = np.asarray(
            [body.height_m for body in scene.dynamic_bodies], dtype=np.float64)
        self.dynamic_cube = np.asarray(
            [body.kind == "cube" for body in scene.dynamic_bodies], dtype=bool)
        self.resolution_m = GRID_RESOLUTION_M
        self.map_half_extent_m = scene.map_half_extent_m
        self.route_inflation_m = UAV_RADIUS_M + .12
        self.route_low_xy = self.static_center[:, :2] \
            - self.static_half[:, :2] - self.route_inflation_m
        self.route_high_xy = self.static_center[:, :2] \
            + self.static_half[:, :2] + self.route_inflation_m
        self.grid_size = int(round(
            2 * self.map_half_extent_m / self.resolution_m)) + 1
        self.blocked = self._build_blocked()
        self.component = self._build_components()

    def _segment_all_boxes_clear(self, a: np.ndarray, b: np.ndarray) -> bool:
        """Vector form of the original closed-slab segment oracle.

        Comparisons at a slab/tangency boundary are intentionally inclusive.
        Values within 1e-12 of a decisive boundary use the scalar oracle so
        vector rearrangement cannot alter the smoothed route.
        """
        if not len(self.route_low_xy):
            return True
        if self.static_cylinder.any():
            a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
            direction=b-a;denom=float(direction@direction);centers=self.static_center[self.static_cylinder,:2]
            t=np.zeros(len(centers)) if denom==0. else np.clip(((centers-a)*direction).sum(-1)/denom,0.,1.)
            squared=((a+t[:,None]*direction-centers)**2).sum(-1)
            radii=self.static_half[self.static_cylinder,0]+self.route_inflation_m
            if np.any(squared<=radii**2):return False
            return all(_segment_aabb_clear_exact(a,b,box,self.route_inflation_m) for box in self.scene.static_boxes if box.kind=="box")
        a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
        direction = b - a
        enter = np.zeros(len(self.route_low_xy), dtype=np.float64)
        leave = np.ones(len(self.route_low_xy), dtype=np.float64)
        possible = np.ones(len(self.route_low_xy), dtype=bool)
        near_boundary = False
        for axis in range(2):
            if abs(float(direction[axis])) <= EPS:
                inside = ((self.route_low_xy[:, axis] <= a[axis])
                          & (a[axis] <= self.route_high_xy[:, axis]))
                possible &= inside
                near_boundary |= bool(np.any(np.minimum(
                    np.abs(a[axis] - self.route_low_xy[:, axis]),
                    np.abs(a[axis] - self.route_high_xy[:, axis])) <= 1e-12))
                continue
            t0 = (self.route_low_xy[:, axis] - a[axis]) / direction[axis]
            t1 = (self.route_high_xy[:, axis] - a[axis]) / direction[axis]
            enter = np.maximum(enter, np.minimum(t0, t1))
            leave = np.minimum(leave, np.maximum(t0, t1))
        possible &= enter <= leave
        near_boundary |= bool(np.any(np.abs(enter - leave) <= 1e-12))
        if near_boundary:
            return all(_segment_aabb_clear_exact(
                a, b, box, self.route_inflation_m)
                for box in self.scene.static_boxes)
        return not bool(possible.any())

    def segment_static_clear(self, a: np.ndarray, b: np.ndarray) -> bool:
        """Closed-segment static proof used by the cached route follower."""
        a = np.asarray(a, dtype=np.float64).reshape(2)
        b = np.asarray(b, dtype=np.float64).reshape(2)
        limit = self.scene.map_half_extent_m - self.route_inflation_m
        if (not np.isfinite(a).all() or not np.isfinite(b).all()
                or np.max(np.abs(a)) > limit or np.max(np.abs(b)) > limit):
            return False
        return self._segment_all_boxes_clear(a, b)

    def _grid_point(self, cell) -> np.ndarray:
        return (np.array(cell, dtype=float) * self.resolution_m
                - self.map_half_extent_m)

    def _grid_index(self, point) -> tuple[int, int]:
        raw = np.rint((np.asarray(point, dtype=np.float64)
                       + self.map_half_extent_m) / self.resolution_m).astype(int)
        return int(raw[0]), int(raw[1])

    def _build_blocked(self) -> np.ndarray:
        blocked = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        for box in self.scene.static_boxes:
            radius = box.horizontal_radius_m + self.route_inflation_m
            lo = np.maximum(0, np.floor(
                (box.center[:2] - radius + self.map_half_extent_m)
                / self.resolution_m).astype(int))
            hi = np.minimum(self.grid_size - 1, np.ceil(
                (box.center[:2] + radius + self.map_half_extent_m)
                / self.resolution_m).astype(int))
            for ix in range(lo[0], hi[0] + 1):
                for iy in range(lo[1], hi[1] + 1):
                    blocked[ix, iy] |= (
                        np.linalg.norm(self._grid_point((ix, iy))
                                       - box.center[:2]) <= radius)
        return blocked

    def _build_components(self) -> np.ndarray:
        component = np.full(self.blocked.shape, -1, dtype=np.int32)
        neighbours = tuple((dx, dy) for dx in (-1, 0, 1)
                           for dy in (-1, 0, 1) if dx or dy)
        label = 0
        for ix in range(self.grid_size):
            for iy in range(self.grid_size):
                if self.blocked[ix, iy] or component[ix, iy] >= 0:
                    continue
                component[ix, iy] = label; queue = deque(((ix, iy),))
                while queue:
                    cell = queue.popleft()
                    for dx, dy in neighbours:
                        nxt = cell[0] + dx, cell[1] + dy
                        if not (0 <= nxt[0] < self.grid_size
                                and 0 <= nxt[1] < self.grid_size) \
                                or self.blocked[nxt] or component[nxt] >= 0:
                            continue
                        if dx and dy and (
                                self.blocked[cell[0] + dx, cell[1]]
                                or self.blocked[cell[0], cell[1] + dy]):
                            continue
                        component[nxt] = label; queue.append(nxt)
                label += 1
        return component

    def point_clearance(self, point: np.ndarray, *,
                        dynamic_positions: np.ndarray | None = None) -> float:
        point = np.asarray(point, dtype=np.float64).reshape(3)
        centers = self.dynamic_initial if dynamic_positions is None \
            else np.asarray(dynamic_positions, dtype=np.float64)
        if centers.shape != self.dynamic_initial.shape \
                or not np.isfinite(centers).all():
            raise ValueError("dynamic_positions must be finite [N,3]")
        wall = min(self.scene.map_half_extent_m - UAV_RADIUS_M
                   - abs(float(point[0])),
                   self.scene.map_half_extent_m - UAV_RADIUS_M
                   - abs(float(point[1])),
                   float(point[2]) - UAV_RADIUS_M,
                   4.5 - UAV_RADIUS_M - float(point[2]))
        if len(self.static_center):
            static=float(static_signed_clearances(point[None],self.scene.static_boxes,UAV_RADIUS_M).min())
        else:
            static = math.inf
        if len(centers):
            dz = np.abs(point[2] - centers[:, 2]) - self.dynamic_height * .5
            dxy = np.abs(point[None, :2] - centers[:, :2]) \
                - self.dynamic_width[:, None] * .5
            cube_delta = np.column_stack((dxy, dz))
            cube_sdf = (np.linalg.norm(np.maximum(cube_delta, 0.), axis=-1)
                        + np.minimum(cube_delta.max(-1), 0.) - UAV_RADIUS_M)
            radial = np.linalg.norm(point[None, :2] - centers[:, :2], axis=-1) \
                - self.dynamic_width * .5
            cylinder_delta = np.column_stack((radial, dz))
            cylinder_sdf = (
                np.linalg.norm(np.maximum(cylinder_delta, 0.), axis=-1)
                + np.minimum(cylinder_delta.max(-1), 0.) - UAV_RADIUS_M)
            dynamic = float(np.where(
                self.dynamic_cube, cube_sdf, cylinder_sdf).min())
        else:
            dynamic = math.inf
        return min(float(wall), static, dynamic)

    def plan_route(self, start, goal) -> tuple[np.ndarray, ...]:
        """Original A* order/smoothing with immutable grid construction cached."""
        start = np.asarray(start, dtype=np.float64).reshape(2)
        goal = np.asarray(goal, dtype=np.float64).reshape(2)
        if not np.isfinite(start).all() or not np.isfinite(goal).all():
            raise ValueError("route endpoints must be finite XY")
        start_cell, goal_cell = self._grid_index(start), self._grid_index(goal)
        if self.blocked[start_cell] or self.blocked[goal_cell] \
                or self.component[start_cell] != self.component[goal_cell]:
            return ()
        queue = [(0.0, start_cell)]; cost = {start_cell: 0.0}; parent = {}
        neighbours = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                      if dx or dy]
        while queue:
            _, cell = heapq.heappop(queue)
            if cell == goal_cell: break
            base = cost[cell]
            for dx, dy in neighbours:
                nxt = cell[0] + dx, cell[1] + dy
                if not (0 <= nxt[0] < self.grid_size
                        and 0 <= nxt[1] < self.grid_size) \
                        or self.blocked[nxt]:
                    continue
                if dx and dy and (self.blocked[cell[0] + dx, cell[1]]
                                  or self.blocked[cell[0], cell[1] + dy]):
                    continue
                candidate = base + math.hypot(dx, dy) * self.resolution_m
                if candidate + 1e-12 < cost.get(nxt, math.inf):
                    cost[nxt] = candidate; parent[nxt] = cell
                    heuristic = np.linalg.norm(
                        np.array(nxt) - np.array(goal_cell)) * self.resolution_m
                    heapq.heappush(queue, (candidate + heuristic, nxt))
        if goal_cell not in cost: return ()
        cells = [goal_cell]
        while cells[-1] != start_cell: cells.append(parent[cells[-1]])
        raw = [start] + [self._grid_point(c) for c in reversed(cells[1:-1])] \
            + [goal]
        smooth = [raw[0]]; cursor = 0
        while cursor < len(raw) - 1:
            target = len(raw) - 1
            while target > cursor + 1:
                if self._segment_all_boxes_clear(raw[cursor], raw[target]):
                    break
                target -= 1
            smooth.append(raw[target]); cursor = target
        if any(not self._segment_all_boxes_clear(a, b)
               for a, b in zip(smooth[:-1], smooth[1:])):
            return ()
        return tuple(np.asarray(p, dtype=np.float64).copy() for p in smooth)

    def sample(self, num_lanes: int, *, seed: int,
               dynamic_positions: np.ndarray | None = None):
        return _sample_with_sampler(
            self, num_lanes, seed=seed,
            dynamic_positions=dynamic_positions)

    def sample_with_routes(self, num_lanes: int, *, seed: int,
                           dynamic_positions: np.ndarray | None = None):
        return _sample_with_sampler(
            self, num_lanes, seed=seed,
            dynamic_positions=dynamic_positions, return_routes=True)


def _sample_with_sampler(sampler: PlanarMissionSampler, num_lanes: int, *,
        seed: int, dynamic_positions: np.ndarray | None = None,
        return_routes: bool = False):
    scene, config = sampler.scene, sampler.config
    if type(num_lanes) is not int or num_lanes <= 0:
        raise ValueError("num_lanes must be a positive integer")
    if dynamic_positions is not None:
        dynamic_positions = np.asarray(dynamic_positions, dtype=np.float64)
        if dynamic_positions.shape != sampler.dynamic_initial.shape \
                or not np.isfinite(dynamic_positions).all():
            raise ValueError("dynamic_positions must be finite [N,3]")
    rng = np.random.default_rng(int(seed))
    starts = np.empty((num_lanes, 3), dtype=np.float64)
    goals = np.empty_like(starts)
    clearances = np.empty((num_lanes, 2), dtype=np.float64)
    route_waypoint_counts = np.empty(num_lanes, dtype=np.int64)
    accepted_routes = []
    attempts = 0
    rejected_separation=0;rejected_route=0
    mixture = config.mission_sampling_schema == "navrl_edges_opposite50_xy_v1"
    short_full = config.mission_sampling_schema in ("navrl_edges_short_full50_xy_v1","navrl_edges_short_full30_xy_v1","navrl_edges_short_full20_xy_v1","navrl_edges_short_full15_xy_v1","navrl_edges_short_full10_xy_v1","navrl_edges_short_full05_xy_v1","navrl_edges_short_full00_xy_v1")
    short_probability = 0. if config.mission_sampling_schema == "navrl_edges_short_full00_xy_v1" else .1 if config.mission_sampling_schema == "navrl_edges_short_full10_xy_v1" else .05 if config.mission_sampling_schema == "navrl_edges_short_full05_xy_v1" else .15 if config.mission_sampling_schema == "navrl_edges_short_full15_xy_v1" else .2 if config.mission_sampling_schema == "navrl_edges_short_full20_xy_v1" else .3 if config.mission_sampling_schema == "navrl_edges_short_full30_xy_v1" else .5
    accepted_short_components=[];component_pair_proposals=[0,0];component_rejections=[0,0]
    accepted_forced_opposite = 0
    accepted_components = []
    accepted_opposite = 0
    pair_proposals = 0
    forced_pair_proposals = 0

    def sample_point(height_m: float, forced_side=None) -> tuple[np.ndarray, float]:
        nonlocal attempts
        while attempts < config.maximum_sampling_attempts:
            attempts += 1
            if config.mission_sampling_schema in ('navrl_edges_xy_v1','navrl_edges_opposite50_xy_v1','navrl_edges_short1to3_xy_v1','navrl_edges_short_full50_xy_v1','navrl_edges_short_full30_xy_v1','navrl_edges_short_full20_xy_v1','navrl_edges_short_full15_xy_v1','navrl_edges_short_full10_xy_v1','navrl_edges_short_full05_xy_v1','navrl_edges_short_full00_xy_v1'):
                side=int(rng.integers(4)) if forced_side is None else int(forced_side)
                other=rng.uniform(-24.,24.)
                point=np.array([other,other,height_m],dtype=np.float64)
                point[side//2]=24. if side%2 else -24.
            else:
                point = np.array([
                    rng.uniform(-config.mission_xy_half_extent_m,
                                config.mission_xy_half_extent_m),
                    rng.uniform(-config.mission_xy_half_extent_m,
                                config.mission_xy_half_extent_m),
                    height_m], dtype=np.float64)
            clearance = sampler.point_clearance(
                point, dynamic_positions=dynamic_positions)
            # The vectorized SDF is algebraically identical.  At the admission
            # threshold retain the scalar oracle's operation order so no seed
            # stream decision can flip from floating-point rearrangement.
            if abs(clearance - config.mission_clearance_m) <= 1e-12:
                clearance = mission_point_clearance(
                    scene, point, dynamic_positions=dynamic_positions)
            if clearance >= config.mission_clearance_m:
                return point, clearance
        raise RuntimeError("free planar mission sampling exhausted its attempt budget")

    for lane in range(num_lanes):
        short_component=bool(rng.random()<short_probability) if short_full and short_probability>0 else False
        easy = config.easy_mission_probability > 0 and rng.random() < config.easy_mission_probability
        min_distance = 3. if easy else config.mission_min_separation_m
        max_distance = 5. if easy else config.mission_max_separation_m
        if short_full:min_distance,max_distance=(1.,3.) if short_component else (.5,70.)
        forced = mixture and rng.random() < .5
        while True:
            pair_proposals += 1
            if short_full:component_pair_proposals[int(short_component)]+=1
            forced_pair_proposals += int(forced)
            start, start_clearance = sample_point(config.mission_start_z_m)
            start_side = (0 if start[0] == -24. else 1 if start[0] == 24.
                          else 2 if start[1] == -24. else 3)
            goal, goal_clearance = sample_point(
                config.mission_goal_z_m, start_side ^ 1 if forced else None)
            if not min_distance <= np.linalg.norm(goal[:2]-start[:2]) <= max_distance:
                rejected_separation+=1
                if short_full:component_rejections[int(short_component)]+=1
                continue
            route = sampler.plan_route(start[:2], goal[:2])
            if route:
                break
            rejected_route+=1
            if short_full:component_rejections[int(short_component)]+=1
        if mixture:
            goal_side = (0 if goal[0] == -24. else 1 if goal[0] == 24.
                         else 2 if goal[1] == -24. else 3)
            accepted_components.append(bool(forced))
            accepted_forced_opposite += int(forced)
            accepted_opposite += int(goal_side == (start_side ^ 1))
        if short_full:accepted_short_components.append(short_component)
        starts[lane] = start; goals[lane] = goal
        clearances[lane] = (start_clearance, goal_clearance)
        route_waypoint_counts[lane] = len(route)
        accepted_routes.append(tuple(point.copy() for point in route))
    mission_config = asdict(config)
    if config.mission_sampling_schema=="interior_v1": mission_config.pop("mission_sampling_schema")
    if config.mission_max_separation_m == 60. and config.easy_mission_probability == 0.:
        # Preserve the legacy default mission contract byte-for-byte. New
        # curriculum settings carry their explicit range and mixture metadata.
        mission_config.pop("mission_max_separation_m")
        mission_config.pop("easy_mission_probability")
    contract = {
        "schema": "navacc_large_map_planar_missions_v1",
        "seed": int(seed), "num_lanes": num_lanes,
        "attempts": attempts, "config": mission_config,
        "minimum_sampled_clearance_m": float(clearances.min()),
        "minimum_sampled_xy_separation_m": float(np.linalg.norm(
            goals[:, :2] - starts[:, :2], axis=1).min()),
        "initial_static_route_reachability_checked": True,
        "route_waypoint_count_min": int(route_waypoint_counts.min()),
        "route_waypoint_count_max": int(route_waypoint_counts.max()),
    }
    if config.mission_sampling_schema=='navrl_edges_xy_v1':
        contract.update(sampling_distribution='independent uniform four-side proposals; conditioned on net clearance>=.3, separation>=.5 and static reachability',rejected_separation=rejected_separation,rejected_static_unreachable=rejected_route)
    if config.mission_sampling_schema=='navrl_edges_short1to3_xy_v1':
        contract.update(sampling_distribution='independent uniform four-side proposals conditioned only on1<=XYdistance<=3, clearance>=.3 and reachability; same/adjacent edges permitted',rejected_separation=rejected_separation,rejected_static_unreachable=rejected_route)
    if mixture:
        contract.update(
            sampling_distribution="50% mission-component selection independent four-side + 50% opposite-goal; component held across rejection, conditioned on clearance/separation/static reachability",
            opposite_component_probability=.5, accepted_forced_components=accepted_components, pair_proposals=pair_proposals,
            forced_pair_proposals=forced_pair_proposals,
            accepted_forced_opposite=accepted_forced_opposite,
            accepted_opposite=accepted_opposite,
            accepted_forced_opposite_fraction=accepted_forced_opposite/num_lanes,
            accepted_opposite_fraction=accepted_opposite/num_lanes,
            rejected_separation=rejected_separation, rejected_static_unreachable=rejected_route)
    if short_full:
        contract.update(sampling_distribution=f'per episode{short_probability:.0%}short1to3+{1-short_probability:.0%}fullfour-edge.5to70; component fixed across clearance/separation/route rejection',short_component_probability=short_probability,accepted_short_components=accepted_short_components,accepted_short_component_count=sum(accepted_short_components),accepted_full_component_count=num_lanes-sum(accepted_short_components),component_pair_proposals_full_short=component_pair_proposals,component_rejections_full_short=component_rejections,rejected_separation=rejected_separation,rejected_static_unreachable=rejected_route)
    if return_routes:
        return starts, goals, contract, tuple(accepted_routes)
    return starts, goals, contract


def sample_free_planar_missions(scene: LargeMapScene, num_lanes: int, *,
        seed: int, config: PlanarSceneConfig = PlanarSceneConfig(),
        dynamic_positions: np.ndarray | None = None,
        sampler: PlanarMissionSampler | None = None,
        return_routes: bool = False):
    """Sample deterministic free interior missions for one episode/reset set."""
    sampler = PlanarMissionSampler(scene, config) if sampler is None else sampler
    if sampler.scene is not scene or sampler.config != config:
        raise ValueError("mission sampler must match the exact scene and config")
    method = sampler.sample_with_routes if return_routes else sampler.sample
    return method(num_lanes, seed=seed, dynamic_positions=dynamic_positions)


def validate_planar_scene(scene: LargeMapScene, *,
        config: PlanarSceneConfig = PlanarSceneConfig()) -> dict:
    traces = np.stack([body.velocity_trace for body in scene.dynamic_bodies]) \
        if scene.dynamic_bodies else np.empty((0, 0, 3))
    maximum_trace_vz = float(np.abs(traces[..., 2]).max(initial=0.0))
    if maximum_trace_vz > config.dynamic_z_tolerance_m:
        raise ValueError("planar dynamic velocity trace contains vertical motion")
    for time_s in (0.0, scene.dynamic_period_s * .5,
                   scene.dynamic_period_s, scene.deadline_s):
        truth = scene.dynamic_state(time_s)
        expected_z = np.asarray(
            [body.initial_position[2] for body in scene.dynamic_bodies])
        if (not np.allclose(truth.position[:, 2], expected_z, atol=0, rtol=0)
                or not np.allclose(truth.velocity[:, 2], 0.0, atol=0, rtol=0)):
            raise ValueError("planar dynamic truth changed physical z or vz")
    return {
        "schema": "navacc_large_map_planar_scene_contract_v1",
        "static_count": len(scene.static_boxes),
        "dynamic_count": len(scene.dynamic_bodies),
        "dynamic_fixed_z": True,
        "maximum_velocity_trace_abs_vz_m_s": maximum_trace_vz,
        "dynamic_period_s": scene.dynamic_period_s,
        "global_clock_reset_with_episode": False,
    }



def convert_scene_to_all_cylinders(scene):
    """Explicit geometry conversion: width becomes diameter; keep centers/heights/truth."""
    if any(b.size[0]!=b.size[1] for b in scene.static_boxes):
        raise ValueError("width-preserving conversion requires square legacy static XY")
    boxes=tuple(StaticBox(b.center,b.size,"cylinder") for b in scene.static_boxes)
    bodies=tuple(DynamicBody(b.body_id,b.class_id,"cylinder",b.width_m,b.height_m,
                             b.initial_position,b.velocity_trace) for b in scene.dynamic_bodies)
    return LargeMapScene(scene.seed,boxes,bodies,scene.starts,scene.goals,scene.deadline_s,
        scene.map_half_extent_m,scene.obstacle_half_extent_m,scene.dynamic_period_s,
        scene.static_height_sampling,"navacc-navrl-shared50-cylinders-v1","all_cylinders_width_diameter_v1")

def generate_planar_scene(seed: int, num_lanes: int, *,
        trace_duration_s: float, config: PlanarSceneConfig = PlanarSceneConfig(),
        static_count: int = 350, dynamic_count: int = 120,
        obstacle_geometry_mode: str = "legacy_box_mixed_v1",
        ) -> tuple[LargeMapScene, dict]:
    """Generate a versioned scene without altering the legacy generator."""
    base = generate_scene(
        seed, num_lanes, static_count=static_count, dynamic_count=dynamic_count,
        trace_duration_s=trace_duration_s)
    if obstacle_geometry_mode not in ("legacy_box_mixed_v1","all_cylinders_width_diameter_v1"):
        raise ValueError("unsupported obstacle geometry mode")
    if obstacle_geometry_mode=="all_cylinders_width_diameter_v1":base=convert_scene_to_all_cylinders(base)
    bodies = []
    for body in base.dynamic_bodies:
        trace = np.array(body.velocity_trace, copy=True)
        trace[:, 2] = 0.0
        bodies.append(DynamicBody(
            body.body_id, body.class_id, body.kind, body.width_m,
            body.height_m, body.initial_position.copy(), trace))
    placeholder = LargeMapScene(
        seed=base.seed, static_boxes=base.static_boxes,
        dynamic_bodies=tuple(bodies), starts=base.starts, goals=base.goals,
        deadline_s=base.deadline_s,
        map_half_extent_m=base.map_half_extent_m,
        obstacle_half_extent_m=base.obstacle_half_extent_m,
        dynamic_period_s=base.dynamic_period_s,
        static_height_sampling=base.static_height_sampling,format_version=base.format_version,obstacle_geometry_mode=base.obstacle_geometry_mode)
    # A disjoint RNG stream keeps static/dynamic geometry identical to the
    # historical generator for the same seed.
    sampler = PlanarMissionSampler(placeholder, config)
    starts, goals, mission = sample_free_planar_missions(
        placeholder, num_lanes, seed=int(seed) ^ 0x504C414E, config=config,
        sampler=sampler)
    scene = LargeMapScene(
        seed=base.seed, static_boxes=base.static_boxes,
        dynamic_bodies=tuple(bodies), starts=starts, goals=goals,
        deadline_s=base.deadline_s,
        map_half_extent_m=base.map_half_extent_m,
        obstacle_half_extent_m=base.obstacle_half_extent_m,
        dynamic_period_s=base.dynamic_period_s,
        static_height_sampling=base.static_height_sampling,format_version=base.format_version,obstacle_geometry_mode=base.obstacle_geometry_mode)
    contract = validate_planar_scene(scene, config=config)
    contract.update({
        "format_version": ("navacc-navrl-shared50-planar-cylinders-v1" if obstacle_geometry_mode!="legacy_box_mixed_v1" else "navacc-navrl-shared50-planar-v1"),
        "geometry_parent_generator": "navacc_large_map.generate_scene",
        "xy_geometry_and_trace_distribution_preserved": obstacle_geometry_mode=="legacy_box_mixed_v1",
        "mission": mission,
        "trace_duration_s": float(trace_duration_s),
    })
    if obstacle_geometry_mode!="legacy_box_mixed_v1":
        contract.update(obstacle_geometry_mode=obstacle_geometry_mode,cylinder_width_semantics="diameter",cylinder_axis="Z",center_height_original_xy_trace_preserved=True,class_id_semantics="original_size_height_category_not_current_shape",native_geometry_verified=False)
    return scene, contract


def save_planar_scene(scene: LargeMapScene, scene_path: str | Path,
                      contract: dict, contract_path: str | Path) -> dict:
    scene_path = Path(scene_path); contract_path = Path(contract_path)
    if contract_path.exists():
        raise FileExistsError(contract_path)
    scene_sha = save_scene(scene, scene_path)
    payload = dict(contract)
    payload["scene_path"] = str(scene_path.resolve())
    payload["scene_sha256"] = scene_sha
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    contract_path.write_text(text)
    return {"scene_sha256": scene_sha,
            "contract_sha256": hashlib.sha256(text.encode()).hexdigest()}


__all__ = ["PlanarMissionSampler", "PlanarSceneConfig", "generate_planar_scene",
           "mission_point_clearance", "sample_free_planar_missions",
           "save_planar_scene", "validate_planar_scene"]

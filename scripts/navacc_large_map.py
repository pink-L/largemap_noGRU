#!/usr/bin/env python3
"""Pure-CPU 50 m shared-map scene, planner and NavACC safety controller.

The file has no Isaac/Hydra/Torch dependency.  It is the executable contract
for the native environment: 350 shared static boxes, 120 shared kinematic
obstacles, a cached grid A* route, the frozen structured Actor observation,
and bounded acceleration selection.  Dynamic velocity changes are
discontinuous every two seconds, so C3BF row feasibility is logged separately
from the conservative reachability gate and is never called a continuous-time
certificate.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import heapq
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from navacc_engineering import solve_disk_halfplanes


MAP_HALF_EXTENT_M = 25.0
OBSTACLE_HALF_EXTENT_M = 20.0
GRID_RESOLUTION_M = 0.25
DYNAMIC_PERIOD_S = 2.0
DYNAMIC_SPEED_MIN_M_S = 0.5
DYNAMIC_SPEED_MAX_M_S = 1.5
MISSION_DEADLINE_S = 120.0
UAV_RADIUS_M = 0.10
EPS = 1.0e-9


def _finite_vec(value: Sequence[float], size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite shape ({size},)")
    return result


def _unit(value: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    return np.zeros_like(value) if norm <= EPS else value / norm


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return value


def dynamic_center_region_status(
        positions: np.ndarray, half_extent_m: float,
        tolerance_m: float = 1.0e-4) -> tuple[float, bool]:
    """Return runtime center-region margin and fail-closed violation state."""
    points = np.asarray(positions, dtype=np.float64)
    if (points.ndim != 2 or points.shape[1] != 3
            or not math.isfinite(float(half_extent_m)) or half_extent_m <= 0
            or not math.isfinite(float(tolerance_m)) or tolerance_m < 0
            or not np.isfinite(points).all()):
        return -math.inf, True
    if len(points) == 0:
        return math.inf, False
    margin = float(half_extent_m - np.max(np.abs(points[:, :2])))
    return margin, margin < -tolerance_m


@dataclass(frozen=True)
class StaticBox:
    center: np.ndarray
    size: np.ndarray
    kind: str = "box"

    def __post_init__(self):
        center = _finite_vec(self.center, 3, "box center").copy(); center.setflags(write=False)
        size = _finite_vec(self.size, 3, "box size").copy(); size.setflags(write=False)
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "size", size)
        if np.any(self.size <= 0):
            raise ValueError("box size must be positive")
        if self.kind not in ("box", "cylinder"):
            raise ValueError("static kind must be box or cylinder")
        if self.kind == "cylinder" and self.size[0] != self.size[1]:
            raise ValueError("cylinder size XY must be equal diameters")

    @property
    def horizontal_radius_m(self) -> float:
        return (float(self.size[0] * .5) if self.kind == "cylinder"
                else float(np.linalg.norm(self.size[:2] * .5)))


def static_signed_clearances(points, boxes, inflation=0.0):
    """Exact finite upright cylinder/box SDF, preserving old all-box arithmetic."""
    points=np.asarray(points,dtype=np.float64).reshape(-1,3)
    if not boxes:return np.empty((len(points),0))
    centers=np.stack([b.center for b in boxes]);half=np.stack([b.size for b in boxes])*.5
    delta=np.abs(points[:,None]-centers[None])-half[None]
    sdf=np.linalg.norm(np.maximum(delta,0.),axis=-1)+np.minimum(delta.max(-1),0.)
    mask=np.array([b.kind=="cylinder" for b in boxes],dtype=bool)
    if mask.any():
        radial=np.linalg.norm(points[:,None,:2]-centers[None,:,:2],axis=-1)[:,mask]-half[mask,0][None]
        dz=np.abs(points[:,None,2]-centers[None,:,2])[:,mask]-half[mask,2][None]
        dc=np.stack((radial,dz),-1)
        sdf[:,mask]=np.linalg.norm(np.maximum(dc,0.),axis=-1)+np.minimum(dc.max(-1),0.)
    return sdf-inflation


def _segment_disc_clear(a,b,center,radius):
    direction=b-a;denom=float(direction@direction)
    t=0. if denom==0. else float(np.clip((center-a)@direction/denom,0.,1.))
    delta=a+t*direction-center
    return float(delta@delta)>radius*radius


def _ray_cylinder_distances(origin,directions,centers,half,max_range):
    """Radial quadratic interval intersected with finite Z slab; inside hit=0."""
    oc=np.asarray(origin)[None,None]-centers[None]
    d=np.asarray(directions)[:,None]
    aa=(d[...,:2]**2).sum(-1);bb=(oc[...,:2]*d[...,:2]).sum(-1)
    cc=(oc[...,:2]**2).sum(-1)-half[None,:,0]**2
    parallel=aa<=EPS**2;safea=np.where(parallel,1.,aa);disc=bb**2-aa*cc
    root=np.sqrt(np.maximum(disc,0.));lo=(-bb-root)/safea;hi=(-bb+root)/safea
    lo=np.where(parallel,-np.inf,lo);hi=np.where(parallel,np.inf,hi)
    radial_valid=np.where(parallel,cc<=0.,disc>=0.)
    dz=d[...,2];pz=oc[...,2];parallelz=np.abs(dz)<=EPS;safez=np.where(parallelz,1.,dz)
    z0=(-half[None,:,2]-pz)/safez;z1=(half[None,:,2]-pz)/safez
    zlo=np.where(parallelz,-np.inf,np.minimum(z0,z1));zhi=np.where(parallelz,np.inf,np.maximum(z0,z1))
    zvalid=(~parallelz)|(np.abs(pz)<=half[None,:,2])
    enter=np.maximum(lo,zlo);leave=np.minimum(hi,zhi)
    hit=radial_valid&zvalid&(leave>=np.maximum(enter,0.))&(enter<=max_range)
    return np.where(hit,np.maximum(enter,0.),max_range)


@dataclass(frozen=True)
class DynamicBody:
    body_id: int
    class_id: int
    kind: str
    width_m: float
    height_m: float
    initial_position: np.ndarray
    velocity_trace: np.ndarray

    def __post_init__(self):
        initial = _finite_vec(self.initial_position, 3, "dynamic initial_position").copy()
        initial.setflags(write=False); object.__setattr__(self, "initial_position", initial)
        trace = np.asarray(self.velocity_trace, dtype=np.float64)
        if trace.ndim != 2 or trace.shape[1] != 3 or not np.isfinite(trace).all():
            raise ValueError("velocity trace must be finite [segments,3]")
        trace = trace.copy(); trace.setflags(write=False)
        object.__setattr__(self, "velocity_trace", trace)
        if self.kind not in ("cube", "cylinder"):
            raise ValueError("dynamic kind must be cube or cylinder")
        if (not math.isfinite(self.width_m) or not math.isfinite(self.height_m)
                or self.width_m <= 0 or self.height_m <= 0):
            raise ValueError("dynamic dimensions must be positive")
        if np.linalg.norm(trace, axis=1).max(initial=0.0) > DYNAMIC_SPEED_MAX_M_S + 1e-8:
            raise ValueError("velocity trace exceeds declared 1.5 m/s reach bound")

    @property
    def horizontal_radius_m(self) -> float:
        # A cube is protected at its corners; cylinder width is its diameter.
        return (self.width_m / math.sqrt(2.0) if self.kind == "cube"
                else self.width_m * .5)


@dataclass(frozen=True)
class DynamicState:
    position: np.ndarray
    velocity: np.ndarray
    time_to_velocity_switch_s: float
    segment_index: int
    time_to_discontinuity_s: np.ndarray


@dataclass(frozen=True)
class LargeMapScene:
    seed: int
    static_boxes: tuple[StaticBox, ...]
    dynamic_bodies: tuple[DynamicBody, ...]
    starts: np.ndarray
    goals: np.ndarray
    deadline_s: float = MISSION_DEADLINE_S
    map_half_extent_m: float = MAP_HALF_EXTENT_M
    obstacle_half_extent_m: float = OBSTACLE_HALF_EXTENT_M
    dynamic_period_s: float = DYNAMIC_PERIOD_S
    static_height_sampling: str = (
        "mixture_uniform_intervals_[1,1.5,2,4,6]_p_[.1,.15,.2,.55]")
    format_version: str = "navacc-navrl-shared50-v1"
    obstacle_geometry_mode: str = "legacy_box_mixed_v1"
    _dynamic_segment_starts: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if self.obstacle_geometry_mode not in ("legacy_box_mixed_v1","all_cylinders_width_diameter_v1"):
            raise ValueError("unsupported obstacle geometry mode")
        if self.obstacle_geometry_mode=="all_cylinders_width_diameter_v1":
            if any(b.kind!="cylinder" for b in self.static_boxes+self.dynamic_bodies):
                raise ValueError("all-cylinder mode requires every static/dynamic body to be cylinder")
        elif any(b.kind!="box" for b in self.static_boxes):
            raise ValueError("new static cylinder geometry requires explicit all-cylinder mode")
        required_format=("navacc-navrl-shared50-cylinders-v1" if self.obstacle_geometry_mode=="all_cylinders_width_diameter_v1" else "navacc-navrl-shared50-v1")
        if self.format_version != required_format:
            raise ValueError(f"scene format/geometry mode mismatch: {self.format_version}")
        starts = np.asarray(self.starts, dtype=np.float64)
        goals = np.asarray(self.goals, dtype=np.float64)
        if (starts.ndim != 2 or starts.shape[1] != 3 or goals.shape != starts.shape
                or not np.isfinite(starts).all() or not np.isfinite(goals).all()):
            raise ValueError("starts/goals must be aligned finite [lanes,3]")
        if len(starts) < 1:
            raise ValueError("scene must contain at least one lane")
        object.__setattr__(self, "starts", starts)
        object.__setattr__(self, "goals", goals)
        if not math.isfinite(self.deadline_s) or self.deadline_s <= 0:
            raise ValueError("deadline must be finite positive")
        if (not math.isfinite(self.map_half_extent_m) or self.map_half_extent_m <= 0
                or not math.isfinite(self.obstacle_half_extent_m)
                or self.obstacle_half_extent_m <= 0
                or self.obstacle_half_extent_m >= self.map_half_extent_m
                or not math.isfinite(self.dynamic_period_s) or self.dynamic_period_s <= 0):
            raise ValueError("scene map and motion bounds must be finite and ordered")
        if (np.abs(starts[:, :2]) > self.map_half_extent_m).any() or \
                (np.abs(goals[:, :2]) > self.map_half_extent_m).any() or \
                (starts[:, 2] < 0).any() or (starts[:, 2] > 4.5).any() or \
                (goals[:, 2] < 0).any() or (goals[:, 2] > 4.5).any():
            raise ValueError("start/goal lies outside declared map")
        ids = [body.body_id for body in self.dynamic_bodies]
        if ids != list(range(len(ids))):
            raise ValueError("dynamic body IDs must be unique tuple-order indices 0..N-1")
        if self.dynamic_bodies:
            required_endpoint_segment = int(self.deadline_s // self.dynamic_period_s)
            if min(len(b.velocity_trace) for b in self.dynamic_bodies) <= required_endpoint_segment:
                raise ValueError("dynamic trace does not cover the mission deadline endpoint")
        if self.dynamic_bodies:
            segments = min(len(b.velocity_trace) for b in self.dynamic_bodies)
            starts_cache = np.empty((len(self.dynamic_bodies), segments, 3), dtype=np.float64)
            for body_index, body in enumerate(self.dynamic_bodies):
                pos = body.initial_position.copy()
                for segment in range(segments):
                    starts_cache[body_index, segment] = pos
                    pos, _ = _reflect_3d(pos, body.velocity_trace[segment],
                                         self.dynamic_period_s,
                                         self.obstacle_half_extent_m)
        else:
            starts_cache = np.empty((0, 0, 3), dtype=np.float64)
        starts_cache.setflags(write=False)
        starts.setflags(write=False); goals.setflags(write=False)
        object.__setattr__(self, "_dynamic_segment_starts", starts_cache)

    def to_dict(self) -> dict:
        return {
            "format_version": self.format_version,
            "seed": int(self.seed),
            "map": {"half_extent_m": self.map_half_extent_m,
                    "obstacle_half_extent_m": self.obstacle_half_extent_m,
                    "grid_resolution_m": GRID_RESOLUTION_M},
            "mission": {"starts": self.starts.tolist(), "goals": self.goals.tolist(),
                        "deadline_s": self.deadline_s,
                        "historical_35_2s_deadline_applicable": False},
            "static_height_sampling": self.static_height_sampling,
            "static_boxes": [{"center": b.center.tolist(), "size": b.size.tolist(),
                              **({"kind": b.kind} if b.kind != "box" else {})}
                             for b in self.static_boxes],
            "dynamic_bodies": [
                {"body_id": b.body_id, "class_id": b.class_id, "kind": b.kind,
                 "width_m": b.width_m, "height_m": b.height_m,
                 "initial_position": b.initial_position.tolist(),
                 "velocity_trace": b.velocity_trace.tolist()}
                for b in self.dynamic_bodies],
            "motion": {
                "period_s": self.dynamic_period_s,
                "speed_range_m_s": [DYNAMIC_SPEED_MIN_M_S, DYNAMIC_SPEED_MAX_M_S],
                "xy_clamp_m": self.obstacle_half_extent_m,
                "z_clamp_m": [0.0, 4.5],
                "global_clock": "continuous_not_episode_reset",
                "future_velocity_oracle_used_by_controller": False,
                "continuous_acceleration_certificate_supported": False,
            },
            "geometry": {
                "static_planner": ("shape_discs_grid_astar" if self.obstacle_geometry_mode=="all_cylinders_width_diameter_v1" else "circumscribed_box_discs_grid_astar"),
                "execution_gate": ("finite_box_or_cylinder_3d_geometry" if self.obstacle_geometry_mode=="all_cylinders_width_diameter_v1" else "full_box_and_cube_or_cylinder_3d_geometry"),
                **({"schema": "all_cylinders_width_as_diameter_zaxis_v1",
                    "cylinder_axis": "Z", "width_semantics": "diameter",
                    "obstacle_geometry_mode": self.obstacle_geometry_mode,
                    "static_cylinder_count": sum(b.kind == "cylinder" for b in self.static_boxes),
                    "dynamic_cylinder_count": sum(b.kind == "cylinder" for b in self.dynamic_bodies)}
                   if self.obstacle_geometry_mode=="all_cylinders_width_diameter_v1" else {}),
                "shared_world_objects": True,
            },
        }

    @classmethod
    def from_dict(cls, value: Mapping) -> "LargeMapScene":
        mission, map_cfg = value["mission"], value["map"]
        motion = value["motion"]
        return cls(
            seed=int(value["seed"]),
            static_boxes=tuple(StaticBox(v["center"], v["size"], v.get("kind", "box"))
                               for v in value["static_boxes"]),
            dynamic_bodies=tuple(DynamicBody(
                int(v["body_id"]), int(v["class_id"]), str(v["kind"]),
                float(v["width_m"]), float(v["height_m"]),
                v["initial_position"], v["velocity_trace"])
                for v in value["dynamic_bodies"]),
            starts=mission["starts"], goals=mission["goals"],
            deadline_s=float(mission["deadline_s"]),
            map_half_extent_m=float(map_cfg["half_extent_m"]),
            obstacle_half_extent_m=float(map_cfg["obstacle_half_extent_m"]),
            dynamic_period_s=float(motion["period_s"]),
            static_height_sampling=str(value["static_height_sampling"]),
            format_version=str(value["format_version"]),
            obstacle_geometry_mode=str(value.get("geometry",{}).get("obstacle_geometry_mode","legacy_box_mixed_v1")),
        )

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"),
                          allow_nan=False)

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()

    @property
    def num_lanes(self) -> int:
        return int(len(self.starts))

    def dynamic_state(self, global_time_s: float) -> DynamicState:
        """Return deterministic shared-world state at absolute global time."""
        t = float(global_time_s)
        if not math.isfinite(t) or t < 0:
            raise ValueError("global_time_s must be finite and nonnegative")
        if not self.dynamic_bodies:
            return DynamicState(np.empty((0, 3)), np.empty((0, 3)),
                                self.dynamic_period_s, int(t // self.dynamic_period_s),
                                np.empty(0))
        segment = int(t // self.dynamic_period_s)
        local = t - segment * self.dynamic_period_s
        positions, velocities = [], []
        for body_index, body in enumerate(self.dynamic_bodies):
            if segment >= len(body.velocity_trace):
                raise ValueError("scene velocity trace does not cover requested global time")
            pos = self._dynamic_segment_starts[body_index, segment].copy()
            pos, velocity = _reflect_3d(pos, body.velocity_trace[segment], local,
                                        self.obstacle_half_extent_m)
            positions.append(pos); velocities.append(velocity)
        scheduled_remaining = self.dynamic_period_s - local
        body_remaining = np.full(len(positions), scheduled_remaining, dtype=np.float64)
        # Wall/ceiling reflections are also discontinuities.  Expose the first
        # possible event so the controller's reach envelope begins there rather
        # than assuming that only scheduled 2 s changes exist.
        for body_index, (pos, velocity) in enumerate(zip(positions, velocities)):
            for axis, (low, high) in enumerate(((-self.obstacle_half_extent_m,
                                                  self.obstacle_half_extent_m),
                                                 (-self.obstacle_half_extent_m,
                                                  self.obstacle_half_extent_m),
                                                 (0.0, 4.5))):
                if abs(velocity[axis]) > EPS:
                    hit = ((high - pos[axis]) / velocity[axis] if velocity[axis] > 0
                           else (low - pos[axis]) / velocity[axis])
                    if hit >= 0:
                        body_remaining[body_index] = min(body_remaining[body_index], float(hit))
        return DynamicState(np.asarray(positions), np.asarray(velocities),
                            scheduled_remaining, segment, body_remaining)


def _reflect_axis(position: float, velocity: float, duration: float,
                  limit: float) -> tuple[float, float]:
    x, v, remaining = float(position), float(velocity), float(duration)
    while remaining > 1e-12 and abs(v) > EPS:
        boundary = limit if v > 0 else -limit
        hit = (boundary - x) / v
        if hit < -1e-10:
            x = min(limit, max(-limit, x)); v = -v; continue
        if hit >= remaining:
            x += v * remaining; remaining = 0.0
        else:
            x = boundary; remaining -= max(hit, 0.0); v = -v
    return min(limit, max(-limit, x)), v


def _reflect_xy(position: np.ndarray, velocity: np.ndarray, duration: float,
                limit: float) -> tuple[np.ndarray, np.ndarray]:
    px, vx = _reflect_axis(position[0], velocity[0], duration, limit)
    py, vy = _reflect_axis(position[1], velocity[1], duration, limit)
    return np.array([px, py]), np.array([vx, vy])


def _reflect_3d(position: np.ndarray, velocity: np.ndarray, duration: float,
                xy_limit: float) -> tuple[np.ndarray, np.ndarray]:
    px, vx = _reflect_axis(position[0], velocity[0], duration, xy_limit)
    py, vy = _reflect_axis(position[1], velocity[1], duration, xy_limit)
    # Shift z into a symmetric interval for the same exact reflection helper.
    pz, vz = _reflect_axis(position[2] - 2.25, velocity[2], duration, 2.25)
    return np.array([px, py, pz + 2.25]), np.array([vx, vy, vz])


def _sample_static_height(rng: np.random.Generator) -> float:
    edges = np.array([1.0, 1.5, 2.0, 4.0, 6.0])
    index = int(rng.choice(4, p=[.1, .15, .2, .55]))
    return float(rng.uniform(edges[index], edges[index + 1]))


def generate_scene(seed: int, num_lanes: int = 4, *, static_count: int = 350,
                   dynamic_count: int = 120,
                   forced_static: Sequence[Mapping] | None = None,
                   trace_duration_s: float = MISSION_DEADLINE_S + 4.0,
                   dynamic_class_allocation: Sequence[int] | None = None
                   ) -> LargeMapScene:
    """Generate the deterministic shared 50 m benchmark scene."""
    if num_lanes <= 0 or static_count < 0 or dynamic_count < 0:
        raise ValueError("scene counts must be valid")
    if dynamic_class_allocation is None:
        if dynamic_count and dynamic_count % 8:
            raise ValueError("dynamic_count must be zero or divisible by eight")
    elif dynamic_count!=20 or tuple(dynamic_class_allocation)!=(3,2,3,2,2,3,2,3):
        raise ValueError("explicit phase allocation requires registered exact20 class counts")
    rng = np.random.default_rng(int(seed))
    xs = np.linspace(-18.0, 18.0, num_lanes + 2)[1:-1]
    starts = np.column_stack((xs, np.full(num_lanes, 24.0), np.full(num_lanes, 2.0)))
    goals = np.column_stack((xs, np.full(num_lanes, -24.0), np.full(num_lanes, 2.0)))
    boxes: list[StaticBox] = []
    if forced_static is not None:
        boxes.extend(StaticBox(v["center"], v["size"]) for v in forced_static)
    attempts = 0
    while len(boxes) < static_count:
        attempts += 1
        if attempts > max(10000, static_count * 100):
            raise RuntimeError("could not place static boxes")
        width = float(rng.choice([.4, .8])); height = _sample_static_height(rng)
        center = np.array([rng.uniform(-19.5, 19.5), rng.uniform(-19.5, 19.5), height / 2])
        candidate = StaticBox(center, np.array([width, width, height]))
        # Only prevent physical overlap; dense placement remains NavRL-like.
        if all(np.any(np.abs(candidate.center[:2] - b.center[:2]) >=
                      (candidate.size[:2] + b.size[:2]) * .5 + .05) for b in boxes):
            boxes.append(candidate)
    segments = int(math.ceil(trace_duration_s / DYNAMIC_PERIOD_S)) + 1
    bodies: list[DynamicBody] = []
    per_class = dynamic_count // 8 if dynamic_count else 0
    widths = (.25, .5, .75, 1.0)
    for class_id in range(8):
        kind = "cube" if class_id < 4 else "cylinder"
        width = widths[class_id % 4]; height = 1.0 if kind == "cube" else 5.0
        for _ in range(per_class if dynamic_class_allocation is None else dynamic_class_allocation[class_id]):
            body_id = len(bodies)
            z = float(rng.uniform(height * .5, max(height * .5, 4.5)))
            z = min(4.5, z)
            initial = np.array([rng.uniform(-20, 20), rng.uniform(-20, 20), z])
            angle = rng.uniform(-math.pi, math.pi, size=segments)
            speed = rng.uniform(DYNAMIC_SPEED_MIN_M_S, DYNAMIC_SPEED_MAX_M_S,
                                size=segments)
            if dynamic_class_allocation is not None:
                # Versioned exact20 phase: actual planar speed .5--1.5 for
                # every height class; legacy None draw order is unchanged.
                if class_id>=4:initial[2]=2.5
                trace=np.column_stack((np.cos(angle)*speed,np.sin(angle)*speed,np.zeros(segments)))
            elif kind == "cube":
                elevation = rng.uniform(-math.pi / 3, math.pi / 3, size=segments)
                trace = np.column_stack((np.cos(elevation) * np.cos(angle) * speed,
                                         np.cos(elevation) * np.sin(angle) * speed,
                                         np.sin(elevation) * speed))
            else:
                initial[2] = 2.5
                trace = np.column_stack((np.cos(angle) * speed, np.sin(angle) * speed,
                                         np.zeros(segments)))
            bodies.append(DynamicBody(body_id, class_id, kind, width, height,
                                      initial, trace))
    return LargeMapScene(int(seed), tuple(boxes), tuple(bodies), starts, goals)


def save_scene(scene: LargeMapScene, path: str | Path) -> str:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(scene.to_dict(), sort_keys=True, indent=2, allow_nan=False) + "\n"
    path.write_text(text, encoding="utf-8")
    return scene.sha256()


def load_scene(path: str | Path) -> LargeMapScene:
    return LargeMapScene.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _segment_aabb_clear(a: np.ndarray, b: np.ndarray, box: StaticBox,
                        inflation_m: float) -> bool:
    if box.kind == "cylinder":
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


def plan_grid_route(start: Sequence[float], goal: Sequence[float],
                    boxes: Sequence[StaticBox], *, resolution_m: float = GRID_RESOLUTION_M,
                    map_half_extent_m: float = MAP_HALF_EXTENT_M,
                    inflation_m: float = UAV_RADIUS_M + .12) -> tuple[np.ndarray, ...]:
    """8-neighbour grid A* followed by collision-checked line smoothing."""
    start = _finite_vec(start, 2, "route start"); goal = _finite_vec(goal, 2, "route goal")
    if not math.isfinite(resolution_m) or resolution_m <= 0:
        raise ValueError("resolution must be finite positive")
    size = int(round(2 * map_half_extent_m / resolution_m)) + 1
    def index(point):
        raw = np.rint((point + map_half_extent_m) / resolution_m).astype(int)
        return int(raw[0]), int(raw[1])
    def point(cell):
        return np.array(cell, dtype=float) * resolution_m - map_half_extent_m
    start_cell, goal_cell = index(start), index(goal)
    blocked = np.zeros((size, size), dtype=bool)
    for box in boxes:
        # Deliberately conservative circumscribed discs for route planning.
        radius = box.horizontal_radius_m + inflation_m
        lo = np.maximum(0, np.floor((box.center[:2] - radius + map_half_extent_m)
                                    / resolution_m).astype(int))
        hi = np.minimum(size - 1, np.ceil((box.center[:2] + radius + map_half_extent_m)
                                         / resolution_m).astype(int))
        for ix in range(lo[0], hi[0] + 1):
            for iy in range(lo[1], hi[1] + 1):
                blocked[ix, iy] |= (np.linalg.norm(point((ix, iy)) - box.center[:2])
                                    <= radius)
    if blocked[start_cell] or blocked[goal_cell]:
        return ()
    queue = [(0.0, start_cell)]; cost = {start_cell: 0.0}; parent = {}
    neighbours = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                  if dx or dy]
    while queue:
        _, cell = heapq.heappop(queue)
        if cell == goal_cell:
            break
        base = cost[cell]
        for dx, dy in neighbours:
            nxt = cell[0] + dx, cell[1] + dy
            if not (0 <= nxt[0] < size and 0 <= nxt[1] < size) or blocked[nxt]:
                continue
            if dx and dy and (blocked[cell[0] + dx, cell[1]]
                              or blocked[cell[0], cell[1] + dy]):
                continue
            candidate = base + math.hypot(dx, dy) * resolution_m
            if candidate + 1e-12 < cost.get(nxt, math.inf):
                cost[nxt] = candidate; parent[nxt] = cell
                heuristic = np.linalg.norm(np.array(nxt) - np.array(goal_cell)) * resolution_m
                heapq.heappush(queue, (candidate + heuristic, nxt))
    if goal_cell not in cost:
        return ()
    cells = [goal_cell]
    while cells[-1] != start_cell:
        cells.append(parent[cells[-1]])
    raw = [start] + [point(c) for c in reversed(cells[1:-1])] + [goal]
    smooth = [raw[0]]; cursor = 0
    while cursor < len(raw) - 1:
        target = len(raw) - 1
        while target > cursor + 1:
            if all(_segment_aabb_clear(raw[cursor], raw[target], box, inflation_m)
                   for box in boxes):
                break
            target -= 1
        smooth.append(raw[target]); cursor = target
    if any(not all(_segment_aabb_clear(a, b, box, inflation_m) for box in boxes)
           for a, b in zip(smooth[:-1], smooth[1:])):
        return ()
    return tuple(np.asarray(p, dtype=np.float64).copy() for p in smooth)


@dataclass(frozen=True)
class VariableRadiusRows:
    q: np.ndarray
    b: np.ndarray
    h: np.ndarray
    inflated_radius_m: np.ndarray
    reasons: tuple[str, ...]
    domain_valid: bool
    continuous_acceleration_certificate_supported: bool = False


def build_variable_radius_rows(position: Sequence[float], velocity: Sequence[float],
                               obstacle_positions: np.ndarray,
                               obstacle_velocities: np.ndarray,
                               bodies: Sequence[DynamicBody], *,
                               uav_radius_m: float = UAV_RADIUS_M,
                               extra_margin_m: float = .05,
                               gamma_per_s: float = 1.2,
                               min_speed_m_s: float = 1e-4) -> VariableRadiusRows:
    position = _finite_vec(position, 2, "position"); velocity = _finite_vec(velocity, 2, "velocity")
    op = np.asarray(obstacle_positions, dtype=np.float64).reshape(-1, 2)
    ov = np.asarray(obstacle_velocities, dtype=np.float64).reshape(-1, 2)
    if len(op) != len(bodies) or ov.shape != op.shape or not np.isfinite(op).all() or not np.isfinite(ov).all():
        raise ValueError("dynamic row inputs must be finite and aligned")
    qs, bs, hs, reasons, radii = [], [], [], [], []
    for p_obs, v_obs, body in zip(op, ov, bodies):
        radius = body.horizontal_radius_m + uav_radius_m + extra_margin_m
        radii.append(radius); p = p_obs - position; w = v_obs - velocity
        rho2 = float(p @ p - radius * radius); speed = float(np.linalg.norm(w))
        if rho2 <= 1e-4:
            reasons.append("inside_or_near_inflated_radius"); continue
        if speed <= min_speed_m_s:
            reasons.append("low_relative_speed"); continue
        rho = math.sqrt(rho2); q = p + rho * w / speed; qnorm = float(np.linalg.norm(q))
        if qnorm <= .01:
            reasons.append("degenerate_q"); continue
        h = float(p @ w + rho * speed)
        # Current velocity is valid inside this piecewise-constant segment.
        # There is no finite obstacle-acceleration premise at its boundaries.
        b = (speed / rho + gamma_per_s) * h
        qs.append(q); bs.append(b); hs.append(h); reasons.append("valid_current_segment")
    domain = all(v == "valid_current_segment" for v in reasons)
    return VariableRadiusRows(np.asarray(qs).reshape(-1, 2), np.asarray(bs).reshape(-1, 1),
                              np.asarray(hs), np.asarray(radii), tuple(reasons), domain)


def _ray_aabb_distance(origin: np.ndarray, direction: np.ndarray, box: StaticBox,
                       max_range: float) -> float:
    if box.kind == "cylinder":
        return float(_ray_cylinder_distances(origin, direction[None],box.center[None],(box.size*.5)[None],max_range)[0,0])
    low, high = box.center - box.size * .5, box.center + box.size * .5
    enter, leave = 0.0, max_range
    for axis in range(3):
        if abs(direction[axis]) <= EPS:
            if not low[axis] <= origin[axis] <= high[axis]:
                return max_range
            continue
        t0 = (low[axis] - origin[axis]) / direction[axis]
        t1 = (high[axis] - origin[axis]) / direction[axis]
        enter = max(enter, min(t0, t1)); leave = min(leave, max(t0, t1))
        if enter > leave:
            return max_range
    return enter if 0 <= enter <= max_range else max_range


def _static_lidar(scene: LargeMapScene, origin: np.ndarray, max_range: float = 4.0
                  ) -> np.ndarray:
    """Vectorized 144-ray versus all-box slab intersection."""
    azimuth = np.deg2rad(np.repeat(np.arange(0, 360, 10), 4))
    elevation = np.deg2rad(np.tile(np.array([-10, 0, 10, 20]), 36))
    directions = np.column_stack((np.cos(elevation) * np.cos(azimuth),
                                  np.cos(elevation) * np.sin(azimuth),
                                  np.sin(elevation)))
    # Four vertical virtual boundary slabs keep the world-aligned 4 m sensor
    # aware of the 50 m arena edge without adding physical wall prims.
    wall_h = 20.0
    wall_t = .02
    boundary_centers = np.array([[scene.map_half_extent_m, 0, 2.25],
                                 [-scene.map_half_extent_m, 0, 2.25],
                                 [0, scene.map_half_extent_m, 2.25],
                                 [0, -scene.map_half_extent_m, 2.25]])
    boundary_sizes = np.array([[wall_t, wall_h * 3, 4.5], [wall_t, wall_h * 3, 4.5],
                               [wall_h * 3, wall_t, 4.5], [wall_h * 3, wall_t, 4.5]])
    centers = (np.stack([b.center for b in scene.static_boxes])
               if scene.static_boxes else np.empty((0, 3)))
    sizes = (np.stack([b.size for b in scene.static_boxes])
             if scene.static_boxes else np.empty((0, 3)))
    centers = np.concatenate((centers, boundary_centers)); sizes = np.concatenate((sizes, boundary_sizes))
    half = sizes * .5
    low, high = centers - half, centers + half
    d = directions[:, None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        t0 = (low[None] - origin[None, None]) / d
        t1 = (high[None] - origin[None, None]) / d
    parallel = np.abs(d) <= EPS
    outside = (origin[None, None] < low[None]) | (origin[None, None] > high[None])
    invalid_parallel = np.any(parallel & outside, axis=-1)
    t_enter = np.max(np.where(parallel, -np.inf, np.minimum(t0, t1)), axis=-1)
    t_leave = np.min(np.where(parallel, np.inf, np.maximum(t0, t1)), axis=-1)
    hit = (~invalid_parallel) & (t_leave >= np.maximum(t_enter, 0.0)) & (t_enter <= max_range)
    distances=np.where(hit,np.maximum(t_enter,0.0),max_range)
    cylinder_slots=[i for i,b in enumerate(scene.static_boxes) if b.kind=="cylinder"]
    if cylinder_slots:
        distances[:,cylinder_slots]=_ray_cylinder_distances(origin,directions,centers[cylinder_slots],half[cylinder_slots],max_range)
    distance = np.min(distances, axis=1)
    return (2.0 * (1.0 - distance / max_range)).reshape(36, 4)


def actor_observation(scene: LargeMapScene, position: Sequence[float],
                      velocity: Sequence[float], quaternion_wxyz: Sequence[float],
                      gyro_body: Sequence[float], yaw_hold_rad: float,
                      goal: Sequence[float], *, global_time_s: float,
                      progress_fraction: float,
                      local_goal: Sequence[float] | None = None) -> dict[str, np.ndarray]:
    """Frozen state16/static36x4/dynamic4x8 Actor layout."""
    pos = _finite_vec(position, 3, "position"); vel = _finite_vec(velocity, 3, "velocity")
    quat = _finite_vec(quaternion_wxyz, 4, "quaternion"); gyro = _finite_vec(gyro_body, 3, "gyro")
    goal = _finite_vec(goal, 3, "goal")
    if not math.isfinite(yaw_hold_rad) or not math.isfinite(progress_fraction):
        raise ValueError("actor scalars must be finite")
    # The large task uses a 5 m path lookahead, then clips it to the policy's
    # original 3 m goal-position domain before normalization.
    delta = goal - pos
    selected_goal = (pos + _unit(delta) * min(5.0, float(np.linalg.norm(delta)))
                     if local_goal is None else _finite_vec(local_goal, 3, "local goal"))
    local_delta = selected_goal - pos
    norm = float(np.linalg.norm(local_delta))
    if norm > 3.0:
        local_delta *= 3.0 / norm
    state = np.concatenate((local_delta / 3.0, quat, vel / 3.0, gyro / 10.0,
                            [math.sin(yaw_hold_rad), math.cos(yaw_hold_rad),
                             np.clip(progress_fraction, 0, 1)]))
    rays = _static_lidar(scene, pos)
    truth = scene.dynamic_state(global_time_s)
    rel = truth.position[:, :2] - pos[:2]
    distances = np.linalg.norm(rel, axis=1)
    nearest = np.argsort(distances, kind="stable")[:4]
    dynamic = np.zeros((4, 8), dtype=np.float64)
    # truth-current is sampled this frame; the Actor's measurement-age field is
    # therefore zero.  Velocity-segment age is separate scene metadata.
    age = 0.0
    for slot, index in enumerate(nearest):
        if distances[index] > 4.0:
            continue
        body = scene.dynamic_bodies[int(index)]
        dynamic[slot] = [rel[index, 0] / 3.0, rel[index, 1] / 3.0,
                         (truth.velocity[index, 0] - vel[0]) / 3.0,
                         (truth.velocity[index, 1] - vel[1]) / 3.0,
                         body.horizontal_radius_m, 1.0, age, 1.0]
    return {"state": state, "static_lidar": rays, "dynamic_obstacle": dynamic,
            "nearest_dynamic_indices": nearest.astype(np.int64)}


@dataclass(frozen=True)
class LargeMapControllerConfig:
    route_mode: str = "interior_astar"
    dt_s: float = .01
    vmax_m_s: float = .8
    axy_max_m_s2: float = .8
    az_max_m_s2: float = .5
    total_accel_max_m_s2: float = 1.0
    policy_residual_max_m_s2: float = .05
    route_inflation_m: float = UAV_RADIUS_M + .12
    route_resolution_m: float = GRID_RESOLUTION_M
    gate_horizon_s: float = .75
    dynamic_hard_horizon_s: float = 2.5
    gate_step_s: float = .05
    dynamic_margin_m: float = .05
    static_gate_margin_m: float = .12
    lookahead_m: float = 5.0
    motor_lag_s: float = .05
    emergency_ttc_trigger_s: float = 1.5
    emergency_latch_s: float = .30
    emergency_horizon_s: float = 1.8
    emergency_step_s: float = .05
    emergency_heading_count: int = 24
    emergency_fast_speed_m_s: float = .8
    emergency_slow_speed_m_s: float = .55
    emergency_xy_gain: float = 2.0
    emergency_z_gain: float = 1.5
    emergency_z_damping: float = 1.4
    outer_reference_m: float = 21.75
    outer_geofence_m: float = 21.25
    outer_corner_position_tolerance_m: float = .10
    outer_corner_speed_tolerance_m_s: float = .10

    def __post_init__(self):
        if self.route_mode not in ("interior_astar", "outer_ring"):
            raise ValueError("route_mode must be interior_astar or outer_ring")
        values = asdict(self); values.pop("route_mode")
        if any(not math.isfinite(float(v)) or float(v) <= 0 for v in values.values()):
            raise ValueError("controller configuration must be finite positive")
        if self.emergency_heading_count < 4:
            raise ValueError("emergency heading count must be at least four")


@dataclass(frozen=True)
class LargeMapCommand:
    acceleration: np.ndarray
    nominal_acceleration: np.ndarray
    policy_residual: np.ndarray
    route: tuple[tuple[float, float], ...]
    local_goal: np.ndarray
    status: str
    failure: bool
    failure_reason: str | None
    c3bf_domain_valid: bool
    c3bf_defined_rows_feasible: bool | None
    c3bf_min_margin: float | None
    geometry_clear: bool
    geometry_min_clearance_m: float
    velocity_switch_reach_uncertainty_m: float
    static_braking_horizon_s: float
    dynamic_hard_horizon_s: float
    brake_tail_terminal_speed_m_s: float | None
    brake_tail_stop_complete: bool
    emergency_active: bool
    emergency_cv_predictive_clear: bool | None
    emergency_cv_min_clearance_m: float | None
    emergency_cv_collision_free_time_s: float | None
    emergency_cv_horizon_s: float | None
    emergency_rows_relaxed: bool
    route_mode: str
    dynamic_rows_considered: int
    dynamic_rows_excluded_reachable_region: int
    known_motion_region_dependency: bool
    outer_geofence_clear: bool | None
    outer_geofence_margin_m: float | None
    terminal_hold_active: bool
    terminal_hold_z_integral_accel_m_s2: float | None
    continuous_acceleration_certificate_supported: bool
    controller_obstacle_scope: str = "all_geometrically_relevant_not_actor_nearest4"
    truth_dependency: str = "current_shared_sim_position_velocity_geometry_global_clock"


class LargeMapController:
    def __init__(self, scene: LargeMapScene,
                 config: LargeMapControllerConfig = LargeMapControllerConfig(),
                 route_cache: dict[int, tuple[np.ndarray, ...]] | None = None):
        self.scene, self.config = scene, config
        self._routes = {} if route_cache is None else route_cache
        self._waypoints: dict[int, int] = {}
        self._previous_acceleration: dict[int, np.ndarray] = {}
        self.failure_count = 0
        self._emergency_until_s: dict[int, float] = {}
        self._emergency_reference: dict[int, np.ndarray] = {}
        self._terminal_z_integral_accel: dict[int, float] = {}
        self._static_centers = (np.stack([b.center for b in scene.static_boxes])
                                if scene.static_boxes else np.empty((0, 3)))
        self._static_half = (np.stack([b.size for b in scene.static_boxes]) * .5
                             if scene.static_boxes else np.empty((0, 3)))
        self._dynamic_cube = np.array([b.kind == "cube" for b in scene.dynamic_bodies], dtype=bool)
        self._dynamic_width = np.array([b.width_m for b in scene.dynamic_bodies])
        self._dynamic_height = np.array([b.height_m for b in scene.dynamic_bodies])
        if config.route_mode == "outer_ring":
            if abs(scene.obstacle_half_extent_m - 20.0) > 1e-9:
                raise ValueError("outer_ring requires the validated +/-20 m motion region")
            max_body_half_width = max(
                (body.width_m * .5 for body in scene.dynamic_bodies), default=0.0)
            required_geofence = (scene.obstacle_half_extent_m
                                  + max_body_half_width + UAV_RADIUS_M)
            if not (required_geofence < config.outer_geofence_m
                    < config.outer_reference_m
                    < scene.map_half_extent_m - UAV_RADIUS_M):
                raise ValueError(
                    "outer_ring requires motion region + body half-width + UAV radius "
                    "< geofence < reference < map boundary - UAV radius")
            if not (config.outer_corner_position_tolerance_m
                    < config.outer_reference_m - config.outer_geofence_m):
                raise ValueError(
                    "outer corner tolerance must be smaller than reference-geofence reserve")
            if any((np.abs(body.initial_position[:2]) > 20.0 + 1e-9).any()
                   or body.width_m * .5 > .5 + 1e-9
                   for body in scene.dynamic_bodies):
                raise ValueError("outer_ring dynamic body exceeds known reachable region")
            if (np.abs(scene.starts[:, :2]) + UAV_RADIUS_M
                    > scene.map_half_extent_m).any() or \
                    (np.abs(scene.goals[:, :2]) + UAV_RADIUS_M
                     > scene.map_half_extent_m).any() or \
                    (scene.starts[:, 2] < UAV_RADIUS_M).any() or \
                    (scene.starts[:, 2] > 4.5 - UAV_RADIUS_M).any() or \
                    (scene.goals[:, 2] < UAV_RADIUS_M).any() or \
                    (scene.goals[:, 2] > 4.5 - UAV_RADIUS_M).any():
                raise ValueError("outer_ring start/goal UAV hull must remain inside the map")
            reference = config.outer_reference_m
            if any(start[1] < reference or goal[1] > -reference
                   or abs(start[0]) >= reference or abs(goal[0]) >= reference
                   for start, goal in zip(scene.starts, scene.goals)):
                raise ValueError("outer_ring only supports top-edge to bottom-edge missions")

    def reset_lane(self, lane: int) -> None:
        self._waypoints[lane] = 1
        self._previous_acceleration.pop(lane, None)
        self._emergency_until_s.pop(lane, None)
        self._emergency_reference.pop(lane, None)
        self._terminal_z_integral_accel.pop(lane, None)

    def jump_reach_uncertainty(self, time_to_switch_s: float) -> float:
        # After a switch, old-CV and any new velocity in the 1.5 m/s ball may
        # separate at up to 3 m/s.  This uses only period/current clock.
        return 2.0 * DYNAMIC_SPEED_MAX_M_S * max(
            self.config.gate_horizon_s - float(time_to_switch_s), 0.0)

    def _route(self, lane: int, position: np.ndarray,
               velocity: np.ndarray) -> tuple[np.ndarray, ...]:
        if lane not in self._routes:
            if self.config.route_mode == "outer_ring":
                start = self.scene.starts[lane]
                goal = self.scene.goals[lane]
                reference = self.config.outer_reference_m
                if (start[1] < reference or goal[1] > -reference
                        or abs(start[0]) >= reference or abs(goal[0]) >= reference):
                    raise ValueError(
                        "outer_ring only supports top-edge to bottom-edge missions")
                side = -1.0 if start[0] < 0 else 1.0
                self._routes[lane] = tuple(np.asarray(point, dtype=np.float64) for point in (
                    start[:2].copy(), np.array([side * reference, reference]),
                    np.array([side * reference, -reference]), goal[:2].copy()))
            else:
                self._routes[lane] = plan_grid_route(
                    position[:2], self.scene.goals[lane, :2], self.scene.static_boxes,
                    resolution_m=self.config.route_resolution_m,
                    map_half_extent_m=self.scene.map_half_extent_m,
                    inflation_m=self.config.route_inflation_m)
            self._waypoints[lane] = 1
        if lane not in self._waypoints:
            self._waypoints[lane] = 1
        route = self._routes[lane]
        if self.config.route_mode == "outer_ring":
            waypoint = self._waypoints[lane]
            if waypoint in (1, 2):
                target = route[waypoint]
                at_corner = (np.linalg.norm(position[:2] - target)
                             <= self.config.outer_corner_position_tolerance_m
                             and np.linalg.norm(velocity[:2])
                             <= self.config.outer_corner_speed_tolerance_m_s)
                both_faces = (abs(position[0]) >= self.config.outer_geofence_m
                              and (position[1] >= self.config.outer_geofence_m
                                   if waypoint == 1 else
                                   position[1] <= -self.config.outer_geofence_m))
                if at_corner and both_faces:
                    self._waypoints[lane] += 1
        else:
            while (route and self._waypoints[lane] < len(route) - 1
                   and np.linalg.norm(position[:2] - route[self._waypoints[lane]]) < .35):
                self._waypoints[lane] += 1
        return route

    def _outer_geofence_margin(self, point: np.ndarray) -> float:
        return float(np.max(np.abs(point[:2])) - self.config.outer_geofence_m)

    def _outer_segment_margin(self, start: np.ndarray, end: np.ndarray) -> float:
        """Conservative continuous margin to the outside of the inner square.

        The safe set max(|x|, |y|) >= g is non-convex.  Endpoint checks alone
        miss diagonal corner cuts, so the 1-Lipschitz endpoint lower bound also
        subtracts half the horizontal segment length.
        """
        return (min(self._outer_geofence_margin(start),
                    self._outer_geofence_margin(end))
                - .5 * float(np.linalg.norm(end[:2] - start[:2])))

    def _bounded(self, value: np.ndarray) -> np.ndarray:
        out = value.copy(); cfg = self.config
        out[2] = np.clip(out[2], -cfg.az_max_m_s2, cfg.az_max_m_s2)
        radius = min(cfg.axy_max_m_s2,
                     math.sqrt(max(cfg.total_accel_max_m_s2 ** 2 - out[2] ** 2, 0)))
        norm = float(np.linalg.norm(out[:2]))
        if norm > radius:
            out[:2] *= radius / norm
        return out

    def _local_goal(self, position: np.ndarray, route: tuple[np.ndarray, ...], lane: int) -> np.ndarray:
        if not route:
            return position.copy()
        # Stay on the currently visible route segment.  Looking five metres
        # through a blind corner would turn A* into a straight-line controller.
        point = route[min(self._waypoints[lane], len(route) - 1)]
        delta = point - position[:2]
        xy = position[:2] + _unit(delta) * min(self.config.lookahead_m,
                                                float(np.linalg.norm(delta)))
        return np.array([xy[0], xy[1], self.scene.goals[lane, 2]])

    @staticmethod
    def _point_box_clearance(point: np.ndarray, box: StaticBox, uav_radius: float) -> float:
        return float(static_signed_clearances(point[None],(box,),uav_radius)[0,0])

    def _box_clearances(self, point: np.ndarray, inflation: float) -> np.ndarray:
        if not len(self._static_centers):
            return np.empty(0)
        return static_signed_clearances(point[None],self.scene.static_boxes,inflation)[0]

    @staticmethod
    def _point_body_clearance(point: np.ndarray, center: np.ndarray,
                              body: DynamicBody, uav_radius: float) -> float:
        dz = abs(float(point[2] - center[2])) - body.height_m * .5
        if body.kind == "cube":
            dxy = np.abs(point[:2] - center[:2]) - body.width_m * .5
            delta = np.array([dxy[0], dxy[1], dz]); outside = np.maximum(delta, 0)
            return float(np.linalg.norm(outside) + min(float(np.max(delta)), 0) - uav_radius)
        radial = float(np.linalg.norm(point[:2] - center[:2])) - body.width_m * .5
        outside = np.maximum([radial, dz], 0)
        return float(np.linalg.norm(outside) + min(max(radial, dz), 0) - uav_radius)

    def _body_clearances(self, point: np.ndarray, centers: np.ndarray,
                         inflation: float | np.ndarray) -> np.ndarray:
        if not len(centers):
            return np.empty(0)
        inflation = np.broadcast_to(np.asarray(inflation, dtype=np.float64), (len(centers),))
        result = np.empty(len(centers)); dz = np.abs(point[2] - centers[:, 2]) - self._dynamic_height * .5
        if self._dynamic_cube.any():
            idx = self._dynamic_cube
            dxy = np.abs(point[None, :2] - centers[:, :2])[idx] - self._dynamic_width[idx, None] * .5
            delta = np.column_stack((dxy, dz[idx])); outside = np.maximum(delta, 0)
            result[idx] = np.linalg.norm(outside, axis=1) + np.minimum(np.max(delta, axis=1), 0) - inflation[idx]
        if (~self._dynamic_cube).any():
            idx = ~self._dynamic_cube
            radial = np.linalg.norm(point[None, :2] - centers[:, :2], axis=1)[idx] - self._dynamic_width[idx] * .5
            delta = np.column_stack((radial, dz[idx])); outside = np.maximum(delta, 0)
            result[idx] = np.linalg.norm(outside, axis=1) + np.minimum(np.max(delta, axis=1), 0) - inflation[idx]
        return result

    def _body_clearances_batch(self, points: np.ndarray, centers: np.ndarray,
                               inflation: float) -> np.ndarray:
        """Exact body SDF for K points against M shared dynamic bodies."""
        if not len(centers):
            return np.empty((len(points), 0))
        result = np.empty((len(points), len(centers)))
        dz = np.abs(points[:, None, 2] - centers[None, :, 2]) \
            - self._dynamic_height[None] * .5
        if self._dynamic_cube.any():
            idx = self._dynamic_cube
            dxy = np.abs(points[:, None, :2] - centers[None, :, :2])[:, idx] \
                - self._dynamic_width[idx][None, :, None] * .5
            delta = np.concatenate((dxy, dz[:, idx, None]), axis=-1)
            result[:, idx] = (np.linalg.norm(np.maximum(delta, 0), axis=-1)
                              + np.minimum(np.max(delta, axis=-1), 0) - inflation)
        if (~self._dynamic_cube).any():
            idx = ~self._dynamic_cube
            radial = np.linalg.norm(points[:, None, :2]
                                    - centers[None, :, :2], axis=-1)[:, idx] \
                - self._dynamic_width[idx][None] * .5
            delta = np.stack((radial, dz[:, idx]), axis=-1)
            result[:, idx] = (np.linalg.norm(np.maximum(delta, 0), axis=-1)
                              + np.minimum(np.max(delta, axis=-1), 0) - inflation)
        return result

    def _geometry_gate(self, position: np.ndarray, velocity: np.ndarray,
                       acceleration: np.ndarray, truth: DynamicState,
                       measured_acceleration: np.ndarray
                       ) -> tuple[bool, float, float, float | None]:
        cfg = self.config
        # .20 s candidate hold + worst hold acceleration + motor-lag settling.
        # This is long enough for the modeled horizontal brake tail to reach
        # <=.05 m/s at the declared envelope; the terminal speed is returned
        # and a non-stopping tail is rejected rather than called verified.
        speed0 = float(np.linalg.norm(velocity[:2]))
        horizon = max(cfg.gate_horizon_s,
                      min(.20 + (speed0 + .20 * cfg.axy_max_m_s2)
                          / cfg.axy_max_m_s2 + .30, 2.5))
        checked_dynamic_horizon = min(cfg.dynamic_hard_horizon_s, horizon)
        uncertainty = (0.0 if cfg.route_mode == "outer_ring" else
            (float(np.max(2 * DYNAMIC_SPEED_MAX_M_S * np.maximum(
                checked_dynamic_horizon - truth.time_to_discontinuity_s, 0.0)))
             if len(truth.time_to_discontinuity_s) else 0.0))
        p = position.copy(); v = velocity.copy(); actual = measured_acceleration.copy()
        minimum = math.inf; previous_t = 0.0; previous_p = p.copy()
        previous_dynamic = truth.position.copy()
        previous_static_clear = self._box_clearances(
            p, UAV_RADIUS_M + cfg.static_gate_margin_m)
        previous_dynamic_clear = self._body_clearances(
            p, truth.position, UAV_RADIUS_M + cfg.dynamic_margin_m)
        previous_possible = np.zeros(len(self.scene.dynamic_bodies), dtype=np.float64)
        for t in np.append(np.arange(0, horizon, cfg.gate_step_s), horizon):
            step = float(t - previous_t)
            if step > 0:
                alpha = math.exp(-step / cfg.motor_lag_s)
                requested = acceleration
                if previous_t >= .20:
                    requested = np.array([-cfg.axy_max_m_s2 * _unit(v[:2])[0],
                                          -cfg.axy_max_m_s2 * _unit(v[:2])[1],
                                          np.clip(-1.4 * v[2], -cfg.az_max_m_s2,
                                                  cfg.az_max_m_s2)])
                actual = alpha * actual + (1 - alpha) * requested
                next_v = v + actual * step
                p = p + .5 * (v + next_v) * step; v = next_v
            if (np.any(np.abs(p[:2]) + UAV_RADIUS_M > self.scene.map_half_extent_m)
                    or not UAV_RADIUS_M <= p[2] <= 4.5 - UAV_RADIUS_M):
                return False, min(minimum, -1.0), uncertainty, None
            if cfg.route_mode == "outer_ring":
                swept_outer_margin = self._outer_segment_margin(previous_p, p)
                minimum = min(minimum, swept_outer_margin)
                if swept_outer_margin < 0.0:
                    return False, minimum, uncertainty, None
            movement = float(np.linalg.norm(p - previous_p))
            current_static_clear = self._box_clearances(
                p, UAV_RADIUS_M + cfg.static_gate_margin_m)
            if len(current_static_clear):
                swept_static = np.minimum(previous_static_clear, current_static_clear) - .5 * movement
                minimum = min(minimum, float(swept_static.min()))
                if (swept_static < 0).any():
                    return False, minimum, uncertainty, None
            # Current-velocity propagation plus a reach envelope after a possible
            # unknown switch.  All 120 are collision checked; range filtering only
            # skips objects whose reachable balls cannot intersect the UAV horizon.
            current_dynamic = truth.position + truth.velocity * t
            possible = np.maximum(0.0, t - truth.time_to_discontinuity_s) \
                * 2 * DYNAMIC_SPEED_MAX_M_S
            if cfg.route_mode == "outer_ring":
                # Scene validation constrains every dynamic center to the
                # reflected +/-20 m box and every horizontal half-extent to
                # <=.5 m.  With UAV radius .1, an ego swept center outside
                # 21.25 m cannot intersect their complete reachable region.
                current_dynamic_clear = previous_dynamic_clear
            elif t <= checked_dynamic_horizon + 1e-9:
                current_dynamic_clear = self._body_clearances(
                    p, current_dynamic, UAV_RADIUS_M + cfg.dynamic_margin_m + possible)
                if len(current_dynamic_clear):
                    relative_movement = np.linalg.norm(
                        (p[None] - current_dynamic) - (previous_p[None] - previous_dynamic), axis=1)
                    # Re-evaluate the previous endpoint with this segment's larger
                    # reach radius; radius growth is part of the swept Lipschitz bound.
                    radius_growth = possible - previous_possible
                    swept_dynamic = np.minimum(previous_dynamic_clear - radius_growth,
                                                current_dynamic_clear) - .5 * relative_movement
                    minimum = min(minimum, float(swept_dynamic.min()))
                    if (swept_dynamic < 0).any():
                        return False, minimum, uncertainty, None
            else:
                current_dynamic_clear = previous_dynamic_clear
            previous_t = float(t); previous_p = p.copy(); previous_dynamic = current_dynamic
            previous_static_clear = current_static_clear
            previous_dynamic_clear = current_dynamic_clear
            previous_possible = possible
        terminal_speed = float(np.linalg.norm(v[:2]))
        return terminal_speed <= .05, minimum, uncertainty, terminal_speed

    def _current_cv_ttc(self, position: np.ndarray, velocity: np.ndarray,
                        truth: DynamicState, horizon_s: float = 1.5) -> float:
        """Earliest sampled collision under current velocities only."""
        if self.config.route_mode == "outer_ring" or not self.scene.dynamic_bodies:
            return math.inf
        for t in np.arange(0.0, horizon_s + 1e-9, .05):
            drone = position + velocity * t
            centers = truth.position + truth.velocity * t
            if (self._body_clearances(drone, centers, UAV_RADIUS_M
                                      + self.config.dynamic_margin_m) <= 0).any():
                return float(t)
        return math.inf

    def _emergency_cv_candidate(self, position: np.ndarray, velocity: np.ndarray,
                                measured: np.ndarray, goal: np.ndarray,
                                forward: np.ndarray, truth: DynamicState,
                                rows: VariableRadiusRows,
                                preferred_vref: np.ndarray | None = None):
        """Choose an active escape; every returned choice remains unverified."""
        cfg = self.config
        horizon = cfg.emergency_horizon_s; dt = cfg.emergency_step_s
        candidates, references, relaxed_flags = [], [], []
        if preferred_vref is None:
            references_to_try = [np.array([math.cos(angle) * target_speed,
                                           math.sin(angle) * target_speed, goal[2]])
                                 for angle in np.linspace(
                                     0.0, 2 * math.pi, cfg.emergency_heading_count,
                                     endpoint=False)
                                 for target_speed in (cfg.emergency_fast_speed_m_s,
                                                      cfg.emergency_slow_speed_m_s)]
        else:
            references_to_try = [_finite_vec(preferred_vref, 3, "preferred emergency vref")]
        for vref in references_to_try:
            raw = self._bounded(np.array([
                cfg.emergency_xy_gain * (vref[0] - velocity[0]),
                cfg.emergency_xy_gain * (vref[1] - velocity[1]),
                cfg.emergency_z_gain * (goal[2] - position[2])
                - cfg.emergency_z_damping * velocity[2]]))
            candidate = raw; rows_relaxed = False
            if len(rows.q):
                solution = solve_disk_halfplanes(
                    raw, rows.q, rows.b, axy_max=self.config.axy_max_m_s2,
                    az_max=self.config.az_max_m_s2,
                    total_max=self.config.total_accel_max_m_s2)
                if solution.feasible:
                    candidate = solution.acceleration
                else:
                    rows_relaxed = True
            candidates.append(candidate); references.append(vref)
            relaxed_flags.append(rows_relaxed)
        if not candidates:
            return None
        candidate_array = np.asarray(candidates); reference_array = np.asarray(references)
        relaxed_flags = np.asarray(relaxed_flags, dtype=bool); count = len(candidate_array)
        p = np.repeat(position[None], count, axis=0)
        v = np.repeat(velocity[None], count, axis=0)
        actual = np.repeat(measured[None], count, axis=0)
        previous_p = p.copy(); previous_centers = truth.position.copy()

        def static_clear(points):
            if not len(self._static_centers): return np.empty((len(points), 0))
            return static_signed_clearances(points,self.scene.static_boxes,
                                           UAV_RADIUS_M+cfg.static_gate_margin_m)

        previous_static = static_clear(p)
        dynamic_centers = (truth.position if cfg.route_mode != "outer_ring"
                           else np.empty((0, 3)))
        dynamic_velocity = (truth.velocity if cfg.route_mode != "outer_ring"
                            else np.empty((0, 3)))
        previous_dynamic = self._body_clearances_batch(
            p, dynamic_centers, UAV_RADIUS_M + self.config.dynamic_margin_m)
        min_dynamic = (previous_dynamic.min(axis=1) if previous_dynamic.shape[1]
                       else np.full(count, math.inf))
        cv_clear = min_dynamic > 0; collision_time = np.full(count, horizon)
        static_valid = np.ones(count, dtype=bool)
        previous_outer_margin = (np.max(np.abs(p[:, :2]), axis=1)
                                 - cfg.outer_geofence_m)
        alpha = math.exp(-dt / self.config.motor_lag_s)
        for step_index, t in enumerate(np.arange(dt, horizon + 1e-9, dt), start=1):
            feedback = np.empty_like(candidate_array)
            feedback[:, :2] = cfg.emergency_xy_gain * (
                reference_array[:, :2] - v[:, :2])
            feedback[:, 2] = (cfg.emergency_z_gain * (reference_array[:, 2] - p[:, 2])
                              - cfg.emergency_z_damping * v[:, 2])
            feedback[:, 2] = np.clip(feedback[:, 2], -self.config.az_max_m_s2,
                                      self.config.az_max_m_s2)
            xy_norm = np.linalg.norm(feedback[:, :2], axis=1)
            xy_radius = np.minimum(self.config.axy_max_m_s2,
                np.sqrt(np.maximum(self.config.total_accel_max_m_s2 ** 2
                                   - feedback[:, 2] ** 2, 0)))
            scale = np.minimum(1.0, xy_radius / np.maximum(xy_norm, EPS))
            feedback[:, :2] *= scale[:, None]
            requested = candidate_array if step_index == 1 else feedback
            actual = alpha * actual + (1.0 - alpha) * requested
            next_v = v + actual * dt
            p = p + .5 * (v + next_v) * dt; v = next_v
            boundary = ((np.abs(p[:, :2]) + UAV_RADIUS_M
                         <= self.scene.map_half_extent_m).all(axis=1)
                        & (p[:, 2] >= UAV_RADIUS_M)
                        & (p[:, 2] <= 4.5 - UAV_RADIUS_M))
            if cfg.route_mode == "outer_ring":
                current_outer_margin = (np.max(np.abs(p[:, :2]), axis=1)
                                        - cfg.outer_geofence_m)
                swept_outer_margin = (np.minimum(previous_outer_margin,
                                                 current_outer_margin)
                                      - .5 * np.linalg.norm(
                                          p[:, :2] - previous_p[:, :2], axis=1))
                boundary &= swept_outer_margin >= 0.0
                previous_outer_margin = current_outer_margin
            movement = np.linalg.norm(p - previous_p, axis=1)
            current_static = static_clear(p)
            if current_static.shape[1]:
                swept_static = np.minimum(previous_static, current_static) \
                    - .5 * movement[:, None]
                boundary &= (swept_static >= 0).all(axis=1)
            static_valid &= boundary
            centers = dynamic_centers + dynamic_velocity * t
            current_dynamic = self._body_clearances_batch(
                p, centers, UAV_RADIUS_M + self.config.dynamic_margin_m)
            if current_dynamic.shape[1]:
                relative_move = np.linalg.norm(
                    (p[:, None] - centers[None])
                    - (previous_p[:, None] - previous_centers[None]), axis=-1)
                swept_dynamic = np.minimum(previous_dynamic, current_dynamic) \
                    - .5 * relative_move
                step_min = swept_dynamic.min(axis=1)
                min_dynamic = np.minimum(min_dynamic, step_min)
                first_hit = cv_clear & (step_min <= 0)
                collision_time[first_hit] = float(t); cv_clear[first_hit] = False
            previous_p = p.copy(); previous_centers = centers
            previous_static = current_static; previous_dynamic = current_dynamic

        strict, relaxed = [], []
        for index in np.flatnonzero(static_valid):
            progress = float(forward @ reference_array[index, :2])
            escape = float(np.linalg.norm(reference_array[index, :2]))
            score = (int(cv_clear[index]), float(collision_time[index]),
                     .25 * progress + .05 * escape, float(min_dynamic[index]))
            record = (score, candidate_array[index], bool(cv_clear[index]),
                      float(min_dynamic[index]), float(collision_time[index]),
                      bool(relaxed_flags[index]), reference_array[index].copy())
            (relaxed if relaxed_flags[index] else strict).append(record)
        strict_safe = [value for value in strict if value[2]]
        relaxed_safe = [value for value in relaxed if value[2]]
        pool = strict_safe or relaxed_safe or strict or relaxed
        if not pool:
            return None
        return max(pool, key=lambda value: value[0])

    def command(self, lane: int, position: Sequence[float], velocity: Sequence[float],
                measured_acceleration: Sequence[float], policy_acceleration: Sequence[float],
                global_time_s: float, *, terminal_hold: bool = False) -> LargeMapCommand:
        if not 0 <= int(lane) < self.scene.num_lanes:
            raise ValueError("lane index out of range")
        position = _finite_vec(position, 3, "position"); velocity = _finite_vec(velocity, 3, "velocity")
        measured = _finite_vec(measured_acceleration, 3, "measured acceleration")
        policy = _finite_vec(policy_acceleration, 3, "policy acceleration")
        truth = self.scene.dynamic_state(global_time_s)
        route = self._route(int(lane), position, velocity)
        local_goal = self._local_goal(position, route, int(lane))
        delta = local_goal - position; distance_goal = float(np.linalg.norm(self.scene.goals[lane] - position))
        waypoint_distance = float(np.linalg.norm(delta[:2]))
        speed = min(self.config.vmax_m_s, 2.0 * distance_goal,
                    2.0 * waypoint_distance,
                    math.sqrt(max(2 * self.config.axy_max_m_s2 * distance_goal, 0)),
                    math.sqrt(max(2 * self.config.axy_max_m_s2 * waypoint_distance, 0)))
        if route and self._waypoints[int(lane)] < len(route) - 1:
            actual_corner_distance = float(np.linalg.norm(
                route[self._waypoints[int(lane)]] - position[:2]))
            corner_speed = (.10 if self.config.route_mode == "outer_ring" else .40)
            turn_bound = math.sqrt(corner_speed ** 2 + 2 * self.config.axy_max_m_s2
                                   * max(actual_corner_distance - .35, 0.0))
            speed = min(speed, turn_bound)
        desired_velocity = _unit(delta) * speed
        lane_index = int(lane)
        terminal_integral_candidate = self._terminal_z_integral_accel.get(
            lane_index, 0.0)
        if terminal_hold:
            z_error = float(self.scene.goals[lane, 2] - position[2])
            terminal_integral_candidate = float(np.clip(
                terminal_integral_candidate + .8 * z_error * self.config.dt_s,
                -.2, .2))
            goal_error = self.scene.goals[lane] - position
            base = np.array([1.5 * goal_error[0] - 2.0 * velocity[0],
                             1.5 * goal_error[1] - 2.0 * velocity[1],
                             1.5 * goal_error[2] - 1.4 * velocity[2]
                             + terminal_integral_candidate])
            residual = np.zeros(3)
        else:
            base = np.array([2.0 * (desired_velocity[0] - velocity[0]),
                             2.0 * (desired_velocity[1] - velocity[1]),
                             1.5 * (self.scene.goals[lane, 2] - position[2])
                             - 1.4 * velocity[2]])
            residual = policy.copy(); norm = float(np.linalg.norm(residual))
            if norm > self.config.policy_residual_max_m_s2:
                residual *= self.config.policy_residual_max_m_s2 / norm
            residual *= min(distance_goal / .5, waypoint_distance / .5, 1.0)
        nominal = self._bounded(base + residual)
        # Rows use every body whose vertical span overlaps the UAV and which is
        # close enough to influence the gate.  Radius varies per actual class.
        vertical = np.array([abs(position[2] - p[2]) <= b.height_m * .5 + UAV_RADIUS_M
                             + DYNAMIC_SPEED_MAX_M_S * self.config.gate_horizon_s
                             for p, b in zip(truth.position, self.scene.dynamic_bodies)],
                            dtype=bool)
        reach = np.linalg.norm(truth.position[:, :2] - position[:2], axis=1) <= 8.0
        indices = (np.empty(0, dtype=np.int64) if self.config.route_mode == "outer_ring"
                   else np.flatnonzero(vertical & reach))
        excluded_reachable_region = (len(self.scene.dynamic_bodies)
                                     if self.config.route_mode == "outer_ring" else 0)
        bodies = [self.scene.dynamic_bodies[int(i)] for i in indices]
        rows = build_variable_radius_rows(position[:2], velocity[:2],
                                          truth.position[indices, :2], truth.velocity[indices, :2],
                                          bodies, uav_radius_m=UAV_RADIUS_M,
                                          extra_margin_m=self.config.dynamic_margin_m)
        proposed = nominal.copy(); feasible = None; margin = None

        def project_and_gate(candidate):
            local_feasible, local_margin = None, None
            if len(rows.q):
                solution = solve_disk_halfplanes(candidate, rows.q, rows.b,
                                                 axy_max=self.config.axy_max_m_s2,
                                                 az_max=self.config.az_max_m_s2,
                                                 total_max=self.config.total_accel_max_m_s2)
                local_feasible = solution.feasible
                if not local_feasible:
                    return candidate, False, local_margin, False, -math.inf, 0.0, None
                candidate = solution.acceleration; local_margin = solution.min_margin
            local_clear, local_clearance, local_uncertainty, local_terminal_speed = self._geometry_gate(
                position, velocity, candidate, truth, measured)
            return (candidate, local_feasible, local_margin, local_clear,
                    local_clearance, local_uncertainty, local_terminal_speed)

        proposed, feasible, margin, clear, clearance, uncertainty, terminal_speed = project_and_gate(proposed)
        # A first normal-gate failure goes directly to the explicitly
        # unverified receding-horizon escape selector below.  Re-running dozens
        # of robust stop envelopes with different headings was both expensive
        # and ineffective when the same velocity-jump reach set blocked all of
        # them.
        robust_clear = bool(clear and feasible is not False)
        emergency_active = False; emergency_cv_clear = None
        emergency_cv_clearance = None; emergency_cv_time = None
        emergency_rows_relaxed = False
        ttc = self._current_cv_ttc(
            position, velocity, truth, self.config.emergency_ttc_trigger_s)
        emergency_expiry = self._emergency_until_s.get(lane_index, -math.inf)
        emergency_latched = global_time_s < emergency_expiry
        emergency_hazard = (not robust_clear) or ttc < self.config.emergency_ttc_trigger_s
        emergency_trigger = emergency_hazard or emergency_latched
        failure = False; reason = None
        if not route:
            failure, reason = True, "no_static_route"
        if emergency_trigger and route:
            forward = _unit(delta[:2])
            preferred = (self._emergency_reference.get(lane_index)
                         if emergency_latched else None)
            emergency = self._emergency_cv_candidate(
                position, velocity, measured, self.scene.goals[lane], forward, truth, rows,
                preferred_vref=preferred)
            # A latched direction is kept for command continuity only while it
            # remains current-CV safe.  Static/boundary rejection returns None;
            # dynamic rejection returns a false predictive flag.  Either case
            # triggers a fresh all-heading escape search without extending the
            # original latch.
            if preferred is not None and (emergency is None or not emergency[2]):
                emergency = self._emergency_cv_candidate(
                    position, velocity, measured, self.scene.goals[lane], forward,
                    truth, rows, preferred_vref=None)
            if emergency is not None:
                _, proposed, emergency_cv_clear, emergency_cv_clearance, \
                    emergency_cv_time, emergency_rows_relaxed, emergency_vref = emergency
                emergency_active = True; failure = True
                reason = ("emergency_cv_rows_relaxed_unverified" if emergency_rows_relaxed
                          else "emergency_cv_unverified")
                # Start a fixed-duration latch once.  A persistent hazard may
                # enter a new latch after expiry, but an active latch never
                # renews itself on every frame.
                if not emergency_latched:
                    self._emergency_until_s[lane_index] = (
                        global_time_s + self.config.emergency_latch_s)
                self._emergency_reference[lane_index] = emergency_vref.copy()
                if len(rows.q):
                    residuals = rows.b.reshape(-1) - rows.q @ proposed[:2]
                    margin = float(residuals.min())
                    feasible = bool((residuals >= -1e-7).all())
                else:
                    feasible = None
                # The robust stopping envelope is re-evaluated for the command
                # that will actually execute.  Its result is separate from the
                # current-velocity MPC prediction and may legitimately fail.
                clear, clearance, uncertainty, terminal_speed = self._geometry_gate(
                    position, velocity, proposed, truth, measured)
        if not emergency_trigger:
            self._emergency_until_s.pop(lane_index, None)
            self._emergency_reference.pop(lane_index, None)
        if not emergency_active and ((feasible is False) or not clear):
            brake = self._bounded(np.array([-2 * velocity[0], -2 * velocity[1],
                                            1.5 * (self.scene.goals[lane, 2] - position[2])
                                            - 1.4 * velocity[2]]))
            raw_brake = brake.copy()
            raw_clear, raw_clearance, raw_uncertainty, raw_terminal_speed = self._geometry_gate(
                position, velocity, raw_brake, truth, measured)
            if len(rows.q):
                solution = solve_disk_halfplanes(brake, rows.q, rows.b,
                                                 axy_max=self.config.axy_max_m_s2,
                                                 az_max=self.config.az_max_m_s2,
                                                 total_max=self.config.total_accel_max_m_s2)
                if solution.feasible:
                    brake = solution.acceleration; feasible = True; margin = solution.min_margin
            clear, clearance, uncertainty, terminal_speed = self._geometry_gate(
                position, velocity, brake, truth, measured)
            # If no command satisfies both constraint sets, execute the bounded
            # physical brake with the better swept geometry.  It remains an
            # explicit unsolved failure; this choice only avoids steering a
            # failed C3BF projection into a static obstacle.
            if raw_clearance > clearance:
                brake, clear, clearance, uncertainty, terminal_speed = (
                    raw_brake, raw_clear, raw_clearance, raw_uncertainty,
                    raw_terminal_speed)
                if len(rows.q):
                    feasible = bool(np.all(rows.q @ brake[:2] <= rows.b.reshape(-1) + 1e-7))
            proposed = brake; failure = True
            reason = reason or ("no_gated_brake" if not clear else "active_brake")
        # Telemetry is always evaluated on the command that actually leaves
        # this controller.  This also covers the raw-brake branch, where a
        # rejected projected command may have carried a stale positive margin.
        if len(rows.q):
            final_residuals = rows.b.reshape(-1) - rows.q @ proposed[:2]
            margin = float(final_residuals.min())
            feasible = bool((final_residuals >= -1e-7).all())
        else:
            margin = None; feasible = None
        if failure:
            self.failure_count += 1
        elif terminal_hold:
            # Commit only an actually gated terminal command.  Invalid input,
            # infeasible rows, or geometry failure never winds up the integral.
            self._terminal_z_integral_accel[lane_index] = terminal_integral_candidate
        self._previous_acceleration[int(lane)] = proposed.copy()
        return LargeMapCommand(
            acceleration=proposed, nominal_acceleration=nominal, policy_residual=residual,
            route=tuple(tuple(float(v) for v in p) for p in route), local_goal=local_goal,
            status=("terminal_hold" if terminal_hold and not failure else
                    ("emergency_cv_unverified" if emergency_active else
                    ("active_brake" if failure and clear and feasible is not False
                     else ("unsolved" if failure else "filtered")))),
            failure=failure, failure_reason=reason, c3bf_domain_valid=rows.domain_valid,
            c3bf_defined_rows_feasible=feasible, c3bf_min_margin=margin,
            geometry_clear=bool(clear),
            geometry_min_clearance_m=float(clearance),
            velocity_switch_reach_uncertainty_m=uncertainty,
            static_braking_horizon_s=max(self.config.gate_horizon_s,
                min(.20 + (float(np.linalg.norm(velocity[:2]))
                    + .20 * self.config.axy_max_m_s2) / self.config.axy_max_m_s2
                    + .30, 2.5)),
            dynamic_hard_horizon_s=min(self.config.dynamic_hard_horizon_s,
                max(self.config.gate_horizon_s,
                    min(.20 + (float(np.linalg.norm(velocity[:2]))
                        + .20 * self.config.axy_max_m_s2) / self.config.axy_max_m_s2
                        + .30, 2.5))),
            brake_tail_terminal_speed_m_s=terminal_speed,
            brake_tail_stop_complete=(bool(clear) and terminal_speed is not None
                                      and terminal_speed <= .05
                                      and not emergency_active),
            emergency_active=emergency_active,
            emergency_cv_predictive_clear=emergency_cv_clear,
            emergency_cv_min_clearance_m=emergency_cv_clearance,
            emergency_cv_collision_free_time_s=emergency_cv_time,
            emergency_cv_horizon_s=(self.config.emergency_horizon_s
                                    if emergency_active else None),
            emergency_rows_relaxed=emergency_rows_relaxed,
            route_mode=self.config.route_mode,
            dynamic_rows_considered=len(indices),
            dynamic_rows_excluded_reachable_region=excluded_reachable_region,
            known_motion_region_dependency=(self.config.route_mode == "outer_ring"),
            outer_geofence_clear=(self._outer_geofence_margin(position) >= 0
                                  if self.config.route_mode == "outer_ring" else None),
            outer_geofence_margin_m=(self._outer_geofence_margin(position)
                                      if self.config.route_mode == "outer_ring" else None),
            terminal_hold_active=bool(terminal_hold),
            terminal_hold_z_integral_accel_m_s2=(
                float(self._terminal_z_integral_accel.get(lane_index, 0.0))
                if terminal_hold and not failure else None),
            continuous_acceleration_certificate_supported=False,
            controller_obstacle_scope=(
                "all_reachable;dynamic120_excluded_by_known_region_outer_ring"
                if self.config.route_mode == "outer_ring" else
                "all_geometrically_relevant_not_actor_nearest4"))


def collision_flags(scene: LargeMapScene, positions: np.ndarray,
                    global_time_s: float) -> tuple[np.ndarray, np.ndarray]:
    """All-geometry collision check for [lanes,3] positions."""
    positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    if not np.isfinite(positions).all():
        raise ValueError("collision positions must be finite")
    static = np.zeros(len(positions), dtype=bool)
    dynamic = np.zeros(len(positions), dtype=bool)
    if scene.static_boxes:
        static = (static_signed_clearances(positions,scene.static_boxes,UAV_RADIUS_M)<=0).any(axis=1)
    truth = scene.dynamic_state(global_time_s)
    if scene.dynamic_bodies:
        widths = np.array([b.width_m for b in scene.dynamic_bodies])
        heights = np.array([b.height_m for b in scene.dynamic_bodies])
        cubes = np.array([b.kind == "cube" for b in scene.dynamic_bodies])
        dz = np.abs(positions[:, None, 2] - truth.position[None, :, 2]) - heights[None] * .5
        clearance = np.empty((len(positions), len(scene.dynamic_bodies)))
        if cubes.any():
            dxy = np.abs(positions[:, None, :2] - truth.position[None, :, :2])[:, cubes] \
                - widths[cubes][None, :, None] * .5
            delta = np.concatenate((dxy, dz[:, cubes, None]), axis=-1)
            clearance[:, cubes] = (np.linalg.norm(np.maximum(delta, 0.0), axis=-1)
                                    + np.minimum(np.max(delta, axis=-1), 0.0))
        if (~cubes).any():
            radial = np.linalg.norm(positions[:, None, :2]
                                    - truth.position[None, :, :2], axis=-1)[:, ~cubes] \
                - widths[~cubes][None] * .5
            delta = np.stack((radial, dz[:, ~cubes]), axis=-1)
            clearance[:, ~cubes] = (np.linalg.norm(np.maximum(delta, 0.0), axis=-1)
                                     + np.minimum(np.max(delta, axis=-1), 0.0))
        dynamic = (clearance <= UAV_RADIUS_M).any(axis=1)
    return static, dynamic


__all__ = [
    "DYNAMIC_PERIOD_S", "GRID_RESOLUTION_M", "LargeMapCommand", "LargeMapController",
    "LargeMapControllerConfig", "LargeMapScene", "MISSION_DEADLINE_S", "StaticBox",
    "DynamicBody", "DynamicState", "UAV_RADIUS_M", "VariableRadiusRows",
    "actor_observation", "build_variable_radius_rows", "collision_flags", "generate_scene",
    "load_scene", "plan_grid_route", "save_scene",
]

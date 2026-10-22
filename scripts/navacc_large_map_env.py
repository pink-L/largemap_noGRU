#!/usr/bin/env python3
"""Native Isaac environment for the shared 50 m NavACC engineering scene.

This module is loaded explicitly by ``run_navacc_large_map.py``.  It adds a new
task class and does not alter registration or behavior of historical tasks.
All 470 obstacle prims are global shared objects; GridCloner creates drones and
target markers only.  Clone collision filtering keeps lanes physically
independent while every lane observes the same moving world.
"""
from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np
import torch
from tensordict import TensorDict
from pxr import Gf, PhysicsSchemaTools, PhysxSchema, UsdGeom, UsdLux

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from navacc_large_map import (  # noqa: E402
    MISSION_DEADLINE_S,
    actor_observation,
    collision_flags,
    dynamic_center_region_status,
    load_scene,
    plan_grid_route,
)
from navacc_large_map_local_ground import (  # noqa: E402
    GROUND_COLOR,
    GROUND_LIGHT_PATH,
    GROUND_MATERIAL_PATH,
    GROUND_PATH,
    GROUND_SIZE_M,
    observe_local_ground_stage,
    validate_local_ground_observation,
)
from omni_drones.envs.single.hover import Hover  # noqa: E402
from omni_drones.envs.single.nav_acc_sim_pilot import NavAccSimPilot  # noqa: E402
from omni_drones.envs.utils import create_obstacle  # noqa: E402
from omni_drones.views import RigidPrimView  # noqa: E402
from omni_drones.utils.torch import quat_rotate_inverse  # noqa: E402
from omni_drones.utils.torchrl.compat import (  # noqa: E402
    CompositeSpec,
    DiscreteTensorSpec,
    UnboundedContinuousTensorSpec,
)


class NavAccNavRLShared(NavAccSimPilot):
    """Reference-inspired 50 m shared-world evaluation task.

    Dynamic motion and analytic event geometry are driven by a single absolute
    clock.  Resetting a lane never resets or duplicates the shared world.
    """

    @staticmethod
    def _validate_frozen_config(cfg) -> None:
        raise RuntimeError("Use the explicit portable LargeMapNoGRU task")

    def _build_actor_routes(self):
        """Build legacy actor routes; subclasses may return None if unused."""
        if self._outer_ring_enabled:
            self._actor_routes = []
            for lane in range(self.large_scene.num_lanes):
                start = self.large_scene.starts[lane, :2]
                goal = self.large_scene.goals[lane, :2]
                side = -1.0 if start[0] < 0 else 1.0
                self._actor_routes.append(tuple(np.asarray(point, dtype=np.float64) for point in (
                    start.copy(), np.array([side * self._outer_reference_m,
                                            self._outer_reference_m]),
                    np.array([side * self._outer_reference_m,
                              -self._outer_reference_m]), goal.copy())))
        else:
            self._actor_routes = [plan_grid_route(
                self.large_scene.starts[lane, :2], self.large_scene.goals[lane, :2],
                self.large_scene.static_boxes) for lane in range(self.large_scene.num_lanes)]
        return self._actor_routes

    @staticmethod
    def _validate_native_scene_protocol(scene, cfg):
        raise RuntimeError("Use the explicit portable LargeMapNoGRU task")

    def __init__(self, cfg, headless):
        self.large_scene = load_scene(Path(str(cfg.task.large_map.scene_path)).expanduser())
        from navacc_goal_curriculum import active as _curriculum_active,capacity_scene
        self._curriculum_capacity_scene=capacity_scene(cfg) if _curriculum_active(cfg) else None
        self._global_time_s = 0.0
        self._large_dynamic_paths: list[str] = []
        super().__init__(cfg, headless)
        self._validate_native_scene_protocol(self.large_scene, cfg)
        expected_static = int(cfg.task.large_map.static_count)
        expected_dynamic = int(cfg.task.large_map.dynamic_count)
        self._dynamic_view = RigidPrimView(
            "/World/navrl_dynamic/body_*", reset_xform_properties=False,
            shape=[len(self._curriculum_capacity_scene.dynamic_bodies) if self._curriculum_capacity_scene is not None else len(self.large_scene.dynamic_bodies)]) if expected_dynamic or self._curriculum_capacity_scene is not None else None
        if self._dynamic_view is not None: self._dynamic_view.initialize()
        if self._curriculum_capacity_scene is not None:
            from navacc_goal_curriculum import ActiveDynamicView
            self._dynamic_view=ActiveDynamicView(self._dynamic_view,expected_dynamic)
            self._curriculum_static_view=RigidPrimView("/World/navrl_static/box_*",reset_xform_properties=False,shape=[len(self._curriculum_capacity_scene.static_boxes)]);self._curriculum_static_view.initialize()
        self._lane_index = torch.arange(self.num_envs, device=self.device) % self.large_scene.num_lanes
        self._world_goals = torch.as_tensor(self.large_scene.goals, device=self.device,
                                            dtype=torch.float32)[self._lane_index].unsqueeze(1)
        self._large_static_latched = torch.zeros(self.num_envs, 1, dtype=torch.bool,
                                                 device=self.device)
        self._large_dynamic_latched = torch.zeros_like(self._large_static_latched)
        self._outer_ring_enabled = str(cfg.task.large_map.get(
            "route_mode", "interior_astar")) == "outer_ring"
        self._outer_geofence_m = float(cfg.task.large_map.get(
            "outer_geofence_m", 21.25))
        self._outer_reference_m = float(cfg.task.large_map.get(
            "outer_reference_m", 21.75))
        self._outer_corner_position_tolerance_m = float(cfg.task.large_map.get(
            "outer_corner_position_tolerance_m", .10))
        self._outer_corner_speed_tolerance_m_s = float(cfg.task.large_map.get(
            "outer_corner_speed_tolerance_m_s", .10))
        self._known_dynamic_center_half_extent_m = float(cfg.task.large_map.get(
            "known_dynamic_center_half_extent_m", 20.0))
        self._large_geofence_latched = torch.zeros_like(self._large_static_latched)
        self._large_geofence_min_margin = torch.full_like(
            self._large_static_latched, float("inf"), dtype=torch.float32)
        self._large_previous_positions = self.drone.get_state(
            env_frame=False)[..., :3].reshape(self.num_envs, 3).clone()
        # This is a global shared-world contract latch.  Partial lane resets
        # intentionally never clear it.
        self._large_motion_region_violation_latched = torch.zeros(
            1, 1, dtype=torch.bool, device=self.device)
        self._large_motion_region_min_margin = torch.full(
            (1, 1), float("inf"), dtype=torch.float32, device=self.device)
        self._large_arrival_dwell = torch.zeros(self.num_envs, 1, dtype=torch.long,
                                                device=self.device)
        self._large_arrived = torch.zeros(self.num_envs, 1, dtype=torch.bool,
                                          device=self.device)
        self._actor_routes = self._build_actor_routes()
        if self._actor_routes is not None and any(
                not route for route in self._actor_routes):
            raise RuntimeError("large scene has a lane without a static route")
        self._actor_waypoint = torch.ones(self.num_envs, dtype=torch.long, device=self.device)
        self._sync_dynamic_prims()
        # Validate the actual shared USD objects and view ordering rather than
        # treating the configured 350+120 counts as runtime evidence.
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        local_ground_contract = self._observe_and_validate_local_ground(stage)
        actual_static = len(stage.GetPrimAtPath("/World/navrl_static").GetChildren()) if stage.GetPrimAtPath("/World/navrl_static").IsValid() else 0
        actual_dynamic = len(stage.GetPrimAtPath("/World/navrl_dynamic").GetChildren()) if stage.GetPrimAtPath("/World/navrl_dynamic").IsValid() else 0
        view_positions, view_rotations = (self._dynamic_view.get_world_poses(clone=True)
            if self._dynamic_view is not None else (torch.empty(0,3,device=self.device),torch.empty(0,4,device=self.device)))
        expected = torch.as_tensor(
            self.large_scene.dynamic_state(self._global_time_s).position,
            device=view_positions.device, dtype=view_positions.dtype)
        if actual_static != (len(self._curriculum_capacity_scene.static_boxes) if self._curriculum_capacity_scene is not None else expected_static) or actual_dynamic != (len(self._curriculum_capacity_scene.dynamic_bodies) if self._curriculum_capacity_scene is not None else expected_dynamic):
            raise RuntimeError(
                f"shared prim count mismatch: static={actual_static}, dynamic={actual_dynamic}")
        if tuple(view_positions.shape) != (expected_dynamic, 3) or tuple(view_rotations.shape) != (expected_dynamic, 4):
            raise RuntimeError(
                f"dynamic view shape mismatch: {view_positions.shape}, {view_rotations.shape}")
        pose_error = float(torch.max(torch.abs(view_positions - expected)).item()) if expected_dynamic else 0.
        if not math.isfinite(pose_error) or pose_error > 1e-5:
            raise RuntimeError(f"dynamic view ordering/pose mismatch: {pose_error}")
        self.native_scene_contract = {
            "expected_static_prims": len(self._curriculum_capacity_scene.static_boxes) if self._curriculum_capacity_scene is not None else expected_static, "actual_static_prims": actual_static,
            "active_static_count":expected_static,"active_dynamic_count":expected_dynamic,"curriculum_capacity_mode":self._curriculum_capacity_scene is not None,
            "expected_dynamic_prims": len(self._curriculum_capacity_scene.dynamic_bodies) if self._curriculum_capacity_scene is not None else expected_dynamic, "actual_dynamic_prims": actual_dynamic,
            "dynamic_view_position_shape": list(view_positions.shape),
            "dynamic_view_rotation_shape": list(view_rotations.shape),
            "initial_truth_pose_max_abs_error_m": pose_error,
            "local_ground": local_ground_contract,
        }

    def _create_ground_plane(self):
        """Create the SDK's local visual Mesh and analytic physical Plane."""
        from omni.isaac.core.materials import PhysicsMaterial
        from omni.isaac.core.prims import GeometryPrim
        import omni.isaac.core.utils.prims as prim_utils
        import omni.usd

        for path in (GROUND_PATH, GROUND_MATERIAL_PATH, GROUND_LIGHT_PATH):
            if prim_utils.is_prim_path_valid(path):
                raise RuntimeError(f"refusing to reuse pre-existing local-ground prim: {path}")

        # Keep the material outside the ground subtree, then bind only the
        # analytic collisionPlane.  The SDK's /geom Mesh remains visual-only.
        material = PhysicsMaterial(
            GROUND_MATERIAL_PATH, static_friction=1.0,
            dynamic_friction=1.0, restitution=0.0)
        physx_material = PhysxSchema.PhysxMaterialAPI.Apply(material.prim)
        physx_material.CreateFrictionCombineModeAttr().Set("multiply")
        physx_material.CreateRestitutionCombineModeAttr().Set("multiply")
        stage = omni.usd.get_context().get_stage()
        PhysicsSchemaTools.addGroundPlane(
            stage, GROUND_PATH, "Z", GROUND_SIZE_M, Gf.Vec3f(0.0, 0.0, 0.0),
            Gf.Vec3f(*GROUND_COLOR))
        plane = GeometryPrim(
            f"{GROUND_PATH}/collisionPlane", disable_stablization=False,
            collision=True)
        plane.apply_physics_material(material)
        light = UsdLux.DistantLight.Define(stage, GROUND_LIGHT_PATH)
        light.CreateIntensityAttr().Set(600.0)
        self._large_local_ground = plane
        return plane

    @staticmethod
    def _observe_and_validate_local_ground(stage):
        return validate_local_ground_observation(observe_local_ground_stage(stage))

    def _design_scene(self):
        global_paths = list(Hover._design_scene(self) or [])
        import omni.isaac.core.utils.prims as prim_utils

        def set_existing_scale(prim, value):
            """Set create_prim's existing scale op without duplicating xformOp."""
            attribute = prim.GetAttribute("xformOp:scale")
            if not attribute.IsValid():
                UsdGeom.Xformable(prim).AddScaleOp().Set(Gf.Vec3f(*value))
            else:
                attribute.Set(Gf.Vec3f(*value))

        prim_utils.define_prim("/World/navrl_static", "Xform")
        prim_utils.define_prim("/World/navrl_dynamic", "Xform")
        if self._curriculum_capacity_scene is not None:
            from navacc_goal_curriculum import parking_positions,parking_geometry_contract
            parking_geometry_contract(self._curriculum_capacity_scene)
            static_parking=parking_positions('static',len(self._curriculum_capacity_scene.static_boxes)).numpy();dynamic_parking=parking_positions('dynamic',len(self._curriculum_capacity_scene.dynamic_bodies)).numpy()
        for index, box in enumerate((self._curriculum_capacity_scene or self.large_scene).static_boxes):
            spawn_center=(static_parking[index] if self._curriculum_capacity_scene is not None and index>=int(self.cfg.task.large_map.static_count) else box.center)
            path = f"/World/navrl_static/box_{index:03d}"
            if box.kind == "cylinder":
                create_obstacle(path,"Cylinder",spawn_center.tolist(),
                                {"radius":box.size[0]*.5,"height":box.size[2],"axis":"Z"})
            else:
                prim = create_obstacle(path, "Cube", spawn_center.tolist(), {"size": 1.0})
                usd_prim = prim if hasattr(prim, "GetAttribute") else prim_utils.get_prim_at_path(path)
                set_existing_scale(usd_prim, box.size.tolist())
            global_paths.append(path)
        for index, body in enumerate((self._curriculum_capacity_scene or self.large_scene).dynamic_bodies):
            spawn_center=(dynamic_parking[index] if self._curriculum_capacity_scene is not None and index>=int(self.cfg.task.large_map.dynamic_count) else body.initial_position)
            path = f"/World/navrl_dynamic/body_{body.body_id:03d}"
            if body.kind == "cylinder":
                create_obstacle(path, "Cylinder", spawn_center.tolist(),
                                {"radius": body.width_m * .5, "height": body.height_m,"axis":"Z"})
            else:
                prim = create_obstacle(path, "Cube", spawn_center.tolist(), {"size": 1.0})
                usd_prim = prim if hasattr(prim, "GetAttribute") else prim_utils.get_prim_at_path(path)
                set_existing_scale(
                    usd_prim, (body.width_m, body.width_m, body.height_m))
            self._large_dynamic_paths.append(path); global_paths.append(path)
        return global_paths

    def _set_specs(self):
        super()._set_specs()
        e, d = self.num_envs, self.device
        self.observation_spec["agents", "observation"] = CompositeSpec({
            "state": UnboundedContinuousTensorSpec((e, 1, 16), device=d),
            "static_lidar": UnboundedContinuousTensorSpec((e, 1, 36, 4), device=d),
            "dynamic_obstacle": UnboundedContinuousTensorSpec((e, 1, 4, 8), device=d),
        }, shape=(e,), device=d)
        scalar = lambda: UnboundedContinuousTensorSpec((e, 1), device=d)
        boolean = lambda: DiscreteTensorSpec(2, (e, 1), dtype=torch.bool, device=d)
        for key in ("large_collision_static", "large_collision_dynamic", "large_arrived",
                    "large_outer_geofence_violation",
                    "large_dynamic_motion_region_violation"):
            self.observation_spec["info", key] = boolean()
        for key in ("large_global_time_s", "large_goal_distance_m",
                    "large_outer_geofence_margin_m",
                    "large_dynamic_motion_region_margin_m"):
            self.observation_spec["info", key] = scalar()
        self.observation_spec["info", "large_nearest_dynamic_indices"] = \
            UnboundedContinuousTensorSpec((e, 4), device=d, dtype=torch.long)
        self.observation_spec["info", "large_dynamic_state"] = \
            UnboundedContinuousTensorSpec((e, 120, 8), device=d)
        self.info = self.observation_spec["info"].zero()
        for key in ("arrival", "collision_static", "collision_dynamic", "timeout",
                    "outer_geofence_violation", "dynamic_motion_region_violation"):
            self.observation_spec["stats", key] = scalar()
        self.stats = self.observation_spec["stats"].zero()

    @property
    def global_time_s(self) -> float:
        return float(self._global_time_s)

    def _sync_dynamic_prims(self):
        truth = self.large_scene.dynamic_state(self._global_time_s)
        positions = torch.as_tensor(np.array(truth.position, copy=True),
                                    dtype=torch.float32, device=self.device)
        rotations = torch.zeros(len(positions), 4, device=self.device)
        rotations[:, 0] = 1.0
        self._dynamic_view.set_world_poses(positions, rotations)

    def _reset_idx(self, env_ids):
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).reshape(-1)
        # Parent reset owns firmware/PID telemetry and keeps the original [1,3]
        # construction target.  We then move only the selected drones to their
        # assigned edge starts without touching the shared obstacle clock.
        super()._reset_idx(ids)
        current_pos, current_rot = self.drone.get_world_poses(clone=True)
        starts = torch.as_tensor(self.large_scene.starts, device=self.device,
                                 dtype=current_pos.dtype)[self._lane_index[ids]].unsqueeze(1)
        self.drone.set_world_poses(starts, current_rot[ids], ids)
        self.drone.set_velocities(torch.zeros_like(self.init_vels[ids]), ids)
        target_rot = torch.zeros(len(ids), 4, device=self.device); target_rot[:, 0] = 1
        self.target_vis.set_world_poses(self._world_goals[ids, 0], target_rot, env_indices=ids)
        if hasattr(self, "_large_static_latched"):
            self._large_static_latched[ids] = False
            self._large_dynamic_latched[ids] = False
            self._large_geofence_latched[ids] = False
            self._large_geofence_min_margin[ids] = float("inf")
            self._large_arrival_dwell[ids] = 0
            self._large_arrived[ids] = False
            self._actor_waypoint[ids] = 1
            self._large_previous_positions[ids] = starts[:, 0]

    def _capture_large_collisions(self):
        state = self.drone.get_state(env_frame=False)
        positions = state[..., :3].reshape(self.num_envs, 3).detach().cpu().numpy()
        static, dynamic = collision_flags(self.large_scene, positions, self._global_time_s)
        self._large_static_latched |= torch.as_tensor(static[:, None], device=self.device)
        self._large_dynamic_latched |= torch.as_tensor(dynamic[:, None], device=self.device)
        actual_dynamic_positions, _ = self._dynamic_view.get_world_poses(clone=True)
        actual_region_margin_value, actual_region_violation = \
            dynamic_center_region_status(
                actual_dynamic_positions.detach().cpu().numpy(),
                self._known_dynamic_center_half_extent_m)
        actual_region_margin = torch.tensor(
            actual_region_margin_value, device=self.device,
            dtype=self._large_motion_region_min_margin.dtype)
        self._large_motion_region_min_margin = torch.minimum(
            self._large_motion_region_min_margin,
            actual_region_margin.reshape(1, 1).to(
                dtype=self._large_motion_region_min_margin.dtype))
        # Allow only a small float32 pose-write tolerance.  A violation breaks
        # the spatial invariant used to exclude all 120 dynamic rows outside.
        self._large_motion_region_violation_latched |= actual_region_violation
        if self._outer_ring_enabled:
            previous_positions = self._large_previous_positions.detach().cpu().numpy()
            endpoint_margin = (np.max(np.abs(previous_positions[:, :2]), axis=1)
                               - self._outer_geofence_m)
            current_margin = (np.max(np.abs(positions[:, :2]), axis=1)
                              - self._outer_geofence_m)
            # Outside of a square is non-convex.  This Lipschitz lower bound
            # catches a segment that cuts through a corner while both endpoints
            # remain outside.
            margin = (np.minimum(endpoint_margin, current_margin)
                      - .5 * np.linalg.norm(
                          positions[:, :2] - previous_positions[:, :2], axis=1))
            margin_t = torch.as_tensor(margin[:, None], device=self.device,
                                       dtype=self._large_geofence_min_margin.dtype)
            self._large_geofence_min_margin = torch.minimum(
                self._large_geofence_min_margin, margin_t)
            self._large_geofence_latched |= margin_t < 0
        self._large_previous_positions.copy_(state[..., :3].reshape(self.num_envs, 3))

    def _step(self, tensordict):
        for substep in range(self.substeps):
            self._global_time_s += float(self.cfg.sim.dt)
            self._sync_dynamic_prims()
            NavAccSimPilot._pre_sim_step(self, tensordict)
            self.sim.step(self._should_render(substep))
            self._capture_large_collisions()
        NavAccSimPilot._post_sim_step(self, tensordict)
        self.progress_buf += 1
        result = TensorDict({}, self.batch_size, device=self.device)
        result.update(self._compute_state_and_obs())
        result.update(self._compute_reward_and_done())
        return result

    def _compute_state_and_obs(self):
        self.drone_state = self.drone.get_state(env_frame=False)
        self.info["drone_state"].copy_(self.drone_state[..., :13])
        position = self.drone_state[..., :3].reshape(self.num_envs, 3)
        quat = self.drone_state[..., 3:7].reshape(self.num_envs, 4)
        velocity = self.drone_state[..., 7:10].reshape(self.num_envs, 3)
        gyro = quat_rotate_inverse(quat, self.drone_state[..., 10:13].reshape(self.num_envs, 3))
        states, lidars, dynamics, nearest = [], [], [], []
        for env_id in range(self.num_envs):
            lane = int(self._lane_index[env_id])
            route = self._actor_routes[lane]
            position_xy = position[env_id, :2].detach().cpu().numpy()
            waypoint = int(self._actor_waypoint[env_id])
            if self._outer_ring_enabled and waypoint in (1, 2):
                horizontal_speed = float(torch.linalg.vector_norm(
                    velocity[env_id, :2]).item())
                both_faces = (abs(position_xy[0]) >= self._outer_geofence_m
                              and (position_xy[1] >= self._outer_geofence_m if waypoint == 1
                                   else position_xy[1] <= -self._outer_geofence_m))
                if (np.linalg.norm(route[waypoint] - position_xy)
                        <= self._outer_corner_position_tolerance_m
                        and horizontal_speed <= self._outer_corner_speed_tolerance_m_s
                        and both_faces):
                    waypoint += 1
            else:
                while (waypoint < len(route) - 1
                       and np.linalg.norm(route[waypoint] - position_xy) <= .35):
                    waypoint += 1
            self._actor_waypoint[env_id] = waypoint
            segment = route[waypoint] - position_xy
            local_xy = position_xy + segment / max(
                float(np.linalg.norm(segment)), 1e-9) * min(float(np.linalg.norm(segment)), 5.0)
            local_goal = np.array([local_xy[0], local_xy[1], self.large_scene.goals[lane, 2]])
            obs = actor_observation(
                self.large_scene, position[env_id].detach().cpu().numpy(),
                velocity[env_id].detach().cpu().numpy(), quat[env_id].detach().cpu().numpy(),
                gyro[env_id].detach().cpu().numpy(),
                float(self.info["yaw_hold_W"][env_id, 0, 0]),
                self._world_goals[env_id, 0].detach().cpu().numpy(),
                global_time_s=self._global_time_s,
                progress_fraction=float(self.progress_buf[env_id] / self.max_episode_length),
                local_goal=local_goal)
            states.append(obs["state"]); lidars.append(obs["static_lidar"])
            dynamics.append(obs["dynamic_obstacle"]); nearest.append(obs["nearest_dynamic_indices"])
        truth = self.large_scene.dynamic_state(self._global_time_s)
        dynamic_state = np.zeros((120, 8), dtype=np.float32)
        dynamic_state[:, :3] = truth.position; dynamic_state[:, 3:6] = truth.velocity
        dynamic_state[:, 6] = [b.horizontal_radius_m for b in self.large_scene.dynamic_bodies]
        dynamic_state[:, 7] = [b.class_id for b in self.large_scene.dynamic_bodies]
        self.info["large_dynamic_state"].copy_(torch.as_tensor(
            dynamic_state, device=self.device).unsqueeze(0).expand(self.num_envs, -1, -1))
        self.info["large_nearest_dynamic_indices"].copy_(torch.as_tensor(
            np.asarray(nearest, dtype=np.int64), device=self.device))
        self.info["large_global_time_s"].fill_(self._global_time_s)
        self.info["large_collision_static"].copy_(self._large_static_latched)
        self.info["large_collision_dynamic"].copy_(self._large_dynamic_latched)
        self.info["large_arrived"].copy_(self._large_arrived)
        self.info["large_outer_geofence_violation"].copy_(self._large_geofence_latched)
        self.info["large_outer_geofence_margin_m"].copy_(self._large_geofence_min_margin)
        self.info["large_dynamic_motion_region_violation"].copy_(
            self._large_motion_region_violation_latched.expand(self.num_envs, 1))
        self.info["large_dynamic_motion_region_margin_m"].copy_(
            self._large_motion_region_min_margin.expand(self.num_envs, 1))
        self.info["large_goal_distance_m"].copy_(torch.linalg.vector_norm(
            self._world_goals[:, 0] - position, dim=-1, keepdim=True))
        return TensorDict({
            "agents": {"observation": {
                "state": torch.as_tensor(np.asarray(states), device=self.device,
                                         dtype=position.dtype).unsqueeze(1),
                "static_lidar": torch.as_tensor(np.asarray(lidars), device=self.device,
                                                dtype=position.dtype).unsqueeze(1),
                "dynamic_obstacle": torch.as_tensor(np.asarray(dynamics), device=self.device,
                                                     dtype=position.dtype).unsqueeze(1),
            }, "intrinsics": self.drone.intrinsics},
            "info": self.info.clone(), "stats": self.stats.clone(),
        }, self.batch_size)

    def _compute_reward_and_done(self):
        position = self.drone_state[..., :3].reshape(self.num_envs, 3)
        velocity = self.drone_state[..., 7:10].reshape(self.num_envs, 3)
        distance = torch.linalg.vector_norm(self._world_goals[:, 0] - position, dim=-1, keepdim=True)
        speed = torch.linalg.vector_norm(velocity, dim=-1, keepdim=True)
        in_goal = (distance <= .2) & (speed <= .2)
        self._large_arrival_dwell = torch.where(in_goal, self._large_arrival_dwell + 1,
                                                torch.zeros_like(self._large_arrival_dwell))
        self._large_arrived |= self._large_arrival_dwell >= 50
        quat = self.drone_state[..., 3:7].reshape(self.num_envs, 4)
        tilt_cos = (1.0 - 2.0 * (quat[:, 1:2].square() + quat[:, 2:3].square())).clamp(-1, 1)
        actual_tilt_crash = torch.acos(tilt_cos) > math.radians(40.0)
        crash = ((position[:, 2:3] < .15) | actual_tilt_crash
                 | ~torch.isfinite(self.drone_state).all(-1))
        oob = ((position[:, :2].abs() > 25.0).any(-1, keepdim=True)
               | (position[:, 2:3] > 4.5) | self._large_geofence_latched
               | self._large_motion_region_violation_latched.expand(self.num_envs, 1))
        collision = self._large_static_latched | self._large_dynamic_latched
        terminated = self._large_arrived | collision | crash | oob
        timeout = (self.progress_buf >= self.max_episode_length).unsqueeze(-1) & ~terminated
        reward = (-.01 * distance - .002 * self.current_a_nom.square().sum(-1)
                  + 20.0 * self._large_arrived.float() - 20.0 * (collision | crash | oob).float())
        self.stats["return"] += reward
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(-1)
        self.stats["pos_error"].copy_(distance)
        self.stats["speed"].copy_(speed)
        self.stats["crash"].copy_(crash.float())
        self.stats["oob"].copy_(oob.float())
        self.stats["arrival"].copy_(self._large_arrived.float())
        self.stats["collision_static"].copy_(self._large_static_latched.float())
        self.stats["collision_dynamic"].copy_(self._large_dynamic_latched.float())
        self.stats["timeout"].copy_(timeout.float())
        self.stats["outer_geofence_violation"].copy_(self._large_geofence_latched.float())
        self.stats["dynamic_motion_region_violation"].copy_(
            self._large_motion_region_violation_latched.float().expand(self.num_envs, 1))
        self.info["large_arrived"].copy_(self._large_arrived)
        reason = torch.zeros_like(self.info["termination_reason"])
        reason = torch.where(self._large_arrived, torch.full_like(reason, 5), reason)
        reason = torch.where(collision, torch.full_like(reason, 6), reason)
        reason = torch.where(crash, torch.ones_like(reason), reason)
        reason = torch.where(oob, torch.full_like(reason, 2), reason)
        reason = torch.where(timeout, torch.full_like(reason, 4), reason)
        self.info["termination_reason"].copy_(reason)
        self.previous_a_nom.copy_(self.current_a_nom); self.previous_a_nom_valid.fill_(True)
        return TensorDict({
            "agents": {"reward": reward.unsqueeze(-1)},
            "done": terminated | timeout, "terminated": terminated,
            "truncated": timeout, "info": self.info.clone(), "stats": self.stats.clone(),
        }, self.batch_size)


__all__ = ["NavAccNavRLShared"]

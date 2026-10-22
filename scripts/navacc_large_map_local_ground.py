#!/usr/bin/env python3
"""Pure contract validation for the native large-map local ground plane."""
from __future__ import annotations

import math
from typing import Any, Mapping


GROUND_PATH = "/World/defaultGroundPlane"
GROUND_MATERIAL_PATH = "/World/navaccLargeGroundMaterial"
GROUND_LIGHT_PATH = "/World/navaccLargeGroundLight"
GROUND_SIZE_M = 60.0
GROUND_COLOR = (0.065, 0.0725, 0.080)


def validate_local_ground_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed on an observed USD ground-plane contract."""
    required = {
        "ground_path", "ground_root_valid", "ground_root_type",
        "collision_prim_path", "collision_prim_type", "axis", "world_z_m",
        "world_normal", "collision_api_applied", "collision_enabled",
        "collision_purpose", "visual_prim_path", "visual_prim_type",
        "visual_points_authored", "visual_world_bounds_min_m",
        "visual_world_bounds_max_m", "visual_face_vertex_counts",
        "visual_face_vertex_indices", "visual_color_rgb",
        "visual_collision_api_applied", "visual_collision_enabled",
        "stage_up_axis", "meters_per_unit", "material_path",
        "static_friction", "dynamic_friction", "restitution",
        "friction_combine_mode", "restitution_combine_mode", "light_path",
        "light_type", "light_intensity",
    }
    missing = sorted(required - set(observation))
    if missing:
        raise ValueError(f"local ground observation missing fields: {missing}")
    value = dict(observation)
    if value["ground_path"] != GROUND_PATH or not bool(value["ground_root_valid"]):
        raise ValueError("local ground root path/validity mismatch")
    if value["ground_root_type"] != "Xform":
        raise ValueError("local ground root must be Xform")
    if value["collision_prim_path"] != f"{GROUND_PATH}/collisionPlane" \
            or value["collision_prim_type"] != "Plane":
        raise ValueError("local ground collision prim must be /collisionPlane Plane")
    if str(value["axis"]).upper() != "Z":
        raise ValueError("local ground plane axis must be Z")
    normal = tuple(float(x) for x in value["world_normal"])
    if len(normal) != 3 or not all(math.isfinite(x) for x in normal):
        raise ValueError("local ground world normal is invalid")
    norm = math.sqrt(sum(x * x for x in normal))
    if norm <= 0 or normal[2] / norm < 1.0 - 1e-6:
        raise ValueError("local ground world normal must point along +Z")
    if value["collision_api_applied"] is not True \
            or value["collision_enabled"] is not True:
        raise ValueError("local ground physical Plane collision must be enabled")
    if str(value["collision_purpose"]) != "guide":
        raise ValueError("local ground physical Plane purpose must be guide")
    if value["visual_prim_path"] != f"{GROUND_PATH}/geom" \
            or value["visual_prim_type"] != "Mesh":
        raise ValueError("local ground visual prim must be /geom Mesh")
    if value["visual_points_authored"] is not True:
        raise ValueError("local ground visual Mesh points must be authored")
    try:
        bounds_min = tuple(float(x) for x in value["visual_world_bounds_min_m"])
        bounds_max = tuple(float(x) for x in value["visual_world_bounds_max_m"])
        color = tuple(float(x) for x in value["visual_color_rgb"])
        face_counts = tuple(int(x) for x in value["visual_face_vertex_counts"])
        face_indices = tuple(int(x) for x in value["visual_face_vertex_indices"])
    except (TypeError, ValueError):
        raise ValueError("local ground visual Mesh observation is invalid") from None
    if (len(bounds_min) != 3 or len(bounds_max) != 3
            or not all(math.isfinite(x) for x in bounds_min + bounds_max)):
        raise ValueError("local ground visual Mesh world bounds are invalid")
    if (bounds_max[0] - bounds_min[0] < GROUND_SIZE_M
            or bounds_max[1] - bounds_min[1] < GROUND_SIZE_M
            or max(abs(bounds_min[2]), abs(bounds_max[2])) > 1e-6):
        raise ValueError("local ground visual Mesh bounds do not cover the z=0 map")
    if face_counts != (4,) or face_indices != (0, 1, 2, 3):
        raise ValueError("local ground visual Mesh must be one authored quad")
    if (len(color) != 3 or not all(math.isfinite(x) for x in color)
            or any(abs(a - b) > 1e-6 for a, b in zip(color, GROUND_COLOR))):
        raise ValueError("local ground visual Mesh color mismatch")
    if (value["visual_collision_api_applied"] is True
            and value["visual_collision_enabled"] is not False):
        raise ValueError("local ground visual Mesh must not be a second collider")
    if str(value["stage_up_axis"]).upper() != "Z" \
            or float(value["meters_per_unit"]) != 1.0:
        raise ValueError("local ground requires a Z-up meter stage")
    numeric = ("world_z_m", "static_friction",
               "dynamic_friction", "restitution", "light_intensity",
               "meters_per_unit")
    if not all(isinstance(value[key], (int, float)) and math.isfinite(float(value[key]))
               for key in numeric):
        raise ValueError("local ground numeric observation is nonfinite")
    if abs(float(value["world_z_m"])) > 1e-6:
        raise ValueError("local ground must be at z=0")
    if value["material_path"] != GROUND_MATERIAL_PATH:
        raise ValueError("local ground material binding mismatch")
    if (float(value["static_friction"]) != 1.0
            or float(value["dynamic_friction"]) != 1.0
            or float(value["restitution"]) != 0.0):
        raise ValueError("local ground friction/restitution mismatch")
    if (str(value["friction_combine_mode"]) != "multiply"
            or str(value["restitution_combine_mode"]) != "multiply"):
        raise ValueError("local ground material combine mode mismatch")
    if (value["light_path"] != GROUND_LIGHT_PATH
            or value["light_type"] != "DistantLight"
            or float(value["light_intensity"]) != 600.0):
        raise ValueError("local ground light mismatch")
    value["schema"] = "navacc_large_local_ground_v2"
    value["provenance"] = (
        "PhysicsSchemaTools local /collisionPlane plus non-colliding /geom Mesh; no remote USD")
    value["requested_size_m"] = GROUND_SIZE_M
    value["requested_color_rgb"] = list(GROUND_COLOR)
    for key in numeric:
        value[key] = float(value[key])
    value["world_normal"] = list(normal)
    value["axis"] = str(value["axis"]).upper()
    value["stage_up_axis"] = str(value["stage_up_axis"]).upper()
    value["collision_enabled"] = bool(value["collision_enabled"])
    value["collision_api_applied"] = bool(value["collision_api_applied"])
    value["visual_collision_api_applied"] = bool(value["visual_collision_api_applied"])
    value["visual_world_bounds_min_m"] = list(bounds_min)
    value["visual_world_bounds_max_m"] = list(bounds_max)
    value["visual_face_vertex_counts"] = list(face_counts)
    value["visual_face_vertex_indices"] = list(face_indices)
    value["visual_color_rgb"] = list(color)
    value["friction_combine_mode"] = str(value["friction_combine_mode"])
    value["restitution_combine_mode"] = str(value["restitution_combine_mode"])
    return value


def observe_local_ground_stage(stage) -> dict[str, Any]:
    """Extract the exact dual-prim SDK ground layout from a USD stage.

    USD imports stay local so pure dictionary validation remains usable without
    Isaac/USD installed.  No Plane width/length fallback is treated as physical
    extent; the authored visual Mesh points provide the finite display bounds.
    """
    from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

    root = stage.GetPrimAtPath(GROUND_PATH)
    collision_path = f"{GROUND_PATH}/collisionPlane"
    collision = stage.GetPrimAtPath(collision_path)
    visual_path = f"{GROUND_PATH}/geom"
    visual = stage.GetPrimAtPath(visual_path)
    collision_world = UsdGeom.Xformable(collision).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()) if collision.IsValid() else Gf.Matrix4d(1.0)
    world_normal = collision_world.TransformDir(Gf.Vec3d(0.0, 0.0, 1.0))
    mesh = UsdGeom.Mesh(visual)
    points_attr = mesh.GetPointsAttr()
    points = points_attr.Get() if visual.IsValid() else None
    visual_world = UsdGeom.Xformable(visual).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()) if visual.IsValid() else Gf.Matrix4d(1.0)
    world_points = ([visual_world.Transform(Gf.Vec3d(*point)) for point in points]
                    if points else [])
    bounds_min = ([min(point[index] for point in world_points) for index in range(3)]
                  if world_points else [float("nan")] * 3)
    bounds_max = ([max(point[index] for point in world_points) for index in range(3)]
                  if world_points else [float("nan")] * 3)
    color_attr = mesh.GetDisplayColorAttr()
    colors = color_attr.Get() if visual.IsValid() else None
    color = list(colors[0]) if colors and len(colors) == 1 else []
    collision_api = UsdPhysics.CollisionAPI(collision)
    visual_collision_applied = bool(
        visual.IsValid() and visual.HasAPI(UsdPhysics.CollisionAPI))
    visual_collision = UsdPhysics.CollisionAPI(visual)
    bound_material, _ = UsdShade.MaterialBindingAPI(
        collision).ComputeBoundMaterial(materialPurpose="physics")
    material_path = (str(bound_material.GetPath())
                     if bound_material and bound_material.GetPrim().IsValid() else None)
    material_prim = stage.GetPrimAtPath(GROUND_MATERIAL_PATH)
    material = UsdPhysics.MaterialAPI(material_prim)
    light_prim = stage.GetPrimAtPath(GROUND_LIGHT_PATH)
    light = UsdLux.DistantLight(light_prim)
    plane = UsdGeom.Plane(collision)
    return {
        "ground_path": GROUND_PATH,
        "ground_root_valid": bool(root.IsValid()),
        "ground_root_type": root.GetTypeName() if root.IsValid() else None,
        "collision_prim_path": collision_path,
        "collision_prim_type": collision.GetTypeName() if collision.IsValid() else None,
        "axis": plane.GetAxisAttr().Get() if collision.IsValid() else None,
        "world_z_m": float(collision_world.ExtractTranslation()[2]),
        "world_normal": [float(x) for x in world_normal],
        "collision_api_applied": bool(
            collision.IsValid() and collision.HasAPI(UsdPhysics.CollisionAPI)),
        "collision_enabled": (collision_api.GetCollisionEnabledAttr().Get()
                              if collision.IsValid() else None),
        "collision_purpose": (UsdGeom.Imageable(collision).GetPurposeAttr().Get()
                              if collision.IsValid() else None),
        "visual_prim_path": visual_path,
        "visual_prim_type": visual.GetTypeName() if visual.IsValid() else None,
        "visual_points_authored": bool(
            visual.IsValid() and points_attr.HasAuthoredValueOpinion()),
        "visual_world_bounds_min_m": bounds_min,
        "visual_world_bounds_max_m": bounds_max,
        "visual_face_vertex_counts": (list(mesh.GetFaceVertexCountsAttr().Get() or [])
                                      if visual.IsValid() else []),
        "visual_face_vertex_indices": (list(mesh.GetFaceVertexIndicesAttr().Get() or [])
                                       if visual.IsValid() else []),
        "visual_color_rgb": color,
        "visual_collision_api_applied": visual_collision_applied,
        "visual_collision_enabled": (visual_collision.GetCollisionEnabledAttr().Get()
                                     if visual_collision_applied else None),
        "stage_up_axis": str(UsdGeom.GetStageUpAxis(stage)),
        "meters_per_unit": float(UsdGeom.GetStageMetersPerUnit(stage)),
        "material_path": material_path,
        "static_friction": material.GetStaticFrictionAttr().Get(),
        "dynamic_friction": material.GetDynamicFrictionAttr().Get(),
        "restitution": material.GetRestitutionAttr().Get(),
        "friction_combine_mode": material_prim.GetAttribute(
            "physxMaterial:frictionCombineMode").Get(),
        "restitution_combine_mode": material_prim.GetAttribute(
            "physxMaterial:restitutionCombineMode").Get(),
        "light_path": GROUND_LIGHT_PATH,
        "light_type": light_prim.GetTypeName() if light_prim.IsValid() else None,
        "light_intensity": light.GetIntensityAttr().Get() if light_prim.IsValid() else None,
    }


__all__ = ["GROUND_PATH", "GROUND_MATERIAL_PATH", "GROUND_LIGHT_PATH",
           "GROUND_SIZE_M", "GROUND_COLOR", "observe_local_ground_stage",
           "validate_local_ground_observation"]

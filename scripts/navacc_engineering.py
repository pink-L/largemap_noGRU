from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import math,numpy as np
EPS=1e-9
def _vec(value: Sequence[float], size: int, name: str) -> np.ndarray:
    out = np.asarray(value, dtype=np.float64).reshape(-1)
    if out.shape != (size,) or not np.isfinite(out).all():
        raise ValueError(f"{name} must be finite shape ({size},)")
    return out

@dataclass(frozen=True)
class HalfplaneSolution:
    acceleration: np.ndarray
    feasible: bool
    status: str
    min_margin: float | None
    candidates_checked: int

def _circle_line_intersections(q: np.ndarray, b: float, radius: float
                               ) -> Iterable[np.ndarray]:
    norm2 = float(q @ q)
    if norm2 <= EPS:
        return ()
    foot = q * (b / norm2)
    remaining = radius * radius - float(foot @ foot)
    if remaining < -1e-10:
        return ()
    tangent = np.array([-q[1], q[0]]) / math.sqrt(norm2)
    offset = math.sqrt(max(remaining, 0.0))
    return (foot + tangent * offset, foot - tangent * offset)

def solve_disk_halfplanes(nominal_acceleration: Sequence[float], q: np.ndarray,
                          b: np.ndarray, *, axy_max: float, az_max: float,
                          total_max: float, tolerance: float = 1.0e-7
                          ) -> HalfplaneSolution:
    """Closest enumerated point in a 2-D disk intersected with all half-planes."""
    nominal = _vec(nominal_acceleration, 3, "nominal acceleration")
    q = np.asarray(q, dtype=np.float64).reshape(-1, 2)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if len(q) != len(b) or not np.isfinite(q).all() or not np.isfinite(b).all():
        raise ValueError("half-plane rows must be finite and aligned")
    z = float(np.clip(nominal[2], -az_max, az_max))
    xy_radius = min(float(axy_max), math.sqrt(max(total_max ** 2 - z ** 2, 0.0)))
    original_xy = nominal[:2].copy()
    xy = original_xy.copy()
    xy_norm = float(np.linalg.norm(xy))
    if xy_norm > xy_radius:
        xy *= xy_radius / xy_norm
    candidates = [xy, np.zeros(2)]
    for row, bound in zip(q, b):
        norm2 = float(row @ row)
        if norm2 > EPS:
            candidates.append(original_xy - (float(row @ original_xy - bound)) * row / norm2)
            candidates.extend(_circle_line_intersections(row, float(bound), xy_radius))
    for i in range(len(q)):
        for j in range(i + 1, len(q)):
            matrix = np.stack((q[i], q[j]))
            determinant = float(np.linalg.det(matrix))
            if abs(determinant) > 1.0e-10:
                candidates.append(np.linalg.solve(matrix, np.array([b[i], b[j]])))
    usable: list[np.ndarray] = []
    for candidate in candidates:
        if (np.isfinite(candidate).all()
                and np.linalg.norm(candidate) <= xy_radius + tolerance
                and np.all(q @ candidate <= b + tolerance)):
            usable.append(candidate)
    if not usable:
        return HalfplaneSolution(np.array([0.0, 0.0, z]), False, "infeasible",
                                 None, len(candidates))
    best = min(usable, key=lambda value: float(np.sum((value - nominal[:2]) ** 2)))
    output = np.array([best[0], best[1], z], dtype=np.float64)
    margin = None if len(q) == 0 else float(np.min(b - q @ best))
    return HalfplaneSolution(output, True, "feasible", margin, len(candidates))

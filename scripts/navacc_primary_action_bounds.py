"""Validation for configurable GoalFrame primary action bounds, in m/s².

Observation history normalization remains the historical (.8,.8,.5); changing
physical action bounds must not silently change checkpoint observation semantics.
XY radial and z limits derive from the symmetric actor mapping scale. CBF candidate
bounds are diagnostic only under the existing unbounded execution protocol.
"""
import math

def validate_action_bounds(action):
    scale=list(action.acceleration_scale_m_s2)
    total=float(action.total_acceleration_max_m_s2)
    if len(scale)!=3 or any(not math.isfinite(float(x)) or float(x)<=0 for x in scale):
        raise ValueError("primary actor scale must contain three finite positive m/s² values")
    if float(scale[0])!=float(scale[1]):
        raise ValueError("primary actor X/Y scales must be equal for the XY radial bound")
    if not math.isfinite(total) or total<=float(scale[2]):
        raise ValueError("primary total acceleration limit must exceed the positive Z limit")
    return tuple(float(x) for x in scale),total

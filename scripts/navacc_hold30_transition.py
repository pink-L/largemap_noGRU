"""Exact perimeter admission, not actor guidance."""
import numpy as np
def route_witness(sampler,start,goal,*,guidance_mode):
 from navacc_large_map_planar_sampling3m import fixed_global_initial_route_witness
 return fixed_global_initial_route_witness(sampler,start,goal,protocol=None,guidance_mode=guidance_mode)

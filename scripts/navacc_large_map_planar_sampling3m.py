"""Portable finite-horizon and exact-route admission helpers."""
import numpy as np
def evaluation_deadline_s(cfg):
    from largemap_no_gru.config import validate_config
    validate_config(cfg)
    return 32.0
def fixed_global_initial_route_witness(sampler,start,goal,*,protocol,guidance_mode):
    """Admission proof only for registered sampling3m global-goal evaluation.

    The cached grid route can reject a legal mission when its final grid leg
    crosses an inflated box. A perimeter witness is accepted only if the SAME
    exact inflated-box segment oracle proves every segment. This route is
    never fed to the actor; training and route-guidance modes cannot use it.
    """
    import numpy as np
    if guidance_mode!='global_goal_v1':return ()
    a=np.asarray(start,dtype=np.float64);b=np.asarray(goal,dtype=np.float64)
    if a.shape!=(2,) or b.shape!=(2,) or not np.isfinite(a).all() or not np.isfinite(b).all():return ()
    cfg=sampler.config
    separation=float(np.linalg.norm(b-a))
    if not cfg.mission_min_separation_m<=separation<=cfg.mission_max_separation_m:return ()
    for point,z in ((a,cfg.mission_start_z_m),(b,cfg.mission_goal_z_m)):
        if sampler.point_clearance(np.array([point[0],point[1],z]))<cfg.mission_clearance_m:return ()
    # Fixed opposite requests use top->bottom at ±24. Both perimeter sides
    # are tried with exact closed-segment checks and original wall inflation.
    for side in (24.,-24.):
        points=(a,np.array([side,24.]),np.array([side,-24.]),b)
        if all(sampler.segment_static_clear(x,y) for x,y in zip(points[:-1],points[1:])):
            return tuple(p.copy() for p in points)
    return ()

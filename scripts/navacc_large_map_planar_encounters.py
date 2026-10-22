"""Passive obstacle telemetry; physical net SDF differs from XY activation."""
import torch

def _sdf_xy(points,centers,half,inflation=.1):
    delta=(points[:,None,:2]-centers[None,:,:2]).abs()-half[None,:,:2]
    return delta.clamp_min(0).norm(dim=-1)+delta.amax(-1).clamp_max(0)-inflation

def obstacle_observations(backend,position,dynamic_position):
    e=len(position);d=position.device;dtype=position.dtype
    result={}
    for name,centers in [('static',backend.static_centers),('dynamic',dynamic_position)]:
        defined=len(centers)>0 and bool(torch.isfinite(centers).all()) and bool(torch.isfinite(position).all())
        if not defined:
            minimum3=minimumxy=torch.zeros(e,device=d,dtype=dtype);count=torch.zeros(e,device=d,dtype=torch.long)
        elif name=='static':
            minimum3=backend._static_sdf(position,.1).amin(-1)
            minimumxy=backend._static_sdf_xy(position,.1).amin(-1)
            count=((position[:,None,:2]-centers[None,:,:2]).norm(dim=-1)<=3.).sum(-1)
        else:
            minimum3=backend._dynamic_sdf(position,centers,.1).amin(-1)
            radial=(position[:,None,:2]-centers[None,:,:2]).norm(dim=-1)-backend.dynamic_width[None]*.5-.1
            half=torch.stack((backend.dynamic_width*.5,backend.dynamic_width*.5),-1)
            cube=_sdf_xy(position,centers,half)
            minimumxy=torch.where(backend.dynamic_cube[None],cube,radial).amin(-1)
            count=((position[:,None,:2]-centers[None,:,:2]).norm(dim=-1)<=3.).sum(-1)
        result[f'planar_{name}_clearance_3d_m']=minimum3[:,None]
        result[f'planar_{name}_clearance_xy_m']=minimumxy[:,None]
        result[f'planar_{name}_encounter_center3m_count']=count[:,None]
        result[f'planar_{name}_clearance_defined']=torch.full((e,1),defined,device=d,dtype=torch.bool)
    return result

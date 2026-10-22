"""Three-stage exact20M continuations with persistent stage coordinates and fixed LR."""
from copy import deepcopy
from pathlib import Path
import json,math,hashlib
import torch
from omegaconf import OmegaConf
SCHEMA='largemap_no_gru_v1'
MODE='obstacle_course_continuous_v1'
REWARD_SCHEMA='s280_first80_hold1_execsmooth02_corr60_v1'
NAMES=('S280D50',)
STAGES=((280,50),)
HORIZONS=((32,32),)
SAMPLERS=('navrl_edges_short_full00_xy_v1',)
CAPACITY_STATIC=280;CAPACITY_DYNAMIC=50;TRACE_SECONDS=4000.
STAGE_FRAMES=20000000;SEGMENT_FRAMES=10000000;SWITCH_FRAMES=10000000;LOCAL_CP=(5111808,10000000,15111808,20000000)
CONTRACT={'schema':SCHEMA,'reward_schema':REWARD_SCHEMA,'historical30_max_ordinal':42,'formal_total_new_frames':60000000,'stage_new_frames':STAGE_FRAMES,'lr_schedule':'constant_1e-4_v1','lr_anchor':.0001,'horizons':[32],'rollout_steps':128,'num_envs':1024,'frames_per_batch':131072,'num_minibatches':64,'ppo_epochs':4,'checkpoint_local_frames':list(LOCAL_CP),'switch_after_frames':SWITCH_FRAMES,'physical':'capacity280_50_fixed_enabled_shapes_unique_farfield_parking_v3','trace_duration_s':TRACE_SECONDS,'eval_distribution':'full_four_edge_min.5_max70','sampled_only':True,'selection':'safe_desc_tailhold_desc_collision_asc_localframes_desc','no_80percent_gate':True}
PARK_SPACING_M=20.;PARK_Z_M=10.;CONTACT_OFFSET_M=.02

def active(cfg):return cfg.task.large_map.get('goal_curriculum_schema')==SCHEMA


def context(cfg):return OmegaConf.to_container(cfg.task.large_map.course_context,resolve=True)

def validate_context(c):
 if c.get('density')!='S280D50' or c.get('horizon_s')!=32 or c.get('historical_through_ordinal')!=-1:raise ValueError('Portable fresh course context differs')
 return c


def capacity_scene(cfg):
 from navacc_large_map import load_scene
 return load_scene(cfg.task.large_map.scene_path)


def validate_config(cfg):
 from largemap_no_gru.config import validate_config as validate
 return validate(cfg)

def validate_short_scene(scene):
 import numpy as np
 if scene.deadline_s!=32 or len(scene.static_boxes)!=280 or len(scene.dynamic_bodies)!=50:raise ValueError('Current profile scene differs')
 if len(scene.starts)!=len(scene.goals):raise ValueError('Mission cardinality differs')



def parking_positions(kind,count,*,device='cpu',dtype=torch.float32):
 if kind not in ('static','dynamic'):raise ValueError('parking bank kind differs')
 out=torch.empty(count,3,device=device,dtype=dtype);out[:,0]=-1000.-PARK_SPACING_M*torch.arange(count,device=device,dtype=dtype);out[:,1]=-1000. if kind=='static' else -2000.;out[:,2]=PARK_Z_M;return out

def parking_geometry_contract(scene):
 radii=[float(b.size[0])*.5 if b.kind=='cylinder' else math.hypot(float(b.size[0]),float(b.size[1]))*.5 for b in scene.static_boxes]+[float(b.width_m)*.5 for b in scene.dynamic_bodies]
 heights=[float(b.size[2]) for b in scene.static_boxes]+[float(b.height_m) for b in scene.dynamic_bodies]
 radius=max(radii);height=max(heights);gap=PARK_SPACING_M-2*radius-2*CONTACT_OFFSET_M;ground=PARK_Z_M-height*.5-CONTACT_OFFSET_M
 if not gap>0 or not ground>0 or not PARK_SPACING_M>height:raise ValueError('parking bank clearance insufficient for actual geometry')
 return {'spacing_m':PARK_SPACING_M,'center_z_m':PARK_Z_M,'max_radius_m':radius,'max_height_m':height,'contact_offset_m':CONTACT_OFFSET_M,'minimum_surface_gap_including_offsets_m':gap,'minimum_ground_clearance_including_offset_m':ground,'between_bank_y_separation_m':1000.}

class ActiveDynamicView:
 """Full native capacity storage, active-only public truth and writes."""
 def __init__(self,view,active_count,capacity=CAPACITY_DYNAMIC):self.view=view;self.active_count=active_count;self.capacity=capacity
 def initialize(self):self.view.initialize()
 def get_world_poses(self,clone=True):
  p,q=self.view.get_world_poses(clone=clone);return p[:self.active_count],q[:self.active_count]
 def get_velocities(self,clone=True):return self.view.get_velocities(clone=clone)[:self.active_count]
 def set_world_poses(self,p,q):
  if p.shape!=(self.active_count,3) or q.shape!=(self.active_count,4):raise ValueError('active native pose shape mismatch')
  allp=parking_positions('dynamic',self.capacity,device=p.device,dtype=p.dtype);allq=torch.zeros(self.capacity,4,device=q.device,dtype=q.dtype);allq[:,0]=1;allp[:self.active_count]=p;allq[:self.active_count]=q;self.view.set_world_poses(allp,allq)
 def set_velocities(self,v):
  if v.shape!=(self.active_count,6):raise ValueError('active native velocity shape mismatch')
  allv=torch.zeros(self.capacity,6,device=v.device,dtype=v.dtype);allv[:self.active_count]=v;self.view.set_velocities(allv)

def audit_native_activation(base,stage):
 import omni.usd
 from pxr import Usd,UsdPhysics
 usd=omni.usd.get_context().get_stage();full=capacity_scene(base.cfg);s,d=STAGES[stage][:2];flags=[]
 for kind,total,count in (('static',CAPACITY_STATIC,s),('dynamic',CAPACITY_DYNAMIC,d)):
  for i in range(total):
   path=f'/World/navrl_static/box_{i:03d}' if kind=='static' else f'/World/navrl_dynamic/body_{full.dynamic_bodies[i].body_id:03d}'
   prim=usd.GetPrimAtPath(path)
   if not prim.IsValid():raise RuntimeError('curriculum capacity prim missing '+path)
   found=False
   for child in Usd.PrimRange(prim):
    if child.HasAPI(UsdPhysics.CollisionAPI):
     if not bool(UsdPhysics.CollisionAPI(child).GetCollisionEnabledAttr().Get()):raise RuntimeError('fixed capacity collision topology was mutated')
     found=True
   if not found:raise RuntimeError('curriculum prim lacks collision API '+path)
   flags.append({'path':path,'collision_enabled':True,'task_active':i<count})
 inactive={}
 for kind,total,count,view in (('static',CAPACITY_STATIC,s,base._curriculum_static_view),('dynamic',CAPACITY_DYNAMIC,d,base._dynamic_view.view)):
  actual,_=view.get_world_poses(clone=True);expected=parking_positions(kind,total,device=actual.device,dtype=actual.dtype)
  parked=actual[count:]
  if not torch.allclose(parked,expected[count:],atol=1e-4,rtol=0):raise RuntimeError('inactive physical parking pose differs '+kind)
  minimum=float(torch.linalg.vector_norm(parked[:,:2],dim=-1).min()) if len(parked) else None
  separation=float(torch.pdist(parked).min()) if len(parked)>1 else None
  if minimum is not None and minimum<=900:raise RuntimeError('inactive physical object inside task vicinity')
  if separation is not None and separation<PARK_SPACING_M-1e-4:raise RuntimeError('inactive parking slots overlap')
  inactive[kind]={'count':total-count,'minimum_xy_distance_m':minimum,'minimum_center_separation_m':separation,'physical_pose_readback':True}
 proof={'stage':stage,'physical_strategy':CONTRACT['physical'],'active_static':s,'active_dynamic':d,'capacity_static':CAPACITY_STATIC,'capacity_dynamic':CAPACITY_DYNAMIC,'collision_topology_fixed_enabled':True,'collision_flags':flags,'inactive_physical_audit':inactive,'geometry_clearance':parking_geometry_contract(full)}
 base._curriculum_physics_proof=proof;return proof

def set_native_activation(base,stage):
 # Shapes/collision API never change after creation. Only native tensor poses move.
 full=capacity_scene(base.cfg);s,d=STAGES[stage][:2];parking_geometry_contract(full)
 positions=parking_positions('static',CAPACITY_STATIC,device=base.device);rotations=torch.zeros(CAPACITY_STATIC,4,device=base.device);rotations[:,0]=1
 if s:positions[:s]=torch.tensor([b.center.tolist() for b in full.static_boxes[:s]],device=base.device)
 base._curriculum_static_view.set_world_poses(positions,rotations);base._curriculum_static_view.set_velocities(torch.zeros(CAPACITY_STATIC,6,device=base.device))
 base._dynamic_view.active_count=d;base._sync_dynamic_prims()
 return audit_native_activation(base,stage)

def route_witness(sampler,start,goal,*,guidance_mode,cfg):
 """Exact static admission witness; never an actor route or dynamic guarantee."""
 import numpy as np
 if not active(cfg) or guidance_mode!='global_goal_v1' or (len(sampler.scene.static_boxes),len(sampler.scene.dynamic_bodies))!=STAGES[int(cfg.task.large_map.curriculum_stage)][:2]:return ()
 a=np.asarray(start,dtype=np.float64);b=np.asarray(goal,dtype=np.float64)
 if a.shape!=(2,) or b.shape!=(2,) or not np.isfinite(a).all() or not np.isfinite(b).all():return ()
 cfg=sampler.config
 if not cfg.mission_min_separation_m<=float(np.linalg.norm(b-a))<=cfg.mission_max_separation_m:return ()
 for side in (24.,-24.):
  points=(a,np.array([side,24.]),np.array([side,-24.]),b)
  if all(sampler.segment_static_clear(x,y) for x,y in zip(points[:-1],points[1:])):return tuple(x.copy() for x in points)
 # Random-edge training requests may connect any two mission edges. Search
 # only the four perimeter corners; every accepted segment uses the same oracle.
 points=[a,b]+[np.array(x,dtype=np.float64) for x in ((-24.,-24.),(-24.,24.),(24.,24.),(24.,-24.))]
 queue=[(0,)];seen={0}
 while queue:
  chain=queue.pop(0)
  for j in range(1,len(points)):
   if j not in seen and sampler.segment_static_clear(points[chain[-1]],points[j]):
    candidate=chain+(j,)
    if j==1:return tuple(points[k].copy() for k in candidate)
    seen.add(j);queue.append(candidate)
 return ()

"""Portable course mission/cohort state; preserve old outcomes and annotate phase origins."""
from copy import deepcopy
import torch
from navacc_large_map_hold48 import Hold48State
from navacc_hold30_continuous import ContinuousCohort as OldCohort,_summary as summary30,validate_cohort_state as validate30
STATE_SCHEMA='obstacle_course_mission_state_v1'
COHORT_SCHEMA='obstacle_course_lane_ordinal_cohort_v1'

def validate_hold_state(x,n):
 from navacc_hold50_continuous import validate_hold_state as validate50
 y=deepcopy(x)
 if y.get('schema')!=STATE_SCHEMA:raise ValueError('course mission schema invalid')
 y['schema']='hold50_mission_state_v1';validate50(y,n)

class CourseState(Hold48State):
 def state_dict(self):
  x=super().state_dict();x['schema']=STATE_SCHEMA;return x
 def load_state_dict(self,x):
  validate_hold_state(x,self.ever.numel())
  for k in ('ever','first_time','hold_time','first_speed'):getattr(self,k).copy_(x[k].to(self.ever.device))

def row_context(c):return {k:c[k] for k in ('phase_id','density','horizon_s','historical_through_ordinal')}

def summary(cid,n,events,c):
 x=summary30(cid,n,events);x['schema']=COHORT_SCHEMA;x['origin_context']=row_context(c);x['task_horizon_s']=c['horizon_s'];x['may_include_prior_phase_episodes']=cid<=c['historical_through_ordinal'];return x

def validate_cohort_state(x,n):
 if x.get('schema')!=COHORT_SCHEMA or 'context' not in x:raise ValueError('course cohort context/schema missing')
 from navacc_goal_curriculum import validate_context
 validate_context(x['context'])
 y=deepcopy(x);y.pop('context');y['schema']='continuous_lane_ordinal_cohort30s_v1'
 for i,row in enumerate(x['records']):
  events=row['episodes'];schema=row['schema']
  if schema==COHORT_SCHEMA:
   expected=summary(i,n,events,row['origin_context'])
  elif schema=='continuous_lane_ordinal_cohort30s_v1':expected=summary30(i,n,events)
  elif schema=='continuous_lane_ordinal_cohort50s_with30s_boundary_v1':
   from navacc_hold50_continuous import _summary
   expected=_summary(i,n,events)
  else:raise ValueError('unregistered historical cohort schema')
  if row!=expected:raise ValueError('course historical aggregates differ')
  y['records'][i]=summary30(i,n,events)
 validate30(y,n)

class CourseCohort(OldCohort):
 def __init__(self,n,target_size=None,context=None):
  super().__init__(n,target_size);self.context=deepcopy(context)
 def _publish_ready(self):
  while len(self.pending.get(self.cohort_id,{}))==self.num_envs:
   events=[self.pending[self.cohort_id][i] for i in range(self.num_envs)];self.records.append(summary(self.cohort_id,self.num_envs,events,self.context));del self.pending[self.cohort_id];self.cohort_id+=1
 def state_dict(self):
  x=super().state_dict();x.update(schema=COHORT_SCHEMA,context=deepcopy(self.context));return x
 def load_state_dict(self,x):
  validate_cohort_state(x,self.num_envs)
  if self.context!=x['context']:raise ValueError('native cohort context must equal effective phase')
  y=deepcopy(x);y.pop('context');y['schema']='continuous_lane_ordinal_cohort30s_v1';y['records']=[summary30(i,self.num_envs,v['episodes']) for i,v in enumerate(x['records'])]
  super().load_state_dict(y);self.records=deepcopy(x['records'])


def update_tail_low_speed(counter,active,distance,velocity):
 if counter.ndim!=1 or active.shape!=counter.shape or distance.shape!=counter.shape or velocity.shape!=(counter.numel(),3):raise ValueError('tail-hold shape differs')
 qualifies=(distance<=.5)&(torch.linalg.vector_norm(velocity,dim=-1)<=.3)&torch.isfinite(velocity).all(-1)&torch.isfinite(distance)
 return torch.where(active,torch.where(qualifies,counter+1,torch.zeros_like(counter)),counter)

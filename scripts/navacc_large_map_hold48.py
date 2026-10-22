"""Opt-in finite 48s mission state and registered asynchronous cohort contract.

This module is simulator independent. Reserved tickets admit only at a real
mission reset. Fast failed lanes can continue training without replacing the
slow requests in an existing cohort. Publications are immutable checkpointed
records, never destructive logger reads.
"""
from __future__ import annotations
from copy import deepcopy
import json
import math
import torch

MODE='hold48_v1'
OBS_MODE='public21_xy1_z1_logd_yaw_hold48_v1'
OBS_SCHEMA='worldgoal_xy1_z1_logd_public21_ever_reached_hold48_yaw144_v1'
POLICY_IDENTITY='navacc_hold48_public21_actor213_publiccritic213_v1'
CRITIC_SCHEMA='structured213_public21_hold48_no_private_distance_v1'
STATE_SCHEMA='hold48_mission_state_v1'
COHORT_SCHEMA='hold48_registered_async_cohort_v1'
EVAL_SCHEMA='fixed_finite48s_hold_end_v1'


class Hold48State:
    def __init__(self,n,device='cpu',dtype=torch.float32):
        self.ever=torch.zeros(n,device=device,dtype=torch.bool)
        self.first_time=torch.full((n,),-1.,device=device,dtype=dtype)
        self.hold_time=torch.zeros(n,device=device,dtype=dtype)
        self.first_speed=torch.full((n,),-1.,device=device,dtype=dtype)
    def reset(self,ids):
        self.ever[ids]=False;self.first_time[ids]=-1;self.hold_time[ids]=0;self.first_speed[ids]=-1
    def step(self,in_goal,fatal,speed,elapsed,dt=.01,rate=1.,vref=.5):
        if dt<=0 or rate<0 or vref<=0 or not all(math.isfinite(v) for v in (dt,rate,vref)):
            raise ValueError('invalid hold reward constants')
        if not bool(torch.isfinite(speed).all()):
            if bool((~fatal & ~torch.isfinite(speed)).any()):raise ValueError("nonfinite speed without fatal")
            speed=torch.nan_to_num(speed)
        first=in_goal & ~self.ever & ~fatal
        eligible=in_goal & self.ever & ~fatal
        self.first_time[first]=elapsed[first].to(speed.dtype)*dt
        self.first_speed[first]=speed[first]
        self.ever|=first;self.hold_time+=eligible.to(speed.dtype)*dt
        return first,dt*rate*eligible.to(speed.dtype)*torch.exp(-(speed/vref).square())
    def state_dict(self):
        return {'schema':STATE_SCHEMA,**{k:getattr(self,k).detach().cpu().clone() for k in ('ever','first_time','hold_time','first_speed')}}
    def load_state_dict(self,x):
        validate_hold_state(x,self.ever.numel())
        for k in ('ever','first_time','hold_time','first_speed'):getattr(self,k).copy_(x[k].to(self.ever.device))


def validate_hold_state(x,n):
    if not isinstance(x,dict) or set(x)!={'schema','ever','first_time','hold_time','first_speed'} or x['schema']!=STATE_SCHEMA:
        raise ValueError('hold48 mission schema differs')
    for k in ('ever','first_time','hold_time','first_speed'):
        v=x[k]
        if not torch.is_tensor(v) or v.shape!=(n,) or (v.dtype!=torch.bool if k=='ever' else not v.is_floating_point() or not bool(torch.isfinite(v).all())):
            raise ValueError('invalid hold48 '+k)
    if bool(((x['hold_time']<0)|(x['hold_time']>48.01)).any()) or bool((x['hold_time'][~x['ever']]!=0).any()) or bool((x['first_time'][~x['ever']]!=-1).any()) or bool((x['first_speed'][~x['ever']]!=-1).any()) or bool(((x['first_time'][x['ever']]<0)|(x['first_time'][x['ever']]>48.01)).any()) or bool((x['first_speed'][x['ever']]<0).any()):
        raise ValueError('inconsistent hold48 mission state')


class RegisteredCohort:
    def __init__(self,num_envs,target_size=1024):
        if type(num_envs)!=int or type(target_size)!=int or num_envs<1 or target_size<num_envs or target_size%num_envs:
            raise ValueError('cohort requires integer equal per-lane quotas')
        self.num_envs=num_envs;self.target_size=target_size;self.cohort_id=0
        self.records=[];self.published_through=-1;self.last_seen_reset_epoch=[-1]*num_envs;self._reserve()
    def _reserve(self):
        self.tickets=[{'cohort':self.cohort_id,'lane':lane,'slot':slot,'reset_epoch':None,'status':'RESERVED_NOT_ADMITTED','metrics':None,'interruption_reason':None}
            for lane in range(self.num_envs) for slot in range(self.target_size//self.num_envs)]
    def _finish_cohort(self):
        if not all(t['status'] in ('DONE','INTERRUPTED') for t in self.tickets):return
        done=[t['metrics'] for t in self.tickets if t['status']=='DONE']
        record={'schema':COHORT_SCHEMA,'cohort_id':self.cohort_id,'registered_count':self.target_size,
            'completed_count':len(done),'interrupted_unknown':self.target_size-len(done),
            'interrupted_admitted_count':sum(t['status']=='INTERRUPTED' and t['reset_epoch'] is not None for t in self.tickets),
            'interrupted_reserved_count':sum(t['status']=='INTERRUPTED' and t['reset_epoch'] is None for t in self.tickets),'tickets':deepcopy(self.tickets)}
        for key in ('raw_in_goal_end','safe_reach_goal_end','ever_reached','fatal'):
            record[key+'_count']=sum(int(m[key]) for m in done)
            record[key+'_rate']=record[key+'_count']/self.target_size
        for key in ('collision','crash','oob','deadline_unfinished','horizon_reached'):
            record[key+'_count']=sum(int(m.get(key,False)) for m in done)
            record[key+'_rate']=record[key+'_count']/self.target_size
        for key in ('first_reach_time_s','arrival_speed_first'):
            values=[float(m[key]) for m in done if m.get('ever_reached') and m.get(key) is not None and m[key]>=0]
            record[key+'_sample_count']=len(values);record[key+'_mean']=sum(values)/len(values) if values else None
        record['arrival_end']=record['safe_reach_goal_end_rate']
        record['ever_reached_rate_denominator']=self.target_size
        self.records.append(record);self.cohort_id+=1;self._reserve()
    def on_reset(self,ids,epochs):
        ids=list(ids);epochs=list(epochs)
        if len(ids)!=len(epochs) or len(set(ids))!=len(ids):raise ValueError('invalid reset batch')
        if any(type(lane)!=int or not 0<=lane<self.num_envs or type(epoch)!=int or epoch<=self.last_seen_reset_epoch[lane] for lane,epoch in zip(ids,epochs)):
            raise ValueError('cohort reset epochs must be strictly increasing')
        # Validate the entire notification before mutating any ticket.
        for lane,epoch in zip(ids,epochs):
            self.last_seen_reset_epoch[lane]=epoch
            for t in self.tickets:
                if t['lane']==lane and t['status']=='ACTIVE':t.update(status='INTERRUPTED',interruption_reason='reset_while_active')
        self._finish_cohort()
        for lane,epoch in zip(ids,epochs):
            reserved=next((t for t in self.tickets if t['lane']==lane and t['status']=='RESERVED_NOT_ADMITTED'),None)
            if reserved is not None:reserved.update(status='ACTIVE',reset_epoch=epoch)
    def on_step(self,done,epochs,metrics):
        for lane in torch.where(done.reshape(-1))[0].tolist():
            active=[t for t in self.tickets if t['lane']==lane and t['status']=='ACTIVE']
            if not active:continue
            t=active[0]
            if t['reset_epoch']!=int(epochs[lane]):raise ValueError('cohort episode epoch mismatch')
            result={k:(v[lane].item() if torch.is_tensor(v) else v[lane]) for k,v in metrics.items()}
            for k,v in list(result.items()):
                if isinstance(v,float) and not math.isfinite(v):result[k]=None;result[k+'_valid']=False
            t.update(status='DONE',metrics=result)
        self._finish_cohort()
    def restart_episodes(self):
        # No state claims exact simulator restoration. Pending requests become
        # explicitly unknown, including reserved tasks never admitted pre-save.
        for t in self.tickets:
            if t['status'] not in ('DONE','INTERRUPTED'):
                t.update(interruption_reason=('physical_restart_active' if t['status']=='ACTIVE' else 'restart_reserved_never_admitted'),status='INTERRUPTED')
        self._finish_cohort()
    def pending_records(self):return deepcopy([r for r in self.records if r['cohort_id']>self.published_through])
    def acknowledge(self,cohort_id):
        if type(cohort_id)!=int or cohort_id not in (self.published_through,self.published_through+1) or not -1<=cohort_id<len(self.records):raise ValueError('invalid cohort publication acknowledgement')
        self.published_through=cohort_id
    def state_dict(self):
        return deepcopy({'schema':COHORT_SCHEMA,'num_envs':self.num_envs,'target_size':self.target_size,
            'cohort_id':self.cohort_id,'tickets':self.tickets,'records':self.records,'published_through':self.published_through,'last_seen_reset_epoch':self.last_seen_reset_epoch})
    def load_state_dict(self,x):
        validate_cohort_state(x,self.num_envs)
        if x['target_size']!=self.target_size:raise ValueError('cohort target mismatch')
        for k in ('cohort_id','tickets','records','published_through','last_seen_reset_epoch'):setattr(self,k,deepcopy(x[k]))


def validate_cohort_state(x,n):
    expected={'schema','num_envs','target_size','cohort_id','tickets','records','published_through','last_seen_reset_epoch'}
    if not isinstance(x,dict) or set(x)!=expected or x['schema']!=COHORT_SCHEMA or x['num_envs']!=n:
        raise ValueError('cohort runtime schema differs')
    target=x['target_size'];cid=x['cohort_id'];ack=x['published_through'];epochs=x['last_seen_reset_epoch']
    if type(target)!=int or target<n or target%n or type(cid)!=int or cid<0 or type(ack)!=int or not -1<=ack<max(1,cid) or (cid==0 and ack!=-1):
        raise ValueError('cohort runtime dimensions/publication invalid')
    if not isinstance(epochs,list) or len(epochs)!=n or any(type(e)!=int or e< -1 for e in epochs):raise ValueError('invalid cohort epoch history')
    def tickets_valid(tickets,cohort,historical=False):
        if not isinstance(tickets,list) or len(tickets)!=target:raise ValueError('cohort quota differs')
        seen=set();active=set()
        for t in tickets:
            if not isinstance(t,dict) or set(t)!={'cohort','lane','slot','reset_epoch','status','metrics','interruption_reason'} or t['cohort']!=cohort or type(t['lane'])!=int or not 0<=t['lane']<n or type(t['slot'])!=int or not 0<=t['slot']<target//n:
                raise ValueError('invalid cohort ticket')
            key=(t['lane'],t['slot'])
            if key in seen:raise ValueError('duplicate cohort ticket')
            seen.add(key);status=t['status'];epoch=t['reset_epoch']
            if status not in (('DONE','INTERRUPTED') if historical else ('RESERVED_NOT_ADMITTED','ACTIVE','DONE','INTERRUPTED')):raise ValueError('invalid ticket status')
            if epoch is not None and (type(epoch)!=int or epoch<0 or epoch>epochs[t['lane']]):raise ValueError('invalid ticket epoch')
            if status=='INTERRUPTED':
                if t['interruption_reason'] not in ('reset_while_active','physical_restart_active','restart_reserved_never_admitted') or ((epoch is None)!=(t['interruption_reason']=='restart_reserved_never_admitted')):raise ValueError('invalid interruption provenance')
            elif t['interruption_reason'] is not None:raise ValueError('unexpected interruption reason')
            if status in ('ACTIVE','DONE') and epoch is None:raise ValueError('missing reset epoch')
            if status=='RESERVED_NOT_ADMITTED' and epoch is not None:raise ValueError('reserved ticket already admitted')
            if status=='ACTIVE':
                if t['lane'] in active:raise ValueError('multiple active lane tickets')
                active.add(t['lane'])
            m=t['metrics']
            if status=='DONE':
                if not isinstance(m,dict) or not {'raw_in_goal_end','safe_reach_goal_end','ever_reached','fatal'}<=set(m):raise ValueError('missing cohort metrics')
                if any(type(m[k]) not in (bool,int) or m[k] not in (0,1) for k in ('raw_in_goal_end','safe_reach_goal_end','ever_reached','fatal')):raise ValueError('invalid cohort bool metrics')
                if bool(m['safe_reach_goal_end']) != (bool(m['raw_in_goal_end']) and not bool(m['fatal'])):raise ValueError('unsafe success metric')
                if any(v is not None and (type(v) not in (bool,int,float) or not math.isfinite(v)) for v in m.values()):raise ValueError('nonfinite cohort metric')
                if any(v is None and m.get(k+'_valid') is not False for k,v in m.items()):raise ValueError('null metric missing invalid flag')
            elif m is not None:raise ValueError('unfinished ticket has metrics')
    tickets_valid(x['tickets'],cid)
    if not isinstance(x['records'],list) or len(x['records'])!=cid:raise ValueError('cohort publication sequence differs')
    for i,r in enumerate(x['records']):
        if not isinstance(r,dict) or r.get('schema')!=COHORT_SCHEMA or r.get('cohort_id')!=i or r.get('registered_count')!=target:raise ValueError('invalid cohort record identity')
        tickets_valid(r.get('tickets'),i,True)
        done=[t['metrics'] for t in r['tickets'] if t['status']=='DONE']
        if r.get('completed_count')!=len(done) or r.get('interrupted_unknown')!=target-len(done):raise ValueError('cohort count differs')
        for k in ('raw_in_goal_end','safe_reach_goal_end','ever_reached','fatal'):
            count=sum(int(m[k]) for m in done)
            if r.get(k+'_count')!=count or r.get(k+'_rate')!=count/target:raise ValueError('cohort derived metric differs')
        for k in ('collision','crash','oob','deadline_unfinished','horizon_reached'):
            count=sum(int(m.get(k,False)) for m in done)
            if r.get(k+'_count')!=count or r.get(k+'_rate')!=count/target:raise ValueError('cohort principal outcome differs')
        for k in ('first_reach_time_s','arrival_speed_first'):
            values=[float(m[k]) for m in done if m.get('ever_reached') and m.get(k) is not None and m[k]>=0]
            expected=sum(values)/len(values) if values else None
            if r.get(k+'_sample_count')!=len(values) or r.get(k+'_mean')!=expected:raise ValueError('cohort reached subset differs')
        if r.get('interrupted_admitted_count')!=sum(t['status']=='INTERRUPTED' and t['reset_epoch'] is not None for t in r['tickets']) or r.get('interrupted_reserved_count')!=sum(t['status']=='INTERRUPTED' and t['reset_epoch'] is None for t in r['tickets']):raise ValueError('interruption origin counts differ')
        if r.get('arrival_end')!=r['safe_reach_goal_end_rate'] or r.get('ever_reached_rate_denominator')!=target:raise ValueError('cohort denominator differs')

    last=[-1]*n
    for ticket_set in [r['tickets'] for r in x['records']]+[x['tickets']]:
        for t in sorted(ticket_set,key=lambda t:(t['lane'],t['slot'])):
            if t['reset_epoch'] is None:continue
            lane=t['lane'];epoch=t['reset_epoch']
            if epoch<=last[lane]:raise ValueError('cohort admitted reset epochs repeat or regress')
            last[lane]=epoch
            if t['status']=='ACTIVE' and epoch!=epochs[lane]:raise ValueError('active ticket epoch is stale')




def validate_config(cfg):
 from largemap_no_gru.config import validate_config as validate
 return validate(cfg)



# Authorized continuation anchors.  Each is a run's own durable ledger; a ledger
# that stops on a partial tail batch is not a regular batch boundary and cannot
# be continued with the generic exact-resume rule, so the boundary is stated
# explicitly here instead of relaxing that rule.
CONTINUATION_ANCHOR_20M={'learned':20000000,'collected':20021248,'updates':611,'target':50000000}
CONTINUATION_ANCHOR_50M={'learned':50000000,'collected':50036736,'updates':1527,'target':60000000}
# Same S0D0 50M parent continued to a larger destination: the 160-cylinder stage
# run adds 611 whole windows (20,021,248 frames) on top of the published 50M
# checkpoint, so the anchor names that destination explicitly.
CONTINUATION_ANCHOR_50M_TO70M={'learned':50000000,'collected':50036736,'updates':1527,'target':70021248}
# The s24d0u run that the 5M pause rule stopped: its last complete checkpoint
# (idle boundary after PPO update 305) is the declared continuation point for
# the +5M extension, evaluated from the recorded durable ledger.
CONTINUATION_ANCHOR_60M={'learned':59994240,'collected':60030976,'updates':1832,'target':65007744}
CONTINUATION_ANCHOR_50M_TO100M={**CONTINUATION_ANCHOR_50M_TO70M,"target":100003968}
CONTINUATION_ANCHORS=(CONTINUATION_ANCHOR_50M_TO100M,CONTINUATION_ANCHOR_20M,CONTINUATION_ANCHOR_50M,CONTINUATION_ANCHOR_60M,
                      CONTINUATION_ANCHOR_50M_TO70M)

# Obstacle stages of the same 48x48 m random-edge mission distribution.
STAGE_S0D0='s0d0';STAGE_S24D0='s24d0';STAGE_S24D0U='s24d0u';STAGE_S160D0U='s160d0u'
# Declared execution contract of a stage.  Stages without the marker run the
# explicitly named legacy bounded contract; stages with it must materialise the
# complete unbounded contract (checked by validate_execution_contract).
UNBOUNDED_CONTRACT='unbounded_raw_top1_v1'
_S24D0_SCENE={'static_count':24,'dynamic_count':0,'obstacle_geometry_mode':'all_cylinders_width_diameter_v1',
              'static_safety_mode':'static_c3bf'}
# Same mission family and same corrected execution contract as s24d0u, with the
# static obstacle field raised from 24 to 160 cylinders.  Nothing else changes:
# dynamic_count stays 0, the geometry semantics stay width-as-diameter on the Z
# axis and the execution contract is the corrected unbounded raw top-1.
_S160D0_SCENE={**_S24D0_SCENE,'static_count':160}
STAGE_SPECS={
    STAGE_S0D0:{'static_count':0,'dynamic_count':0,'obstacle_geometry_mode':'legacy_box_mixed_v1',
                'static_safety_mode':'lidar144'},
    STAGE_S24D0:dict(_S24D0_SCENE),
    # Same 24-cylinder scene with the CORRECTED C3BF execution contract:
    # nominal keeps its original bounds, the single selected row is projected in
    # closed form with NO magnitude bound, R = obstacle radius + UAV radius with no
    # extra inflation, the float64 projection is re-verified in float32 at 1e-4, the
    # command tilt rejection is off (actuator saturation executes and is recorded;
    # crash only above 40 deg of actual tilt) and the correction reward uses 0.30 on
    # the raw candidate-minus-nominal delta.
    STAGE_S24D0U:{**_S24D0_SCENE,'execution_contract':UNBOUNDED_CONTRACT},
    STAGE_S160D0U:{**_S160D0_SCENE,'execution_contract':UNBOUNDED_CONTRACT},
}
GEOMETRY_DERIVED={'all_cylinders_width_diameter_v1':{'obstacle_geometry_schema':'all_cylinders_width_as_diameter_zaxis_v1',
                                                     'obstacle_width_semantics':'diameter','obstacle_cylinder_axis':'Z'}}








UNBOUNDED_FORK_BUDGET_PATHS=(('task', 'large_map', 'learned_frame_target'),)
"""The one frame-budget key a new training stage must rename.

A fork starts a *new* learning stage, so the continued frame target moves
(50M -> 60M).  That is a budget declaration, not a semantic change: it is
admissible only together with an explicit ``budget_increment_frames`` receipt
field that must equal the delta observed in the live tree.
"""






UNBOUNDED_MIGRATION_TOP_LEVEL=('source_sha256','scene_file_sha256','scene_contract_file_sha256',
                               'trace_duration_s','effective_config_sha256',
                               # the three identity keys the corrected contract adds to a
                               # legacy checkpoint's identity map
                               'execution_protocol','cbf_reward_schema','action_semantics')










# Kept as the name used by the trainer for the obstacle-free stage.
extend_s0d0_contract=lambda payload,runtime,**kw:extend_hold48_contract(
    payload,runtime,stage=STAGE_S0D0,**kw)


MISSION_DISTRIBUTION_KEYS=('mission_sampling_schema','mission_xy_half_extent_m','mission_start_z_m',
                           'mission_goal_z_m','mission_clearance_m','mission_min_separation_m',
                           'mission_max_separation_m','easy_mission_probability')




AMENDMENT_SCHEMA='hold48_identity_amendment_v1'
# Evaluation-only migration of a legacy (bounded-contract) checkpoint onto the
# corrected unbounded execution contract.  Its config allowlist is restricted to
# the execution-contract paths; it can never authorize scene/reward/observation or
# any other semantic change, and it is rejected outside evaluation mode.
UNBOUNDED_MIGRATION_KIND='hold48_unbounded_execution_migration'
# Training fork: resuming (not evaluating) a legacy-contract checkpoint into the
# corrected contract as a new stage.  Restricted to the execution-contract paths
# plus the stage scene keys, and rejected outside resume mode.
UNBOUNDED_TRAINING_FORK_KIND='hold48_unbounded_training_fork'
# Stage migration: evaluating a legacy-contract checkpoint onto the corrected
# contract *inside a new obstacle stage* (so the stage scene keys legitimately
# differ as well).  Like the plain migration it is evaluation-only and locked to
# the execution-contract paths, the stage scene keys and the two keys the
# standard hold48 evaluation flow resolves differently; it can never authorize a
# reward, observation or physics change.
UNBOUNDED_STAGE_MIGRATION_KIND='hold48_unbounded_stage_migration'
AMENDMENT_KINDS=('hold48_s0d0_trace_budget_continuation','hold48_stage_continuation',
                 'hold48_fixed48s_evaluation',UNBOUNDED_MIGRATION_KIND,
                 UNBOUNDED_TRAINING_FORK_KIND,UNBOUNDED_STAGE_MIGRATION_KIND)







"""Versioned50s mission configuration and lossless per-lane ordinal cohorts.

The public21 tensor leaves are byte-compatible with Hold48. Task/horizon,
policy, mission-state and cohort identities are explicitly versioned here.
"""
from copy import deepcopy
import math
import torch
from navacc_large_map_hold48 import Hold48State,STATE_SCHEMA as OLD_STATE_SCHEMA

MODE='hold50_continuous_v1'
STATE_SCHEMA='hold50_mission_state_v1'
COHORT_SCHEMA='continuous_lane_ordinal_cohort50s_with30s_boundary_v1'
EVAL_SCHEMA='fixed_finite50s_hold_end_v1'
POLICY_IDENTITY='navacc_hold50_public21_actor213_publiccritic213_continuous_v1'
FLAGS=('raw_in_goal_end','safe_reach_goal_end','ever_reached','fatal','collision','crash','oob','deadline_unfinished','horizon_reached')
VALUES=('first_reach_time_s','hold_time_s','arrival_speed_first','end_distance','end_speed')


class Hold50State(Hold48State):
    def state_dict(self):
        result=super().state_dict();result['schema']=STATE_SCHEMA;return result
    def load_state_dict(self,state):
        validate_hold_state(state,self.ever.numel())
        for key in ('ever','first_time','hold_time','first_speed'):getattr(self,key).copy_(state[key].to(self.ever.device))


def validate_hold_state(x,n):
    if not isinstance(x,dict) or set(x)!={'schema','ever','first_time','hold_time','first_speed'} or x['schema']!=STATE_SCHEMA:
        raise ValueError('hold50 mission schema differs')
    for k in ('ever','first_time','hold_time','first_speed'):
        v=x[k]
        if not torch.is_tensor(v) or v.shape!=(n,) or (v.dtype!=torch.bool if k=='ever' else not v.is_floating_point() or not bool(torch.isfinite(v).all())):
            raise ValueError('invalid hold50 '+k)
    if bool(((x['hold_time']<0)|(x['hold_time']>50.01)).any()) or bool((x['hold_time'][~x['ever']]!=0).any()) or bool((x['first_time'][~x['ever']]!=-1).any()) or bool((x['first_speed'][~x['ever']]!=-1).any()) or bool(((x['first_time'][x['ever']]<0)|(x['first_time'][x['ever']]>50.01)).any()) or bool((x['first_speed'][x['ever']]<0).any()):
        raise ValueError('inconsistent hold50 mission state')


def _pack_metric(metrics,lane):
    flags=0;values=[];valid=0
    for index,key in enumerate(FLAGS):
        value=metrics.get(key)
        value=(value[lane].item() if torch.is_tensor(value) else value[lane]) if value is not None else False
        if type(value) not in (bool,int) or value not in (0,1):raise ValueError('nonbinary continuous outcome '+key)
        flags |= int(value)<<index
    for index,key in enumerate(VALUES):
        value=metrics.get(key)
        value=(value[lane].item() if torch.is_tensor(value) else value[lane]) if value is not None else None
        if value is not None and math.isfinite(float(value)):valid |= 1<<index;values.append(float(value))
        else:values.append(0.)
    return flags,tuple(values),valid


def _summary(cid,n,events):
    done=[event for event in events if event[2]==0]
    row={'schema':COHORT_SCHEMA,'cohort_id':cid,'registered_count':n,'completed_count':len(done),'interrupted_unknown':n-len(done),
         'interrupted_admitted_count':n-len(done),'interrupted_reserved_count':0,
         # Compact lane/epoch/status/flags/value/validity records retain every result.
         'episodes':deepcopy(events)}
    for index,key in enumerate(FLAGS):
        count=sum((event[3]>>index)&1 for event in done);row[key+'_count']=count;row[key+'_rate']=count/n
    starts=[event[6] for event in events if event[6] is not None];ends=[event[7] for event in events if event[7] is not None]
    row.update(timestamped_episode_count=sum(event[6] is not None and event[7] is not None for event in events),
        phase_clock_start_min_s=min(starts) if starts else None,phase_clock_start_max_s=max(starts) if starts else None,
        phase_clock_end_min_s=min(ends) if ends else None,phase_clock_end_max_s=max(ends) if ends else None,
        phase_clock_coverage_s=max(ends)-min(starts) if starts and ends else None)
    row['arrival_end']=row['safe_reach_goal_end_rate'];row['ever_reached_rate_denominator']=n
    for key,index in [('first_reach_time_s',0),('arrival_speed_first',2)]:
        sample=[event[4][index] for event in done if (event[3]>>2)&1 and (event[5]>>index)&1 and event[4][index]>=0]
        row[key+'_sample_count']=len(sample);row[key+'_mean']=sum(sample)/len(sample) if sample else None
    from navacc_goal_curriculum import CONTRACT
    row['task_horizon_s']=50.;row['may_include_historical30s_episodes']=cid<=CONTRACT['historical30_max_ordinal']
    return row


class ContinuousCohort:
    """Every admitted lane episode belongs to its own ordinal cohort."""
    def __init__(self,num_envs,target_size=None):
        if type(num_envs)!=int or num_envs<1 or (target_size is not None and target_size!=num_envs):raise ValueError('continuous cohort requires exactlyone episode perlane')
        self.num_envs=num_envs;self.target_size=num_envs;self.cohort_id=0;self.published_through=-1
        self.next_ordinal=[0]*num_envs;self.active=[None]*num_envs;self.last_seen_reset_epoch=[-1]*num_envs
        self.pending={};self.records=[]
    def _store(self,ordinal,event):
        lane=event[0];rows=self.pending.setdefault(ordinal,{})
        if lane in rows:raise ValueError('duplicate continuous episode outcome')
        rows[lane]=event
    def _publish_ready(self):
        while len(self.pending.get(self.cohort_id,{}))==self.num_envs:
            events=[self.pending[self.cohort_id][lane] for lane in range(self.num_envs)]
            self.records.append(_summary(self.cohort_id,self.num_envs,events));del self.pending[self.cohort_id];self.cohort_id+=1
    def on_reset(self,ids,epochs,*,clock_s=None):
        ids=list(ids);epochs=list(epochs)
        if len(ids)!=len(epochs) or len(set(ids))!=len(ids) or any(type(lane)!=int or not 0<=lane<self.num_envs or type(epoch)!=int or epoch<=self.last_seen_reset_epoch[lane] for lane,epoch in zip(ids,epochs)):raise ValueError('continuous reset epoch/order invalid')
        for lane,epoch in zip(ids,epochs):
            if self.active[lane] is not None:
                ordinal,old_epoch,start=self.active[lane];self._store(ordinal,(lane,old_epoch,1,0,(0.,)*5,0,start,clock_s))
            ordinal=self.next_ordinal[lane];self.next_ordinal[lane]+=1;self.active[lane]=(ordinal,epoch,clock_s);self.last_seen_reset_epoch[lane]=epoch
        self._publish_ready()
    def on_step(self,done,epochs,metrics,*,clock_s=None):
        staged=[]
        for lane in torch.where(done.reshape(-1))[0].tolist():
            active=self.active[lane]
            if active is None or active[1]!=int(epochs[lane]):raise ValueError('continuous completion missing admission/epoch or duplicate')
            flags,values,valid=_pack_metric(metrics,lane)
            if bool((flags>>1)&1)!=(bool(flags&1) and not bool((flags>>3)&1)):raise ValueError('continuous unsafe success outcome')
            staged.append((lane,active[0],(lane,active[1],0,flags,values,valid,active[2],clock_s)))
        for lane,ordinal,event in staged:self._store(ordinal,event);self.active[lane]=None
        self._publish_ready()
    def restart_episodes(self,*,clock_s=None):
        # No reserved future admissions exist. Retain all already DONE futures.
        for lane,active in enumerate(self.active):
            if active is not None:
                ordinal,epoch,start=active;self._store(ordinal,(lane,epoch,2,0,(0.,)*5,0,start,clock_s));self.active[lane]=None
        self._publish_ready()
    def pending_records(self):return deepcopy([row for row in self.records if row['cohort_id']>self.published_through])
    def acknowledge(self,cohort_id):
        if type(cohort_id)!=int or cohort_id not in (self.published_through,self.published_through+1) or not -1<=cohort_id<len(self.records):raise ValueError('continuous acknowledgement outoforder')
        self.published_through=cohort_id
    def state_dict(self):
        rows=[(ordinal,*event) for ordinal,events in sorted(self.pending.items()) for _,event in sorted(events.items())]
        return deepcopy({'schema':COHORT_SCHEMA,'num_envs':self.num_envs,'target_size':self.target_size,'cohort_id':self.cohort_id,
            'published_through':self.published_through,'next_ordinal':self.next_ordinal,'active':self.active,'last_seen_reset_epoch':self.last_seen_reset_epoch,
            'records':self.records,'pending_index':torch.tensor([row[:4] for row in rows],dtype=torch.long).reshape(-1,4),
            'pending_flags':torch.tensor([row[4] for row in rows],dtype=torch.int16),
            'pending_values':torch.tensor([row[5] for row in rows],dtype=torch.float64).reshape(-1,5),
            'pending_valid':torch.tensor([row[6] for row in rows],dtype=torch.uint8),
            'pending_clock':torch.tensor([[0. if row[7] is None else row[7],0. if row[8] is None else row[8]] for row in rows],dtype=torch.float64).reshape(-1,2),
            'pending_clock_valid':torch.tensor([[row[7] is not None,row[8] is not None] for row in rows],dtype=torch.bool).reshape(-1,2)})
    def load_state_dict(self,state):
        validate_cohort_state(state,self.num_envs)
        for key in ('cohort_id','published_through','next_ordinal','active','last_seen_reset_epoch','records'):setattr(self,key,deepcopy(state[key]))
        self.pending={}
        for index,row in enumerate(state['pending_index'].tolist()):
            ordinal,lane,epoch,status=row;self._store(ordinal,(lane,epoch,status,int(state['pending_flags'][index]),tuple(state['pending_values'][index].tolist()),int(state['pending_valid'][index]),
                float(state['pending_clock'][index,0]) if state['pending_clock_valid'][index,0] else None,float(state['pending_clock'][index,1]) if state['pending_clock_valid'][index,1] else None))


def validate_cohort_state(state,n):
    expected={'schema','num_envs','target_size','cohort_id','published_through','next_ordinal','active','last_seen_reset_epoch','records','pending_index','pending_flags','pending_values','pending_valid','pending_clock','pending_clock_valid'}
    if not isinstance(state,dict) or set(state)!=expected or state['schema']!=COHORT_SCHEMA or state['num_envs']!=n or state['target_size']!=n:raise ValueError('continuous checkpoint schema/count differs')
    cid=state['cohort_id'];ack=state['published_through']
    if type(cid)!=int or cid<0 or type(ack)!=int or not -1<=ack<max(1,cid) or (cid==0 and ack!=-1):raise ValueError('continuous ordinal/ack invalid')
    for key in ('next_ordinal','active','last_seen_reset_epoch'):
        if not isinstance(state[key],list) or len(state[key])!=n:raise ValueError('continuous lane dimensions differ')
    if any(type(x)!=int or x<0 for x in state['next_ordinal']) or any(type(x)!=int or x< -1 for x in state['last_seen_reset_epoch']):raise ValueError('continuous lane counters invalid')
    index=state['pending_index'];flags=state['pending_flags'];values=state['pending_values'];valid=state['pending_valid']
    if not torch.is_tensor(index) or index.dtype!=torch.long or index.ndim!=2 or index.shape[1]!=4:raise ValueError('continuous pending index invalid')
    count=len(index)
    if not torch.is_tensor(flags) or flags.dtype!=torch.int16 or flags.shape!=(count,) or not torch.is_tensor(values) or values.dtype!=torch.float64 or values.shape!=(count,5) or not bool(torch.isfinite(values).all()) or not torch.is_tensor(valid) or valid.dtype!=torch.uint8 or valid.shape!=(count,):raise ValueError('continuous packed outcomes invalid')
    clocks=state['pending_clock'];clock_valid=state['pending_clock_valid']
    if not torch.is_tensor(clocks) or clocks.dtype!=torch.float64 or clocks.shape!=(count,2) or not bool(torch.isfinite(clocks).all()) or not torch.is_tensor(clock_valid) or clock_valid.dtype!=torch.bool or clock_valid.shape!=(count,2):raise ValueError('continuous timestamp arrays invalid')
    seen=[{} for _ in range(n)]
    def event_valid(ordinal,event):
        if not isinstance(event,(list,tuple)) or len(event)!=8:raise ValueError('continuous event fields differ')
        lane,epoch,status,bits,vals,mask,start,end=event
        if type(lane)!=int or not 0<=lane<n or type(epoch)!=int or epoch<0 or epoch>state['last_seen_reset_epoch'][lane] or status not in (0,1,2) or type(bits)!=int or not 0<=bits<512 or type(mask)!=int or not 0<=mask<32 or len(vals)!=5 or any(type(v) not in (int,float) or not math.isfinite(v) for v in vals):raise ValueError('continuous event values invalid')
        if any(value is not None and (type(value) not in (int,float) or not math.isfinite(value) or value<0) for value in (start,end)) or (start is not None and end is not None and end<start):raise ValueError('continuous episode phaseclock invalid')
        if ordinal in seen[lane]:raise ValueError('duplicate continuous ordinal')
        if status==0 and bool((bits>>1)&1)!=(bool(bits&1) and not bool((bits>>3)&1)):raise ValueError('continuous success invalid')
        if status and (bits or mask or any(vals)):raise ValueError('interrupted event carries fabricated metrics')
        seen[lane][ordinal]=epoch
    if not isinstance(state['records'],list) or len(state['records'])!=cid:raise ValueError('continuous history sequence differs')
    for ordinal,record in enumerate(state['records']):
        events=record.get('episodes',[])
        if len(events)!=n or {event[0] for event in events}!=set(range(n)):raise ValueError('continuous complete cohort lacks one result perlane')
        for event in events:event_valid(ordinal,event)
        from navacc_hold30_continuous import _summary as historical30_summary
        expected=historical30_summary(ordinal,n,events) if record.get('schema')=='continuous_lane_ordinal_cohort30s_v1' else _summary(ordinal,n,events)
        if record!=expected:raise ValueError('continuous history aggregates differ')
    pending_counts={}
    for i,(ordinal,lane,epoch,status) in enumerate(index.tolist()):
        if ordinal<cid:raise ValueError('continuous future pending regressed')
        event_valid(ordinal,(lane,epoch,status,int(flags[i]),tuple(values[i].tolist()),int(valid[i]),float(clocks[i,0]) if clock_valid[i,0] else None,float(clocks[i,1]) if clock_valid[i,1] else None))
        pending_counts[ordinal]=pending_counts.get(ordinal,0)+1
    if pending_counts.get(cid,0)==n:raise ValueError('complete continuous cohort not published')
    for lane,active in enumerate(state['active']):
        if active is not None:
            if not isinstance(active,(tuple,list)) or len(active)!=3:raise ValueError('continuous active fields invalid')
            ordinal,epoch,start=active
            if start is not None and (type(start) not in (int,float) or not math.isfinite(start) or start<0):raise ValueError("continuous active phaseclock invalid")
            if type(ordinal)!=int or ordinal in seen[lane] or epoch!=state['last_seen_reset_epoch'][lane]:raise ValueError('continuous active ordinal/epoch differs')
            seen[lane][ordinal]=epoch
        if set(seen[lane])!=set(range(state['next_ordinal'][lane])):raise ValueError('continuous admitted episode gap/drop detected')
        epochs=[seen[lane][ordinal] for ordinal in sorted(seen[lane])]
        if any(a>=b for a,b in zip(epochs,epochs[1:])):raise ValueError('continuous epochs regress/repeat')



def validate_config(cfg):
 from largemap_no_gru.config import validate_config as validate
 return validate(cfg)

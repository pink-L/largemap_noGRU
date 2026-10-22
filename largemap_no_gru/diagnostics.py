"""Bounded smoke-only diagnostics; never applied in formal training/evaluation."""
import torch
from .checkpoint import tensor_digest,atomic_json

def snapshot_lane_tensors(base):
    result={}
    for scope,obj in [('base',base),('drone',base.drone),('motor',base.drone.motor_model),('pid',getattr(base.firmware_pid,'legacy',base.firmware_pid))]:
        for k,v in vars(obj).items():
            if torch.is_tensor(v)and v.ndim and v.shape[0]==base.num_envs:result[scope+'.'+k]=v.detach().clone()
    result['pose']=base.drone.get_world_poses()[0].clone();result['quat']=base.drone.get_world_poses()[1].clone();result['velocity']=base.drone.get_velocities().clone()
    return result

def smoke_checks(env,base,policy,out):
    from torchrl.envs.utils import step_mdp,set_exploration_type,ExplorationType
    from .evaluation import _CrossEvalActionNoise,_apply_random_eval_action
    before=snapshot_lane_tensors(base);clock=base._global_time_s
    base._reset_idx(torch.tensor([0],device=base.device));after=snapshot_lane_tensors(base)
    for k,v in before.items():
        if not torch.equal(v[1:],after[k][1:]):raise RuntimeError('Partial reset contaminated other lanes:'+k)
    if base._global_time_s!=clock:raise RuntimeError('Partial reset moved global clock')
    # No physics step in partial reset; a separate explicitly bounded sampled action path follows.
    env.eval();policy.eval();td=env.reset();noise=_CrossEvalActionNoise(93728,base.num_envs,base.device);digest=tensor_digest(policy.state_dict())
    with torch.no_grad(),set_exploration_type(ExplorationType.MODE):
        for _ in range(32):
            policy(td);_apply_random_eval_action(policy,td,noise);transition=env.step(td)
            if not bool(torch.isfinite(transition['next','info','drone_state']).all()):raise RuntimeError('Short sampled eval nonfinite')
            td=step_mdp(transition,keep_other=True)
    if digest!=tensor_digest(policy.state_dict()):raise RuntimeError('Short eval updated weights/VN')
    result={'status':'PASS','diagnostic_only':True,'formal_navigation_metric':False,'partial_reset_ids':[0],'other_lanes_unchanged':True,'global_clock_unchanged_on_partial_reset':True,'sampled_eval_steps':32,'model_VN_unchanged':True,'total_control_steps_per_lane':160}
    atomic_json(out/'SMOKE_CHECKS.json',result);return result

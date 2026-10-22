"""Pure Torch episode-fixed goal frame, public21 SI state and dynamic10.

W and G are right-handed Z-up frames. Vectors use column-vector mathematics:
v_G=Rz(theta).T v_W; v_W=Rz(theta) v_G. No physics/safety truth is modified.
Dynamic slots: [u_G.xyz,dxy,dz,vrel_G.xyz,physical_radius,full_height].
State snapshots use CPU tensors for explicit portable restore to the owner device.
"""
from __future__ import annotations
import math
import torch

SCHEMA = 'episode_goal_frame_dyn10_v1'
EPSILON_M = 1e-6
DYNAMIC_SLOT_DIM = 10
STATE_DIM = 21


def _float(value, name, last=None):
    if not isinstance(value, torch.Tensor) or value.dtype not in (torch.float32, torch.float64):
        raise ValueError(f'{name} must be a float32/float64 tensor')
    if last is not None and (value.ndim < 1 or value.shape[-1] != last):
        raise ValueError(f'{name} last dimension must be {last}')
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f'{name} is nonfinite; invalid truth must fail closed')
    return value


def _compatible(value, reference, name):
    if value.dtype != reference.dtype or value.device != reference.device:
        raise ValueError(f'{name} dtype/device differs')


def _angle(theta, vector):
    theta = _float(theta, 'theta')
    _compatible(theta, vector, 'theta')
    while theta.ndim < vector.ndim - 1:
        theta = theta.unsqueeze(-1)
    try:
        shape = torch.broadcast_shapes(theta.shape, vector.shape[:-1])
    except RuntimeError as exc:
        raise ValueError('theta batch prefix differs') from exc
    if shape != vector.shape[:-1]:
        raise ValueError('theta would expand vector batch prefix')
    return theta


def world_to_goal(vector_W, theta):
    vector_W = _float(vector_W, 'vector_W', 3)
    theta = _angle(theta, vector_W)
    c, s = torch.cos(theta), torch.sin(theta)
    x, y, z = vector_W.unbind(-1)
    return _float(torch.stack((c*x+s*y, -s*x+c*y, z), -1),'rotated vector_G',3)


def goal_to_world(vector_G, theta):
    vector_G = _float(vector_G, 'vector_G', 3)
    theta = _angle(theta, vector_G)
    c, s = torch.cos(theta), torch.sin(theta)
    x, y, z = vector_G.unbind(-1)
    return _float(torch.stack((c*x-s*y, s*x+c*y, z), -1),'rotated vector_W',3)


def normalize_quaternion(q):
    q = _float(q, 'quaternion', 4)
    scale = q.abs().amax(-1, keepdim=True)
    if bool((scale == 0).any()):
        raise ValueError('zero quaternion is invalid')
    # Scale first so finite nonzero very small/large inputs cannot under/overflow.
    q = q / scale
    return q / torch.linalg.vector_norm(q, dim=-1, keepdim=True)


def canonical_quaternion(q):
    q = normalize_quaternion(q)
    undecided = torch.ones(q.shape[:-1], dtype=torch.bool, device=q.device)
    negative = torch.zeros_like(undecided)
    for component in q.unbind(-1):
        negative |= undecided & (component < 0)
        undecided &= component == 0
    return torch.where(negative[..., None], -q, q)


def quaternion_multiply(a, b):
    a = _float(a, 'quaternion a', 4); b = _float(b, 'quaternion b', 4)
    _compatible(a, b, 'quaternion a')
    aw, ax, ay, az = a.unbind(-1); bw, bx, by, bz = b.unbind(-1)
    return torch.stack((aw*bw-ax*bx-ay*by-az*bz,
                        aw*bx+ax*bw+ay*bz-az*by,
                        aw*by-ax*bz+ay*bw+az*bx,
                        aw*bz+ax*by-ay*bx+az*bw), -1)


def relative_quaternion(quaternion_WB, theta):
    q = normalize_quaternion(quaternion_WB)
    theta = _angle(theta, q)
    half = theta / 2
    c, s = torch.cos(half), torch.sin(half)
    # conj(q_WG) left-multiplies q_WB; never swap operand order.
    c = torch.broadcast_to(c,q.shape[:-1]);s = torch.broadcast_to(s,q.shape[:-1])
    zero = torch.zeros_like(c)
    return canonical_quaternion(quaternion_multiply(torch.stack((c,zero,zero,-s),-1), q))


def yaw_from_quaternion(quaternion_WB):
    w,x,y,z = normalize_quaternion(quaternion_WB).unbind(-1)
    return torch.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))


def _scalar(value, prefix, ref, name):
    if isinstance(value,torch.Tensor):
        if value.device!=ref.device or (value.is_floating_point() and value.dtype!=ref.dtype):
            raise ValueError(f'{name} dtype/device differs')
    else:
        value=torch.as_tensor(value,device=ref.device,dtype=ref.dtype)
    if value.ndim == len(prefix)+1 and value.shape[-1] == 1:
        value = value.squeeze(-1)
    try:
        value = torch.broadcast_to(value, prefix)
    except RuntimeError as exc:
        raise ValueError(f'{name} shape differs') from exc
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f'{name} is nonfinite')
    return value.to(dtype=ref.dtype)


def encode_state21(*, position_W, goal_W, velocity_W, quaternion_WB, gyro_B,
                   hold_yaw_W, progress, max_episode_length, history_request_W,
                   ever_reached, theta, history_valid=None):
    position_W = _float(position_W, 'position_W',3)
    prefix = position_W.shape[:-1]
    for name, value, width in (('goal_W',goal_W,3),('velocity_W',velocity_W,3),
                               ('quaternion_WB',quaternion_WB,4),('gyro_B',gyro_B,3),
                               ('history_request_W',history_request_W,3)):
        _float(value,name,width);_compatible(value,position_W,name)
        if value.shape[:-1] != prefix:raise ValueError(f'{name} batch prefix differs')
    delta_W = goal_W-position_W; delta_G = world_to_goal(delta_W,theta)
    direction_xy = delta_G[...,:2] / torch.linalg.vector_norm(delta_G[...,:2],dim=-1,keepdim=True).clamp_min(1.)
    q_GB = relative_quaternion(quaternion_WB,theta)
    hold = _scalar(hold_yaw_W,prefix,position_W,'hold_yaw_W')-_angle(theta,position_W)
    elapsed = _scalar(progress,prefix,position_W,'progress')
    deadline = _scalar(max_episode_length,prefix,position_W,'max_episode_length')
    if bool((deadline<=0).any()) or bool((elapsed<0).any()):raise ValueError('invalid progress/deadline')
    ever = _scalar(ever_reached,prefix,position_W,'ever_reached')
    if bool(((ever!=0)&(ever!=1)).any()):raise ValueError('ever_reached must be0/1')
    history = history_request_W
    if history_valid is not None:
        mask = _scalar(history_valid,prefix,position_W,'history_valid')
        if bool(((mask!=0)&(mask!=1)).any()):raise ValueError('history_valid must be0/1')
        history = torch.where(mask.bool()[...,None],history,torch.zeros_like(history))
    scale = position_W.new_tensor((.8,.8,.5))
    result = torch.cat((direction_xy,delta_G[...,2:3],q_GB,world_to_goal(velocity_W,theta),gyro_B,
                        torch.sin(hold)[...,None],torch.cos(hold)[...,None],
                        (elapsed/deadline).clamp(0,1)[...,None],
                        torch.log1p(torch.linalg.vector_norm(delta_W,dim=-1))[...,None],
                        world_to_goal(history,theta)/scale,ever[...,None]),-1)
    _float(result,'encoded state',STATE_DIM)
    return result


def encode_dynamic10(relative_position_W, relative_velocity_W, radius, height, valid, theta):
    r = _float(relative_position_W,'relative_position_W',3)
    v = _float(relative_velocity_W,'relative_velocity_W',3);_compatible(v,r,'relative_velocity_W')
    if v.shape != r.shape:raise ValueError('relative velocity shape differs')
    if not isinstance(valid,torch.Tensor) or valid.dtype!=torch.bool or valid.device!=r.device or valid.shape!=r.shape[:-1]:
        raise ValueError('valid must be a boolean slot mask matching batch/device')
    radius = _scalar(radius,r.shape[:-1],r,'radius');height=_scalar(height,r.shape[:-1],r,'height')
    if bool(((radius<=0)|(height<=0))[valid].any()):raise ValueError('valid entity radius/full height must be positive')
    r_G=world_to_goal(r,theta);v_G=world_to_goal(v,theta)
    d3=torch.linalg.vector_norm(r_G,dim=-1,keepdim=True)
    result=torch.cat((r_G/d3.clamp_min(EPSILON_M),torch.linalg.vector_norm(r_G[...,:2],dim=-1,keepdim=True),
                      r_G[...,2:3],v_G,radius[...,None],height[...,None]),-1)
    _float(result,'encoded dynamic',DYNAMIC_SLOT_DIM)
    # One final mask covers direction/dxy/dz/relative velocity AND dimensions.
    return torch.where(valid[...,None],result,torch.zeros_like(result))


class GoalFrame:
    """Authoritative fixed per-lane episode frame; reset uses final start/goal."""
    def __init__(self, num_envs, device='cpu', dtype=torch.float32):
        if type(num_envs)!=int or num_envs<=0 or dtype not in (torch.float32,torch.float64):raise ValueError('invalid goal frame batch/dtype')
        self.num_envs=num_envs;self.device=torch.device(device);self.dtype=dtype
        self.theta=torch.zeros(num_envs,device=self.device,dtype=dtype)
        self.start=torch.zeros(num_envs,3,device=self.device,dtype=dtype);self.goal=torch.zeros_like(self.start)
        self.task_epoch=torch.full((num_envs,),-1,device=self.device,dtype=torch.int64)
        self.fallback=torch.zeros(num_envs,device=self.device,dtype=torch.bool)
        self.valid=torch.zeros_like(self.fallback)

    def reset(self, env_ids, start_W, goal_W, actual_yaw_W, task_epoch):
        ids=torch.as_tensor(env_ids,device=self.device)
        if ids.dtype!=torch.int64 or ids.ndim!=1 or bool(((ids<0)|(ids>=self.num_envs)).any()) or ids.unique().numel()!=ids.numel():raise ValueError('invalid/duplicate reset lanes')
        count=ids.numel()
        for name,v in (('start_W',start_W),('goal_W',goal_W)):
            _float(v,name,3);_compatible(v,self.start,name)
            if v.shape!=(count,3):raise ValueError(f'{name} reset shape differs')
        yaw=_float(actual_yaw_W,'actual_yaw_W');_compatible(yaw,self.theta,'actual_yaw_W')
        if yaw.shape!=(count,):raise ValueError('reset yaw shape differs')
        epoch=torch.as_tensor(task_epoch,device=self.device)
        if epoch.dtype!=torch.int64 or epoch.shape!=(count,) or bool((epoch<0).any()):raise ValueError('invalid task_epoch')
        if bool((self.valid[ids] & (epoch<=self.task_epoch[ids])).any()):raise ValueError('new reset must advance task_epoch')
        delta=_float(goal_W-start_W,'reset goal delta',3)
        fallback=torch.linalg.vector_norm(delta[...,:2],dim=-1)<EPSILON_M
        theta=torch.where(fallback,torch.atan2(torch.sin(yaw),torch.cos(yaw)),torch.atan2(delta[...,1],delta[...,0]))
        _float(theta,'reset theta')
        self.start[ids]=start_W;self.goal[ids]=goal_W;self.theta[ids]=theta
        self.task_epoch[ids]=epoch;self.fallback[ids]=fallback;self.valid[ids]=True

    def state_dict(self):
        return {'schema':SCHEMA,'num_envs':self.num_envs,'dtype':str(self.dtype),'epsilon_m':EPSILON_M,
                'theta':self.theta.detach().cpu().clone(),'start_W':self.start.detach().cpu().clone(),
                'goal_W':self.goal.detach().cpu().clone(),'task_epoch':self.task_epoch.detach().cpu().clone(),
                'fallback':self.fallback.detach().cpu().clone(),'valid':self.valid.detach().cpu().clone()}

    def load_state_dict(self, state, *, expected_start=None, expected_goal=None, expected_task_epoch=None):
        # A fallback theta is the saved original reset-yaw authority. Never
        # reconstruct it from the UAV's current (possibly moved) orientation.
        keys={'schema','num_envs','dtype','epsilon_m','theta','start_W','goal_W','task_epoch','fallback','valid'}
        if not isinstance(state,dict) or set(state)!=keys or state['schema']!=SCHEMA or type(state['num_envs']) is not int or state['num_envs']!=self.num_envs or state['dtype']!=str(self.dtype) or state['epsilon_m']!=EPSILON_M:raise ValueError('goal frame schema/batch/dtype differs')
        floats={'theta':(self.num_envs,),'start_W':(self.num_envs,3),'goal_W':(self.num_envs,3)}
        checked={}
        for name,shape in {**floats,'task_epoch':(self.num_envs,),'fallback':(self.num_envs,),'valid':(self.num_envs,)}.items():
            v=state[name];dtype=self.dtype if name in floats else torch.int64 if name=='task_epoch' else torch.bool
            if not isinstance(v,torch.Tensor) or v.device.type!='cpu' or v.dtype!=dtype or tuple(v.shape)!=shape:raise ValueError(f'goal frame {name} shape/dtype/storage device differs')
            if name in floats:_float(v,name)
            checked[name]=v.to(self.device).clone()
        valid=checked['valid'];epoch=checked['task_epoch']
        if bool((epoch[valid]<0).any()) or bool((epoch[~valid]!=-1).any()):raise ValueError('goal frame epoch/valid differs')
        if bool(valid.any()):
            if bool((checked['theta'][valid].abs()>math.pi+4*torch.finfo(self.dtype).eps).any()):
                raise ValueError('goal frame theta is outside wrapped initial-yaw range')
            if any(x is None for x in (expected_start,expected_goal,expected_task_epoch)):raise ValueError('restoring valid frame requires start/goal/task_epoch binding')
            for name,expected in (('start_W',expected_start),('goal_W',expected_goal),('task_epoch',expected_task_epoch)):
                actual=checked[name];expected=torch.as_tensor(expected,device=self.device)
                if expected.dtype!=actual.dtype or expected.shape!=actual.shape or not torch.equal(actual[valid],expected[valid]):raise ValueError(f'restored goal frame {name} task identity differs')
            delta=_float(checked['goal_W']-checked['start_W'],'restored goal delta',3);fallback=torch.linalg.vector_norm(delta[...,:2],dim=-1)<EPSILON_M
            if not torch.equal(fallback[valid],checked['fallback'][valid]):raise ValueError('goal frame fallback differs from task geometry')
            expected_theta=torch.atan2(delta[...,1],delta[...,0]);regular=valid & ~fallback
            if bool((torch.abs(checked['theta'][regular]-expected_theta[regular])>4*torch.finfo(self.dtype).eps).any()):raise ValueError('goal frame theta differs from initial task')
        if bool(checked['fallback'][~valid].any()) or bool((checked['theta'][~valid]!=0).any()) or bool((checked['start_W'][~valid]!=0).any()) or bool((checked['goal_W'][~valid]!=0).any()):raise ValueError('uninitialized goal frame has residual episode state')
        for target,name in (('theta','theta'),('start','start_W'),('goal','goal_W'),('task_epoch','task_epoch'),('fallback','fallback'),('valid','valid')):setattr(self,target,checked[name])
        return self

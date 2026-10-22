"""Versioned paper-reference motor model. SI units; no rotor gyro coupling.

Reference: CrazyflieBrushJAX commit5e449f116b03218e803e728f2a9d8f68f60b05fe.
Exact constant-input motor discretization at caller dt; not author0.1ms bitwise replication.
"""
# MIT License
# 
# Copyright (c) 2025 Data Science in Mechanical Engineering (DSME) 
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from dataclasses import dataclass
import math
import torch
PROFILE = 'brushless_paper_reference_v1'
SOURCE_COMMIT = '5e449f116b03218e803e728f2a9d8f68f60b05fe'

@dataclass(frozen=True)
class BrushlessPaperProfile:
    profile: str = PROFILE
    mass_kg: float = .042
    inertia_kg_m2: tuple = (3.3e-5,3.6e-5,5.9e-5)
    sites_m: tuple = ((.03535,.03535,0.),(-.03535,.03535,0.),(-.03535,-.03535,0.),(.03535,-.03535,0.))
    spin: tuple = (-1.,1.,-1.,1.)
    omega_max_rad_s: float = 2900.
    tau_s: float = .05
    motor_inertia_kg_m2: float = 5e-8
    low_speed_thrust: str = 'max_raw_polynomial_zero_v1'
    rotor_gyro_mode: str = 'omitted_v1'


def _finite(x,name):
    if not isinstance(x,torch.Tensor) or not x.is_floating_point() or not bool(torch.isfinite(x).all()):raise ValueError(name+' must be a finite floating Tensor')

def force_from_s(s):
    _finite(s,'normalizedspeed')
    if bool(((s<0)|(s>1)).any()):raise ValueError('Normalized speed outside[0,1]')
    return ((-.23009526*s+.56176458)*s-.0433191).mul(s).clamp_min(0.)

def aero_torque_from_s(s):
    _finite(s,'normalizedspeed')
    if bool(((s<0)|(s>1)).any()):raise ValueError('Normalized speed outside[0,1]')
    return ((-.0003396*s+.00087032)*s+.0002896)*s

def inverse_force_positive(force_N,*,bounds='reject'):
    _finite(force_N,'force')
    if bounds not in ('reject','clip'):raise ValueError('Unknown inverse bounds')
    maximum=-.23009526+.56176458-.0433191
    if bounds=='reject' and bool(((force_N<0)|(force_N>maximum)).any()):raise ValueError('Force outside actuator curve')
    target=force_N.clamp(0,maximum);lo=torch.zeros_like(target);hi=torch.ones_like(target)
    # The positive-thrust branch is monotone. F=0 is explicitly u=0, not the positive zero root.
    for _ in range(48):
        mid=(lo+hi)*.5;below=force_from_s(mid)<target;lo=torch.where(below,mid,lo);hi=torch.where(below,hi,mid)
    return torch.where(target==0,torch.zeros_like(target),(lo+hi)*.5)

class BrushlessMotorBank:
    def __init__(self,shape,*,device='cpu',dtype=torch.float32,profile=None):
        self.profile=profile or BrushlessPaperProfile();self.omega=torch.zeros(*shape,4,device=device,dtype=dtype)
        self.sites=self.omega.new_tensor(self.profile.sites_m);self.spin=self.omega.new_tensor(self.profile.spin);self.last=None
    def reset(self,indices=None,*,omega_rad_s=0.):
        value=torch.as_tensor(omega_rad_s,device=self.omega.device,dtype=self.omega.dtype)
        _finite(value,'resetomega')
        if bool(((value<0)|(value>self.profile.omega_max_rad_s)).any()):raise ValueError('Reset omega out of range')
        if indices is None:self.omega.copy_(value.expand_as(self.omega))
        else:self.omega[indices]=value
    def step(self,u,dt=.002):
        _finite(u,'motorinput')
        if u.shape!=self.omega.shape or u.device!=self.omega.device or u.dtype!=self.omega.dtype:raise ValueError('Motor input shape/device/dtype differs')
        if bool(((u<0)|(u>1)).any()) or not math.isfinite(dt) or dt<=0:raise ValueError('Motor input must be[0,1],dt>0')
        p=self.profile;previous=self.omega.clone();target=p.omega_max_rad_s*u
        next_omega=previous+(target-previous)*(-math.expm1(-dt/p.tau_s));self.omega.copy_(next_omega)
        derivative=(next_omega-previous)/dt;s=(next_omega/p.omega_max_rad_s).clamp(0,1)
        force=force_from_s(s);q=aero_torque_from_s(s);jm=p.motor_inertia_kg_m2*derivative;reaction=-self.spin*(q+jm)
        each=torch.zeros(*force.shape,3,device=u.device,dtype=u.dtype);each[...,2]=force
        moment=torch.linalg.cross(self.sites.expand_as(each),each,dim=-1).sum(-2);moment[...,2]+=reaction.sum(-1)
        self.last={'u':u.clone(),'omega_previous':previous,'omega_target':target,'omega':next_omega.clone(),'domega_dt':derivative,'force_N':force,'aero_torque_Nm':q,'inertial_torque_Nm':jm,'reaction_z_each_Nm':reaction,'force_B':each.sum(-2),'torque_B':moment}
        return self.last

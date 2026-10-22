"""Opt-in brushless adapter. Legacy PID/mixer limits/signs preserved; curve replaced only here."""
from dataclasses import replace
import torch
from .acceleration_to_rpyt import AccelerationToRPYT,AccelerationToRPYTConfig
from .crazyflie_firmware_pid import CrazyflieFirmwarePID,CrazyflieFirmwarePIDConfig
from omni_drones.actuators.brushless_paper import inverse_force_positive,force_from_s,PROFILE

class BrushlessAccelerationToRPYT(AccelerationToRPYT):
    def __init__(self):super().__init__(replace(AccelerationToRPYTConfig.cf21bl_simulation_reference(),check_tilt=False))
    def force_to_commander(self,total_force_N):
        if not bool(torch.isfinite(total_force_N).all()):raise ValueError('Nonfinite brushless force')
        u=inverse_force_positive(total_force_N/4.,bounds='clip');pwm=torch.round(u*65535).clamp(self.config.commander_thrust_min,self.config.commander_thrust_max)
        return torch.where(total_force_N<=0,torch.zeros_like(pwm),pwm)
    def __call__(self,a_test_W,psi_hold_W):
        if not bool(torch.isfinite(a_test_W).all()) or not bool(torch.isfinite(psi_hold_W).all()):raise ValueError('Nonfinite brushless converter input')
        rpyt,diagnostics=super().__call__(a_test_W,psi_hold_W)
        requested=diagnostics['desired_force_N']/4.;u=inverse_force_positive(requested,bounds='clip');raw=u*65535.;maximum=force_from_s(torch.ones_like(requested))
        diagnostics['commander_thrust_raw']=raw;diagnostics['thrust_saturated']=(requested>maximum)|(raw<self.config.commander_thrust_min)|(raw>self.config.commander_thrust_max)
        diagnostics['brushless_steady_u_unclipped_curve']=u;diagnostics['brushless_per_motor_force_requested_N']=requested
        return rpyt,diagnostics

class BrushlessFirmwarePIDAdapter:
    profile=PROFILE
    input_mapping='simulation_motor_PWM_div65535_assumption_v1'
    def __init__(self):
        self.legacy=CrazyflieFirmwarePID(CrazyflieFirmwarePIDConfig.cf21bl_2025_02());self.config=self.legacy.config
        if self.config.apply_motor_dynamics:raise ValueError('No firmware duplicate motor lag allowed')
    def __getattr__(self,key):return getattr(self.legacy,key)
    def reset(self,*args,**kwargs):return self.legacy.reset(*args,**kwargs)
    def step(self,*args,**kwargs):
        discarded_old_force_command,d=self.legacy.step(*args,**kwargs)
        u=d['simulation_motor_pwm']/65535.
        if not bool(torch.isfinite(u).all()) or bool(((u<0)|(u>1)).any()):raise RuntimeError('Mixer produced invalid normalized motor input')
        d=dict(d)
        for key in ('motor_force_fraction_target','motor_force_fraction_applied'):d.pop(key,None)
        d['brushless_u']=u.clone();d['legacy_force_mapping_discarded']=True
        return u,d

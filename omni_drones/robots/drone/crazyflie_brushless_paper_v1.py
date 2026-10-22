"""Explicit opt-in single-effective-rigid-body brushless paper-reference backend.

No joint rotors, downwash, randomization or rotor gyroscopic cross coupling.
Rigid-body Euler gyroscopic dynamics remain owned by PhysX. Input is direct u[0,1].
"""
from dataclasses import asdict
import torch
from .multirotor import MultirotorBase
from omni_drones.robots.robot import RobotBase,ASSET_PATH
from omni_drones.robots.config import RobotCfg,RigidBodyPropertiesCfg
from omni_drones.actuators.brushless_paper import BrushlessMotorBank,BrushlessPaperProfile,inverse_force_positive,force_from_s,aero_torque_from_s
from omni_drones.utils.torch import quat_axis,quat_rotate_inverse
from omni_drones.utils.torchrl.compat import BoundedTensorSpec

class CrazyflieBrushlessPaperV1(MultirotorBase):
    usd_path=ASSET_PATH+'/usd/crazyflie_brushless_paper_v1.usda'
    param_path=ASSET_PATH+'/usd/crazyflie_brushless_paper_v1.yaml'
    def __init__(self,name=None,cfg=None,is_articulation=False):
        if is_articulation:raise ValueError('BrushlessPaperV1 is a single rigid body, not articulation')
        cfg=cfg or RobotCfg(rigid_props=RigidBodyPropertiesCfg(linear_damping=0.,angular_damping=0.))
        if cfg.rigid_props.linear_damping!=0 or cfg.rigid_props.angular_damping!=0 or cfg.rigid_props.disable_gravity:raise ValueError('Brushless profile requires damping0 and gravityenabled')
        self.profile=BrushlessPaperProfile();super().__init__(name,cfg,is_articulation=False)
        if float(self.dt)!=.002:raise ValueError('Phase1 native profile requires physics dt=.002')
    @property
    def action_spec(self):
        if not hasattr(self,'_action_spec'):self._action_spec=BoundedTensorSpec(0.,1.,4,device=self.device)
        return self._action_spec
    def initialize(self,prim_paths_expr=None,track_contact_forces=False):
        # Deliberately bypass MultirotorBase's articulated rotor/mass/lag setup.
        RobotBase.initialize(self,prim_paths_expr)
        self.base_link=self._view;self.motor_model=BrushlessMotorBank(self.shape,device=self.device);self.last_motor=None
        self.masses=self._view.get_masses().clone();self.inertias=self._view.get_inertias().reshape(*self.shape,3,3).diagonal(0,-2,-1)
        expected=self.inertias.new_tensor(self.profile.inertia_kg_m2)
        if not bool((self.masses-.042).abs().le(1e-7).all()) or not torch.allclose(self.inertias,expected.expand_as(self.inertias),atol=1e-9,rtol=1e-5):raise RuntimeError('Native single-body mass/inertia differ from brushless profile')
        self.KF=torch.full((*self.shape,4),float(force_from_s(torch.tensor(1.))),device=self.device)
        self.KM=torch.full_like(self.KF,float(aero_torque_from_s(torch.tensor(1.))));self.drag_coef=torch.zeros(*self.shape,1,device=self.device)
        self.tau_up=torch.full_like(self.KF,.05);self.tau_down=self.tau_up.clone();self.use_motor_time_constant=True
        self.throttle=self.motor_model.omega/2900.;self.thrusts=torch.zeros(*self.shape,4,3,device=self.device);self.forces=torch.zeros(*self.shape,3,device=self.device);self.torques=torch.zeros_like(self.forces)
        self.throttle_difference=torch.zeros(*self.shape,device=self.device);self.vel=self.vel_w=torch.zeros(*self.shape,6,device=self.device);self.vel_b=torch.zeros_like(self.vel)
        self.acc=self.acc_w=torch.zeros_like(self.vel);self.acc_b=torch.zeros_like(self.vel);self.heading=torch.zeros(*self.shape,3,device=self.device);self.up=torch.zeros_like(self.heading)
        self.MASS_0=self.masses[0].clone();self.INERTIA_0=self.inertias[0].clone();self.intrinsics=self.intrinsics_spec.expand(self.shape).zero()
        self.mass_contract_audit={'schema':'single_effective_rigid_body_brushless_paper_v1','mass_kg':.042,'body_count_per_robot':1,'rotor_joint_count':0,'inertia_kg_m2':self.profile.inertia_kg_m2}
        self._reset_idx(torch.arange(self.shape[0],device=self.device),False)
    def _reset_idx(self,env_ids=None,train=True):
        if env_ids is None:env_ids=torch.arange(self.shape[0],device=self.device)
        if self.randomization:raise ValueError('Brushless phase1 forbids randomization')
        hover=inverse_force_positive(self.motor_model.omega.new_tensor(.042*9.81/4.))
        self.motor_model.reset(env_ids,omega_rad_s=hover*2900.)
        self.throttle=self.motor_model.omega/2900.;self.thrusts[env_ids]=0;self.forces[env_ids]=0;self.torques[env_ids]=0
        return env_ids
    def setup_randomization(self,cfg):raise ValueError('Brushless phase1 has no randomization')
    def apply_action(self,actions):
        if actions.shape!=(*self.shape,4):raise ValueError('Brushless input requires exact shape(*robotshape,4)')
        before=self.throttle.clone();self.last_motor=self.motor_model.step(actions,self.dt);self.throttle=self.motor_model.omega/2900.
        self.thrusts.zero_();self.thrusts[...,2]=self.last_motor['force_N'];self.forces.copy_(self.last_motor['force_B']);self.torques.copy_(self.last_motor['torque_B'])
        indices=torch.arange(self._view.count,device=self.device,dtype=torch.int32)
        # One COM wrench; lever-arm moments already included exactly once in torque_B.
        self._view._physics_view.apply_forces_and_torques_at_position(self.forces.reshape(-1,3),self.torques.reshape(-1,3),position_data=None,indices=indices,is_global=False)
        self.throttle_difference=(self.throttle-before).norm(dim=-1)
        return actions.sum(-1)
    def get_state(self,check_nan=False,env_frame=True):
        self.pos,self.rot=self._view.get_world_poses(clone=True);self.vel=self.vel_w=self._view.get_velocities(clone=True)
        if env_frame and hasattr(self,'_envs_positions'):self.pos=self.pos-self._envs_positions
        self.vel_b=torch.cat((quat_rotate_inverse(self.rot,self.vel[...,:3]),quat_rotate_inverse(self.rot,self.vel[...,3:])),dim=-1)
        self.heading=quat_axis(self.rot,axis=0);self.up=quat_axis(self.rot,axis=2)
        state=torch.cat((self.pos,self.rot,self.vel,self.heading,self.up,self.throttle*2-1),-1)
        if check_nan and not bool(torch.isfinite(state).all()):raise RuntimeError('Nonfinite brushless state')
        return state

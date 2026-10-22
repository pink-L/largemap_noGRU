"""Independent damping0.2 plant; force/motor/reset/state implementations inherited.

This is a controlled simulation ablation, not a new real-hardware calibration.
"""
from dataclasses import replace
from .crazyflie_brushless_paper_v1 import CrazyflieBrushlessPaperV1
from .multirotor import MultirotorBase
from omni_drones.robots.robot import ASSET_PATH
from omni_drones.robots.config import RobotCfg,RigidBodyPropertiesCfg
from omni_drones.actuators.brushless_paper import BrushlessPaperProfile

PROFILE='brushless_paper_reference_damping02_v1'

class CrazyflieBrushlessPaperDampedV1(CrazyflieBrushlessPaperV1):
    usd_path=ASSET_PATH+'/usd/crazyflie_brushless_paper_damped_v1.usda'
    param_path=ASSET_PATH+'/usd/crazyflie_brushless_paper_damped_v1.yaml'
    def __init__(self,name=None,cfg=None,is_articulation=False):
        if is_articulation:raise ValueError('Damped brushless plant remains singlebody/nonarticulated')
        cfg=cfg or RobotCfg(rigid_props=RigidBodyPropertiesCfg(linear_damping=.2,angular_damping=.2))
        if cfg.rigid_props.linear_damping!=.2 or cfg.rigid_props.angular_damping!=.2 or cfg.rigid_props.disable_gravity:
            raise ValueError('Damped profile requires exactlylinear/angular0.2 and gravityenabled')
        self.profile=replace(BrushlessPaperProfile(),profile=PROFILE)
        # Intentionally bypass only the zero-profile constructor's damping lock.
        # All initialized numerical motor/force/state/reset paths remain inherited.
        MultirotorBase.__init__(self,name,cfg,is_articulation=False)
        if float(self.dt)!=.002:raise ValueError('Damped profile requires physics dt=.002')

    def initialize(self,*args,**kwargs):
        super().initialize(*args,**kwargs)
        self.mass_contract_audit=dict(schema='single_effective_rigid_body_brushless_paper_v1',
            profile=PROFILE,mass_kg=.042,body_count_per_robot=1,rotor_joint_count=0,
            inertia_kg_m2=self.profile.inertia_kg_m2,linear_damping=.2,angular_damping=.2,
            shared_motor_law_profile=self.motor_model.profile.profile,
            motor_law_semantics='same paper motor numeric component; whole-vehicle damping02 identity is separate')

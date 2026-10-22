"""Explicit portable task; inherited numerical reset/reward/control implementation."""
import torch
from navacc_large_map_planar_env import NavAccNavRLSharedPlanar
from navacc_large_map_env import NavAccNavRLShared
from omni_drones.robots.robot import ASSET_PATH
from omni_drones.controllers.brushless_paper_adapter import BrushlessAccelerationToRPYT,BrushlessFirmwarePIDAdapter
from omni_drones.robots.drone.crazyflie_brushless_paper_damped_v1 import CrazyflieBrushlessPaperDampedV1
from .config import validate_config
class LargeMapNoGRU(NavAccNavRLSharedPlanar):
    @staticmethod
    def _validate_frozen_config(cfg):return validate_config(cfg)
    def __init__(self,cfg,headless=True):
        self._portable_constructed=False
        NavAccNavRLSharedPlanar.__init__(self,cfg,headless)
        self.acceleration_to_rpyt=self._make_acceleration_to_rpyt()
        self._assert_brushless_backend()
        self._portable_constructed=True
    def _validate_native_scene_protocol(self,scene,cfg):
        validate_config(cfg)
        if (len(scene.static_boxes),len(scene.dynamic_bodies))!=(cfg.task.large_map.static_count,cfg.task.large_map.dynamic_count):raise RuntimeError('Native count mismatch')
        if scene.deadline_s!=32. or scene.map_half_extent_m!=25. or scene.dynamic_period_s!=2.:raise RuntimeError('Native geometry/deadline differs')
    def _design_scene(self):
        return NavAccNavRLShared._design_scene(self)
    def _reset_idx(self,env_ids):
        if not self._portable_constructed:raise RuntimeError('Reset before construction completed')
        self._assert_brushless_backend()
        return NavAccNavRLSharedPlanar._reset_idx(self,env_ids)
    def _pre_sim_step(self,td):
        if not self._portable_constructed:raise RuntimeError('Step before construction completed')
        return NavAccNavRLSharedPlanar._pre_sim_step(self,td)
    def _make_firmware_pid(self):
        return BrushlessFirmwarePIDAdapter()

    def _make_acceleration_to_rpyt(self):
        return BrushlessAccelerationToRPYT()

    def _target_visual_usd_path(self):
        # This independent articulation is only the retained legacy marker.
        # The real brushless plant remains a single rigid body without joints.
        return ASSET_PATH+'/usd/cf2x_pybullet.usd'

    def _reset_firmware_motor_command(self,env_ids,attitude_quat_wxyz):
        # Plant._reset_idx already reset only these lanes to steady hoverΩ.
        # PID owns no motor dynamics here, so do not feed it throttle²/forcefraction.
        self.firmware_pid.reset(env_ids,attitude_quat_wxyz=attitude_quat_wxyz)
        command=self.drone.motor_model.omega[env_ids]/2900.
        if not bool(torch.isfinite(command).all()) or bool(((command<0)|(command>1)).any()):
            raise RuntimeError('Invalid direct brushless reset motor input')
        return command.clone()
    def _assert_brushless_backend(self):
        if type(self.acceleration_to_rpyt) is not BrushlessAccelerationToRPYT or type(self.firmware_pid) is not BrushlessFirmwarePIDAdapter:
            raise RuntimeError('Damped controller adapter overwritten')
        if self.acceleration_to_rpyt.config.check_tilt or self.acceleration_to_rpyt.config.mass_kg!=.042 or self.firmware_pid.config.apply_motor_dynamics:
            raise RuntimeError('Damped controller mass/tilt/duplicate lag mismatch')
        if type(self.drone) is not CrazyflieBrushlessPaperDampedV1 or self.drone.is_articulation:
            raise RuntimeError('Wrong robot for damping02 task')
        if not bool((self.drone._view.get_masses()-.042).abs().le(1e-6).all()):
            raise RuntimeError('Damped actual mass differs from42g')
        expected=self.drone.inertias.new_tensor([3.3e-5,3.6e-5,5.9e-5])
        actual=self.drone._view.get_inertias().reshape(*self.drone.shape,3,3).diagonal(0,-2,-1)
        if not torch.allclose(actual,expected.expand_as(actual),atol=1e-9,rtol=1e-5):
            raise RuntimeError('Damped actual inertia differs from identical zero backend inertia')
        com=self.drone._view.get_coms()[0]
        if not bool(torch.isfinite(com).all()) or not bool(com.abs().le(1e-8).all()):
            raise RuntimeError('Damped actual COM differs from unchanged zero COM')
        self._ensure_damped_static_audit(force=False)

    def _ensure_damped_static_audit(self,*,force=False):
        import omni.usd
        from pxr import PhysxSchema,Usd,UsdPhysics
        stage=omni.usd.get_context().get_stage()
        paths=tuple(self.drone.prim_paths)
        binding=(id(self.drone),id(self.drone._view),id(stage),paths)
        receipt=getattr(self,'damped_static_audit_receipt',None)
        if receipt is None:
            receipt={'schema':'damped_static_audit_cache_v1','full_scans':0,'construction_scans':0,
                'forced_end_scans':0,'cache_hits':0,'failed_scans':0}
            self.damped_static_audit_receipt=receipt
        if not force and getattr(self,'_damped_static_binding',None)==binding:
            receipt['cache_hits']+=1
            return
        # Invalidate before attempting a fresh audit: failed scans are not cached.
        self._damped_static_binding=None
        try:
            bodies=self._read_damped_static_bodies(stage,paths)
        except Exception:
            receipt['failed_scans']+=1;receipt['last_full_scan_status']='FAIL'
            raise
        receipt['full_scans']+=1
        receipt['forced_end_scans' if force else 'construction_scans']+=1
        receipt['last_full_scan_status']='PASS'
        receipt['bound_plant_count']=len(paths)
        receipt['binding_identity']={'drone':id(self.drone),'view':id(self.drone._view),
            'stage':id(stage),'prim_paths':list(paths)}
        self._damped_static_stage=stage  # retain actualstage identity while cached
        self._damped_static_binding=binding
        self.damped_native_audit={'schema':'damped02_native_admission_v1','status':'PASS',
            'RobotCfg_linear_damping':self.drone.cfg.rigid_props.linear_damping,
            'RobotCfg_angular_damping':self.drone.cfg.rigid_props.angular_damping,
            'mass42g_inertia_COM_unchanged_verified':True,'singlebody_nojoints_verified':True,
            'damping_evidence':'composed Usd PhysxRigidBody attributes plus actual RobotCfg; not an unavailable direct runtime damping getter claim',
            'bodies':bodies,'shared_motor_law_profile':self.drone.motor_model.profile.profile,
            'whole_vehicle_profile':'brushless_paper_reference_damping02_v1','static_audit_cache':receipt}

    def _read_damped_static_bodies(self,stage,paths):
        from pxr import PhysxSchema,Usd,UsdPhysics
        bodies=[]
        for path in paths:
            root=stage.GetPrimAtPath(path)
            prims=list(Usd.PrimRange(root))
            if sum(p.HasAPI(UsdPhysics.RigidBodyAPI) for p in prims)!=1 or any(p.IsA(UsdPhysics.Joint) for p in prims):
                raise RuntimeError('Damped plant must have exactlyonebody and no joints')
            physics=PhysxSchema.PhysxRigidBodyAPI(root)
            values=(physics.GetLinearDampingAttr().Get(),physics.GetAngularDampingAttr().Get())
            if any(abs(float(value)-.2)>1e-6 for value in values):
                raise RuntimeError('Composed physical damping does not match0.2')
            if root.GetAttribute('physxRigidBody:enableGyroscopicForces').Get() is not True:
                raise RuntimeError('Damped body Euler gyroscopic force setting changed')
            authored={}
            for name in ('physxRigidBody:linearDamping','physxRigidBody:angularDamping'):
                attr=root.GetAttribute(name)
                authored[name]=[{'layer':spec.layer.identifier,'path':str(spec.path)} for spec in attr.GetPropertyStack()]
            bodies.append({'path':str(path),'composed_linear_damping':float(values[0]),
                'composed_angular_damping':float(values[1]),'composed_gyro_enabled':True,'authored_property_stack':authored})
        return bodies

    def full_static_audit(self):
        self._assert_brushless_backend()
        self._ensure_damped_static_audit(force=True)
        return self.damped_static_audit_receipt

"""Portable resolved configuration; no campaign identities or archived paths."""
from pathlib import Path
import math
from omegaconf import OmegaConf
ROOT=Path(__file__).resolve().parents[1]
SCHEMA='largemap_no_gru_v1'
def load_config(path=None,*,purpose='train',envs=None):
    cfg=OmegaConf.load(path or ROOT/'config/baseline.yaml')
    cfg.portable.purpose=purpose
    if envs is not None:cfg.env.num_envs=cfg.task.env.num_envs=int(envs)
    mode='eval' if purpose in ('eval','smoke_eval') else 'train'
    cfg.task.large_map.scene_path=str(ROOT/f'resources/s280d50/{mode}/scene.json')
    cfg.task.large_map.evaluation_protocol='fixed_obstacle_course_hold_end_v1' if mode=='eval' else None
    cfg.seed=3725
    validate_config(cfg)
    return cfg

def validate_config(cfg):
    p=cfg.portable;large=cfg.task.large_map;o=cfg.task.observation;r=cfg.task.reward;a=cfg.algo
    checks=[p.schema==SCHEMA,cfg.task.name=='LargeMapNoGRU',p.purpose in ('train','eval','smoke','smoke_eval'),
        cfg.sim.dt==.002,cfg.sim.substeps==5,cfg.env.max_episode_length==3200,
        cfg.task.env.max_episode_length==3200,cfg.env.num_envs==cfg.task.env.num_envs,
        o.mode=='public21_goal_fixed_si_yaw144_dyn10_v1',o.state_dim==21,o.dynamic_slot_dim==10,
        o.velocity_scale==1.,o.gyro_body_scale==1.,o.static_frame=='actual_yaw_only',
        cfg.task.action.actor_frame=='goal_fixed_v1',list(cfg.task.action.acceleration_scale_m_s2)==[1.5,1.5,.5],
        cfg.task.action.nominal_max_norm_m_s2==1.6,large.c3bf_global_selected_row_max==1,
        large.execution_protocol=='nominal_continue_unbounded_raw_top1_v1',
        large.total_mass_kg==.042,large.mass_schema=='singlebody_total_mass_v1',
        cfg.task.drone_model.name=='CrazyflieBrushlessPaperDampedV1',
        p.physical_backend=='brushless_paper_reference_damping02_v1',
        large.episode_mode=='obstacle_course_continuous_v1',r.mode==large.episode_mode,
        large.mission_deadline_s==32.,r.deadline_s==32.,r.cbf_correction_coeff==3.,
        r.goal_radius_m==.5,r.arrival_bonus_once==80.,r.hold_rate==1.,r.velocity_rate==5.,
        r.execution_smooth_rate==.02,r.effort_rate==0.,r.failure_discount_compensation==False,
        a.std_mode=='state_smooth_v1',a.structured_state_dim==21,a.dynamic_slot_dim==10,
        a.probability_action_key=='z_policy',a.trajectory_boundary==True,
        a.train_every==128,a.num_minibatches==64,a.ppo_epochs==4,
        cfg.task.reset.easy_mission_probability==0.,cfg.task.reset.mission_sampling_schema=='navrl_edges_short_full00_xy_v1',
        cfg.task.reset.mission_min_separation_m==.5,cfg.task.reset.mission_max_separation_m==70.,
        all(math.isfinite(float(x))and float(x)>0 for x in (p.actor_lr,p.critic_lr))]
    if not all(checks):raise ValueError('Portable configuration invariant mismatch at checks '+str([i for i,v in enumerate(checks)if not v]))
    n=int(cfg.env.num_envs)
    if n<=0 or (p.purpose in ('train','eval') and n!=1024):raise ValueError('Formal train/eval require1024; explicit smoke permits subset')
    if not Path(large.scene_path).is_file():raise FileNotFoundError('Scene missing: '+str(large.scene_path))
    return {'schema':SCHEMA,'status':'PASS','purpose':p.purpose,'num_envs':n}

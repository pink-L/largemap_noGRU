"""Portable CPU/model constructor. No simulator or campaign imports."""
import torch
from scripts.navacc_large_map_planar_policy import PlanarPrimaryPPOPolicy
from scripts.navacc_large_map_planar_reward import PlanarRewardConfig
def reward_config_from_cfg(cfg) -> PlanarRewardConfig:
    from navacc_goal_curriculum import active as _arrival10_active,validate_config as _validate_arrival10
    if _arrival10_active(cfg):_validate_arrival10(cfg) #before actual numeric reward projection
    from navacc_large_map_planar_sampling3m import evaluation_deadline_s
    return PlanarRewardConfig(
        deadline_s=evaluation_deadline_s(cfg),
        mode=str(cfg.task.reward.get("mode","legacy_pbrs_v2")),
        velocity_rate=float(cfg.task.reward.get("velocity_rate",1.)),
        lidar_risk_rate=float(cfg.task.reward.get("lidar_risk_rate",.1)),
        goal_radius_m=float(cfg.task.reward.get("goal_radius_m",.2)),
        goal_brake_rate=float(cfg.task.reward.get("goal_brake_rate",.2)),
        control_dt_s=float(cfg.sim.dt) * int(cfg.sim.substeps),
        discount_time_constant_s=float(
            cfg.task.reward.discount_time_constant_s),
        gae_time_constant_s=float(cfg.task.reward.gae_time_constant_s),
        height_rate=float(cfg.task.reward.height_interval_rate_per_s),
        height_tolerance_m=float(
            cfg.task.reward.height_interval_tolerance_m),
        potential_scale=float(cfg.task.reward.potential_scale),
        arrival_bonus=float(cfg.task.reward.arrival_bonus_once),
        failure_cost=float(cfg.task.reward.failure_discounted_cost),
        pb_mode=str(cfg.task.reward.get("pb_mode","none")),
        oob_shaping_rate=float(cfg.task.reward.get("oob_shaping_rate",0.)),
        oob_xy_limit_m=float(cfg.task.reward.get("oob_xy_limit_m",24.)),
        near_goal_pbrs_rate=float(
            cfg.task.reward.get("near_goal_pbrs_rate",0.)),
        near_goal_pbrs_max_d_m=float(
            cfg.task.reward.get("near_goal_pbrs_max_d_m",2.5)),
        near_goal_pbrs_inner_m=float(
            cfg.task.reward.get("near_goal_pbrs_inner_m",.5)),
        time_rate=float(cfg.task.reward.get("time_rate",.05)),
        effort_rate=float(cfg.task.reward.get("effort_rate",.02)),
        execution_smooth_rate=float(cfg.task.reward.get("execution_smooth_rate",0.)),
        failure_discount_compensation=cfg.task.reward.get("failure_discount_compensation",True),
        hold_rate=float(cfg.task.reward.get("hold_rate",1.)),
        hold_speed_reference_m_s=float(cfg.task.reward.get("hold_speed_reference_m_s",.5)),
        cbf_correction_coeff=float(
            cfg.task.reward.get("cbf_correction_coeff",.03)))

def build_cpu_policy(cfg):
    """Construct the real planar policy layout without Isaac or CUDA."""
    from torchrl.data import Composite, UnboundedContinuous
    e = 1
    observation = Composite({"agents": Composite({"observation": Composite({
        "state": UnboundedContinuous((1, int(cfg.algo.get("structured_state_dim",16)))),
        "static_lidar": UnboundedContinuous((1, 36, 4)),
        "dynamic_obstacle": UnboundedContinuous((1, 4, int(cfg.algo.get("dynamic_slot_dim",8)))),
    })}), "info": Composite({
        "large_goal_distance_m": UnboundedContinuous((1,)),
    })}).expand(e)
    action = Composite({"agents": Composite({
        "action": UnboundedContinuous((1, 3))})}).expand(e)
    reward = Composite({"agents": Composite({
        "reward": UnboundedContinuous((1, 1))})}).expand(e)
    return PlanarPrimaryPPOPolicy(
        cfg.algo, observation, action, reward, "cpu",
        reward_config=reward_config_from_cfg(cfg))

def build_policy(cfg,device='cpu'):
    policy=build_cpu_policy(cfg)
    policy.device=torch.device(device)
    policy.to(device)
    for group in policy.actor_opt.param_groups:group['lr']=float(cfg.portable.actor_lr)
    for group in policy.critic_opt.param_groups:group['lr']=float(cfg.portable.critic_lr)
    return policy

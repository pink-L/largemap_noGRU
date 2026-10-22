#!/usr/bin/env python3
"""Structured PPO identity for primary-acceleration dense-planar training."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch
import torch.nn as nn
from tensordict.nn import TensorDictModule

from omni_drones.learning.ppo.common import GAE
from omni_drones.learning.ppo.navacc_v4_structured import (
    OBS_KEYS, NavAccV4PPOPolicy, StructuredEncoder, StructuredCritic)
from navacc_large_map_planar_reward import PlanarRewardConfig


POLICY_IDENTITY = \
    "navacc_planar_primary_acceleration_ppo_routegoal_distancecritic_v1"
CRITIC_SCHEMA = "structured208_plus_true_global_distance1_v1"
DISTANCE_KEY = ("info", "large_goal_distance_m")


class PlanarDistanceCritic(nn.Module):
    """Structured critic with one true-global-distance feature; actor unchanged."""

    def __init__(self, scene_max_distance_m: float):
        super().__init__()
        if not torch.isfinite(torch.tensor(scene_max_distance_m)) \
                or scene_max_distance_m <= 0:
            raise ValueError("scene_max_distance_m must be finite and positive")
        self.encoder = StructuredEncoder()
        self.scene_max_distance_m = float(scene_max_distance_m)
        self.head = nn.Sequential(
            nn.Linear(209, 256), nn.LeakyReLU(), nn.LayerNorm(256),
            nn.Linear(256, 256), nn.LeakyReLU(), nn.LayerNorm(256),
            nn.Linear(256, 256), nn.LeakyReLU(), nn.LayerNorm(256),
            nn.Linear(256, 1))

    def forward(self, state, static_lidar, dynamic_obstacle, goal_distance):
        encoded = self.encoder(state, static_lidar, dynamic_obstacle)
        expected = encoded.shape[:-2] + (1,)
        if goal_distance.shape != expected or not goal_distance.is_floating_point() \
                or not bool(torch.isfinite(goal_distance).all()) \
                or bool((goal_distance < 0).any()):
            raise ValueError(
                f"true global goal distance must be finite nonnegative {expected}")
        feature = goal_distance.to(encoded.dtype).unsqueeze(-2) \
            / self.scene_max_distance_m
        return self.head(torch.cat((encoded, feature), -1))


class PlanarPrimaryPPOPolicy(NavAccV4PPOPolicy):
    """Same 16/144/4x8 actor layout with a new action/reward identity."""

    def __init__(self, cfg, observation_spec, action_spec, reward_spec, device,
                 *, reward_config: PlanarRewardConfig = PlanarRewardConfig()):
        if getattr(cfg, "checkpoint_path", None):
            raise ValueError(
                "planar policy accepts actor-only warmstart after construction; "
                "full old PPO checkpoint restore is forbidden")
        super().__init__(cfg, observation_spec, action_spec, reward_spec, device)
        self.public20 = int(getattr(cfg, "structured_state_dim", 16)) in (20, 21)
        self.hold36 = reward_config.mode in ("hold30_continuous_v1","hold50_continuous_v1", "obstacle_course_continuous_v1")
        self.hold48 = reward_config.mode in ("hold48_v1","hold30_continuous_v1","hold50_continuous_v1", "obstacle_course_continuous_v1")
        if self.hold48 != (int(getattr(cfg,"structured_state_dim",16))==21):raise ValueError("hold48 requires exclusive public21 policy")
        if not self.public20 and DISTANCE_KEY not in observation_spec.keys(True, True):
            raise ValueError("planar distance critic requires info.large_goal_distance_m")
        self.critic = TensorDictModule(
            (StructuredCritic(int(getattr(cfg,"structured_state_dim",16)),int(getattr(cfg,"dynamic_slot_dim",8))) if self.public20 else PlanarDistanceCritic(reward_config.scene_max_distance_m)),
            in_keys=list(OBS_KEYS) + ([] if self.public20 else [DISTANCE_KEY]),
            out_keys=["state_value"]).to(self.device)
        fake = observation_spec.zero(); self.critic(fake)
        def init(module):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, .01); nn.init.zeros_(module.bias)
        self.critic.apply(init)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=5e-4)
        continuation_lr = getattr(cfg, "hold36_continuation_lr", None)
        if continuation_lr is not None:
            if cfg.get("goal_curriculum_schema") in ("navacc_s280d50_h32_corr60_continue200m_10m_v1", "navacc_action_bounds_bcd_20m_split10m_v1"):
                if not self.hold36 or float(continuation_lr)!=0.0001 or float(cfg.lr)!=0.0001:raise ValueError("curriculum initial LR differs")
            elif not self.hold36 or float(continuation_lr)!=1e-4 or float(cfg.lr)!=1e-4:
                raise ValueError("unregistered Hold36 continuation learning rate")
            for optimizer in (self.actor_opt, self.critic_opt):
                for group in optimizer.param_groups:group["lr"]=float(continuation_lr)
        gamma = torch.tensor(reward_config.gamma, dtype=torch.float32).item()
        gae_lambda = torch.tensor(
            reward_config.gae_lambda, dtype=torch.float32).item()
        self.gae = GAE(gamma, gae_lambda).to(self.device)
        self.planar_policy_identity = ("navacc_zero_globalgoal_rawtop1_velocity_reward_v1" if reward_config.mode == "zero_curriculum_v1" else POLICY_IDENTITY)
        self.planar_critic_schema = CRITIC_SCHEMA
        if self.public20:
            from navacc_large_map_planar_obs20 import POLICY_IDENTITY as NEW_ID, CRITIC_SCHEMA as NEW_CRITIC
            self.planar_policy_identity = NEW_ID
            self.planar_critic_schema = NEW_CRITIC
            from navacc_large_map_planar_obs20 import GOAL_SCALE_MODE, GOAL_SCALE_POLICY_IDENTITY, GOAL_SCALE_CRITIC_SCHEMA
            self.planar_observation_mode = str(getattr(cfg, 'structured_observation_mode', 'public20_yaw_v1'))
            if self.planar_observation_mode == GOAL_SCALE_MODE:
                self.planar_policy_identity = GOAL_SCALE_POLICY_IDENTITY
                self.planar_critic_schema = GOAL_SCALE_CRITIC_SCHEMA
            elif self.planar_observation_mode == 'public21_xy1_z1_logd_yaw_hold48_v1':
                from navacc_large_map_hold48 import POLICY_IDENTITY as HOLD_ID, CRITIC_SCHEMA as HOLD_CRITIC
                if not self.hold48 or int(cfg.structured_state_dim)!=21:raise ValueError('hold48 policy/reward mismatch')
                self.planar_policy_identity=HOLD_ID;self.planar_critic_schema=HOLD_CRITIC
                if self.hold36:
                    from navacc_hold30_continuous import POLICY_IDENTITY as HOLD36_ID
                    self.planar_policy_identity=HOLD36_ID
            elif self.planar_observation_mode == 'public21_goal_fixed_si_yaw144_dyn10_v1':
                from navacc_goal_frame_profile import POLICY_IDENTITY as GOAL_POLICY,CRITIC_SCHEMA as GOAL_CRITIC,CONTRACT
                if not self.hold36 or int(getattr(cfg,'dynamic_slot_dim',0))!=10:raise ValueError('goal-frame requiresHold30/dyn10')
                self.planar_policy_identity=GOAL_POLICY;self.planar_critic_schema=GOAL_CRITIC;self.goal_frame_contract=dict(CONTRACT)
            elif self.planar_observation_mode != 'public20_yaw_v1':
                raise ValueError('unsupported public20 policy observation mode')
        self.resolved_discount = {"gamma": float(gamma),
                                  "gae_lambda": float(gae_lambda),
                                  "control_dt_s": reward_config.control_dt_s}
        self.std_mode = str(getattr(cfg, "std_mode", "legacy_hard_clamp_v1"))
        if self.std_mode != "legacy_hard_clamp_v1":
            from omni_drones.learning.ppo.navacc_smooth_std import MODES, SCHEMA
            if not self.hold48 or self.std_mode not in MODES:
                raise ValueError("smooth std requires Hold48 and a registered mode")
            self.std_schema = SCHEMA
            self.planar_policy_identity += ":" + self.std_mode
        self.actor_warmstart = None






"""Structured PPO for NavACC V4 (three observation leaves, no ravel adapter)."""
from __future__ import annotations

import torch
import torch.nn as nn
from tensordict.nn import TensorDictModule, TensorDictModuleBase
from torchrl.modules import ProbabilisticActor

from omni_drones.learning.ppo.ppo import PPOPolicy, Actor
from omni_drones.learning.modules.distributions import IndependentNormal
from omni_drones.learning.utils.valuenorm import ValueNorm1
from omni_drones.learning.ppo.common import GAE


OBS_KEYS = (
    ("agents", "observation", "state"),
    ("agents", "observation", "static_lidar"),
    ("agents", "observation", "dynamic_obstacle"),
)
SCHEMA = "NavAcc-V4-Structured-state16-static36x4-dynamic4x8-v1"


class StructuredEncoder(nn.Module):
    """NavRL-style static CNN plus dynamic MLP, preserving prefix/agent axes."""
    output_dim = 208

    def __init__(self, state_dim=16, dynamic_slot_dim=8):
        super().__init__()
        if state_dim not in (16, 20, 21):
            raise ValueError("unsupported structured state width")
        if dynamic_slot_dim not in (8,10):raise ValueError("unsupported dynamic slot width")
        self.dynamic_slot_dim = int(dynamic_slot_dim)
        self.state_dim = state_dim
        self.output_dim = 192 + state_dim
        self.static = nn.Sequential(
            nn.Conv2d(1, 4, (5, 3), stride=(1, 1), padding=(2, 1)), nn.ELU(),
            nn.Conv2d(4, 16, (5, 3), stride=(2, 1), padding=(2, 1)), nn.ELU(),
            nn.Conv2d(16, 16, (5, 3), stride=(2, 2), padding=(2, 1)), nn.ELU(),
            nn.Flatten(), nn.Linear(288, 128), nn.LayerNorm(128),
        )
        self.dynamic = nn.Sequential(
            nn.Flatten(), nn.Linear(4*self.dynamic_slot_dim, 128), nn.ELU(),
            nn.Linear(128, 64), nn.ELU(),
        )

    def _validate(self, state, static_lidar, dynamic_obstacle):
        if state.shape[-2:] != (1, self.state_dim):
            raise ValueError(f"V4 state must end in [1,16], got {tuple(state.shape)}")
        if static_lidar.shape[-3:] != (1, 36, 4):
            raise ValueError("V4 static_lidar must end in [1,36,4]")
        if dynamic_obstacle.shape[-3:] != (1, 4, self.dynamic_slot_dim):
            raise ValueError("V4 dynamic_obstacle must end in [1,4,8]")
        if state.shape[:-2] != static_lidar.shape[:-3] \
                or state.shape[:-2] != dynamic_obstacle.shape[:-3]:
            raise ValueError("V4 observation branches have different prefixes")
        if any(x.device != state.device or x.dtype != state.dtype
               for x in (static_lidar, dynamic_obstacle)):
            raise ValueError("V4 observation branches must share device/dtype")
        if not all(x.is_floating_point() for x in (state, static_lidar, dynamic_obstacle)):
            raise TypeError("V4 observation branches must be floating point")

    def forward(self, state, static_lidar, dynamic_obstacle):
        self._validate(state, static_lidar, dynamic_obstacle)
        prefix = state.shape[:-2]
        static_feature = self.static(static_lidar.reshape(-1, 1, 36, 4))
        dynamic_feature = self.dynamic(dynamic_obstacle.reshape(-1, 1, 4, self.dynamic_slot_dim))
        state_feature = state.reshape(-1, self.state_dim)
        fused = torch.cat((static_feature, dynamic_feature, state_feature), -1)
        return fused.reshape(prefix + (1, self.output_dim))


class StructuredActor(nn.Module):
    def __init__(self, action_dim: int, state_dim=16, std_mode="legacy_hard_clamp_v1", dynamic_slot_dim=8):
        super().__init__()
        self.encoder = StructuredEncoder(state_dim, dynamic_slot_dim)
        self.head = nn.Sequential(
            nn.Linear(192 + state_dim, 256), nn.LeakyReLU(), nn.LayerNorm(256),
            nn.Linear(256, 256), nn.LeakyReLU(), nn.LayerNorm(256),
            nn.Linear(256, 256), nn.LeakyReLU(), nn.LayerNorm(256),
            (Actor(action_dim) if std_mode == "legacy_hard_clamp_v1" else
             __import__("omni_drones.learning.ppo.navacc_smooth_std", fromlist=["SmoothStdActor"]).SmoothStdActor(action_dim, std_mode)),
        )

    def forward(self, state, static_lidar, dynamic_obstacle):
        return self.head(self.encoder(state, static_lidar, dynamic_obstacle))


class StructuredCritic(nn.Module):
    def __init__(self, state_dim=16, dynamic_slot_dim=8):
        super().__init__()
        self.encoder = StructuredEncoder(state_dim, dynamic_slot_dim)
        self.head = nn.Sequential(
            nn.Linear(192 + state_dim, 256), nn.LeakyReLU(), nn.LayerNorm(256),
            nn.Linear(256, 256), nn.LeakyReLU(), nn.LayerNorm(256),
            nn.Linear(256, 256), nn.LeakyReLU(), nn.LayerNorm(256),
            nn.Linear(256, 1),
        )

    def forward(self, state, static_lidar, dynamic_obstacle):
        return self.head(self.encoder(state, static_lidar, dynamic_obstacle))


def validate_structured_spec(observation_spec, state_dim=16, dynamic_slot_dim=8) -> None:
    expected = {OBS_KEYS[0]: (1, state_dim), OBS_KEYS[1]: (1, 36, 4),
                OBS_KEYS[2]: (1, 4, dynamic_slot_dim)}
    missing = [key for key in OBS_KEYS if key not in observation_spec.keys(True, True)]
    if missing:
        flat = observation_spec.get(("agents", "observation"), None)
        width = getattr(flat, "shape", (None,))[-1] if flat is not None else None
        if width == 80:
            raise ValueError("legacy flat80 observation/checkpoint is forbidden for NavACC V4")
        raise ValueError(f"V4 structured observation leaves missing: {missing}")
    for key, suffix in expected.items():
        if tuple(observation_spec[key].shape[-len(suffix):]) != suffix:
            raise ValueError(f"V4 spec {key} has wrong shape {observation_spec[key].shape}")


class NavAccV4PPOPolicy(PPOPolicy):
    """PPO update semantics from PPOPolicy with independent V4 encoders."""
    def __init__(self, cfg, observation_spec, action_spec, reward_spec, device):
        TensorDictModuleBase.__init__(self)
        state_dim = int(getattr(cfg, "structured_state_dim", 16))
        dynamic_slot_dim=int(getattr(cfg,"dynamic_slot_dim",8))
        validate_structured_spec(observation_spec, state_dim, dynamic_slot_dim)
        self.cfg, self.device = cfg, torch.device(device)
        if bool(getattr(cfg, "priv_actor", False)) or bool(getattr(cfg, "priv_critic", False)):
            raise ValueError("NavACC V4 does not support privileged flat encoders")
        if str(getattr(cfg, "critic_input_mode", "legacy")) != "legacy":
            raise ValueError("NavACC V4 critic_input_mode must be legacy")
        if str(getattr(cfg, "probability_action_key", "action")) != "z_policy":
            raise ValueError("NavACC V4 probability action must be agents.z_policy")
        self.critic_input_mode = "legacy"
        self.probability_action_key = ("agents", "z_policy")
        self.use_trajectory_boundary = True
        self.entropy_coef = float(getattr(cfg, "entropy_coef", .001))
        self.clip_param = .1
        self.critic_loss_fn = nn.HuberLoss(delta=10)
        self.n_agents, self.action_dim = action_spec[("agents", "action")].shape[-2:]
        self.gae = GAE(.99, .95)
        actor_module = TensorDictModule(
            StructuredActor(self.action_dim, state_dim, str(getattr(cfg, "std_mode", "legacy_hard_clamp_v1")), dynamic_slot_dim), in_keys=list(OBS_KEYS), out_keys=["loc", "scale"])
        self.actor = ProbabilisticActor(
            module=actor_module, in_keys=["loc", "scale"],
            out_keys=[("agents", "action")], distribution_class=IndependentNormal,
            return_log_prob=True, log_prob_key="sample_log_prob").to(self.device)
        self.critic = TensorDictModule(
            StructuredCritic(state_dim, dynamic_slot_dim), in_keys=list(OBS_KEYS), out_keys=["state_value"]).to(self.device)
        fake = observation_spec.zero()
        self.actor(fake); self.critic(fake)
        self.value_norm = ValueNorm1(reward_spec[("agents", "reward")].shape[-2:]).to(self.device)
        if getattr(cfg, "checkpoint_path", None):
            state = torch.load(str(cfg.checkpoint_path), map_location=self.device, weights_only=True)
            self.load_state_dict(state, strict=True)
        else:
            def init(module):
                if isinstance(module, nn.Linear):
                    nn.init.orthogonal_(module.weight, .01); nn.init.zeros_(module.bias)
            self.actor.apply(init); self.critic.apply(init)
        self.actor_opt = torch.optim.Adam((p for p in self.actor.parameters() if p.requires_grad), lr=5e-4)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=5e-4)


__all__ = ["SCHEMA", "OBS_KEYS", "StructuredEncoder", "StructuredActor",
           "StructuredCritic", "validate_structured_spec", "NavAccV4PPOPolicy"]

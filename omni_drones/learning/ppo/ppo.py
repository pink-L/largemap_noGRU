# MIT License
#
# Copyright (c) 2023 Botian Xu, Tsinghua University
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


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

from torchrl.modules import ProbabilisticActor
from torchrl.envs.transforms import CatTensors
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase, TensorDictModule, TensorDictSequential

from hydra.core.config_store import ConfigStore
from dataclasses import dataclass
from typing import Union
import einops

from ..utils.valuenorm import ValueNorm1
from ..modules.distributions import IndependentNormal
from .common import GAE
from omni_drones.utils.torchrl.compat import CompositeSpec, TensorSpec
from omni_drones.utils.privileged_critic import critic_input_mode

@dataclass
class PPOConfig:
    name: str = "ppo"
    train_every: int = 32
    ppo_epochs: int = 4
    num_minibatches: int = 16
    entropy_coef: float = 0.001   # [M2 2026-09-04] was hardcoded; CLI: algo.entropy_coef=

    # whether to use privileged information
    priv_actor: bool = False
    priv_critic: bool = False
    # legacy preserves priv_critic -> intrinsics; explicit modes never redefine it.
    critic_input_mode: str = "legacy"

    checkpoint_path: Union[str, None] = None
    # Opt-in for transforms that replace agents.action before collection.  The
    # default preserves every existing PPO task and checkpoint contract.
    probability_action_key: str = "action"
    trajectory_boundary: bool = False

cs = ConfigStore.instance()
cs.store("ppo", node=PPOConfig, group="algo")
cs.store("ppo_priv", node=PPOConfig(priv_actor=True, priv_critic=True), group="algo")
cs.store("ppo_priv_critic", node=PPOConfig(priv_critic=True), group="algo")


@torch.no_grad()
def dynamic_cbf_advantage_stats(tensordict: TensorDict, raw_adv: torch.Tensor):
    """Read-only rollout attribution; absent legacy info yields no metrics."""
    cache = tensordict.get(("next", "info", "dynamic_cbf"), None)
    if cache is None:
        return {}
    required = ("cache_valid", "intervened", "delta_u")
    if any(key not in cache.keys() for key in required):
        return {}
    valid = cache["cache_valid"].bool()
    active = cache["intervened"].bool() & valid
    delta_norm = cache["delta_u"].norm(dim=-1)

    def mean_where(value, mask):
        while mask.ndim < value.ndim:
            mask = mask.unsqueeze(-1)
        mask = mask.expand_as(value) & torch.isfinite(value)
        count = mask.sum()
        return torch.where(count > 0, torch.where(mask, value, 0.).sum()
                           / count.clamp_min(1), value.new_zeros(()))

    valid_count = valid.sum()
    probability = torch.where(
        valid_count > 0, active.sum().to(raw_adv.dtype)
        / valid_count.clamp_min(1).to(raw_adv.dtype), raw_adv.new_zeros(()))
    return {
        "cbf_active_probability": probability,
        "cbf_active_count": active.sum().to(raw_adv.dtype),
        "cbf_inactive_count": (valid & ~active).sum().to(raw_adv.dtype),
        "cbf_delta_u_norm_given_active": mean_where(delta_norm, active),
        "raw_advantage_given_cbf_active": mean_where(raw_adv, active),
        "raw_advantage_given_cbf_inactive": mean_where(raw_adv, valid & ~active),
    }


@torch.no_grad()
def ppo_observation_metrics(
    logratio: torch.Tensor,
    ratio: torch.Tensor,
    entropy: torch.Tensor,
    entropy_loss: torch.Tensor,
    values: torch.Tensor,
    returns: torch.Tensor,
    clip_param: float,
    learning_rate: float,
):
    """Return detached diagnostics from tensors already used by a PPO update."""
    # [DLC15 2026-10-03] Passive logratio distribution telemetry: quantiles and
    # extrema over the minibatch, so a KL spike can be traced to the shape of
    # the probability-ratio tail without changing any PPO computation.  The
    # reductions read tensors that were already produced by the update.
    logratio_flat = logratio.detach().float().reshape(-1)
    quantiles = torch.quantile(
        logratio_flat,
        logratio_flat.new_tensor([0.001, 0.01, 0.5, 0.99, 0.999]))
    return {
        "entropy": entropy.mean(),
        "entropy_loss": entropy_loss.detach(),
        "approx_kl": ((ratio - 1.) - logratio).mean(),
        "clip_fraction": ((ratio - 1.).abs() > clip_param).float().mean(),
        "ratio_mean": ratio.mean(),
        "logratio_min": logratio_flat.min(),
        "logratio_max": logratio_flat.max(),
        "logratio_p001": quantiles[0],
        "logratio_p01": quantiles[1],
        "logratio_p50": quantiles[2],
        "logratio_p99": quantiles[3],
        "logratio_p999": quantiles[4],
        # The critic and return target use ValueNorm's normalized training scale.
        "value_mean_normalized": values.mean(),
        "return_mean_normalized": returns.mean(),
        "learning_rate": values.new_tensor(learning_rate),
    }


def make_mlp(num_units):
    layers = []
    for n in num_units:
        layers.append(nn.LazyLinear(n))
        layers.append(nn.LeakyReLU())
        layers.append(nn.LayerNorm(n))
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, action_dim: int) -> None:
        super().__init__()
        self.actor_mean = nn.LazyLinear(action_dim)
        self.actor_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, features: torch.Tensor):
        loc = self.actor_mean(features)
        # [2026-09-08] std 上限 clamp 防熵暴涨: log_std ∈ [ln(0.05), ln(1.8)] -> std ∈ [0.05, 1.8]
        #   (reward 收紧梯度弱时 entropy 正则曾把 std 无限推大 -> 熵 30+ -> 动作被限幅成全随机 ->
        #    μ(mean) 学习信号被噪声淹没. 有界 std 保探索可控, 熵封顶, reward 能把 std 收到低做确定性化)
        log_std = self.actor_std.clamp(min=-2.9957, max=0.5878)   # 0.05 / 1.8
        scale = torch.exp(log_std).expand_as(loc)
        return loc, scale


class IntrinsicsEncoder(nn.Module):
    """Flatten only the named vehicle intrinsic leaves, preserving their meaning."""
    def __init__(self, spec):
        super().__init__()
        self.keys = list(spec.keys(True, True))
        width = sum(spec[key].shape[-1] for key in self.keys)
        self.net = nn.Sequential(nn.LayerNorm(width), make_mlp([64, 64]))

    def forward(self, intrinsics):
        return self.net(torch.cat([intrinsics[key] for key in self.keys], -1))


class PPOPolicy(TensorDictModuleBase):

    def __init__(
        self,
        cfg: PPOConfig,
        observation_spec: CompositeSpec,
        action_spec: CompositeSpec,
        reward_spec: TensorSpec,
        device
    ):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.critic_input_mode = critic_input_mode(cfg)
        probability_leaf = str(getattr(cfg, "probability_action_key", "action"))
        if not probability_leaf or "." in probability_leaf:
            raise ValueError("probability_action_key must be an agents leaf name")
        self.probability_action_key = ("agents", probability_leaf)
        self.use_trajectory_boundary = bool(
            getattr(cfg, "trajectory_boundary", False)
        ) or self.critic_input_mode == "obstacle_privileged"
        if self.critic_input_mode == "obstacle_privileged":
            width = observation_spec["agents", "critic_state"].shape[-1]
            if observation_spec["agents", "observation"].shape[-1] != 70 or (width - 70) % 13 or width <= 70:
                raise ValueError("Invalid NavVelObstacleCritic-v1 spec")

        # [M2 2026-09-04] entropy_coef now configurable (default 0.001, unchanged behaviour)
        self.entropy_coef = getattr(cfg, "entropy_coef", 0.001)
        self.clip_param = 0.1
        self.critic_loss_fn = nn.HuberLoss(delta=10)
        # torchrl >= 0.6: Composite.shape only carries batch dim; use leaf spec shape
        self.n_agents, self.action_dim = action_spec[("agents", "action")].shape[-2:]
        self.gae = GAE(0.99, 0.95)

        fake_input = observation_spec.zero()

        if self.cfg.priv_actor:
            intrinsics_dim = observation_spec[("agents", "intrinsics")].shape[-1]
            actor_module = TensorDictSequential(
                TensorDictModule(make_mlp([128, 128]), [("agents", "observation")], ["feature"]),
                TensorDictModule(
                    nn.Sequential(nn.LayerNorm(intrinsics_dim), make_mlp([64, 64])),
                    [("agents", "intrinsics")], ["context"]
                ),
                CatTensors(["feature", "context"], "feature"),
                TensorDictModule(
                    nn.Sequential(make_mlp([256, 256]), Actor(self.action_dim)),
                    ["feature"], ["loc", "scale"]
                )
            )
        else:
            actor_module=TensorDictModule(
                nn.Sequential(make_mlp([256, 256, 256]), Actor(self.action_dim)),
                [("agents", "observation")], ["loc", "scale"]
            )
        self.actor: ProbabilisticActor = ProbabilisticActor(
            module=actor_module,
            in_keys=["loc", "scale"],
            out_keys=[("agents", "action")],
            distribution_class=IndependentNormal,
            return_log_prob=True,
            log_prob_key="sample_log_prob",
        ).to(self.device)

        if self.critic_input_mode == "intrinsics":
            intrinsics_spec = observation_spec[("agents", "intrinsics")]
            intrinsics_dim = intrinsics_spec.shape[-1]
            intrinsic_encoder = (IntrinsicsEncoder(intrinsics_spec)
                if isinstance(intrinsics_spec, CompositeSpec) else
                nn.Sequential(nn.LayerNorm(intrinsics_dim), make_mlp([64, 64])))
            self.critic = TensorDictSequential(
                TensorDictModule(make_mlp([128, 128]), [("agents", "observation")], ["feature"]),
                TensorDictModule(
                    intrinsic_encoder,
                    [("agents", "intrinsics")], ["context"]
                ),
                CatTensors(["feature", "context"], "feature"),
                TensorDictModule(
                    nn.Sequential(make_mlp([256, 256]), nn.LazyLinear(1)),
                    ["feature"], ["state_value"]
                )
            ).to(self.device)
        else:
            self.critic = TensorDictModule(
                nn.Sequential(make_mlp([256, 256, 256]), nn.LazyLinear(1)),
                [("agents", "critic_state") if self.critic_input_mode == "obstacle_privileged"
                 else ("agents", "observation")], ["state_value"]
            ).to(self.device)

        self.actor(fake_input)
        self.critic(fake_input)

        # Register buffers before loading a full checkpoint, including ValueNorm.
        self.value_norm = ValueNorm1(reward_spec[("agents", "reward")].shape[-2:]).to(self.device)

        if self.cfg.checkpoint_path is not None:
            state_dict = torch.load(self.cfg.checkpoint_path, map_location=self.device)
            self.load_state_dict(state_dict, strict=self.critic_input_mode == "obstacle_privileged")
        else:
            def init_(module):
                if isinstance(module, nn.Linear):
                    nn.init.orthogonal_(module.weight, 0.01)
                    nn.init.constant_(module.bias, 0.)

            self.actor.apply(init_)
            self.critic.apply(init_)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=5e-4)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=5e-4)

    def __call__(self, tensordict: TensorDict):
        self.actor(tensordict)
        if self.probability_action_key != ("agents", "action"):
            # Copy before an inverse action transform can replace agents.action.
            tensordict.set(
                self.probability_action_key,
                tensordict[("agents", "action")].clone(),
            )
        self.critic(tensordict)
        tensordict.exclude("loc", "scale", "feature", inplace=True)
        return tensordict

    def train_op(self, tensordict: TensorDict):
        next_tensordict = tensordict["next"]
        with torch.no_grad():
            next_values = self.critic(next_tensordict)["state_value"]
        rewards = tensordict[("next", "agents", "reward")]
        bootstrap_terminal = tensordict[("next", "terminated")]
        if getattr(self, 'hold48', False):
            explicit = tensordict.get(('next','info','hold48_bootstrap_terminal'), None)
            expected = bootstrap_terminal | tensordict['next','truncated']
            if explicit is None or explicit.dtype != torch.bool or explicit.shape != expected.shape or not torch.equal(explicit, expected):
                raise ValueError('hold48 requires explicit finite-horizon bootstrap mask')
            bootstrap_terminal = explicit
        dones = einops.repeat(
            bootstrap_terminal,
            "t e 1 -> t e a 1",
            a=self.n_agents
        )
        values = tensordict["state_value"]
        values = self.value_norm.denormalize(values)
        next_values = self.value_norm.denormalize(next_values)

        if self.use_trajectory_boundary:
            # Collector evidence: [B,T], names=[None,'time']; never infer from
            # einops letters. Reject a transposed/flattened rollout before GAE.
            if tensordict.ndim != 2 or tensordict.names[1] != "time":
                raise ValueError("obstacle privileged PPO requires [B,T] named time at dim=1")
            boundary = (tensordict["next", "done"] |
                        tensordict["next", "truncated"] |
                        tensordict["next", "terminated"]).clone()
            # Also cut manually reset trajectories when collector IDs reveal them.
            ids = tensordict.get(("collector", "traj_ids"), None)
            if ids is not None:
                boundary[:, :-1] |= (ids[:, 1:] != ids[:, :-1]).unsqueeze(-1)
            boundary = boundary.unsqueeze(-2).expand_as(dones)
            adv, ret = self.gae(rewards, dones, values, next_values,
                                trajectory_boundary=boundary)
        else:
            adv, ret = self.gae(rewards, dones, values, next_values)
        cbf_advantage_metrics = dynamic_cbf_advantage_stats(tensordict, adv)
        plateau_hook = getattr(self, "plateau_diagnostic_hook", None)
        raw_adv = adv.detach().clone() if plateau_hook is not None else None
        adv_mean = adv.mean()
        adv_std = adv.std()
        adv = (adv - adv_mean) / adv_std.clip(1e-7)
        if plateau_hook is not None:
            plateau_hook.begin(tensordict, raw_adv, adv, values, next_values, dones)
        self.value_norm.update(ret)
        ret = self.value_norm.normalize(ret)

        tensordict.set("adv", adv)
        tensordict.set("ret", ret)

        infos = []
        for epoch in range(self.cfg.ppo_epochs):
            batch = make_batch(tensordict, self.cfg.num_minibatches)
            for minibatch in batch:
                infos.append(self._update(minibatch))
            if plateau_hook is not None:
                plateau_hook.epoch(epoch)

        infos: TensorDict = torch.stack(infos).to_tensordict()
        infos = infos.apply(torch.mean, batch_size=[])
        result = {k: v.item() for k, v in infos.items()}
        result.update({k: v.item() for k, v in cbf_advantage_metrics.items()})
        if plateau_hook is not None:
            plateau_hook.finish(result)
        return result

    def _update(self, tensordict: TensorDict):
        dist = self.actor.get_dist(tensordict)
        log_probs = dist.log_prob(tensordict[self.probability_action_key])
        entropy = dist.entropy()

        adv = tensordict["adv"]
        logratio = log_probs - tensordict["sample_log_prob"]
        ratio = torch.exp(logratio).unsqueeze(-1)
        surr1 = adv * ratio
        surr2 = adv * ratio.clamp(1.-self.clip_param, 1.+self.clip_param)
        policy_loss = - torch.mean(torch.min(surr1, surr2)) * self.action_dim
        entropy_loss = - self.entropy_coef * torch.mean(entropy)

        b_values = tensordict["state_value"]
        b_returns = tensordict["ret"]
        values = self.critic(tensordict)["state_value"]
        values_clipped = b_values + (values - b_values).clamp(
            -self.clip_param, self.clip_param
        )
        value_loss_clipped = self.critic_loss_fn(b_returns, values_clipped)
        value_loss_original = self.critic_loss_fn(b_returns, values)
        value_loss = torch.max(value_loss_original, value_loss_clipped)

        loss = policy_loss + entropy_loss + value_loss
        self.actor_opt.zero_grad()
        self.critic_opt.zero_grad()
        loss.backward()
        actor_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.actor.parameters(), 5)
        critic_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.critic.parameters(), 5)
        self.actor_opt.step()
        self.critic_opt.step()
        explained_var = 1 - F.mse_loss(values, b_returns) / b_returns.var()
        metrics = ppo_observation_metrics(
            logratio=logratio.unsqueeze(-1),
            ratio=ratio,
            entropy=entropy,
            entropy_loss=entropy_loss,
            values=values,
            returns=b_returns,
            clip_param=self.clip_param,
            learning_rate=self.actor_opt.param_groups[0]["lr"],
        )
        metrics.update({
            "policy_loss": policy_loss.detach(),
            "value_loss": value_loss.detach(),
            "actor_grad_norm": actor_grad_norm.detach(),
            "critic_grad_norm": critic_grad_norm.detach(),
            "explained_var": explained_var.detach(),
        })
        return TensorDict(metrics, [])


def make_batch(tensordict: TensorDict, num_minibatches: int):
    tensordict = tensordict.reshape(-1)
    perm = torch.randperm(
        (tensordict.shape[0] // num_minibatches) * num_minibatches,
        device=tensordict.device,
    ).reshape(num_minibatches, -1)
    for indices in perm:
        yield tensordict[indices]

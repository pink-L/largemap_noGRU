"""Opt-in bounded Gaussian heads; latent probability semantics stay in PPO."""
import torch
from torch import nn

LOW, HIGH = -2.9957, 0.5878
MID, HALF = (LOW + HIGH) / 2, (HIGH - LOW) / 2
MODES = ('global_smooth_v1', 'state_smooth_v1', 'fixed_smooth_v1')
SCHEMA = 'navacc_smooth_std_abc_v1'


def inverse_parent(raw):
    if not torch.isfinite(raw).all():
        raise ValueError('non-finite parent std')
    effective = raw.clamp(LOW, HIGH)
    if bool(((effective <= LOW) | (effective >= HIGH)).any()):
        raise ValueError('parent effective std touches boundary; migration refused')
    result = torch.atanh((effective - MID) / HALF)
    if not torch.isfinite(result).all():
        raise ValueError('non-finite inverse parent std')
    return effective, result


class SmoothStdActor(nn.Module):
    def __init__(self, action_dim, mode):
        super().__init__()
        if mode not in MODES:
            raise ValueError('unknown std mode')
        self.mode = mode
        # Preserve the historical mean key and its exact computation.
        self.actor_mean = nn.LazyLinear(action_dim)
        if mode == 'state_smooth_v1':
            self.std_head = nn.Linear(256, action_dim)
        elif mode == 'fixed_smooth_v1':
            self.register_buffer('std_raw', torch.zeros(action_dim))
        else:
            self.std_raw = nn.Parameter(torch.zeros(action_dim))

    def raw(self, features):
        return self.std_head(features) if self.mode == 'state_smooth_v1' else self.std_raw.expand(features.shape[:-1] + self.std_raw.shape)

    def statistics(self, features):
        raw = self.raw(features)
        tangent = raw.tanh()
        log_std = MID + HALF * tangent
        return raw, log_std, HALF * (1 - tangent.square())

    def forward(self, features):
        loc = self.actor_mean(features)
        _, log_std, _ = self.statistics(features)
        return loc, log_std.exp().expand_as(loc)

    @torch.no_grad()
    def initialize_parent(self, raw):
        effective, inverse = inverse_parent(raw)
        if self.mode == 'state_smooth_v1':
            self.std_head.weight.zero_()
            self.std_head.bias.copy_(inverse)
        else:
            self.std_raw.copy_(inverse)
        return effective, inverse

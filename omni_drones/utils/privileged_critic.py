"""NavVelObstacleCritic-v1: actor information plus fixed-M cylinder truth.

No simulator dependency. Inputs are environment-local world coordinates.
Scales are PROVISIONAL, fixed (no clipping/running feature normalization).
Slot: p(3)/5m, v(3)/1m/s, a(3)/1m/s², radius/1m,
half_height/2.5m, active, is_dynamic. No identity or global origin feature.
"""
import torch

SCHEMA = "NavVelObstacleCritic-v1"
SLOT_DIM = 13
ACTOR_DIM = 70


def critic_state_dim(capacity):
    if capacity < 1:
        raise ValueError("critic capacity must be positive")
    return ACTOR_DIM + SLOT_DIM * capacity


def critic_input_mode(cfg):
    mode = getattr(cfg, "critic_input_mode", "legacy")
    if mode == "legacy":
        return "intrinsics" if cfg.priv_critic else "symmetric"
    if mode not in ("symmetric", "intrinsics", "obstacle_privileged"):
        raise ValueError(f"Unknown critic_input_mode: {mode}")
    if cfg.priv_critic and mode != "intrinsics":
        raise ValueError("priv_critic retains intrinsics semantics; conflicting mode")
    if mode == "obstacle_privileged" and cfg.priv_actor:
        raise ValueError("obstacle_privileged requires priv_actor=false")
    return mode


def build_critic_state(observation, *, position, velocity, acceleration,
                       radius, half_height, active, is_dynamic):
    """Return fresh storage, including the actor prefix; inactive NaN-safe zeros."""
    batch, capacity = active.shape
    if observation.shape != (batch, 1, ACTOR_DIM):
        raise ValueError("critic requires NavVelDynamicActor-v1 [B,1,70]")
    for name, value, shape in (
        ("position", position, (batch, capacity, 3)),
        ("velocity", velocity, (batch, capacity, 3)),
        ("acceleration", acceleration, (batch, capacity, 3)),
        ("radius", radius, (batch, capacity)),
        ("half_height", half_height, (batch, capacity)),
        ("is_dynamic", is_dynamic, (batch, capacity)),
    ):
        if value.shape != shape or value.device != observation.device:
            raise ValueError(f"{name}: shape/device mismatch")
    if active.dtype != torch.bool or is_dynamic.dtype != torch.bool:
        raise TypeError("active/is_dynamic must be bool")
    slot = torch.cat((position / 5., velocity / 1., acceleration / 1.,
                      radius.unsqueeze(-1) / 1., half_height.unsqueeze(-1) / 2.5,
                      active.unsqueeze(-1), is_dynamic.unsqueeze(-1)), -1)
    slot = torch.where(active.unsqueeze(-1), slot, torch.zeros_like(slot))
    return torch.cat((observation, slot.reshape(batch, 1, SLOT_DIM * capacity)), -1)

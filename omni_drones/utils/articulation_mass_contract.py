"""Pure tensor checks for opt-in total articulation mass semantics."""
import torch

MASS_SCHEMA = "total_articulation_mass_v1"


def reject_mass_randomization(cfg):
    for phase in ("train", "eval"):
        if cfg.get(phase, {}).get("mass_scale", None) is not None:
            raise ValueError("total_articulation_mass_v1 does not support mass randomization")


def plan_total_mass(body_masses, base_index, target_total_kg):
    if body_masses.ndim < 2 or not body_masses.is_floating_point():
        raise ValueError("body masses must be floating [..., bodies]")
    if type(base_index) is not int or not 0 <= base_index < body_masses.shape[-1]:
        raise ValueError("invalid base body index")
    if not bool(torch.isfinite(body_masses).all()) or bool((body_masses <= 0).any()):
        raise ValueError("all physical body masses must be finite and positive")
    target = torch.as_tensor(target_total_kg, dtype=body_masses.dtype, device=body_masses.device)
    if target.numel() != 1 or not bool(torch.isfinite(target)) or float(target) <= 0:
        raise ValueError("target total mass must be finite positive scalar")
    others = body_masses.clone(); others[..., base_index] = 0
    base_target = target - others.sum(-1)
    if bool((base_target <= 1e-6).any()):
        raise ValueError("other bodies leave no physically positive base mass")
    proposed = body_masses.clone(); proposed[..., base_index] = base_target
    return proposed


def validate_total_mass_readback(before, after, base_index, target_total_kg, atol=1e-8):
    expected = plan_total_mass(before, base_index, target_total_kg)
    if after.shape != before.shape or not bool(torch.isfinite(after).all()) or bool((after <= 0).any()):
        raise ValueError("mass readback invalid")
    mask = torch.ones(before.shape[-1], dtype=torch.bool, device=before.device);mask[base_index]=False
    if not torch.equal(before[...,mask],after[...,mask]):
        raise ValueError("non-base body masses changed")
    if not torch.allclose(after,expected,atol=atol,rtol=0):
        raise ValueError("base/total mass readback differs from contract")
    if not torch.allclose(after.sum(-1),torch.full_like(after.sum(-1),float(target_total_kg)),atol=atol,rtol=0):
        raise ValueError("total articulation mass differs from target")
    return after.sum(-1)

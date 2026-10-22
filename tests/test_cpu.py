import sys,json,hashlib
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf
from largemap_no_gru.config import load_config,validate_config,ROOT
from largemap_no_gru.model import build_policy,reward_config_from_cfg
from largemap_no_gru.checkpoint import tensor_digest

def test_real_cpu_import_and_model():
    p=build_policy(load_config());assert len(p.state_dict())==63
    assert not any(k.startswith(('isaacsim','omni.isaac'))for k in sys.modules)
    assert not any(isinstance(m,(torch.nn.GRU,torch.nn.LSTM,torch.nn.RNN))for m in p.modules())
    assert [g['lr']for g in p.actor_opt.param_groups]==[5e-6]
    assert [g['lr']for g in p.critic_opt.param_groups]==[5e-5]
    assert p.gae.gamma.item()==.999833345413208

def test_config_hard_rejects():
    c=load_config();c.task.reward.cbf_correction_coeff=6
    import pytest
    with pytest.raises(ValueError):validate_config(c)

def test_same_scene_and_request_suite():
    inputs=json.loads((ROOT/'provenance/RESOURCE_SOURCES.json').read_text())
    for path,row in inputs.items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==row['sha256']
    s=json.loads((ROOT/'resources/s280d50/eval/scene.json').read_text());assert len(s['mission']['starts'])==1024;assert len(s['static_boxes'])==280;assert len(s['dynamic_bodies'])==50

def test_noise_oracle():
    from largemap_no_gru.evaluation import _CrossEvalActionNoise
    noise=_CrossEvalActionNoise(93728,1024,'cpu')
    for _ in range(3200):noise.draw()
    assert noise.rolling_sha256.hexdigest()==json.loads((ROOT/'resources/action_noise_oracles.json').read_text())['93728']

def test_finite_horizon_and_goal_frame():
    from scripts.navacc_goal_frame import goal_to_world,world_to_goal
    v=torch.tensor([[1.,2.,3.],[-1.,2.,1.]]);theta=torch.tensor([0.,np.pi/2])
    assert torch.allclose(world_to_goal(goal_to_world(v,theta),theta),v,atol=1e-6)
    from omni_drones.learning.ppo.common import GAE
    r=torch.tensor([[[[1.]],[[2.]]]]);v=torch.zeros_like(r);done=torch.tensor([[[[False]],[[True]]]])
    adv,ret=GAE(.9,1.)(r,done,v,v,trajectory_boundary=done)
    assert torch.allclose(ret,torch.tensor([[[[2.8]],[[2.]]]]))

def test_all_lanes_early_terminal_oracle_prefix():
    from largemap_no_gru.evaluation import _CrossEvalActionNoise,validate_noise_receipt
    noise=_CrossEvalActionNoise(93728,1024,'cpu')
    for _ in range(17):noise.draw()
    # Complete first-terminal suite may legitimately finish before3200 physics ticks.
    mock={'completed':1024,'censored':0,'action_noise_ticks':17,'action_noise_sha256':noise.rolling_sha256.hexdigest()}
    r=validate_noise_receipt(mock,93728,1024,'unused-full-oracle')
    assert r['actual_physics_ticks']==17 and r['oracle_scope']=='actual_tick_prefix'
    assert noise.draw_count==17

def test_original_snapshot_numeric_golden():
    import dataclasses
    from scripts import navacc_large_map_planar_reward as reward
    from scripts import navacc_large_map_planar_c3bf as bounds
    from scripts import navacc_large_map_planar_unbounded as cbf
    from omni_drones.controllers.brushless_paper_adapter import BrushlessAccelerationToRPYT
    fixture=json.loads((ROOT/'resources/numeric_regression.json').read_text())
    def decode_values(x):
        if isinstance(x,list):return[decode_values(v)for v in x]
        return float(x)if isinstance(x,str)and x in ('NaN','Infinity','-Infinity')else x
    def unpack(x):return torch.tensor(decode_values(x['values']),dtype=getattr(torch,x['dtype'].split('.')[-1])).reshape(x['shape'])
    def check(actual,expected):
        if torch.is_tensor(actual):torch.testing.assert_close(actual,unpack(expected),rtol=0,atol=0,equal_nan=True)
        elif dataclasses.is_dataclass(actual):
            for key,value in vars(actual).items():check(value,expected[key])
        elif isinstance(actual,dict):
            for key,value in actual.items():check(value,expected[key])
        elif isinstance(actual,(list,tuple)):
            for value,other in zip(actual,expected):check(value,other)
        else:assert actual==expected
    inputs={k:torch.tensor(v,dtype=torch.long if k in ('previous_dwell_steps','completed_elapsed_steps')else torch.bool if isinstance(v[0],bool)else torch.float32)for k,v in fixture['reward_inputs'].items()}
    r=reward.planar_reward_step(**inputs,config=reward.PlanarRewardConfig(**fixture['reward_config']));check(r,fixture['reward_expected'])
    config=bounds.PlanarC3BFConfig(**fixture['bounds_config']);nom=bounds.bound_primary_acceleration(torch.tensor(fixture['raw_nominal']),config);check(nom,fixture['bounded_expected'])
    args={k:unpack(v)for k,v in fixture['cbf_inputs'].items()};check(cbf.unbounded_top1(**args,nominal_bounds=config),fixture['cbf_expected'])
    check(BrushlessAccelerationToRPYT()(nom[:,None,:],torch.tensor(fixture['converter_yaw'])[:,None]),fixture['converter_expected'])

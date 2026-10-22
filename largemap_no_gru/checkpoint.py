"""Explicit one-time U153 import and self-contained learning-state checkpoints."""
from pathlib import Path
from copy import deepcopy
import hashlib,json,os
import torch
from .config import ROOT,SCHEMA
U153_SHA='76ceb7d6f85c46a8694d7fe638b38f1da3bb1337e3e0d41a6daaebdaaf7df527'
CP_SCHEMA='largemap_no_gru_checkpoint_v1'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def source_manifest():
    paths=[]
    for dirname in ('largemap_no_gru','scripts','omni_drones','resources','config'):
        paths.extend(p for p in (ROOT/dirname).rglob('*')if p.is_file()and '__pycache__'not in p.parts and p.suffix!='.pyc')
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)}
def atomic_json(path,x):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('x')as f:json.dump(x,f,indent=2,default=str);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
def tensor_digest(state):
    h=hashlib.sha256()
    for name,tensor in sorted(state.items()):
        t=tensor.detach().cpu().contiguous();h.update(name.encode());h.update(str((t.dtype,tuple(t.shape))).encode());h.update(t.numpy().tobytes())
    return h.hexdigest()
def named_optimizer(policy,label,state=None):
    opt=getattr(policy,label+'_opt');state=opt.state_dict()if state is None else state
    names={id(p):n for n,p in getattr(policy,label).named_parameters()}
    result={}
    if len(state['param_groups'])!=len(opt.param_groups):raise ValueError('Optimizer group mismatch')
    for src_group,dst_group in zip(state['param_groups'],opt.param_groups):
        if len(src_group['params'])!=len(dst_group['params']):raise ValueError('Optimizer parameter count mismatch')
        for sid,p in zip(src_group['params'],dst_group['params']):
            name=names[id(p)];s=state['state'].get(sid,{})
            if s and any(k not in s for k in ('step','exp_avg','exp_avg_sq')):raise ValueError('Incomplete Adam state')
            if s and (s['exp_avg'].shape!=p.shape or s['exp_avg_sq'].shape!=p.shape):raise ValueError('Adam named tensor shape mismatch:'+name)
            result[name]={k:(v.detach().cpu().clone()if torch.is_tensor(v)else v)for k,v in s.items()}
    return result

def assert_equal(a,b):
    if type(a)!=type(b):raise ValueError('Type mismatch')
    if isinstance(a,dict):
        if a.keys()!=b.keys():raise ValueError('Key mismatch')
        for k in a:assert_equal(a[k],b[k])
    elif torch.is_tensor(a):
        if a.dtype!=b.dtype or a.shape!=b.shape or not torch.equal(a,b):raise ValueError('Tensor mismatch')
    elif isinstance(a,(list,tuple)):
        if len(a)!=len(b):raise ValueError('Length mismatch')
        for x,y in zip(a,b):assert_equal(x,y)
    elif a!=b:raise ValueError('Value mismatch')

def restore(policy,payload):
    if len(payload['policy'])!=63:raise ValueError('Expected63tensor complete policy')
    policy.load_state_dict(payload['policy'],strict=True)
    for label in ('actor','critic'):
        state=payload[label+'_optimizer'];before=named_optimizer(policy,label,state)
        getattr(policy,label+'_opt').load_state_dict(deepcopy(state))
        assert_equal(before,named_optimizer(policy,label))
    if tensor_digest(policy.state_dict())!=tensor_digest(payload['policy']):raise ValueError('Policy roundtrip mismatch')
    return policy

def save(path,policy,*,ledger,lineage,config,source=None):
    from omegaconf import OmegaConf
    path=Path(path)
    if path.exists():raise FileExistsError(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    x={'schema':CP_SCHEMA,'model_schema':'state21_static36x4_dyn4x10_state_smooth_v1_noGRU',
       'physics_schema':'brushless_paper_reference_damping02_v1','policy':deepcopy(policy.state_dict()),
       'actor_optimizer':deepcopy(policy.actor_opt.state_dict()),'critic_optimizer':deepcopy(policy.critic_opt.state_dict()),
       'config':OmegaConf.to_container(config,resolve=True),'ledger':deepcopy(ledger),'lineage':deepcopy(lineage),
       'source_sha256':source or source_manifest(),'resume_semantics':'full learning state; fresh physics/RNG/clock/cohort; not bitwise trajectory resume',
       'rng':{'torch_cpu':torch.get_rng_state()},'named_optimizer_parameter_order':{k:list(named_optimizer(policy,k))for k in ('actor','critic')}}
    tmp=path.with_suffix('.tmp')
    with tmp.open('xb')as f:torch.save(x,f);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path);return {'path':str(path),'sha256':sha(path),'ledger':ledger}

def load(path,policy):
    x=torch.load(path,map_location='cpu',weights_only=False)
    if x.get('schema')!=CP_SCHEMA or x.get('physics_schema')!='brushless_paper_reference_damping02_v1':raise ValueError('Import legacy U153 explicitly before using this checkpoint')
    if x.get('source_sha256')!=source_manifest():raise ValueError('Checkpoint implementation/resource manifest differs; explicit migration required')
    restore(policy,x);return x

def import_u153(source,output,cfg):
    from .model import build_policy
    source=Path(source);output=Path(output)
    if sha(source)!=U153_SHA:raise ValueError('Only exact documented U153 checkpoint is accepted')
    x=torch.load(source,map_location='cpu',weights_only=False)
    if x.get('schema')!='navacc_R76_corr30_continue20M_checkpoint_v1':raise ValueError('U153 schema differs')
    if x['ledger']['update']!=153 or x['identity']['actor_lr']!=5e-6:raise ValueError('U153 identity differs')
    policy=build_policy(cfg);restore(policy,x)
    for label,lr in (('actor',5e-6),('critic',5e-5)):
        if any(g['lr']!=lr for g in getattr(policy,label+'_opt').param_groups):raise ValueError('Actual source optimizer LR differs')
    lineage={'source_checkpoint_sha256':U153_SHA,'source_schema':x['schema'],'source_ledger':x['ledger'],
             'source_identity':x['identity'],'migration':'strict63policy_namedAdam_VN_GAE_learning_transfer',
             'historical_source_identity_preserved':True,'old_runtime_imported':False,'historical_training_reproduced':False}
    result=save(output,policy,ledger={'local_updates':0,'local_frames':0,'source_displayed_frames':x['ledger']['displayed_global_frames']},lineage=lineage,config=cfg)
    receipt={**result,'status':'PASS','source_sha256':U153_SHA,'policy_tensor_count':63,'policy_sha256':tensor_digest(policy.state_dict()),
             'optimizers':{k:{'named_parameters':len(named_optimizer(policy,k)),'steps':sorted({float(v['step'])for v in named_optimizer(policy,k).values()if v})}for k in ('actor','critic')},
             'proof':'Exact approved source bytes; strict model+both named Adam moments; VN/GAE included in63 tensors. Not an old whole-runtime or performance replay.'}
    atomic_json(output.with_suffix('.import.json'),receipt);return receipt

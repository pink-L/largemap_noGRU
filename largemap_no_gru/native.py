"""Native launch and complete-batch training. Imported only after explicit CLI action."""
from pathlib import Path
import json,random,math,traceback,hashlib
import numpy as np
import torch
from .config import validate_config
from .checkpoint import atomic_json,source_manifest,save,load,tensor_digest
from .model import build_policy

def subset_scene(cfg,out):
    n=int(cfg.env.num_envs)
    if n==1024:return
    data=json.loads(Path(cfg.task.large_map.scene_path).read_text())
    for key in ('starts','goals'):data['mission'][key]=data['mission'][key][:n]
    atomic_json(out/'diagnostic_scene.json',data)
    cfg.task.large_map.scene_path=str(out/'diagnostic_scene.json')
    cfg.task.large_map.cohort_target_size=n

def seed_all(env,seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);env.set_seed(seed)

def run(args,cfg):
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    app=None;collector=None;policy=None;wandb_run=None;wandb_status={'requested':args.wandb_mode=='online','finished':False,'error':None}
    initial=source_manifest()
    try:
        subset_scene(cfg,out);validate_config(cfg);atomic_json(out/'effective_config.json',__import__('omegaconf').OmegaConf.to_container(cfg,resolve=True))
        from omni_drones import init_simulation_app
        app=init_simulation_app(cfg)
        from .environment import LargeMapNoGRU
        from navacc_large_map_planar_training import build_planar_training_env
        base=LargeMapNoGRU(cfg,headless=True);env,transform=build_planar_training_env(base)
        is_eval=args.command=='eval'
        base._planar_fixed_eval=is_eval
        env.eval()if is_eval else env.train()
        policy=build_policy(cfg,base.device);payload=None
        if args.checkpoint:payload=load(args.checkpoint,policy)
        for label,override in (('actor',args.actor_lr),('critic',args.critic_lr)):
            if override is not None:
                for g in getattr(policy,label+'_opt').param_groups:g['lr']=override
        seed_all(env,args.seed)
        atomic_json(out/'PLAN.json',{'schema':'largemap_no_gru_run_v1','command':args.command,'purpose':cfg.portable.purpose,'seed':args.seed,'source_sha256':initial,
            'checkpoint':str(args.checkpoint)if args.checkpoint else None,'fresh_physics_clock_rng_cohort':True,'historical_campaign_local_KL_guard_inherited':False})
        if args.wandb_mode=='online':
            try:
                import wandb
                wandb_run=wandb.init(project=args.wandb_project,name=args.run_name,dir=str(out),config={'command':args.command,'purpose':cfg.portable.purpose})
                wandb_status['url']=wandb_run.url
            except Exception as e:wandb_status['error']=repr(e)
        if is_eval:
            from .evaluation import run_fixed_zero_evaluation
            before=tensor_digest(policy.state_dict())
            with torch.no_grad():result=run_fixed_zero_evaluation(env,policy,base,3200,progress_path=out/'evaluation_progress.json',action_mode='random'if args.action_mode=='sampled'else'deterministic',action_seed=args.action_seed)
            if tensor_digest(policy.state_dict())!=before:raise RuntimeError('Evaluation changed model or ValueNorm')
            if args.action_mode=='sampled':
                oracles=json.loads((Path(__file__).resolve().parents[1]/'resources/action_noise_oracles.json').read_text())
                expected=oracles[str(args.action_seed)]
                from .evaluation import validate_noise_receipt
                result['noise_oracle_validation']=validate_noise_receipt(result,args.action_seed,1024,expected)
            if result['requests']!=1024 or result['completed']!=1024 or result['censored']!=0:raise RuntimeError('Incomplete formal evaluation')
            atomic_json(out/'evaluation.json',result);ledger=None
        else:
            from omni_drones.utils.torchrl.collector import SyncDataCollector
            total=args.updates*base.num_envs*128
            collector=SyncDataCollector(env,policy=policy,frames_per_batch=base.num_envs*128,total_frames=total,device=base.device,return_same_td=True)
            iterator=iter(collector);prior=payload['ledger']if payload else {'local_updates':0,'local_frames':0,'source_displayed_frames':0}
            lineage=payload['lineage']if payload else {'initialization':'from_scratch','source_checkpoint_sha256':None}
            before_update={k:tensor_digest(getattr(policy,k).state_dict())for k in ('actor','critic')}
            initial_steps={k:[float(s['step'])for s in getattr(policy,k+'_opt').state.values()if 'step'in s]for k in ('actor','critic')}
            for u in range(1,args.updates+1):
                batch=next(iterator).to_tensordict()
                if tuple(batch.batch_size)!=(base.num_envs,128):raise RuntimeError('Rollout shape differs')
                finite=bool(torch.isfinite(batch['next','info','drone_state']).all());substeps=bool((batch['next','info','substeps_last_outer']==5).all())
                if not finite or not substeps:raise RuntimeError('Nonfinite rollout or substeps mismatch')
                metrics=policy.train_op(batch)
                if not all(math.isfinite(float(v))for v in metrics.values()):raise RuntimeError('Nonfinite PPO loss')
                ledger={'local_updates':prior['local_updates']+u,'local_frames':prior['local_frames']+u*base.num_envs*128,'source_displayed_frames':prior.get('source_displayed_frames',0)}
                row={'update':u,'ledger':ledger,'metrics':metrics,'actor_lr':[g['lr']for g in policy.actor_opt.param_groups],'critic_lr':[g['lr']for g in policy.critic_opt.param_groups]}
                with(out/'metrics.jsonl').open('a')as f:f.write(json.dumps(row)+'\n');f.flush();__import__('os').fsync(f.fileno())
                if u==1:atomic_json(out/'FIRST_BATCH_AUDIT.json',{'status':'PASS','finite':finite,'substeps_all5':substeps,'learning_batch':list(batch.batch_size),'initial_optimizer_steps':initial_steps,'actor_weights_changed':tensor_digest(policy.actor.state_dict())!=before_update['actor'],'critic_weights_changed':tensor_digest(policy.critic.state_dict())!=before_update['critic'],**row})
                if u%args.save_every==0 or u==args.updates:
                    marker=save(out/'checkpoints'/f'update_{ledger["local_updates"]:06d}.pt',policy,ledger=ledger,lineage=lineage,config=cfg,source=initial)
                    with(out/'CHECKPOINTS.jsonl').open('a')as f:f.write(json.dumps(marker)+'\n');f.flush();__import__('os').fsync(f.fileno())
                atomic_json(out/'PROGRESS.json',row)
                if wandb_run:
                    try:wandb_run.log(row,step=ledger['local_updates'])
                    except Exception as e:wandb_status['error']=repr(e)
                print(f'update={u}/{args.updates} frames={ledger["local_frames"]}',flush=True)
            atomic_json(out/'INTERRUPTED_EPISODES.json',{'count':sum(x is not None for x in base._hold_cohort.active),'reason':'phase_boundary','not_success_or_failure':True})
        if args.command=='smoke':
            from .diagnostics import smoke_checks
            smoke_checks(env,base,policy,out)
            restored=build_policy(cfg,base.device);load(marker['path'],restored)
            if tensor_digest(policy.state_dict())!=tensor_digest(restored.state_dict()):raise RuntimeError('Native checkpoint save/restore differs')
        audit=base.full_static_audit()
        if source_manifest()!=initial:raise RuntimeError('Source/resource changed during run')
        if wandb_run:
            try:wandb_run.finish();wandb_status['finished']=True
            except Exception as e:wandb_status['error']=repr(e)
        if wandb_status['error']:atomic_json(out/'WANDB_PENDING.json',wandb_status)
        atomic_json(out/'COMPLETION.json',{'status':'COMPLETE','command':args.command,'purpose':cfg.portable.purpose,'ledger':ledger,'final_static_audit':audit,'source_bytes_unchanged':True,'wandb':wandb_status})
        return 0
    except BaseException as exc:
        atomic_json(out/'FAILURE.json',{'error':repr(exc),'traceback':traceback.format_exc()});raise
    finally:
        if collector:
            try:collector.shutdown()
            except Exception:pass
        if app:app.close()

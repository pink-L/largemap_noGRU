"""python -m largemap_no_gru --help"""
from pathlib import Path
import argparse,json

def main():
    parser=argparse.ArgumentParser(description='Independent brushless LargeMap feed-forward PPO')
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('import-checkpoint');p.add_argument('--source',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p=sub.add_parser('model-check');p.add_argument('--checkpoint',type=Path);p.add_argument('--device',choices=['cpu'],default='cpu')
    for cmd in ('train','eval','smoke'):
        p=sub.add_parser(cmd);p.add_argument('--native-worker',action='store_true',help=argparse.SUPPRESS);p.add_argument('--config',type=Path);p.add_argument('--output',required=True,type=Path)
        if cmd in ('train','smoke'):
            g=p.add_mutually_exclusive_group(required=True);g.add_argument('--from-scratch',action='store_true');g.add_argument('--resume',dest='checkpoint',type=Path)
            p.add_argument('--updates',type=int,default=1);p.add_argument('--save-every',type=int,default=38)
        else:p.add_argument('--checkpoint',required=True,type=Path);p.set_defaults(updates=0)
        p.add_argument('--seed',type=int,default=3725);p.add_argument('--actor-lr',type=float);p.add_argument('--critic-lr',type=float)
        p.add_argument('--wandb-mode',choices=['off','online'],default='off');p.add_argument('--wandb-project',default='LargeMapNoGRU');p.add_argument('--run-name')
        if cmd=='eval':p.add_argument('--action-mode',choices=['sampled','mean'],default='sampled');p.add_argument('--action-seed',type=int,choices=[93728,93729,93730],default=93728);p.add_argument('--suite',choices=['s280d50-full'],default='s280d50-full')
        if cmd=='smoke':p.add_argument('--envs',type=int,choices=[32],default=32);p.add_argument('--steps',type=int,choices=[128],default=128)
    args=parser.parse_args()
    from .config import load_config
    from .model import build_policy
    from .checkpoint import import_u153,load,tensor_digest
    if args.command=='import-checkpoint':print(json.dumps(import_u153(args.source,args.output,load_config()),indent=2));return 0
    if args.command=='model-check':
        import sys
        policy=build_policy(load_config());payload=load(args.checkpoint,policy)if args.checkpoint else None
        print(json.dumps({'status':'PASS','policy_tensors':len(policy.state_dict()),'policy_sha256':tensor_digest(policy.state_dict()),'checkpoint_loaded':payload is not None,'isaac_imported':any(k.startswith(('isaacsim','omni.isaac'))for k in sys.modules)}));return 0
    if args.command=='eval'and args.seed!=3725:parser.error('Formal fixed suite requires environment seed3725')
    if args.command=='smoke'and args.updates!=1:parser.error('smoke permits exactly one update')
    if args.command!='eval'and (args.updates<=0 or args.save_every<=0):parser.error('Positive updates/save-every required')
    purpose='smoke'if args.command=='smoke'else args.command;cfg=load_config(args.config,purpose=purpose,envs=32 if args.command=='smoke'else None);cfg.seed=args.seed
    for k in ('actor_lr','critic_lr'):
        if getattr(args,k) is not None:cfg.portable[k]=getattr(args,k)
    if not args.native_worker:
        # Kit may terminate its worker with exit0 during shutdown even on exceptions.
        # The pure parent binds scientific success to durable completion/failure markers.
        import subprocess,sys
        completed=subprocess.run([sys.executable,'-m','largemap_no_gru',*sys.argv[1:],'--native-worker'])
        output=Path(args.output)
        marker=output/'COMPLETION.json'
        accepted=marker.is_file()and not(output/'FAILURE.json').exists()and json.loads(marker.read_text()).get('status')=='COMPLETE'
        return 0 if completed.returncode==0 and accepted else 1
    from .native import run
    return run(args,cfg)
if __name__=='__main__':raise SystemExit(main())

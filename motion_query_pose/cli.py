"""Portable entry points around the frozen architecture and training loop."""
import argparse,json,os
from pathlib import Path

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['inspect','preflight','calibrate','train','predict'])
    parser.add_argument('--data-dir',type=Path,required=True,help='Prepared cache root; read only')
    parser.add_argument('--output-dir',type=Path,default=Path('runs/current'))
    parser.add_argument('--backend-dir',type=Path,help='Project exposing aps.backend and the native rasterizer')
    parser.add_argument('--scene-dir',type=Path,help='Reconstruction root; required for native preflight')
    parser.add_argument('--methods',nargs='+',choices=['Full','NoRender'],default=['NoRender','Full'])
    parser.add_argument('--splits',nargs='+',choices=['R1','R2','R3'],default=['R1','R2','R3'])
    parser.add_argument('--seeds',nargs='+',type=int,choices=[0,1,2],default=[0,1,2])
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--split',choices=['R1','R2','R3'],default='R1')
    parser.add_argument('--part',choices=['train','val','test'],default='test')
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda',help='Prediction only; fitting/native preflight require CUDA')
    args=parser.parse_args()
    for env,value in [('MQP_DATA_DIR',args.data_dir),('MQP_OUTPUT_DIR',args.output_dir),('MQP_BACKEND_DIR',args.backend_dir),('MQP_SCENE_DIR',args.scene_dir)]:
        if value is not None:os.environ[env]=str(value.resolve())
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    from .settings import DATA,ROOT,RENDER,CONFIG,ALIASES
    from .data import sha,save_json
    if DATA==ROOT or DATA in ROOT.parents:parser.error('Use an output directory outside the prepared data directory')
    from .validation import inspect_cache
    audit=inspect_cache(DATA,args.splits if args.command!='predict' else [args.split])
    if args.command=='inspect':print(json.dumps(audit,indent=2));return
    ROOT.mkdir(parents=True,exist_ok=True)
    for folder in ['weights','curves','predictions']:(ROOT/folder).mkdir(exist_ok=True)
    if args.command=='predict':
        if args.checkpoint is None:parser.error('--checkpoint is required')
        from .inference import predict_checkpoint
        print(json.dumps(predict_checkpoint(args.checkpoint,args.split,args.part,args.device),indent=2));return
    # Avoid creating a CUDA context before detecting other GPU users.
    from .experiment import wait_for_resources,runtime,tensors
    wait_for_resources(args.command)
    if args.command in ['preflight','calibrate']:
        if args.backend_dir is None:parser.error('--backend-dir is required for native rendering')
        if args.command=='preflight':
            if args.scene_dir is None:parser.error('--scene-dir is required for native equivalence checks')
            from .preflight import main as run
        else:
            receipt=json.loads((ROOT/'ours_preflight.json').read_text())
            assert receipt.get('complete') and receipt['data_manifest_sha256']==sha(DATA/'data_manifest.json')
            from .calibrate import main as run
        run();return
    runtime()
    import fcntl,torch
    lock=(ROOT/'training.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    from .training import train_job
    renderer=None;scale=None
    if 'Full' in args.methods:
        if args.backend_dir is None:parser.error('--backend-dir is required for Full training')
        for file in ['ours_preflight.json','render_scale.json']:
            receipt=json.loads((ROOT/file).read_text())
            assert receipt.get('complete') and receipt['data_manifest_sha256']==sha(DATA/'data_manifest.json')
        from .rendering import NativeRenderer,Supervision
        renderer=NativeRenderer();scale=json.loads((ROOT/'render_scale.json').read_text())
    plan={'methods':args.methods,'splits':args.splits,'seeds':args.seeds,'config':CONFIG,'render_config':RENDER,
          'data_manifest_sha256':sha(DATA/'data_manifest.json'),'cache_audit':audit,
          'code_sha256':{p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}}
    save_json(ROOT/'run_manifest.json',plan);completed=[]
    for split in args.splits:
        d=tensors(split);sup=None
        if renderer is not None:
            sup=Supervision(renderer,d['arrays']);sup.denominator=torch.tensor(scale['splits'][split]['denominator'],device='cuda')
        for method in args.methods:
            internal={v:k for k,v in ALIASES.items()}[method]
            for seed in args.seeds:
                completed.append(train_job(internal,split,seed,d,len(completed),sup))
                save_json(ROOT/'training_completed.json',{'training':completed,'planned_jobs':len(args.splits)*len(args.seeds)*len(args.methods)})
        del d,sup;torch.cuda.empty_cache()

if __name__=='__main__':main()

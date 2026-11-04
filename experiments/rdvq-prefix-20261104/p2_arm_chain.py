"""One finite matched arm: train2000, dev, restore, train4000, dev/test.

Every child exits before the next phase. Unknown failure stops only this arm.
No repeated optimizer step or receiver stage is automatically scheduled.
"""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time,traceback
from owned_process import run_role,proc_identity

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def write(p,v):
    p=Path(p);assert not p.exists();p.write_text(json.dumps(v,indent=2));return p

def main():
    rp=Path(sys.argv[1]);r=json.loads(rp.read_text());L=Path(r['root']);S=Path(__file__).resolve().parent;O=Path(r['output_root']);O.mkdir(exist_ok=False)
    (O/'owner.json').write_text(json.dumps(dict(**proc_identity(os.getpid()),started=time.time())))
    assert r['reserved_gpu_hours']==19 and json.loads(Path(r['p1_gate']).read_text())['passed']
    for p,h in r['bindings'].items():assert sha(p)==h,p
    baseenv=os.environ.copy();gpu=r['gpu'];baseenv.update(CUDA_VISIBLE_DEVICES=str(gpu['index']),CUDA_DEVICE_ORDER='PCI_BUS_ID',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',MAX_JOBS='2',PYTHONNOUSERSITE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',
        PYTHONPATH=str(L/'environment/site')+':'+str(L/'dependencies/RDVQ-a2744eedd803d9e20b55b4b0c935ec211a07d184')+':'+str(S),TORCH_EXTENSIONS_DIR=str(L/'extensions_headers_v2'),CPATH=str(L/'dependencies/python310_headers/Include')+':'+str(L/'dependencies/python310_headers'))
    requests=O/'requests';requests.mkdir();ledger=[];eval_records=[]
    def child_owner(role,argv,bindings):
        # Child owner records GPU resources at the leaf, not this CPU wrapper.
        w=run_role(O,role,argv,dict(baseenv,CUDA_VISIBLE_DEVICES=''),list(range(180)),[],21600,bindings)
        assert w['returncode']==0 and w['child_gone'];return w
    def evaluate(checkpoint,step,split):
        name=f'eval_{step:05d}_{split}';root=O/name
        source=json.loads(Path(r['development_manifest' if split=='development' else 'test_manifest']).read_text())['sources']
        assert len(source)==(128 if split=='development' else 500)
        refs=[]
        for v in source:
            sid=v['source_id'];base={k:v[k] for k in ['source_id','split','name','path','sha256']}
            refs.append(dict(**base,reuse_p0=False,indices_path=str(root/'tx'/sid/'indices.npy'),lengths_path=str(root/'tx'/sid/'lengths.json'),source_rgb_path=str(root/'tx'/sid/'source_rgb.npy'),rx_directory=str(root/'rx'/sid),public_directory=str(root/'public'/sid)))
        manifest=write(requests/(name+'_manifest.json'),dict(sources=refs))
        req=dict(root=str(L),output_root=str(root),checkpoint=checkpoint,source_manifest=str(manifest),gpu_index=gpu['index'],gpu_uuid=gpu['uuid'],gpu_pci=gpu['pci'],cpus=gpu['cpus'],tx_seconds=1800,rx_seconds=3600 if split=='development' else 10800,rx_request=str(requests/(name+'_rx.json')))
        rx={k:v for k,v in req.items() if k not in ['source_manifest','rx_request']};rx['rx_denied_roots']=r['source_roots']+[str(requests)]
        write(req['rx_request'],rx);bound=dict(r['bindings'],**{str(manifest):sha(manifest),req['rx_request']:sha(req['rx_request']),checkpoint['path']:checkpoint['sha256']});req['bindings']=bound
        eq=write(requests/(name+'_request.json'),req);child_owner(name+'_owner_control',[sys.executable,'-u',str(S/'p1_shard_owner.py'),str(eq)],dict(bound,**{str(eq):sha(eq)}))
        result=json.loads((root/'complete.json').read_text());assert result['passed'];ledger.append(dict(stage=name,gpu_hours=result['gpu_hours']))
        score_root=O/(name+'_metrics');mq=write(requests/(name+'_metrics.json'),dict(output=str(score_root),manifest=str(manifest),metric_assets=r['metric_assets']))
        w=run_role(O,name+'_metrics',[sys.executable,'-u',str(S/'score_prefixes.py'),'--request',str(mq)],baseenv,gpu['cpus'],[gpu['uuid']],1800 if split=='development' else 5400,dict(bound,**{str(mq):sha(mq)}))
        ledger.append(dict(stage=name+'_metrics',gpu_hours=w['charged_gpu_device_hours']));assert json.loads((score_root/'complete.json').read_text())['passed']
        eval_records.append(dict(step=step,split=split,manifest=str(manifest),scores=str(score_root/'scores.csv'),checkpoint=checkpoint))
        (O/'resource_ledger.json').write_text(json.dumps(ledger,indent=2));(O/'evaluations.json').write_text(json.dumps(eval_records,indent=2))
    try:
        resume=None;train_hours=0
        for end in [2000,4000]:
            if (L/'STOP_P2').exists():write(O/'paused.json',dict(reason='user stop before next segment',completed_evaluations=eval_records));return
            t=dict(r['training'],root=str(L),output=str(O/f'train_to_{end:05d}'),gpu=gpu,bindings=r['bindings'],checkpoint_dir=r['checkpoint_dir'],updates=4000,segment_end=end,training_seconds=16800,reserved_gpu_hours=5,global_batch=16,microbatch=4)
            if resume:t['resume']=resume
            tq=write(requests/f'train_to_{end:05d}.json',t)
            child_owner(f'train_to_{end:05d}_owner_control',[sys.executable,'-u',str(S/'train_owner.py'),str(tq)],dict(r['bindings'],**{str(tq):sha(tq)}))
            wait=json.loads((Path(t['output'])/'actual_wait.json').read_text());assert wait['returncode']==0 and wait['child_gone'];train_hours+=wait['charged_gpu_device_hours'];assert train_hours<=10
            ledger.append(dict(stage=f'train_to_{end}',gpu_hours=wait['charged_gpu_device_hours']))
            if (Path(t['output'])/'paused.json').exists():write(O/'paused.json',dict(reason='training checkpoint pause',stage=t['output']));return
            proof=json.loads((Path(t['output'])/('segment_complete.json' if end==2000 else 'complete.json')).read_text());assert proof['passed']
            c=Path(r['checkpoint_dir']);ident=json.loads((c/f'native_step{end:05d}.json').read_text());assert sha(ident['path'])==ident['sha256'];checkpoint={k:ident[k] for k in ['path','sha256']}
            evaluate(checkpoint,end,'development')
            if end==2000:resume=json.loads((c/'latest.json').read_text())
            else:evaluate(checkpoint,end,'test')
        charged=sum(v['gpu_hours'] for v in ledger);assert charged<=19
        write(O/'complete.json',dict(passed=True,arm=r['training']['arm'],seed=r['training']['seed'],updates=4000,gpu_hours=charged,evaluations=eval_records,scientific_conclusion_pending=True))
    except BaseException:
        (O/'failure.txt').write_text(traceback.format_exc());raise
if __name__=='__main__':main()

"""Export immutable, already completed results at the explicit user pause.

CPU only. Does not execute models, resume training, decode streams, score images
or create bootstrap samples. Creates a new output directory exactly once.
"""
from residual_runtime import *
from fit_common import csvsave
import csv,sqlite3,collections,torch

def csvread(p):return list(csv.DictReader(pathlib.Path(p).open(encoding='utf8')))
def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''and(L/'STOP').exists()
    out=O/'interim_v1';out.mkdir(exist_ok=False);receipt=out/'execution_receipts';receipt.mkdir()
    analysis=read(O/'paused_training_value.json');coarse_audit=read(O/'paused_coarse_identity.json');assert coarse_audit['sources']==200 and coarse_audit['all_equal'];coarse={r['source_id']:r['coarse_bits']for r in coarse_audit['rows']}
    rows=[];curves=[];manifest={};sources=[];resources=[]
    for order in ['scalar','progressive']:
        role='train_'+order+'_to5000';wait=read(L/(role+'_control')/'actual_wait.json');done=read(L/role/'completion.json');assert wait['returncode']==0 and gone(wait['owner'])and gone(wait['process'])and done['status']=='RESUMABLE_BOUNDARY'and done['ledger']['unresolved']==0
        cp=torch.load(checked(done['checkpoint']),map_location='cpu',weights_only=False);assert cp['step']==done['step']and len(cp['history'])==cp['step']
        curves.extend(dict(order=order,**{k:v for k,v in r.items()if k!='source_ids'})for r in cp['history'])
        manifest[order]=dict(latest=done['checkpoint'],step=done['step'],next_update=done['step']+1,remaining_updates=5000-done['step'],next_indices=cp['next_indices'],next_j=cp['next_j'],best_so_far=dict(done['best'],checkpoint=pin(done['best']['path'])),schedule_hash=done['schedule_hash'],scope=cp['scope'],cache_index=cp['cache_index'],format=cp['format'],frozen_models=done['frozen_models'],first_resume_proof=read(L/role/'resume_proof.json'),automatic_resume=False)
        for step in [1000,2000,3000]:
            p=L/role/f'eval_{step:05d}/summary.json';v=read(p);assert len(v['rows'])==600;sources.append(pin(p));(out/(order+f'_eval_{step:05d}.json')).write_bytes(p.read_bytes())
            for r in v['rows']:
                rows.append(dict(method=order,step=step,source_id=r['source_id'],source_index=r['source_index'],budget=r['budget'],j=r['j'],status=r['status'],**{k:r[k]for k in ['psnr_db','lpips_alex','mse','D','actual_bits']}))
    f0=csvread(L/'f0_dev200_v1/source_rd.csv');assert len(f0)==600;sources.extend([pin(L/'f0_dev200_v1/source_rd.csv'),pin(L/'f0_dev200_v1/source_lengths.csv')])
    for r in f0:rows.append(dict(method='F0_FINE_TOKEN',step=0,source_id=r['source_id'],source_index=int(r['source_index']),budget=int(r['budget']),j='m'+r['m']+'K'+r['K'],status=r['status'],**{k:float(r[k])for k in ['psnr_db','lpips_alex','mse','D','actual_bits']}))
    assert len(rows)==4200;csvsave(out/'source_rd.csv',rows);csvsave(out/'training_curve.csv',curves)
    csvsave(out/'summary.csv',[{k:r.get(k)for k in ['method','step','budget','n','psnr_db','lpips_alex','mse','D','actual_bits']}for r in analysis['means']])
    csvsave(out/'paired_differences.csv',[{k:r.get(k)for k in ['comparison','step','budget','n','psnr_db','lpips_alex','mse','D','actual_bits','D_better_fraction','PSNR_better_fraction','LPIPS_better_fraction']}for r in analysis['paired_means']]);csvsave(out/'residual_length_by_step.csv',analysis['rate'])
    csvsave(out/'source_accounting.csv',[dict(method=r['method'],step=r['step'],source_id=r['source_id'],source_index=r['source_index'],budget=r['budget'],actual_bits=r['actual_bits'],coarse_bits=coarse[r['source_id']],finished_length_increment=r['actual_bits']-coarse[r['source_id']],fields=29,padding=r['budget']-r['actual_bits'],N=1024,n=1912,k=r['budget']+29,total_energy=2048,wireless_executed=False)for r in rows])
    matched={(r['method'],r['step'],r['budget'],r['source_id']):r for r in rows};pairs=[]
    for r in rows:
        if r['method']=='F0_FINE_TOKEN':continue
        comparators=[('F0_FINE_TOKEN',0)]+([('scalar',r['step'])]if r['method']=='progressive'else[])
        for method,step in comparators:
            other=matched[(method,step,r['budget'],r['source_id'])];pairs.append(dict(comparison=r['method']+'_minus_'+method,step=r['step'],source_id=r['source_id'],source_index=r['source_index'],budget=r['budget'],**{k:r[k]-other[k]for k in ['psnr_db','lpips_alex','mse','D','actual_bits']}))
    csvsave(out/'paired_source_differences.csv',pairs);save(out/'checkpoint_recovery_manifest.json',manifest)
    expected_failures={'b0','cache_v1'}
    for d in sorted(L.glob('*_control')):
        if not(d/'launch.json').exists():continue
        role=d.name[:-8];w=read(d/'actual_wait.json');assert gone(w['owner'])and gone(w['process']);assert (w['returncode']!=0)==(role in expected_failures)
        dest=receipt/role;dest.mkdir()
        for name in ['owner','launch','actual_wait','failure']:
            p=d/(name+'.json')
            if p.exists():(dest/('control_'+name+'.json')).write_bytes(p.read_bytes())
        for name in ['completion','failure']:
            p=L/role/(name+'.json')
            if p.exists():(dest/('worker_'+name+'.json')).write_bytes(p.read_bytes())
        db=sqlite3.connect('file:'+str(L/role/'calls.sqlite')+'?mode=ro',uri=True);counts=[dict(kind=k,status=s,count=n)for k,s,n in db.execute('select kind,status,count(*) from calls group by kind,status')];db.close()
        reg=read(checked(w['request']));resources.append(dict(role=role,returncode=w['returncode'],seconds=w['seconds'],GPU=bool(reg.get('environment',{}).get('CUDA_VISIBLE_DEVICES')),ledger=counts))
    save(out/'resource_accounting.json',dict(stages=resources,GPU_device_seconds=sum(r['seconds']for r in resources if r['GPU']),conservative_CPU_worker_wall_seconds=sum(r['seconds']for r in resources),GPU_limit_hours=48,CPU_worker_limit_hours=16,interpretation='shared-resource stage wall times, not a latency benchmark; CPU field is a conservative worker-wall proxy'))
    save(out/'delivery_identity.json',dict(status='INTERIM_USER_PAUSED',input_results=sources,training_updates=sum(v['step']for v in manifest.values()),main_exposures=4*sum(v['step']for v in manifest.values()),completed_evaluation_steps=[1000,2000,3000],source_budget_rows=len(rows),development_sources=200,bootstrap='NOT_RUN',wireless_frames=0,new_model_calls=0,new_metric_calls=0,pause=pin(O/'user_pause_20261028_verified.json'),full_B1='NOT_COMPLETE',diagnostic16='REGISTERED_NOT_RUN',B2_B3_formal='NOT_RUN_GATED',created_unix=time.time()))
    print({'status':'INTERIM_EXPORT_COMPLETE','rows':len(rows),'updates':sum(v['step']for v in manifest.values())})
if __name__=='__main__':main()

"""Descriptive finite-development analysis after all registered workers close."""
from residual_runtime import *
from fit_common import csvsave,digest
import csv,collections,torch

def rows(p):return list(csv.DictReader(pathlib.Path(p).open(encoding='utf8')))
def groupmeans(data,keys,metrics):
    groups=collections.defaultdict(list)
    for v in data:groups[tuple(v[k]for k in keys)].append(v)
    return [dict(zip(keys,key),sources=len(v),**{m:float(np.mean([float(x[m])for x in v]))for m in metrics})for key,v in groups.items()]
def closed(role):
    w=read(L/(role+'_control')/'actual_wait.json');assert w['returncode']==0 and gone(w['owner'])and gone(w['process']);done=read(L/role/'completion.json');assert done['ledger']['unresolved']==0;return done
def main():
    out=O/'b1_delivery';out.mkdir(exist_ok=False);chosen=read(O/'b1_selected_checkpoints.json')['selected'];metrics=['psnr_db','lpips_alex','mse','D','actual_bits'];allrows=[];curves=[];training=[];checkpoints={}
    for order in ['scalar','progressive']:
        role='train_'+order+'_to5000';done=closed(role);assert done['step']==5000 and done['new_updates']==4975;assert closed('train_'+order+'_first25')['new_updates']==25
        cp=torch.load(checked(done['checkpoint']),map_location='cpu',weights_only=False);assert len(cp['history'])==5000
        training += [dict(order=order,**{k:v for k,v in x.items()if k!='source_ids'})for x in cp['history']]
        for step in range(1000,5001,1000):
            ss=read(L/role/f'eval_{step:05d}/summary.json');assert len(ss['rows'])==600
            curves+=groupmeans(ss['rows'],['order','step','budget'],metrics)
        selected=read(checked(chosen[order]['summary']))['rows'];assert len({v['source_id']for v in selected})==200
        for v in selected:allrows.append(dict(method='R_'+order,source_id=v['source_id'],source_index=v['source_index'],budget=v['budget'],step=v['step'],j=v['j'],status=v['status'],**{m:v[m]for m in metrics}))
        checkpoints[order]=dict(latest=done['checkpoint'],selected=chosen[order]['checkpoint'],best_step=chosen[order]['step'],objective=chosen[order]['objective'],schedule_hash=done['schedule_hash'],resume_proof=read(L/role/'resume_proof.json'),format=cp['format'],frozen_models=done['frozen_models'],repeated_updates=0)
    f0done=closed('f0_dev200_v1')
    for v in rows(checked(f0done['rows'])):allrows.append(dict(method='F0_FINE_TOKEN',source_id=v['source_id'],source_index=int(v['source_index']),budget=int(v['budget']),step=0,j='m'+v['m']+'K'+v['K'],status=v['status'],**{m:float(v[m])for m in metrics}))
    assert len(allrows)==1800
    csvsave(out/'source_rd.csv',allrows);means=groupmeans(allrows,['method','budget'],metrics);csvsave(out/'source_rd_summary.csv',means);csvsave(out/'checkpoint_curve.csv',curves);csvsave(out/'training_curve.csv',training);save(out/'checkpoint_recovery_manifest.json',checkpoints)
    csvsave(out/'source_accounting.csv',[dict(method=v['method'],source_id=v['source_id'],source_index=v['source_index'],budget=v['budget'],actual_bits=v['actual_bits'],fields=29,information_padding=v['budget']-v['actual_bits'],k=v['budget']+29,n=1912,N=1024,energy=2048,j=v['j'],status=v['status'],wireless_run=False)for v in allrows])
    by={(v['method'],v['source_id'],v['budget']):v for v in allrows};pairs=[]
    for a,b in [('R_progressive','R_scalar'),('R_progressive','F0_FINE_TOKEN'),('R_scalar','F0_FINE_TOKEN')]:
        for v in allrows:
            if v['method']!=a:continue
            other=by[(b,v['source_id'],v['budget'])];pairs.append(dict(comparison=a+'_minus_'+b,source_id=v['source_id'],source_index=v['source_index'],budget=v['budget'],**{m:float(v[m])-float(other[m])for m in metrics}))
    csvsave(out/'paired_source_differences.csv',pairs);pm=groupmeans(pairs,['comparison','budget'],metrics);csvsave(out/'paired_differences.csv',pm)
    diagnostics=[];bits=[];codes=[]
    for order in ['scalar','progressive']:
        role='diagnostic16_'+order+'_v1';done=closed(role);diagnostics+=rows(checked(done['rows']));bits+=rows(checked(done['bits']))
        for p in sorted((L/role).glob('*/code_diagnostic.json')):
            v=read(p);codes.append(dict(order=order,**{k:x for k,x in v.items()if k!='indices'}))
    csvsave(out/'code_usage_diagnostic.csv',codes)
    csvsave(out/'diagnostic_source_rd.csv',diagnostics);csvsave(out/'predicted_bits_vs_actual_bits.csv',bits)
    anchor=[v for v in diagnostics if v['kind']=='anchor'];csvsave(out/'anchor_rd.csv',groupmeans(anchor,['order','step','j'],metrics))
    degradation=[]
    for order in ['scalar','progressive']:
        vv={(v['source_id'],int(v['j'])):v for v in anchor if v['order']==order};ids=sorted({k[0]for k in vv})
        for j in range(16,257,16):
            for metric,sign in [('psnr_db',-1),('lpips_alex',1),('D',1)]:
                changes=[float(vv[(sid,j)][metric])-float(vv[(sid,j-16)][metric])for sid in ids];degradation.append(dict(order=order,j_previous=j-16,j=j,metric=metric,sources=len(ids),degradation_fraction=float(np.mean(np.asarray(changes)*sign>0)),mean_change=float(np.mean(changes))))
    csvsave(out/'anchor_degradation.csv',degradation)
    score={}
    for mode in ['dino','dreamsim','clip']:
        done=closed('diagnostic_metrics_'+mode+'_v1');score[mode]=read(checked(done['scores']))
    events=read(L/'diagnostic_metrics/events.json');extra=[]
    for v in events:
        key=v['pair_key'];z=dict(v,dinov2_vitl14_cosine=score['dino'][key]['value'],dreamsim=score['dreamsim'][key]['dreamsim'],clip_image_cosine=score['clip'][key]['clip_image_cosine']);extra.append(z)
    assert len(extra)==144;csvsave(out/'diagnostic_five_metrics.csv',extra);extra_metrics=['psnr_db','lpips_alex','D','actual_bits','dinov2_vitl14_cosine','dreamsim','clip_image_cosine'];csvsave(out/'diagnostic_five_metric_summary.csv',groupmeans(extra,['method','budget'],extra_metrics))
    statecounts=collections.Counter((v['method'],v['budget'],v['status'],v['j'])for v in allrows);csvsave(out/'endpoint_counts.csv',[dict(method=k[0],budget=k[1],status=k[2],j=k[3],count=n)for k,n in statecounts.items()])
    endpoint_gaps=[]
    for order in ['scalar','progressive']:
        values=[float(x['previous_endpoint_gap'])for x in bits if x['order']==order and int(x['j'])>0];endpoint_gaps.append(dict(method='R_'+order,n=len(values),mean=float(np.mean(values)),median=float(np.median(values)),max=float(np.max(values)),negative_fraction=float(np.mean(np.asarray(values)<0)),zero_fraction=float(np.mean(np.asarray(values)==0))))
    fr=rows(checked(f0done['lengths']));ids={v['source_id']for v in diagnostics};fg=collections.defaultdict(list)
    for v in fr:
        if v['source_id']in ids:fg[v['source_id']].append(v)
    values=[]
    for ss in fg.values():
        ss.sort(key=lambda v:OFFSETS[int(v['m'])]+int(v['K']));values.extend(np.diff([int(v['actual_bits'])for v in ss]).tolist())
    endpoint_gaps.append(dict(method='F0_FINE_TOKEN',n=len(values),mean=float(np.mean(values)),median=float(np.median(values)),max=float(np.max(values)),negative_fraction=float(np.mean(np.asarray(values)<0)),zero_fraction=float(np.mean(np.asarray(values)==0))));csvsave(out/'endpoint_gap_summary.csv',endpoint_gaps)
    receipt=out/'execution_receipts';receipt.mkdir();resources=[]
    for d in sorted(L.glob('*_control')):
        if not(d/'launch.json').exists():continue
        w=read(d/'actual_wait.json');assert gone(w['owner'])and gone(w['process']);role=d.name[:-8];dest=receipt/role;dest.mkdir()
        for name in ['owner','launch','actual_wait','failure']:
            p=d/(name+'.json')
            if p.exists():(dest/('control_'+name+'.json')).write_bytes(p.read_bytes())
        for name in ['completion','failure']:
            p=L/role/(name+'.json')
            if p.exists():(dest/('worker_'+name+'.json')).write_bytes(p.read_bytes())
        launch=read(d/'launch.json');registration=read(checked(w['request']));resources.append(dict(role=role,seconds=w['seconds'],GPU=bool(registration.get('environment',{}).get('CUDA_VISIBLE_DEVICES')),returncode=w['returncode']))
    accounting=dict(rows=resources,conservative_CPU_worker_seconds=sum(x['seconds']for x in resources),GPU_device_seconds=sum(x['seconds']for x in resources if x['GPU']),limits=dict(GPU_hours=48,CPU_worker_hours=16));assert accounting['conservative_CPU_worker_seconds']<16*3600 and accounting['GPU_device_seconds']<48*3600;save(out/'resource_accounting.json',accounting)
    # Gating conclusion is written separately after reviewing all fixed conditions.
    save(out/'completion.json',dict(status='B1_DATA_ASSEMBLED',training_updates=10000,main_sample_exposures=40000,seeds=1,development_sources=200,diagnostic_sources=16,source_budget_rows=1800,extra_metric_logical_rows=144,bootstrap='none; development descriptive paired differences',wireless_frames=0,B2_B3_formal='NOT_RUN_GATED',created_unix=time.time()))
    print({k:chosen[k]['step']for k in chosen});print(means);print(pm)
if __name__=='__main__':main()

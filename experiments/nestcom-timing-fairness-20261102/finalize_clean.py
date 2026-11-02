"""Aggregate the sole clean GPU6 comparison and exact reused CPU-only cells."""
from pathlib import Path
import json,sqlite3,sys
import numpy as np
from summarize import read,save,sha,csvout,gone,aggregate,cpu_telemetry

R=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
D=R/'results/nestcom_timing_fairness_20261102_v1'
OLD=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/nestcom_timing_fairness_20261102_v1')
L=OLD/'clean_gpu6_v1';D6=D/'clean_gpu6_v1'

def main():
    wait=read(L/'queue_owner_control/actual_wait.json')
    assert gone(wait['child'])and gone(read(L/'queue_owner_control/owner.json')['owner'])
    queue=read(L/'queue_control/completion.json')
    history=read(L/'closure_gate/prior_resource_receipts.json')
    total={k:history[k]+queue[k]for k in ['GPU_device_seconds','CPU_process_wall_seconds']}
    assert total['GPU_device_seconds']<=10800 and total['CPU_process_wall_seconds']<=3600
    stages=read(D6/'final_stages.json');rows=[];legacy=[];checks=[];failures=[];telemetry=[]
    assert len(stages)==19 and len({s['name']for s in stages})==19
    prior_controls=[]
    for role in ['queue_owner_control','unlaunched_owner_control','granularity_owner_control']:
        p=OLD/role;w=read(p/'actual_wait.json');owner=read(p/'owner.json')['owner']
        assert gone(w['child'])and gone(owner)
        prior_controls.append(dict(role=role,actual_wait=w,owner=owner))
    mapping=read(D6/'device_mapping.json')
    for st in stages:
        name=st['name'];c=Path(st['control']);o=Path(st['output_root'])/name
        if not(c/'actual_wait.json').exists():
            failures.append(dict(stage=name,diagnostic=read(c/'not_launched.json')if(c/'not_launched.json').exists()else'MISSING_WAIT'));continue
        w=read(c/'actual_wait.json');assert gone(w['child'])
        if w['returncode']!=0:
            failures.append(dict(stage=name,wait=w,failure=read(o/'failure.json')if(o/'failure.json').exists()else'INTERRUPTED_NO_FAILURE_JSON'));continue
        comp=read(o/'completion.json');assert comp['status']=='COMPLETE'and comp['rows']==st['calls']and comp['ledger']['unresolved']==0
        db=sqlite3.connect('file:'+str(o/'calls.sqlite')+'?mode=ro',uri=True)
        counts=db.execute('SELECT status,COUNT(*)FROM calls GROUP BY status').fetchall();db.close();assert counts==[('COMPLETE',st['calls'])]
        env=read(o/'environment.json');resource=read(o/'effective_timing_resources.json')
        assert env['cpu_affinity']==[38,40]and env['threads']==2 and env['interop']==2
        assert len({(p['socket'],p['core'])for p in resource['cpu_topology']})==2
        if st['timing_reuse']:assert env['CPU_only']and name.startswith('main_g12_')
        else:assert env['GPU_UUID']==mapping['CUDA_UUID']and env['registered_UUID']==mapping['NVML_UUID']
        assert sha(st['registration']['path'])==st['registration']['sha256']
        assert sha(st['baseline']['path'])==st['baseline']['sha256']
        current=read(o/'rows.json');old=read(st['baseline']['path']);assert len(current)==len(old)==st['calls']
        assert len({(x['source_id'],x['event'],x['phase'],x['repetition'])for x in current})==st['calls']
        scope='diagnostic'if name.startswith('diag')else'granularity'if name.startswith('grid')else'main'
        for x,y in zip(current,old):
            for k in ['source_id','source_index','event','direction','phase','repetition','method','group','N','snr_db']:assert x[k]==y[k]
            key='emitted_symbols_sha256'if x['direction']=='tx'else'image_sha256'
            assert x[key]==y[key]
            if x['direction']=='rx':
                assert x['status']==y['status']and x['success']==y['success']
                if 'received_tokens_sha256'in y:assert x['received_tokens_sha256']==y['received_tokens_sha256']
            rows.append(dict(stage=name,scope=scope,timing_reuse=st['timing_reuse'],**x))
            legacy.append(dict(stage=name,scope=scope,**y))
        tl=[json.loads(s)for s in(c/'telemetry.jsonl').read_text().splitlines()]
        samples=[];foreign_seen=[]
        for sample in tl:
            if not env['CPU_only']:
                for line in sample['apps'].splitlines():
                    a=line.split(', ')
                    if a[0]==mapping['NVML_UUID']and int(a[1])!=w['child']['pid']:foreign_seen.append(line)
                gpu=next(s.split(', ')for s in sample['gpus'].splitlines()if s.split(', ')[1]==mapping['NVML_UUID'])
                samples.append(dict(temperature=float(gpu[5]),power=float(gpu[6]),clock=float(gpu[7])))
        assert not foreign_seen
        telemetry.append(dict(stage=name,samples=len(tl),cpu_busy_fraction=cpu_telemetry(tl),
                              GPU_ranges={k:[min(v[k]for v in samples),max(v[k]for v in samples)]for k in samples[0]}if samples else None,
                              foreign_GPU_processes_observed=foreign_seen))
        checks.append(dict(stage=name,actual_wait=w,environment=env,calls=len(current),all_output_hashes_match=True,timing_reuse=st['timing_reuse'],operation_counts=comp['in_memory_calls'],peak_GPU_allocated=comp['peak_GPU_allocated'],peak_GPU_reserved=comp['peak_GPU_reserved']))
    out=D/'delivery';out.mkdir(exist_ok=False)
    if not failures:assert wait['returncode']==0 and queue['status']=='COMPLETE' and len(rows)==2440
    save(out/'closure.json',dict(status='COMPLETE'if not failures else'INCOMPLETE',final_actual_wait=wait,final_queue=queue,prior_resources=history,prior_owner_closure=prior_controls,resources=total,checks=checks,failures=failures))
    prior_receipts=[]
    for control in sorted(OLD.glob('*_control')):
        for p in sorted(control.glob('*.json')):
            if p.name in ['actual_wait.json','not_launched.json','failure.json']:
                prior_receipts.append(dict(path=str(p),sha256=sha(p),receipt=read(p)))
    save(out/'preserved_prior_failures_and_waits.json',prior_receipts)
    save(out/'resource_telemetry_summary.json',telemetry);csvout(out/'call_timings.csv',rows)
    # All attempted pipeline calls, including excluded GPU0 trials and interrupted
    # reserved calls, are retained in the accounting. A reserved call is not
    # claimed to have produced a successful output.
    accounting=[]
    for root,label in [(OLD,'initial_GPU0_or_CPU'),(L,'clean_GPU6')]:
        for dbpath in sorted(root.glob('*/calls.sqlite')):
            db=sqlite3.connect('file:'+str(dbpath)+'?mode=ro',uri=True)
            for status,n in db.execute('SELECT status,COUNT(*)FROM calls GROUP BY status'):
                accounting.append(dict(phase=label,stage=dbpath.parent.name,status=status,calls=n))
            db.close()
    csvout(out/'all_attempted_calls.csv',accounting)
    mainrows=[r for r in rows if r['scope']=='main'];mainold=[r for r in legacy if r['scope']=='main']
    means=aggregate(mainrows);oldmeans=aggregate(mainold)
    src=read(read(D6/'main_g03_tx_registration.json')['sources']['path']);order=[s['source_id']for s in src]
    assert len(order)==len(set(order))==32
    bank=np.random.default_rng(2026110201).integers(0,32,(10000,32),dtype=np.int64)
    with(L/'timing_bootstrap_bank.npy').open('xb')as f:np.save(f,bank)
    save(out/'statistics_identity.json',dict(seed=2026110201,resamples=10000,source_order=order,bank_sha256=sha(L/'timing_bootstrap_bank.npy'),bank_PFS_path=str(L/'timing_bootstrap_bank.npy'),python=sys.version,numpy=np.__version__,interval_scope='source variation under the measured runtime, not all possible host load',comparison='new clean GPU6 resource session minus historical E5 GPU1 session; does not isolate affinity from device/session effects'))
    csvout(out/'source_means.csv',[dict(method=m,source_id=s,direction=d,seconds=v,historical_seconds=oldmeans[m,s,d])for(m,s,d),v in means.items()])
    summaries=[];deltas=[]
    for method in read(D/'protocol.json')['methods']:
        for direction in ['tx','rx','total']:
            if any((method,s,direction)not in means for s in order):continue
            x=np.asarray([means[method,s,direction]for s in order]);y=np.asarray([oldmeans[method,s,direction]for s in order])
            ci=np.quantile(x[bank].mean(1),[.025,.975]);di=np.quantile((x-y)[bank].mean(1),[.025,.975])
            calls=[r['seconds']for r in mainrows if r['method']==method and r['direction']==direction and r['phase']=='measured']
            received=[r for r in mainrows if r['method']==method and r['direction']=='rx'and r['phase']=='measured']
            summaries.append(dict(method=method,direction=direction,source_count=32,mean_ms=float(x.mean()*1000),ci_low_ms=float(ci[0]*1000),ci_high_ms=float(ci[1]*1000),historical_mean_ms=float(y.mean()*1000),median_call_ms=float(np.median(calls)*1000)if calls else None,p95_call_ms=float(np.quantile(calls,.95)*1000)if calls else None,measured_RX_trials=len(received),measured_RX_successes=sum(r['success']for r in received)))
            deltas.append(dict(method=method,direction=direction,new_minus_historical_ms=float((x-y).mean()*1000),ci_low_ms=float(di[0]*1000),ci_high_ms=float(di[1]*1000),new_over_historical=float(x.mean()/y.mean())))
    csvout(out/'timing_summary.csv',summaries);csvout(out/'paired_resource_deltas.csv',deltas)
    diag=aggregate([r for r in rows if r['scope']=='diagnostic'])
    csvout(out/'audit_diagnostics.csv',[dict(method=m,source_id=s,direction=d,lean_ms=v*1000,original_audits_ms=means[m,s,d]*1000,delta_ms=(v-means[m,s,d])*1000)for(m,s,d),v in diag.items()if(m,s,d)in means])
    grid=aggregate([dict(r,method=r['group'])for r in rows if r['scope']=='granularity'])
    oldgrid=aggregate([dict(r,method=r['group'])for r in legacy if r['scope']=='granularity'])
    gridout=[]
    for group in dict.fromkeys(k[0]for k in grid):
        if any((group,s,'tx')not in grid for s in order):continue
        x=np.asarray([grid[group,s,'tx']for s in order]);ci=np.quantile(x[bank].mean(1),[.025,.975])
        gridout.append(dict(group=group,mean_ms=float(x.mean()*1000),ci_low_ms=float(ci[0]*1000),ci_high_ms=float(ci[1]*1000),historical_mean_ms=float(np.mean([oldgrid[group,s,'tx']for s in order])*1000),reused_main=False))
    if all(('Token_NeST',s,'tx')in means for s in order):
        x=np.asarray([means['Token_NeST',s,'tx']for s in order]);ci=np.quantile(x[bank].mean(1),[.025,.975])
        gridout.append(dict(group='granularity_token',mean_ms=float(x.mean()*1000),ci_low_ms=float(ci[0]*1000),ci_high_ms=float(ci[1]*1000),historical_mean_ms=float(np.mean([oldmeans['Token_NeST',s,'tx']for s in order])*1000),reused_main=True))
    csvout(out/'granularity_summary.csv',gridout)
    grid_deltas=[]
    if all(('Token_NeST',s,'tx')in means for s in order):
        for group in dict.fromkeys(k[0]for k in grid):
            if any((group,s,'tx')not in grid for s in order):continue
            delta=np.asarray([means['Token_NeST',s,'tx']-grid[group,s,'tx']for s in order]);ci=np.quantile(delta[bank].mean(1),[.025,.975])
            grid_deltas.append(dict(comparison='Token minus '+group,source_count=32,mean_ms=float(delta.mean()*1000),ci_low_ms=float(ci[0]*1000),ci_high_ms=float(ci[1]*1000)))
    csvout(out/'granularity_paired_deltas.csv',grid_deltas)
    save(out/'actual_compute_and_reuse.json',dict(status='COMPLETE'if not failures else'INCOMPLETE',canonical_pipeline_calls=len(rows),canonical_main_calls=len(mainrows),canonical_diagnostic_calls=sum(r['scope']=='diagnostic'for r in rows),canonical_granularity_new_TX_calls=sum(r['scope']=='granularity'for r in rows),warmups=sum(r['phase']=='warmup'for r in rows),measured=sum(r['phase']=='measured'for r in rows),CPU_calls_reused=sum(r['timing_reuse']for r in rows),all_attempted_completed_calls=sum(r['calls']for r in accounting if r['status']=='COMPLETE'),all_attempted_unresolved_or_failed_calls=sum(r['calls']for r in accounting if r['status']!='COMPLETE'),cached_results_inside_timed_path=0,training=0,quality_evaluation=0,calibration=0,all_canonical_outputs_match_historical=True,**total))
    print(json.dumps(dict(status='COMPLETE'if not failures else'INCOMPLETE',canonical_calls=len(rows),failures=failures,summary=summaries,granularity=gridout)))

if __name__=='__main__':main()

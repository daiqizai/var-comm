"""Read-only timing aggregation after actual worker/owner closure."""
from pathlib import Path
import sys, json, csv, hashlib, sqlite3, time
import numpy as np

R=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
D=R/'results/nestcom_timing_fairness_20261102_v1'
L=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/nestcom_timing_fairness_20261102_v1')
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
    with Path(p).open('x')as f:json.dump(v,f,indent=2);f.write('\n')
def csvout(p,data):
    fields=list(dict.fromkeys(k for row in data for k in row))
    with Path(p).open('x',newline='')as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(data)
def gone(q):
    p=Path('/proc',str(q['pid']),'stat')
    return not p.exists()or int(p.read_text().rsplit(')',1)[1].split()[19])!=q['start_ticks']

def aggregate(data):
    result={}
    for x in data:
        if x['phase']!='measured':continue
        key=(x['method'],x['source_id'],x['direction'])
        result.setdefault(key,[]).append(float(x['seconds']))
    for key,v in result.items():assert len(v)==3,key
    means={k:float(np.mean(v))for k,v in result.items()}
    for method,source,direction in list(means):
        if direction=='tx'and(method,source,'rx')in means:
            means[method,source,'total']=means[method,source,'tx']+means[method,source,'rx']
    return means

def cpu_telemetry(lines):
    pairs=[]
    for a,b in zip(lines,lines[1:]):
        aa={x.split()[0]:np.asarray([int(v)for v in x.split()[1:9]],dtype=np.int64)for x in a['cpu']}
        bb={x.split()[0]:np.asarray([int(v)for v in x.split()[1:9]],dtype=np.int64)for x in b['cpu']}
        for c in sorted(aa):
            d=bb[c]-aa[c];total=int(d.sum());busy=total-int(d[3])-int(d[4])
            if total>0:pairs.append(dict(cpu=c,total=total,busy=busy))
    return {c:sum(v['busy']for v in pairs if v['cpu']==c)/sum(v['total']for v in pairs if v['cpu']==c)
            for c in sorted({v['cpu']for v in pairs})}

def main():
    # Exact successful closure, never extrapolate from filenames or row counts.
    outer=read(L/'queue_owner_control/actual_wait.json')
    assert gone(outer['child'])and gone(read(L/'queue_owner_control/owner.json')['owner'])
    queue=read(L/'queue_control/completion.json')
    resource_totals={k:queue[k]for k in ['GPU_device_seconds','CPU_process_wall_seconds']}
    first_launch_closure=None
    if(L/'unlaunched_dispatch.json').exists():
        extra_wait=read(L/'unlaunched_owner_control/actual_wait.json')
        assert gone(extra_wait['child'])and gone(read(L/'unlaunched_owner_control/owner.json')['owner'])
        extra=read(L/'unlaunched_queue_control/completion.json')
        first_launch_closure=dict(actual_wait=extra_wait,queue=extra,evidence=read(L/'unlaunched_supervision/zero_launch_evidence.json'))
        for k in resource_totals:resource_totals[k]+=extra[k]
    granularity_closure=None
    if(L/'granularity_dispatch.json').exists():
        grid_wait=read(L/'granularity_owner_control/actual_wait.json')
        assert gone(grid_wait['child'])and gone(read(L/'granularity_owner_control/owner.json')['owner'])
        grid_queue=read(L/'granularity_queue_control/completion.json')
        granularity_closure=dict(actual_wait=grid_wait,queue=grid_queue,addendum=read(D/'granularity_timing_addendum.json'))
        for k in resource_totals:resource_totals[k]+=grid_queue[k]
    stages=read(D/'stages.json')+(read(D/'granularity_stages.json')if(D/'granularity_stages.json').exists()else[])
    data=[];old=[];grid_old=[];checks=[];telemetry=[];failures=[]
    for stage in stages:
        name=stage['name'];c=L/(name+'_control');o=L/name
        if(c/'not_launched.json').exists()and(L/(name+'_first_launch_control')).exists():
            assert not(c/'child.json').exists()and not(c/'actual_wait.json').exists()
            assert read(c/'not_launched.json')['scientific_calls']==0
            c=L/(name+'_first_launch_control')
        if not(c/'actual_wait.json').exists():
            failures.append(dict(stage=name,reason=read(c/'not_launched.json')if(c/'not_launched.json').exists()else'MISSING_ACTUAL_WAIT'));continue
        wait=read(c/'actual_wait.json');assert gone(wait['child'])
        if wait['returncode']!=0:
            failures.append(dict(stage=name,wait=wait,failure=read(o/'failure.json')if(o/'failure.json').exists()else'NO_WORKER_FAILURE_RECEIPT'));continue
        completion=read(o/'completion.json');assert completion['status']=='COMPLETE'and completion['rows']==stage['calls']and completion['ledger']['unresolved']==0
        db=sqlite3.connect('file:'+str(o/'calls.sqlite')+'?mode=ro',uri=True)
        counts=db.execute('SELECT status,COUNT(*)FROM calls GROUP BY status').fetchall();db.close();assert counts==[('COMPLETE',stage['calls'])]
        rows=read(o/'rows.json');baseline=read(stage['baseline']['path']);assert len(rows)==len(baseline)==stage['calls']
        for x,y in zip(rows,baseline):
            for key in ['source_id','source_index','event','phase','direction']:assert x[key]==y[key]
            key='emitted_symbols_sha256'if x['direction']=='tx'else'image_sha256';assert x[key]==y[key]
            if x['direction']=='rx':
                assert x['status']==y['status']and x['success']==y['success']
                if 'received_tokens_sha256'in y:assert x['received_tokens_sha256']==y['received_tokens_sha256']
            data.append(dict(stage=name,scope='diagnostic'if name.startswith('diag')else'granularity'if name.startswith('grid')else'main',**x))
            if name.startswith('main'):old.append(dict(stage=name,scope='historical',**y))
            if name.startswith('grid'):grid_old.append(y)
        env=read(o/'environment.json');resources=read(o/'effective_timing_resources.json')
        assert env['cpu_affinity']==[38,40]and env['threads']==2 and env['interop']==2
        assert len({(v['socket'],v['core'])for v in resources['cpu_topology']})==2
        tl=[json.loads(v)for v in(c/'telemetry.jsonl').read_text().splitlines()]
        telemetry.append(dict(stage=name,samples=len(tl),cpu_busy_fraction=cpu_telemetry(tl),
                              first=tl[0]if tl else None,last=tl[-1]if tl else None))
        checks.append(dict(stage=name,rows=len(rows),warmups=sum(x['phase']=='warmup'for x in rows),measured=sum(x['phase']=='measured'for x in rows),all_output_hashes_match=True,actual_wait=wait,environment=env))
    out=D/'delivery';out.mkdir(exist_ok=False)
    save(out/'closure.json',dict(status='COMPLETE'if not failures else'INCOMPLETE',original_queue=queue,original_outer_wait=outer,first_launch_closure=first_launch_closure,granularity_closure=granularity_closure,resources=resource_totals,checks=checks,failures=failures))
    save(out/'resource_telemetry_summary.json',telemetry)
    csvout(out/'call_timings.csv',data)
    mainrows=[r for r in data if r['scope']=='main'];newmeans=aggregate(mainrows);oldmeans=aggregate(old)
    source_rows=[dict(method=m,source_id=s,direction=d,seconds=v,old_seconds=oldmeans[m,s,d],new_minus_old_seconds=v-oldmeans[m,s,d])for(m,s,d),v in newmeans.items()]
    csvout(out/'source_means.csv',source_rows)
    protocol=read(D/'protocol.json');src=read(read(D/'main_g03_tx_registration.json')['sources']['path']);order=[s['source_id']for s in src]
    bank=np.random.default_rng(2026110201).integers(0,32,size=(10000,32),dtype=np.int64)
    with(L/'timing_bootstrap_bank.npy').open('xb')as f:np.save(f,bank)
    save(out/'statistics_identity.json',dict(seed=2026110201,resamples=10000,sources=order,bank_path=str(L/'timing_bootstrap_bank.npy'),bank_sha256=sha(L/'timing_bootstrap_bank.npy'),interpretation='source variation; does not estimate all host-load or device variability',paired_delta='new resource session minus historical E5; device/session and affinity both differ'))
    summary=[];deltas=[]
    for method in protocol['methods']:
        for direction in ['tx','rx','total']:
            if any((method,s,direction)not in newmeans for s in order):continue
            x=np.asarray([newmeans[method,s,direction]for s in order]);y=np.asarray([oldmeans[method,s,direction]for s in order])
            ci=np.quantile(x[bank].mean(1),[.025,.975]);di=np.quantile((x-y)[bank].mean(1),[.025,.975])
            summary.append(dict(method=method,direction=direction,source_count=32,measured_trials=96 if direction!='total' else 192,mean_ms=float(x.mean()*1000),median_source_ms=float(np.median(x)*1000),ci_low_ms=float(ci[0]*1000),ci_high_ms=float(ci[1]*1000),historical_mean_ms=float(y.mean()*1000)))
            deltas.append(dict(method=method,direction=direction,new_minus_old_ms=float((x-y).mean()*1000),ci_low_ms=float(di[0]*1000),ci_high_ms=float(di[1]*1000),new_over_old=float(x.mean()/y.mean())))
    csvout(out/'timing_summary.csv',summary);csvout(out/'paired_resource_deltas.csv',deltas)
    diagmeans=aggregate([r for r in data if r['scope']=='diagnostic']);diagnostics=[]
    for(m,s,d),v in diagmeans.items():
        if(m,s,d)in newmeans:diagnostics.append(dict(method=m,source_id=s,direction=d,lean_ms=v*1000,original_audits_ms=newmeans[m,s,d]*1000,difference_ms=(v-newmeans[m,s,d])*1000))
    csvout(out/'audit_diagnostics.csv',diagnostics)
    griddata=[v for v in data if v['scope']=='granularity'];gridmeans=aggregate([dict(v,method=v['group'])for v in griddata]);gridoldmeans=aggregate([dict(v,method=v['group'])for v in grid_old])
    grid_summary=[]
    for group in dict.fromkeys(v['group']for v in griddata):
        if any((group,s,'tx')not in gridmeans for s in order):continue
        x=np.asarray([gridmeans[group,s,'tx']for s in order]);y=np.asarray([gridoldmeans[group,s,'tx']for s in order]);ci=np.quantile(x[bank].mean(1),[.025,.975])
        grid_summary.append(dict(group=group,source_count=32,mean_ms=float(x.mean()*1000),ci_low_ms=float(ci[0]*1000),ci_high_ms=float(ci[1]*1000),historical_mean_ms=float(y.mean()*1000),reuse=False))
    if all(('Token_NeST',s,'tx')in newmeans for s in order):
        x=np.asarray([newmeans['Token_NeST',s,'tx']for s in order]);ci=np.quantile(x[bank].mean(1),[.025,.975])
        grid_summary.append(dict(group='granularity_token_reuses_main_g04_tx',source_count=32,mean_ms=float(x.mean()*1000),ci_low_ms=float(ci[0]*1000),ci_high_ms=float(ci[1]*1000),historical_mean_ms=float(np.mean([oldmeans['Token_NeST',s,'tx']for s in order])*1000),reuse=True))
    csvout(out/'granularity_summary.csv',grid_summary)
    save(out/'actual_compute_and_reuse.json',dict(status='COMPLETE'if not failures else'INCOMPLETE',new_timed_pipeline_calls=len(data),warmup_calls=sum(v['phase']=='warmup'for v in data),measured_calls=sum(v['phase']=='measured'for v in data),main_calls=len(mainrows),diagnostic_calls=sum(v['scope']=='diagnostic'for v in data),granularity_calls=len(griddata),cached_scientific_results_inside_timer=0,training=0,quality_scores=0,selection=0,historical_received_observations_reused_as_RX_inputs=True,model_weights_reused=True,all_completed_outputs_identical=True,maximum_registered_calls=2440 if granularity_closure else 1960,**resource_totals))
    print(json.dumps(dict(status='COMPLETE'if not failures else'INCOMPLETE',rows=len(data),failed_stages=len(failures),summary=summary)))

if __name__=='__main__':main()

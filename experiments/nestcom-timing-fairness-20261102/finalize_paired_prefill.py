"""Close the new numerical route once, using the existing source bootstrap bank."""
from pathlib import Path
import sys,json,csv,hashlib
import numpy as np
from run_once import read,save,sha,same_process

def write_csv(p,rows):
    assert rows
    with Path(p).open('x',newline='')as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main(registration,destination):
    r=read(registration);root=Path(r['root']);D=Path(destination);D.mkdir(exist_ok=False)
    outer=read(root/'outer/actual_wait.json');assert outer['returncode']==0 and outer['process_gone']
    assert not same_process(outer['child'])
    chain=read(root/'chain/completion.json');assert chain['status']=='COMPLETE'
    allrows=[];completions={};setups={}
    for phase in ['qualification','tx','rx']:
        q=root/(phase+'_queue');v=read(q/(phase+'_control')/'actual_wait.json')
        assert v['returncode']==0 and v['final_identity'] is None and not same_process(v['child'])
        c=read(root/phase/'completion.json');assert c['status']=='COMPLETE' and c['ledger']['unresolved']==0
        completions[phase]=c;setups[phase]=read(root/phase/'setup.json')
        if phase!='qualification':
            rows=read(root/phase/'rows.json');assert len(rows)==160
            allrows.extend(rows)
    qual=read(root/'qualification/qualification.json');assert qual['status']=='PASS'
    tx={x['event']:x for x in allrows if x['direction']=='tx'}
    rx={x['event']:x for x in allrows if x['direction']=='rx'}
    assert set(tx)==set(rx) and len(tx)==160
    for event,v in rx.items():
        proof=read(root/'tx/witnesses'/(event+'.json'))
        assert v['success'] and v['status']=='ACCEPTED_SOURCE'
        assert v['received_tokens_sha256']==proof['received_tokens_sha256']
        assert v['received_class']==proof['predicted_class'] and v['endpoint']==proof['endpoint']
    repo=Path(r['R']);old=repo/'results/nestcom_timing_fairness_20261102_v1/delivery'
    ident=read(old/'statistics_identity.json');order=ident['source_order'];assert len(order)==32
    bp=Path(ident['bank_PFS_path']);assert sha(bp)==ident['bank_sha256'];bank=np.load(bp,allow_pickle=False)
    assert bank.shape==(10000,32)
    source_rows=[];values={};summary=[]
    for direction in ['tx','rx']:
        values[direction]=[]
        for sid in order:
            rows=[x for x in allrows if x['direction']==direction and x['source_id']==sid]
            assert len(rows)==5 and {x['repetition']for x in rows}==set(range(5))
            measured=[x['seconds']for x in rows if x['phase']=='measured'];assert len(measured)==3
            value=float(np.mean(measured));values[direction].append(value)
            source_rows.append(dict(source_id=sid,direction=direction,seconds=value,measured_repetitions=3))
    values['total']=(np.array(values['tx'])+np.array(values['rx'])).tolist()
    def stats(a):
        a=np.asarray(a,dtype=np.float64)*1000
        lo,hi=np.quantile(a[bank].mean(1),[.025,.975])
        return dict(mean_ms=float(a.mean()),ci_low_ms=float(lo),ci_high_ms=float(hi))
    previous=list(csv.DictReader((old/'source_means.csv').open()))
    deltas=[]
    for direction,a in values.items():
        summary.append(dict(method='Ada_compiled_paired_prefill_FP32',direction=direction,sources=32,**stats(a)))
        before=[]
        for sid in order:
            if direction=='total':v=sum(float(next(x for x in previous if x['method']=='Ada' and x['source_id']==sid and x['direction']==d)['seconds'])for d in ['tx','rx'])
            else:v=float(next(x for x in previous if x['method']=='Ada' and x['source_id']==sid and x['direction']==direction)['seconds'])
            before.append(v)
        deltas.append(dict(comparison='new_compiled_prefill_minus_serial_audit',direction=direction,sources=32,**stats(np.array(a)-np.array(before))))
    baseline={d:read(Path(read(repo/'results/nestcom_timing_fairness_20261102_v1/clean_gpu6_v1'/('main_g05_'+d+'_registration.json'))['baseline_rows']['path']))for d in ['tx','rx']}
    matches={}
    for d,key in [('tx','emitted_symbols_sha256'),('rx','image_sha256')]:
        expected={v['event']:v for v in baseline[d]}
        rows=[v for v in allrows if v['direction']==d]
        matches[d]=dict(exact_old_output_matches=sum(v['output_sha256']==expected[v['event']][key]for v in rows),total=160)
    write_csv(D/'source_means.csv',source_rows);write_csv(D/'summary.csv',summary);write_csv(D/'paired_deltas.csv',deltas)
    keys=sorted({k for v in allrows for k in v})
    write_csv(D/'call_timings.csv',[{k:v.get(k)for k in keys}for v in allrows])
    save(D/'qualification.json',qual)
    save(D/'statistics_identity.json',dict(**{k:v for k,v in ident.items()if k!='comparison'},
        comparison='compiled paired-prefill numerical implementation minus serial implementation, same source32; source intervals, not host variability',bank_reused=True))
    result=dict(status='COMPLETE_PAIRED_NUMERICAL_IMPLEMENTATION',registration_sha256=sha(registration),
        actual_wait=outer,chain=chain,phases=completions,setups=setups,source_count=32,
        timing_calls=320,warmup_calls=128,measured_calls=192,qualification_reused_from=r['extension']['qualification_reuse'],
        correct_independent_received_prefixes=160,comparison_to_frozen_outputs=matches,
        implementation='Single conditional prefill TX; repeated fixed-shape causal prefill RX with zero unknown suffix and CFG4 for tail only; compiled FP32/math/TF32off.',
        scope='Not the authors released executable; not a replacement of frozen image-quality outputs. CPU classifier and digital link included, cold compile excluded from timings but charged to resource ledger.',
        quality_metrics=0,training=0)
    save(D/'actual_compute_and_reuse.json',result)
    save(D/'closure.json',dict(status='COMPLETE',actual_wait=outer,chain=chain))
    print(json.dumps(dict(status=result['status'],summary=summary,output_matches=matches,resources=chain)))

if __name__=='__main__':main(*sys.argv[1:])

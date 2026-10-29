"""Summarize the closed finite TX run. Does not import a model or resample."""
from pathlib import Path
import csv, hashlib, json, statistics, sys
import numpy as np

L=Path(sys.argv[1]);D=Path(sys.argv[2]);out=L/'timing_v3';owner=L/'timing_owner_v3'
def read(p):return json.loads(p.read_bytes())
def pin(p):
    v=p.read_bytes();return dict(path=str(p),bytes=len(v),sha256=hashlib.sha256(v).hexdigest())
wait=read(owner/'actual_wait.json');assert wait['actual_wait'] and wait['returncode'] in (0,1)
identities=[read(owner/'started.json')['process'],read(owner/'child.json'),read(out/'phy_ready.json')['process']]
assert all(not Path('/proc',str(i['pid'])).exists() for i in identities)
complete=wait['returncode']==0
completion=read(out/('completion.json' if complete else 'failure.json'))
assert read(out/('phy_actual_wait.json' if complete else 'phy_failure_actual_wait.json'))['returncode']==0
phy=read(out/'phy/completion.json');assert phy['ledger']['unresolved']==0
rows=[read(p) for p in sorted((out/'calls').glob('*.json'))]
valid=lambda r:all(r[k] for k in ['tokens_exact','cdfs_exact','streams_exact','endpoint_exact','wave_exact'])
invalid=[r for r in rows if not valid(r)]
if complete:
    assert len(rows)==610 and not invalid and completion['native_counts']=={'prior_scale':6100}
else:
    assert len(rows)==187 and len(invalid)==1 and invalid[0]['key']=='measured/088/4/0'
    assert completion['native_counts']=={'prior_scale':1870}
    assert 'QUALIFICATION_IDENTITY_MISMATCH' in completion['traceback']
measured=[r for r in rows if r['phase']=='measured' and valid(r)]
assert len(measured)==(600 if complete else 176)
positions=sorted({r['source_position'] for r in measured});n_sources=len(positions)
assert all(r['actual_prior_scale_calls']==10 and r['encoded_token_count']==680 for r in rows)
assert all(r['finish_calls']=={4:25,8:49}[r['grid']] for r in rows)
metrics=['tokenizer_ms','prior_cdf_context_ms','ac_encode_ms','endpoint_finish_ms','ac_and_finish_ms','source_traversal_ms','length_lookup_ms','packing_ldpc_qam_ms','phy_with_ipc_ms','total_tx_ms']
summary=[];paired=[];per_source=[]
for grid in [4,8]:
    selected=[r for r in measured if r['grid']==grid];assert len(selected)==(300 if complete else 88)
    for metric in metrics:
        v=[r[metric] for r in selected]
        summary.append(dict(grid='Quarter' if grid==4 else 'Eighth',metric=metric,calls=len(selected),sources=n_sources,mean_ms=statistics.mean(v),median_ms=statistics.median(v),p95_ms=float(np.quantile(v,.95)),minimum_ms=min(v),maximum_ms=max(v)))
for pos in positions:
    p=[r for r in measured if r['source_position']==pos]
    assert len(p)==(6 if complete else 2) and len({r['source_id'] for r in p})==1
    for metric in metrics:
        q=[r[metric] for r in p if r['grid']==4];e=[r[metric] for r in p if r['grid']==8]
        assert len(q)==len(e)==(3 if complete else 1)
        per_source.append(dict(source_position=pos,source_id=p[0]['source_id'],source_index=p[0]['source_index'],metric=metric,quarter_mean_ms=statistics.mean(q),eighth_mean_ms=statistics.mean(e),eighth_minus_quarter_ms=statistics.mean(e)-statistics.mean(q)))
for metric in metrics:
    v=[r for r in per_source if r['metric']==metric];a=[r['eighth_minus_quarter_ms'] for r in v]
    q=statistics.mean(r['quarter_mean_ms'] for r in v);e=statistics.mean(r['eighth_mean_ms'] for r in v)
    paired.append(dict(metric=metric,sources=n_sources,quarter_mean_ms=q,eighth_mean_ms=e,mean_source_paired_delta_ms=statistics.mean(a),median_source_paired_delta_ms=statistics.median(a),p95_source_paired_delta_ms=float(np.quantile(a,.95)),relative_mean_delta_percent=100*(e/q-1),confidence_interval='not computed; descriptive paired timing'))
resources=[read(p) for p in sorted(out.glob('resource_*.json')) if p.name!='resource_admission.json']
counts=[]
for grid in [4,8]:
    rr=[r for r in measured if r['grid']==grid]
    counts.append(dict(grid='Quarter' if grid==4 else 'Eighth',calls=len(rr),prior_scale_calls_per_call=10,encoded_tokens_per_call=680,finish_calls_per_call={4:25,8:49}[grid],mean_length_checks=statistics.mean(r['length_checks'] for r in rr)))
failures=[]
for w,o in [('timing_v1','timing_owner'),('timing_v2','timing_owner_v2')]:
    f=read(L/w/'failure.json');assert f['native_counts']=={}
    failures.append(dict(run=w,failure=pin(L/w/'failure.json'),actual_wait=pin(L/o/'actual_wait.json'),new_scientific_calls=0))
dest=D/'timing';dest.mkdir(exist_ok=False)
def csvout(name,values):
    with (dest/name).open('w',newline='')as f:
        writer=csv.DictWriter(f,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
prefix='' if complete else 'partial_'
csvout('tx_calls.csv',rows);csvout(prefix+'timing_summary.csv',summary);csvout(prefix+'source_paired_deltas.csv',per_source);csvout(prefix+'paired_timing_summary.csv',paired);csvout('operation_counts.csv',counts)
identity=dict(status='COMPLETE' if complete else 'BLOCKED_IDENTITY_MISMATCH',owner_actual_wait=wait,processes_gone=identities,completion=completion,phy_completion=phy,input_call_hashes=[pin(p)for p in sorted((out/'calls').glob('*.json'))],native_identity=read(out/'actual_native_identity.json'),resource_samples=resources,failures_preserved=failures,new_tokenizer_calls=len(rows),new_prior_scale_calls=10*len(rows),new_complete_TX_calls=len(rows),new_RX=0,new_metrics=0,new_noise=0,new_training=0,new_bootstrap=0,valid_measured_calls=len(measured),attempted_measured_calls=len(rows)-10,qualification_calls=4,warmup_calls=6,model_load_completed=1,pre_forward_failed_load_attempts=1,invalid_calls=invalid,full_benchmark_eligible=complete,planned_measured_calls=600,unattempted_TX_calls=610-len(rows),timing_scope='Resident RGB uint8 input to complete host waveform, including preprocessing/H2D/tokenizer/prior/CDF/incremental AC/finalization/lookup/packing/LDPC/QAM and persistent CPU IPC; excluding model load, file I/O and durable audit writes. CPU prior timing is synchronous wall attribution, not isolated kernel time.',resource_scope='Shared PRO6000 server; fixed GPU/affinities/threads and interleaved grid order. Not an exclusive-device speed claim.',registration=pin(D/'timing_registration_routes_v3.json'))
(dest/'actual_compute_and_reuse.json').write_text(json.dumps(identity,indent=2)+'\n')
print(json.dumps(dict(status=identity['status'],total_tx=[r for r in summary if r['metric']=='total_tx_ms'],paired_total=[r for r in paired if r['metric']=='total_tx_ms'],counts=counts)))

from pathlib import Path
import hashlib,json,os,subprocess,time
R=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM');T=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/baseline_strength_20261011/quality_metrics_supplement_20261014_v1');E=R/'experiments/quality-metrics-supplement-20261014/common500_metrics_v1';O=R/'results/quality_metrics_supplement_20261014_v1/common500'
def pin(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def gone(v):
    p=Path('/proc')/str(v['pid'])/'stat';return not p.exists() or int(p.read_text().rsplit(')',1)[1].split()[19])!=v['start_ticks']
evidence={}
for n in ('tests','register'):
    p=T/f'common500_metrics_{n}_control_v1/actual_wait.json';v=json.loads(p.read_text());assert v['actual_wait'] is True and type(v['returncode']) is int and v['returncode']==0 and not any(v[k] for k in ('timed_out','terminated','killed')) and v['metadata_error'] is None and all(gone(v[k]) for k in ('process','parent_process'));evidence[n]=pin(p)
reg=O/'registration.json';v=json.loads(reg.read_text());assert v['logical_pairs']==27000 and v['logical_scalar_scores']==81000 and v['source_count']==500 and v['SNRS']==[4,10,19] and v['new_channel']==0 and v['new_checkpoint_or_policy_selection'] is False
assert pin(E/'common500_metrics.py')==v['scientific_code'] and pin(E/'common500_metrics_control.py')['sha256']=='6b9c20efd6bac679c8727fa79f1bbbaea8c2f27ebf964678262549a65939d3f1'
assert not (T/'common500_metrics_v1').exists() and not (T/'common500_metrics_initial_control_v1').exists() and not (T/'common500_metrics_dispatch_v1').exists()
free=int(subprocess.check_output(['nvidia-smi','--id=3','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip());assert free>=8192
value=dict(schema='COMMON500_SUPPLEMENT_ROOT_PRELAUNCH_REVIEW_V1',status='PASS',reviewer='Codex root source and concrete full-grid review; no independent agent claim',registration=pin(reg),sources={p.name:pin(p) for p in E.iterdir() if p.is_file()},evidence=evidence,
    native_logical_pairs=27000,scalar_positions=81000,unique_pairs=v['unique_pairs'],all_failure_denominators_retained=True,all_9000_original_replay_SHA_match=True,official_weights_preprocessing_frozen_before_grades=True,
    six_methods_fixed=True,paired_source_statistics_fixed=True,no_selection_feedback=True,new_model_or_channel_calls_during_review=0,prelaunch_gpu3_free_MiB=free,observed_unix=time.time())
p=O/'coordinator_review.json';assert not p.exists()
with p.open('x') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
print(json.dumps(dict(review=pin(p),registration=pin(reg),unique_pairs=v['unique_pairs'])))

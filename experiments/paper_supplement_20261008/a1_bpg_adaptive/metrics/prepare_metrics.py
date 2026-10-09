"""Bind actual completed adaptive holdout to the unchanged four-metric runtime."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

ROOT='/home/liulu/projects/VAR_COMM'
REMOTE=ROOT+'/experiments/paper_supplement_20261008/a1_bpg_adaptive/metrics'
METRICS=['psnr_db','lpips_alex','dinov2_vitl14_cosine','convnext_top1_source_prediction']


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def descriptor(local,remote):return {'path':remote,'sha256':sha(local)}


def write(path,value):
    data=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode();p=Path(path)
    if p.exists():
        if p.read_bytes()!=data:raise ValueError('Refusing to replace prepared materials: '+str(p))
    else:p.write_bytes(data)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--workspace',required=True)
    p.add_argument('--adaptive-request',required=True)
    p.add_argument('--adaptive-request-remote',default=ROOT+'/experiments/paper_supplement_20261008/a1_bpg_adaptive/a1b_materials_v1/request.json')
    p.add_argument('--actual',required=True,help='Actual local A1b output mirror containing holdout/completion.json and freeze/{completion,worker_0}.json')
    p.add_argument('--out',required=True)
    p.add_argument('--deadline-unix',type=float,required=True)
    p.add_argument('--max-seconds',type=int,default=7200)
    a=p.parse_args()
    w=Path(a.workspace).resolve();here=Path(__file__).resolve().parent
    oldbase=w/'.research/main_raw64_20261008_paper_supplement'
    template=oldbase/'bpg_metrics_v1/materials_r2/request.json';r=read(template)
    oldexecution=read(oldbase/'bpg_metrics_v1/materials_r2/execution.json')
    oldsource=oldbase/'bpg_metrics_v1/score_bpg_cached_holdout_r2.py'
    original_module=next(x for x in oldexecution['put'] if x['remote_path'].endswith('.py'))
    assert sha(oldsource)==original_module['sha256']==r['worker_sha256']
    q=read(a.adaptive_request);actual=Path(a.actual)
    normal=read(actual/'holdout/completion.json');freeze_normal=read(actual/'freeze/completion.json');freeze=read(actual/'freeze/worker_0.json')
    assert normal['status']=='ADAPTIVE_BPG_STAGE_COMPLETE_V1' and normal['frame_count']==9000
    assert normal['actual_children_waited'] and normal['worker_exit_codes']==[0]*8 and normal['ledger']['unresolved']==0
    assert normal['request_sha256']==sha(a.adaptive_request)==freeze_normal['request_sha256']==freeze['request_sha256']
    assert freeze_normal['actual_children_waited'] and freeze_normal['worker_exit_codes']==[0]
    assert not freeze['holdout_used_for_selection'] and freeze['calibration_source_count']==100
    native_path=oldbase/'bpg_delivery_actual/actual_metrics/completion.json';native=read(native_path)
    main_path=w/'.research/main_raw64_20261007/take_over_v1/current/unified500_actual_source_statistics_r6/completion.json';main=read(main_path)
    assert main['scientific_statistics_completed'] is True
    ids=[x['source_id'] for x in q['records']['holdout']]
    assert ids==native['source_ids']==main['source_ids'] and len(ids)==len(set(ids))==500
    def table(normal,suffix):
        matches=[(p,h) for p,h in normal['outputs'].items() if p.endswith('/'+suffix)]
        assert len(matches)==1
        return {'path':matches[0][0],'sha256':matches[0][1]}
    old_BPG_remote=str(Path(r['out'])).replace('\\','/')
    for key in list(r):
        if key.startswith('BPG_'):del r[key]
    # All original metric/model assets, numerical flags, adapter and statistics
    # are kept verbatim from the actually completed native-BPG metric request.
    code=here/'score_adaptive_cached_holdout.py'
    out_remote=ROOT+'/results/paper_supplement_20261008/a1_bpg_adaptive/metrics_v1'
    request_remote=REMOTE+'/materials_v1/request.json'
    q_remote=a.adaptive_request_remote
    r.update(schema='ADAPTIVE_BPG_CACHED_FOUR_METRIC_REQUEST_V1',worker_sha256=sha(code),out=out_remote,
             adaptive_request=descriptor(a.adaptive_request,q_remote),
             adaptive_holdout_normal=descriptor(actual/'holdout/completion.json',q['out']+'/holdout/completion.json'),
             adaptive_freeze_normal=descriptor(actual/'freeze/completion.json',q['out']+'/freeze/completion.json'),
             adaptive_freeze_worker=descriptor(actual/'freeze/worker_0.json',q['out']+'/freeze/worker_0.json'),
             original_scoring_module={'path':original_module['remote_path'],'sha256':original_module['sha256']},
             original_statistics_completion=descriptor(main_path,ROOT+'/outputs/MAIN-RAW64-20261007/unified500_actual_source_statistics_r6/completion.json'),
             original_source_means=table(main,'source_means.json.gz'),original_summary=table(main,'summary.json'),
             native_BPG_metrics_completion=descriptor(native_path,old_BPG_remote+'/completion.json'),
             native_BPG_source_means=table(native,'source_means.csv'),native_BPG_summary=table(native,'summary.csv'),
             native_BPG_quality_ledger=table(native,'quality_calls.jsonl'),
             source_feature_cache=out_remote+'/source_feature_cache',reference_prepare_cap=500,
             paired_reference_prefixes=['RAW64_PARTIAL_VAR_COMPLETION','P1024','SWIN80K_N1024','BPG_LDPC_N1024'],
             pairing_scope='new adaptive BPG minus each fixed same-source reference; noise means first; no shared-observation claim',
             old_summary_recomputed=False,old_bootstrap_recomputed=False,
             source_feature_prior_audit='Remote targeted check found no persistent prepared-reference feature cache; new first preparation is counted separately',
             deadline_unix=a.deadline_unix,max_seconds=a.max_seconds,
             stop_files=list(dict.fromkeys(r['stop_files']+q['stop_files']+[out_remote+'/STOP'])))
    out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True);write(out/'request.json',r)
    e=dict(status='ACTUAL_METADATA_PREPARED_METRIC_EVALUATION_NOT_EXECUTED',
           put=[dict(local_path=str(code),remote_path=REMOTE+'/'+code.name,sha256=sha(code)),
                dict(local_path=str(out/'request.json'),remote_path=request_remote,sha256=sha(out/'request.json'))],
           argv=[r['python'],'-B',REMOTE+'/'+code.name,'--request',request_remote],
           environment=dict(CUDA_VISIBLE_DEVICES='0',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='6',
                            MKL_NUM_THREADS='6',OPENBLAS_NUM_THREADS='6',PYTHONPATH=':'.join(r['pythonpath'])),
           affinity=[4,5,6,7,8,9],nice=15,max_seconds=a.max_seconds,
           source_reference_prepare_cap=500,reconstructed_unique_quality_cap=9000,
           bootstrap_replicates=10000,bootstrap_seed=2026100701,new_summary_rows=24,new_paired_rows=96,
           new_packet_decodes=0,old_metric_calls=0,old_bootstrap_calls=0,
           phase_guard='Only after actual adaptive holdout wait-zero; never run during a dedicated timing window',
           external_wait_zero_required=True,
           fetch=[dict(remote_path=out_remote+'/'+name,local_path=str(out/'actual'/name))
                  for name in ['completion.json','summary.csv','source_means.csv','paired.csv','failure_breakdown.csv','comparison_summary.csv']])
    write(out/'execution.json',e)
    print(json.dumps({'execution':str(out/'execution.json'),'request_sha256':sha(out/'request.json'),
                      'worker_sha256':sha(code),'source_prepare_cap':500,'new_quality_cap':9000,'new_pairs':96}))


if __name__=='__main__':main()

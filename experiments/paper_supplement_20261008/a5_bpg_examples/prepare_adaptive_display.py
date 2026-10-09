"""Prepare dependent cached scoring before reception finishes; never claim done."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);data=(json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    if p.exists():assert p.read_bytes()==data,'Refuse to overwrite prepared materials'
    else:p.write_bytes(data)


def main():
    p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();w=a.workspace.resolve()
    root='/home/liulu/projects/VAR_COMM';o=root+'/outputs/PAPER-SUPPLEMENT-20261008'
    native_materials=w/'results/paper_supplement_20261008/a5_examples/materials_v2';r=read(native_materials/'request.json')
    native_exec=read(native_materials/'execution.json');new_request=w/'results/paper_supplement_20261008/a5_examples/adaptive_materials_v1/request.json'
    receive=read(new_request);native=w/'paper/figures/a5_six_methods_N1024_13dB_group02'
    done=read(native/'completion.json');assert done['status']=='COMPLETE' and done['remaining_missing_metric_cells']==0
    here=Path(__file__).resolve().parent;worker=here/'adaptive_cached_examples.py'
    core=here/'a5_cached_examples.py';assert sha(core)==r['worker_sha256']==native_exec['worker_sha256']
    runtime=o+'/a5_adaptive_display_runtime_v1/adaptive_cached_examples.py';request=o+'/a5_adaptive_display_materials_v1/request.json'
    r.update(schema='A5_ADAPTIVE_CACHED_FIGURE_REQUEST_V1',worker_sha256=sha(worker),
        core_module=dict(path=native_exec['argv']['collect'][1],sha256=sha(core)),
        receive_request=dict(path=o+'/a5_adaptive_fixed16_materials_v1/request.json',sha256=sha(new_request)),
        receive_completion_path=receive['out']+'/completion.json',
        out=root+'/paper/figures/a5_adaptive_six_methods_N1024_13dB_group02',
        max_quality_calls=16,max_reference_feature_preparations=16,max_seconds=1800,
        original_five_columns_metric_cells_reused=64,source_reference_cells_not_scored=16,
        reference_ConvNeXt_reuse='Reuse native16 same-source prediction after exact source hash and evaluator identity verification',
        receive_status_at_preparation='DEPENDENT_COMPLETION_NOT_ASSUMED; consumer requires actual16 wait-zero before any score',
        old_native_figure_modified=False,new_reconstructions=0,new_PHY_calls=0,new_bootstrap=0,policy_selection=False)
    for key,name in [('native_figure_completion','completion.json'),('native_collection','collection.json'),('native_metrics_completion','bpg_metrics_completion.json')]:
        remote=root+'/paper/figures/a5_six_methods_N1024_13dB_group02/'+name
        r[key]=dict(path=remote,sha256=sha(native/name))
    local=a.out/'request.json';save(local,r)
    e=dict(status='PREPARED_NOT_EXECUTED_WAITING_ACTUAL_RECEIVE_COMPLETION',request_sha256=sha(local),worker_sha256=sha(worker),
        upload=[dict(local=str(worker),remote=runtime),dict(local=str(local.resolve()),remote=request)],
        argv={mode:[r['python'],runtime,'--request',request,'--mode',mode] for mode in ['collect','score','render','export']},
        score_environment=native_exec['score_only_environment'],suggested_score_cpu_affinity=[4,5,6,7,8,9],suggested_score_nice=15,
        new_quality_calls_cap=16,new_reference_feature_preparations_cap=16,new_original_ConvNeXt_predictions=0,max_seconds=1800,
        dependency='Actual adaptive_fixed16 completion must be present with16 sources, profile3009, wait[0,0], resolved <=32 ledger.',
        existing_file_dependencies_verify_only=[r['core_module'],r['native_figure_completion'],r['native_collection'],r['native_metrics_completion']],
        output_directory=r['out'],old_native_figure_retained=True)
    save(a.out/'execution.json',e)
    print(json.dumps(dict(request=str(local),request_sha256=sha(local),worker_sha256=sha(worker),execution=str(a.out/'execution.json')),indent=2))


if __name__=='__main__':main()

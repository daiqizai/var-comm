"""Local preparation only; hashes completed A3 native BPG and the old figure."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--version',default='v2');a=p.parse_args();w=a.workspace.resolve()
    assert a.version.startswith('v') and a.version[1:].isdigit()
    remote='/home/liulu/projects/VAR_COMM';o=remote+'/outputs/PAPER-SUPPLEMENT-20261008'
    worker=w/'experiments/paper_supplement_20261008/a5_bpg_examples/a5_cached_examples.py'
    template=w/'.research/main_raw64_20261008_paper_supplement/bpg_metrics_v1/materials_r2/request.json'
    t=read(template)
    a3=w/'.research/main_raw64_20261008_paper_supplement/a3_fixed16/materials_bpg_v1/request.json'
    done=w/'results/paper_supplement_20261008/a3_timing/actual_v1/results/BPG_LDPC_native256/completion.json'
    actual=read(done)
    assert actual['status']=='COMPLETE' and actual['request_sha256']==sha(a3)
    fig=w/'paper/figures/hifi_development_N1024_13dB_group02'
    old=read(fig/'selection_and_integrity.json')
    assert old['display_images_sha256']==sha(fig/'display_images.npz')
    r={k:t[k] for k in ('root','metric_assets','metric_bindings','original_metric_request','pythonpath',
        'validation_module','original_frozen_policy','suite_adapter_module','original_FINAL_FREEZE','visual_lock','python')}
    r.update(schema='A5_NATIVE_BPG_CACHED_EXAMPLES_REQUEST_V1',worker_sha256=sha(worker),
        out=remote+'/paper/figures/a5_six_methods_N1024_13dB_group02',
        existing_score_module=dict(path=o+'/bpg_metrics_runtime_v1/score_bpg_cached_holdout_r2.py',
            sha256='d8e9f85c83f63a288cf0500e53857bc6e81c205cec93897e4d46f9f177c79d8e'),
        a3_request=dict(path=o+'/a3_timing_v1/materials_bpg/request.json',sha256=sha(a3)),
        a3_completion=dict(path=o+'/a3_timing_v1/results/BPG_LDPC_native256/completion.json',sha256=sha(done)),
        stop_files=[remote+'/STOP',o+'/STOP',o+'/a5_native_bpg_metrics_STOP'],
        source_count=16,SNRs=[13],selected_group02=[4,21,24,29],
        new_quality_call_cap=16,new_reference_preparation_cap=16,max_seconds=1800,
        training_updates=0,new_PHY_calls=0,new_reconstruction_model_calls=0,new_bootstrap=0,
        holdout_used=False,policy_selection=False,old_results_modified=False)
    for key,name in [('old_plan','selection_and_integrity.json'),('old_images','display_images.npz'),('old_provenance','provenance.csv')]:
        r[key]=dict(path=remote+'/paper/figures/hifi_development_N1024_13dB_group02/'+name,sha256=sha(fig/name))
    a.out.mkdir(parents=True,exist_ok=True);dest=a.out/'request.json'
    data=(json.dumps(r,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    if dest.exists():assert dest.read_bytes()==data,'Prepared request exists with different identity'
    else:dest.write_bytes(data)
    runtime=o+'/a5_cached_examples_runtime_'+a.version+'/a5_cached_examples.py';request=o+'/a5_cached_examples_materials_'+a.version+'/request.json'
    commands={mode:[r['python'],runtime,'--request',request,'--mode',mode] for mode in ['collect','score','render','export']}
    execution=dict(status='PREPARED_NOT_EXECUTED',request_sha256=sha(dest),worker_sha256=sha(worker),
        upload=[dict(local=str(worker),remote=runtime),dict(local=str(dest),remote=request)],
        existing_dependencies_verify_only=[r[key] for key in ['old_plan','old_images','old_provenance']],
        argv=commands,score_only_environment={'CUDA_VISIBLE_DEVICES':'0','CUBLAS_WORKSPACE_CONFIG':':4096:8','OMP_NUM_THREADS':'6','MKL_NUM_THREADS':'6'},
        suggested_score_cpu_affinity=[4,5,6,7,8,9],suggested_score_nice=15,
        schedule='CPU collect/render/export; root admits score under existing shared_visual.lock. No automatic background launch.',
        metric_scope='Only BPG16 at 13dB; old five columns read caches, missing fields stay missing; source column is not scored.')
    ep=a.out/'execution.json';ed=(json.dumps(execution,sort_keys=True,indent=2)+'\n').encode()
    if ep.exists():assert ep.read_bytes()==ed
    else:ep.write_bytes(ed)
    print(json.dumps({'request':str(dest),'request_sha256':sha(dest),'worker_sha256':sha(worker),'execution':str(ep)},indent=2))


if __name__=='__main__':main()

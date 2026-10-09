"""Pin the completed A1 freeze/qualification and fixed A3 development sources."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);data=(json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    if p.exists():assert p.read_bytes()==data,'Refuse to overwrite frozen materials'
    else:p.write_bytes(data)
def desc(local,remote):return dict(path=remote,sha256=sha(local))


def main():
    p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--deadline-unix',type=float,required=True);a=p.parse_args();w=a.workspace.resolve()
    assert time.time()<a.deadline_unix<=time.time()+3600,'Prepare a near-term finite window; owner runtime remains <=1800s'
    root='/home/liulu/projects/VAR_COMM';o=root+'/outputs/PAPER-SUPPLEMENT-20261008'
    original=w/'results/paper_supplement_20261008/a1_bpg_adaptive/a1b_materials_v1/request.json';q=read(original)
    actual=w/'results/paper_supplement_20261008/a1_bpg_adaptive/a1b_adaptive_v1'
    freeze=read(actual/'freeze/worker_0.json');qual=read(actual/'qualification/worker_0.json')
    assert freeze['selected_profile_ids']['13']==3009 and freeze['request_sha256']==qual['request_sha256']==sha(original)
    a3=w/'.research/main_raw64_20261008_paper_supplement/a3_fixed16/materials_bpg_v1/request.json';sources=read(a3)
    here=Path(__file__).resolve().parent;worker=here/'adaptive_fixed16.py'
    remote_materials=o+'/a5_adaptive_fixed16_materials_v1';remote_runtime=o+'/a5_adaptive_fixed16_runtime_v1/adaptive_fixed16.py'
    remote_out=o+'/a5_adaptive_fixed16_v1';a.out.mkdir(parents=True,exist_ok=True)
    empty=a.out/'empty_native_cache_index.json'
    save(empty,dict(status='NO_DEVELOPMENT_NATIVE_STREAM_CACHE_AVAILABLE',outputs={},scientific_result=False))
    r=dict(schema='A5_ADAPTIVE_FIXED16_RECEIVE_REQUEST_V1',worker_sha256=sha(worker),
        original_adaptive_request=desc(original,root+'/experiments/paper_supplement_20261008/a1_bpg_adaptive/a1b_materials_v1/request.json'),
        original_adaptive_freeze=desc(actual/'freeze/worker_0.json',q['out']+'/freeze/worker_0.json'),
        original_adaptive_freeze_completion=desc(actual/'freeze/completion.json',q['out']+'/freeze/completion.json'),
        original_adaptive_qualification=desc(actual/'qualification/worker_0.json',q['out']+'/qualification/worker_0.json'),
        original_adaptive_qualification_completion=desc(actual/'qualification/completion.json',q['out']+'/qualification/completion.json'),
        a3_request=desc(a3,o+'/a3_timing_v1/materials_bpg/request.json'),
        empty_native_cache_index=desc(empty,remote_materials+'/empty_native_cache_index.json'),
        disabled_native_cache_root=remote_out+'/NO_INHERITED_NATIVE_CODEC_CACHE',out=remote_out,
        independent_ledger=remote_out+'/a5_phy_budget.sqlite',phase_caps={'development':32},
        source_indices=sources['source_indices'],records=sources['records'],SNRs=[13],noise_seeds=[2001],
        noise_namespace='A3_DEVELOPMENT_TIMING',profile=next(x for x in q['catalogue'] if x['profile_id']==3009),
        source_search_rule='Unchanged frozen adaptive source_rows: all15 catalogue capacities, coarse QPs and capacity-straddling fine QPs; send only frozen13dB profile3009.',
        source_case_cap=16*4*52,source_case_cap_per_image=208,source_selection_truth_scope='Source encoding only; receiver receives only actual parsed bytes.',
        resources=dict(workers=2,threads=2,nice=15,affinities=[[10,11],[12,13]]),
        deadline_unix=a.deadline_unix,max_seconds=1800,
        stop_files=list(dict.fromkeys([root+'/STOP',o+'/STOP',remote_out+'/STOP'])),
        new_qualification_calls=0,training_updates=0,policy_selection=False,new_metric_calls=0,
        original_A1_runtime_modified=False,original_A1_ledger_modified=False,holdout_used=False,
        old_native_display_retained=True,external_parent_wait_zero_required=True)
    request=a.out/'request.json';save(request,r)
    e=dict(status='PREPARED_NOT_EXECUTED',request_sha256=sha(request),worker_sha256=sha(worker),
        upload=[dict(local=str(worker),remote=remote_runtime),dict(local=str(request.resolve()),remote=remote_materials+'/request.json'),
                dict(local=str(empty.resolve()),remote=remote_materials+'/empty_native_cache_index.json')],
        argv=[q['python']['path'],'-B',remote_runtime,'--request',remote_materials+'/request.json'],
        environment=dict(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'),
        child_workers=2,child_affinities=[[10,11],[12,13]],nice=15,maximum_NEW_PHY=32,maximum_total_frames=16,
        maximum_source_cases=3328,max_seconds=1800,out=remote_out,deadline_unix=a.deadline_unix,
        output_contract='completion.json after real wait-zero; sources/{historical_source_index:04d}.json; reconstructions/{historical_source_index:04d}.npz contains float32 CHW rgb + source_rgb.',
        next_phase='Root schedules GPU score of only16 new adaptive outputs. Export a separate adaptive six-column figure; retain native256 figure unchanged.')
    save(a.out/'execution.json',e)
    print(json.dumps(dict(request=str(request),request_sha256=sha(request),worker_sha256=sha(worker),execution=str(a.out/'execution.json')),indent=2))


if __name__=='__main__':main()

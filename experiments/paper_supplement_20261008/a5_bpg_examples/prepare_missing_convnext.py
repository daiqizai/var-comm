"""Prepare the independent missing-only job after the main v2 request exists."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,value):
    data=(json.dumps(value,sort_keys=True,indent=2)+'\n').encode()
    if p.exists():assert p.read_bytes()==data,'Prepared output differs'
    else:p.write_bytes(data)


def main():
    p=argparse.ArgumentParser();p.add_argument('--main-materials',type=Path,required=True);a=p.parse_args()
    d=a.main_materials;main_request=d/'request.json';r=read(main_request);execution=read(d/'execution.json')
    worker=Path(__file__).with_name('fill_missing_convnext.py').resolve()
    core=Path(__file__).with_name('a5_cached_examples.py').resolve()
    assert sha(core)==r['worker_sha256']==execution['worker_sha256'],'Main request/core mismatch'
    argv=execution['argv']['collect'];remote_worker=str(Path(argv[1]).parent/'fill_missing_convnext.py').replace('\\','/')
    remote_request=str(Path(argv[3]).with_name('missing_convnext_request.json')).replace('\\','/')
    s=dict(schema='A5_MISSING_ONLY_CONVNEXT_REQUEST_V1',worker_sha256=sha(worker),
        main_request=dict(path=argv[3],sha256=sha(main_request)),core_module=dict(path=argv[1],sha256=sha(core)),
        max_seconds=600,max_reconstruction_predictions=8,max_reference_predictions=4,
        allowed_methods=['SwinJSCC80k','HiFiDiffCom'],allowed_sources=[4,21,24,29],
        metric='convnext_top1_source_prediction',existing_metrics_recomputed=0,old_metric_files_modified=False,
        new_PHY_calls=0,new_reconstructions=0,new_bootstrap=0,policy_selection=False,training_updates=0)
    out=d/'missing_convnext_request.json';save(out,s)
    e=dict(status='PREPARED_NOT_EXECUTED',request_sha256=sha(out),worker_sha256=sha(worker),
        upload=[dict(local=str(worker),remote=remote_worker),dict(local=str(out.resolve()),remote=remote_request)],
        argv=[r['python'],remote_worker,'--request',remote_request],
        environment=execution['score_only_environment'],suggested_cpu_affinity=execution['suggested_score_cpu_affinity'],
        suggested_nice=15,expected_GB_model='ConvNeXt-Tiny only; no FullSuite or reconstruction model load',
        order='After main collect; preferably after BPG score to reuse 4 matching source predictions; before main export. Main render is CPU-only.',
        max_reconstruction_predictions=8,max_reference_predictions=4,max_seconds=600)
    save(d/'missing_convnext_execution.json',e)
    print(json.dumps(dict(request=str(out),request_sha256=sha(out),worker_sha256=sha(worker)),indent=2))


if __name__=='__main__':main()

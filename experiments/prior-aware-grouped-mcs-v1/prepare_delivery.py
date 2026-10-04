"""Prepare an explicit lightweight publication manifest after real completion."""
from pathlib import Path
import argparse
import gzip
import shutil
from uep_common import read,write,seal,sha,require
from release_tables import copy_light

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);p.add_argument('--out',required=True,type=Path);a=p.parse_args()
    root=a.root.resolve();out=a.out.resolve();result=root/'results/prior_aware_uep_20261004';code=root/'experiments/prior-aware-grouped-mcs-v1'
    report_manifest=result/'publication_manifest.json';report=read(report_manifest)
    require(report['status']=='UEP_STATIC_REPORT_COMPLETE','Real report is incomplete')
    require(read(out/'actual_scores/completion.json')['status']=='UEP_ACTUAL_SCORING_COMPLETE','Actual scores incomplete')
    require(read(out/'p1024_scores/completion.json')['status']=='P1024_BASELINE_EVALUATION_COMPLETE','P1024 evaluation incomplete')
    require(read(out/'receiver_cost/timing.json')['status']=='UEP_UNCACHED_RECEIVER_COST_COMPLETE','Required online receiver timing incomplete')
    for path,digest in report['outputs'].items():require(sha(path)==digest,'Report output changed')
    for name in ('ldpc_qualification.json','stage_a_cost_decision.json','final_codebook.json','final_freeze.json','owner_config_r2.json','environment_completion.json','ldpc_requirements_lock.txt'):
        source=out/name;require(source.exists(),'Delivery receipt missing: '+name);shutil.copyfile(source,result/name)
    shutil.copyfile(out/'receiver_cost/timing.json',result/'receiver_cost.json')
    for name in ('candidate_profiles_N1024.json','candidate_profiles_N2048.json','resource_ledger.csv','K_generation.json','resource_enumeration.json'):
        copy_light(out/'stage_a'/name,result/('resources_'+name))
    gate=read(out/'extension_gate.json')
    note='\n\n2026-10-04: Prior-aware UEP N1024 actual-link evaluation, independent ConvNeXt validation, matched P1024 10 dB, fixed examples and cost measurements completed. '+str(gate['status'])+'. See [UEP report](reports/uep_prior_aware_20261004.md). No new model training or holdout access.\n'
    for name in ('PROGRESS.md','RESEARCH_STATUS.md','EXPERIMENTS.md'):
        path=root/name;text=path.read_text()
        if note.strip() not in text:path.write_text(text+note,encoding='utf-8')
    files=list(result.rglob('*'))+list(code.glob('*'))+[root/'reports/uep_prior_aware_20261004.md',*[root/n for n in ('PROGRESS.md','RESEARCH_STATUS.md','EXPERIMENTS.md')]]
    files=sorted({f.resolve() for f in files if f.is_file()})
    require(all(root in f.parents and f.stat().st_size<10_000_000 for f in files),'Publication artifact size/path invalid')
    paths=[str(f.relative_to(root)) for f in files]
    value=dict(status='FINAL_PUBLICATION_PREPARED',publish_paths=paths,file_sha256={str(f.relative_to(root)):sha(f) for f in files},
        report_manifest_sha256=sha(report_manifest),extension_gate_sha256=sha(out/'extension_gate.json'),training_updates=0)
    seal(out/'publish_manifest.json',value);print('Prepared',len(paths),'explicit publication files')

if __name__=='__main__':main()

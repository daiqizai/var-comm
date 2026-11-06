"""One CPU registration after exact final checkpoint closure, before test scores."""
import json,subprocess,time
from pathlib import Path
from runtime import *
setup()
import numpy as np
from formal_codec_v1 import pixels,digest

def main():
    F=L/'formal500_v1';assert not F.exists(),'Formal registration exists; do not repeat'
    old_paused();actual=json.loads((L/'roles/training_continue_v1/actual_wait.json').read_text())
    assert actual['returncode']==0 and actual['child_gone']
    complete=json.loads((L/'continuation_v1/segment_complete.json').read_text());assert complete['step']==5000 and complete['finished_tranche']
    ck=dict(path=str(CK/'continuation_v1/step005000.pt'),sha256='adaf831899b3b420bb66419fbbd9e0725bbd286451702a2a4a5adcca1b110980',step=5000)
    assert sha(ck['path'])==ck['sha256']
    mp=B/'multirate_reconstruction_20261105_v1/posttrain_v1/source_manifest.json'
    assert sha(mp)=='897f6cff72ed173b07c7912ce8cd44869b2982f14c74431da6ed3168d20c654c'
    manifest=json.loads(mp.read_text());sources=manifest['sources'];assert len(sources)==500
    testhash={v['native_rgb_sha256'] for v in sources};assert len(testhash)==500
    split_proof={}
    for name in ['training_sources.json','development_sources.json','qualification_sources.json']:
        p=L/name;rows=json.loads(p.read_text())['sources'];hashes={v['native_rgb_sha256'] for v in rows}
        assert not(testhash&hashes);split_proof[name]=dict(path=str(p),sha256=sha(p),n_sources=len(rows),native_pixel_overlap=0)
    for row in sources:pixels(row)
    qual=json.loads((L/'qualification_sources.json').read_text())
    for row in qual['sources']:row['preprocessed_sha256']={'256':digest(pixels(row))}
    bank=OLD/'p1_source_analysis_v1/source_bootstrap_bank.npy';bp=OLD/'p1_source_analysis_v1/statistics_identity.json'
    assert sha(bank)=='64080ca2036a9f156a6d88e53b2fa48471ac2bbd837dd2b8a13052a1af7838b0'
    ident=json.loads(bp.read_text());order=[v['source_id'] for v in sources]
    assert ident['sources']==order,(ident['sources'][:2],order[:2])
    b=np.load(bank,allow_pickle=False);assert b.shape==(10000,500) and digest(b)==ident['array_sha256']
    metrics=json.loads((OLD/'metric_assets.json').read_text())
    man=json.loads(Path(metrics['four_config']['manifest']['path']).read_text());clip=man['clip']
    clip['implementation']['path']=str(R.parents[1]/'var_comm_runtime_20261010/assets/clip_i_frozen_v1/sources/clip')
    clip['weights']['path']='/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-checkpoint/liulu/var_comm/code_assets_20261028/var_comm_runtime_20261010/assets/clip_i_frozen_v1/weights/ViT-L-14.pt'
    for name,h in clip['implementation']['files'].items():assert sha(Path(clip['implementation']['path'])/name)==h
    assert sha(clip['weights']['path'])==clip['weights']['sha256'];metrics['clip']=clip
    F.mkdir();write(F/'source_manifest.json',manifest);write(F/'qualification_manifest.json',qual);write(F/'metric_assets.json',metrics)
    write(F/'split_audit.json',dict(training_dev_qualification=split_proof,heldout500_native_and_preprocessed_hashes_verified=True,
        historical_manifest=dict(path=str(mp),sha256=sha(mp)),bootstrap_order_exact=True))
    binding_paths=list(S.glob('formal_*_v1.py'))+[S/'register_formal_v1.py',S/'runtime.py',S/'model_v2.py',S/'resize_gradient_v2.py',S/'packet.py',
        L/'model_routes.json',R/'experiments/rdvq-prefix-20261104/owned_process.py',R/'experiments/rdvq-prefix-20261104/metrics_v2.py',
        R/'experiments/rdvq-prefix-20261104/training_core.py',R/'src/var_comm/next_scale_prior.py',
        R/'experiments/scale-causal-partial-residual-20261002/partial_receiver.py',
        R/'experiments/wcl-evidence-closure-20261009/scripts/t1_entropy_core.py',
        R/'experiments/var-latent-enhancement-20260917/src/latent_enhancement/latent.py',
        F/'source_manifest.json',F/'qualification_manifest.json',F/'metric_assets.json',bp]
    bindings={str(p):sha(p) for p in binding_paths}
    now=time.time();cpus=json.loads((L/'continuation_v1.json').read_text())['rank_cpus']
    r=dict(schema='NEST_RD_TRANSFER_FORMAL500_V1',registered=now,deadline=now+4*3600,
        checkpoint=ck,model_routes=str(L/'model_routes.json'),frozen_quantizer_sha256='902e2d17a8a8d1d96cde263484a47f89326126866b053c2ac14dab3f9335693e',
        methods=['original_nest','trained5000'],manifest=str(F/'source_manifest.json'),source_capacities=[591,1273,1911,2548],
        endpoint_counts=list(range(30,681)),packet_overhead=dict(profile=12,length=13,crc=16),
        full_condition='All680 tokens, independently decoded canonical unbounded arithmetic stream; not an equal-bit fixed frame',
        channel='clean recovered information packet, no new PHY or random noise repetitions',
        statistics='10000 direct source-paired bootstrap; 50 mean CI and25 paired CI; all500 denominators',
        bank=str(bank),bank_sha256=sha(bank),bank_identity=str(bp),metric_assets=str(F/'metric_assets.json'),
        numerical='FP32, TF32off, mathSDPA, batch1, deterministic, same physical device per source for both models',
        qualification_sources=2,formal_sources=500,formal_logical_results=5000,
        gpu_device_hour_limit=12,cpu_process_hour_limit=48,qualification_max_seconds=3600,formal_shard_max_seconds=4500,
        checkpoint_selection='User fixed terminal step5000 before formal evaluation; no quality-based checkpoint selection',
        execution_failure='Preserve evidence; do not retry automatically, drop sources, or convert execution errors into gray images',
        old_jobs_remain_paused=True,bindings=bindings,
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip())
    write(F/'protocol.json',r);bindings=dict(bindings);bindings[str(F/'protocol.json')]=sha(F/'protocol.json')
    denied={str(Path(v['path']).parent.absolute()) for v in sources+qual['sources']}
    denied|={str((F/'qualification/tx').absolute())}|{str((F/f'shard{i}/tx').absolute()) for i in range(8)}
    requests=[]
    for label,shard,shards,qualification in [('qualification',0,1,True)]+[(f'shard{i}',i,8,False) for i in range(8)]:
        q=dict(r,bindings=bindings,qualification=qualification,shard=shard,shards=shards,
            manifest=str(F/('qualification_manifest.json' if qualification else 'source_manifest.json')),
            output=str(F/label),gpu=shard,cpus=cpus[str(shard)],rx_denied_roots=sorted(denied),
            max_worker_seconds=3600 if qualification else 4500)
        p=F/'requests'/(label+'.json');write(p,q);requests.append(dict(path=str(p),sha256=sha(p)))
    write(F/'registration_complete.json',dict(passed=True,protocol_sha256=sha(F/'protocol.json'),requests=requests,scientific_calls=0,time=time.time()))
    print(json.dumps(dict(protocol_sha256=sha(F/'protocol.json'),checkpoint=ck,sources=500,logical_results=5000,split_disjoint=True,bank_order_exact=True)))

if __name__=='__main__':main()

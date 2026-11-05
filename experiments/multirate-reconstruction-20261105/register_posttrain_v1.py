"""One-time CPU identity registration before any final-checkpoint quality result."""
from pathlib import Path
import csv,json,time,hashlib,sys
from runtime import setup,L,OLD,R,sha,write
setup()
import numpy as np
from PIL import Image
from torchvision.transforms import functional as TF
from packet import SourcePacket
from collect_snapshot import collect

def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def main():
    root=L/'posttrain_v1';root.mkdir(exist_ok=False)
    for n in ['roles','worker_requests','owner_requests']:(root/n).mkdir()
    snap=collect();assert snap['active_training_role']=='training_world8_v1' and snap['old_P2_stop']
    assert snap['ledger']['training_world8_v1']['owner_live'] and snap['ledger']['training_world8_v1']['child_live']
    for path,h in snap['source_bindings'].items():assert sha(path)==h
    previous=OLD/'p1_source_analysis_v1';metric=OLD/'p1_source_metrics_v1'
    for folder in [previous,metric]:
        wait=json.loads((folder/'actual_wait.json').read_text());assert wait['returncode']==0 and wait['child_gone']
        assert json.loads((folder/'complete.json').read_text())['passed']
    manifest=OLD/'p1_registration_v1/evaluation_manifest.json';all_rows=json.loads(manifest.read_text())['sources']
    test=[v for v in all_rows if v['split']=='test'];assert len(test)==500 and len({v['source_id'] for v in test})==500
    native={v['source_id']:v for v in json.loads((OLD/'data_union_verified_v1/test_manifest.json').read_text())['sources']}
    trainpath=Path(snap['training']['latest_checkpoint.json']['corpus']);train=json.loads(trainpath.read_text())['sources'];trainhash={v['native_rgb_sha256'] for v in train}
    scores={(v['source'],int(v['slices'])):v for v in csv.DictReader((metric/'scores.csv').open())}
    from posttrain_codec_v1 import BUDGETS,METRICS
    sources=[];baseline=[];checks=0
    for i,row in enumerate(test):
        sid=row['source_id'];n=native[sid];assert row['sha256']==n['sha256']==sha(row['path'])
        with Image.open(row['path']) as im:
            im=im.convert('RGB');pixels=np.asarray(im).copy();assert digest(pixels)==n['native_rgb_sha256'] and digest(pixels) not in trainhash
            hashes={str(size):digest(np.asarray(TF.center_crop(TF.resize(im,size,interpolation=TF.InterpolationMode.BICUBIC,antialias=True),[size,size])).copy()) for size in [256,512]}
        source=dict(source_id=sid,split='test',path=row['path'],sha256=row['sha256'],native_rgb_sha256=n['native_rgb_sha256'],preprocessed_sha256=hashes)
        cached=np.load(row['source_rgb_path'],allow_pickle=False);assert cached.dtype==np.uint8 and digest(cached)==hashes['256']
        reference=np.ascontiguousarray(cached.transpose(2,0,1).astype(np.float32)/255);rh=digest(reference)
        lengths={0:72,**{v['slices']:v['packet_bits'] for v in json.loads(Path(row['lengths_path']).read_text())}};assert set(lengths)==set(range(29))
        tx=np.load(row['indices_path'],allow_pickle=False);mask=np.load(Path(row['indices_path']).parent/'slice_mask.npy',allow_pickle=False)
        verified={}
        for budget in BUDGETS:
            ep=max(k for k,v in lengths.items() if v<=budget)
            if ep not in verified:
                wirepath=Path(row['public_directory'])/f's{ep:02d}.bin';wire=wirepath.read_bytes();packet=SourcePacket.from_bytes(wire);assert packet.slices==ep and len(wire)*8==lengths[ep]
                folder=Path(row['rx_directory']);receipt=json.loads((folder/f's{ep:02d}_complete.json').read_text())
                yp=folder/f's{ep:02d}_rgb.npy';y=np.load(yp,allow_pickle=False);yh=digest(y);score=scores[(sid,ep)]
                assert sha(wirepath)==receipt['packet_sha256'] and yh==receipt['rgb_sha256']==score['reconstruction_sha256'] and score['reference_sha256']==rh
                if ep:
                    got=np.load(folder/f's{ep:02d}_indices.npy',allow_pickle=False);assert np.array_equal(tx[:,mask<ep],got[:,mask<ep])
                verified[ep]=dict(score=score,rgb_path=str(yp),rgb_sha256=yh,tokens=int(np.sum(mask<ep)))
                checks+=1
            v=verified[ep]
            baseline.append(dict(source=sid,size=256,method='original',budget_bits=budget,packet_bits=lengths[ep],endpoint=ep,received_tokens=v['tokens'],null=int(ep==0),reference_sha256=rh,reconstruction_sha256=v['rgb_sha256'],**{m:float(v['score'][m]) for m in METRICS},origin='identity_verified_existing_RDVQ',reconstruction_path=v['rgb_path']))
        sources.append(source)
        write(root/'registration_progress.json',dict(sources=i+1,total=500,unique_existing_rx_verified=checks))
    write(root/'source_manifest.json',dict(sources=sources,preprocessing='PIL RGB; shorter-side bicubic antialias resize and center crop to registered resolution; identical source pixels for both methods',training_manifest_sha256=sha(trainpath),all_native_pixels_disjoint_from_training=True))
    with (root/'original256.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(baseline[0]));w.writeheader();w.writerows(baseline)
    bank=previous/'source_bootstrap_bank.npy';stat=previous/'statistics_identity.json';identity=json.loads(stat.read_text())
    assert identity['sources']==[v['source_id'] for v in sources]
    arr=np.load(bank,allow_pickle=False);assert arr.shape==(10000,500) and digest(arr)==identity['array_sha256']
    source=Path(__file__).resolve().parent
    files=[p for p in source.glob('posttrain_*_v1.py')]+[source/'register_posttrain_v1.py']
    files+=[source/n for n in ['runtime.py','model.py','codec.py','train_world8_v1.py','world_migration_v1.py']]
    files+=[R/'experiments/rdvq-prefix-20261104'/n for n in ['codec_adapter.py','packet.py','metrics_v2.py','training_core.py','owned_process.py']]
    files += [root/'source_manifest.json',root/'original256.csv',OLD/'metric_assets.json',OLD/'checkpoint.json',stat,bank]
    bindings={str(p):sha(p) for p in files}
    current=json.loads((L/'training_world8_v1/request.json').read_text())
    protocol=dict(schema='multirate-posttraining-v1',root=str(root),registered=time.time(),deadline=time.time()+7*86400,
        user_request='Run tests immediately after training finishes; prior full 100k256+50k512 schedule unchanged',
        target_step=150000,initial_training_role='training_world8_v1',max_new_training_segments=8,
        training_segment_owner_max_seconds=43200,training_segment_max_GPU_device_hours=96,maximum_additional_training_GPU_device_hours=768,
        evaluation_max_GPU_device_hours=96,evaluation_role_max_seconds=7200,evaluation_roles=48,resource_wait_max_seconds=3600,
        manifest=str(root/'source_manifest.json'),baseline256=str(root/'original256.csv'),metric_assets=str(OLD/'metric_assets.json'),
        bootstrap_bank=str(bank),bootstrap_bank_sha256=sha(bank),bootstrap_identity=str(stat),rank_cpus=current['rank_cpus'],bindings=bindings,
        source_roots=[str(OLD/'data'),str(OLD/'data_priority'),str(L/'data'),str(L/'corpus_coco116k_v2'),str(L/'corpus_openimages100k_v1')],
        resolutions=[256,512],budgets256=list(BUDGETS),budgets512=[4*b for b in BUDGETS],test_sources=500,
        metrics=list(METRICS),bootstrap_resamples=10000,comparison='new final checkpoint minus original public codec; actual complete source-packet budget',
        overhead_bits=dict(new=136,original=72),original512='native four 256-patch encoding, one 28-slice rANS stream; fixed shared resolution',
        numerical_route='FP32 inference / TF32 off / math SDPA, same registered metric assets; training remains unchanged BF16 E-D',
        no_new_training_or_tuning_in_test=True,no_new_PHY=True,no_old_P2_resume=True,unknown_failure='retain evidence; no automatic retry; independent resolution may continue',
        test_selection='scheduled final checkpoint only, no test-based choice',new_data_admission=False)
    write(root/'protocol.json',protocol)
    write(root/'registration_complete.json',dict(passed=True,sources=500,baseline_logical_cache_reuse=4000,unique_native_RX_checks=checks,
        new_model_calls=0,new_metric_calls=0,training_running=True,old_four_paused=True,protocol_sha256=sha(root/'protocol.json'),bindings=len(bindings)))
    print(json.dumps(dict(protocol=str(root/'protocol.json'),sources=500,baseline_cache_positions=4000,existing_RX_verified=checks)))

if __name__=='__main__':main()

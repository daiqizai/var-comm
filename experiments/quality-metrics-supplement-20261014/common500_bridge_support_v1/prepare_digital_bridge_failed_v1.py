"""Byte-only selection of already sealed RAW/whole/BPG native RGB outputs."""
from pathlib import Path, PurePosixPath
import collections,csv,hashlib,json,os,tarfile,time,traceback
import numpy as np
HERE=Path(__file__).resolve().parent
DRIVE=Path('D:/Lu_Data/VAR_COMM_BASELINE_STRENGTH_20261011/quality_metrics_supplement_20261014_v1')
DEST=DRIVE/'digital_input_bridge_v1'
REMOTE=PurePosixPath('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/baseline_strength_20261011/quality_metrics_supplement_20261014_v1/digital_input_bridge_v1')
CONTEXT=HERE/'context'
METHODS=('RAW_PARTIAL','EC_VAR_WHOLE','BPG_ADAPTIVE_DOWNSAMPLING_LDPC_N1024')
SNRS=(4,10,19)
def long(p):return Path('\\\\?\\'+str(Path(p).resolve()))
def sha(p):
    h=hashlib.sha256()
    with long(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def pin(p):return dict(path=str(p),sha256=sha(p),bytes=long(p).stat().st_size)
def save(p,v):
    with long(p).open('x',encoding='utf8',newline='\n') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def read(p):return json.loads(long(p).read_text(encoding='utf8'))
def ash(a):
    a=np.ascontiguousarray(a);return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()
def bsha(a):return hashlib.sha256(b'float32:3,256,256:RGB\0'+np.ascontiguousarray(a).tobytes()).hexdigest()
def pixels(a):
    assert a.dtype==np.float32 and a.shape==(3,256,256) and np.isfinite(a).all() and a.min()>=0 and a.max()<=1
    return np.ascontiguousarray(a)
def rows(name):
    with (CONTEXT/name).open(encoding='utf8',newline='') as f:return list(csv.DictReader(f))
def main():
    assert not DEST.exists(),'Do not repeat/overwrite an existing bridge attempt'
    DEST.mkdir();(DEST/'images').mkdir();started=time.monotonic()
    plan=read(DRIVE/'restore_plan.json');restored=read(DRIVE/'input_restore/completion.json');assert restored['all_selected_file_SHA_PASS'] is True
    paths={r['old_path']:(DRIVE/'input_restore'/r['path'],r) for r in plan['files']}
    whole_done=read(DRIVE/'whole19_restore_v1/completion.json');assert whole_done['all_file_SHA_PASS'] is True
    for r in whole_done['files']:paths[r['old_path']]=(DRIVE/'whole19_restore_v1'/r['restore_path'],r)
    verified={}
    def original(old):
        p,r=paths[old]
        if old not in verified:
            before=long(p).stat();assert before.st_size==r['bytes'] and sha(p)==r['sha256'];after=long(p).stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
            verified[old]=dict(old_path=old,bytes=r['bytes'],sha256=r['sha256'])
        return long(p)
    roster=read(CONTEXT/'source_records.json')['records'];assert len(roster)==500 and [s['source_index'] for s in roster]==list(range(500))
    refs={}
    for r in rows('P100000_frame_rows.csv'):
        i=int(r['source_index']);identity=(r['source_id'],r['reference_sha256'])
        assert i not in refs or refs[i]==identity;refs[i]=identity
    whole=collections.defaultdict(list);bpg=collections.defaultdict(list)
    for r in rows('whole_frame_rows.csv'):
        if r['family']=='EC_VAR_WHOLE' and int(r['snr_db']) in SNRS:whole[int(r['source_index'])].append(r)
    for r in rows('bpg_frame_rows.csv'):
        if int(r['snr_db']) in SNRS:bpg[int(r['source_index'])].append(r)
    frames=[];sources=[];archives=[]
    rawbase='/home/liulu/projects/VAR_COMM/outputs/MAIN-RAW64-20261007/unified500_same_rx_parallel_images_r2'
    bpgbase='/home/liulu/projects/VAR_COMM/results/paper_supplement_20261008/a1_bpg_adaptive/a1b_adaptive_v1/holdout/reconstructions'
    for i,source in enumerate(roster):
        sid=source['source_id'];assert refs[i][0]==sid
        raw_image_old=f'{rawbase}/images/{i:04d}.npz';raw_row_old=f'{rawbase}/sources/{i:04d}.json';bpg_old=f'{bpgbase}/{i:04d}.npz'
        raw_rows=read(original(raw_row_old))
        with np.load(original(raw_image_old),allow_pickle=False) as z:raw_images=z['images'].copy()
        with np.load(original(bpg_old),allow_pickle=False) as z:bp_images=z['images'].copy();source_rgb=pixels(z['source_rgb'].copy());row_ids=z['row_ids'].tolist();slots=z['image_slots'].tolist()
        assert len(row_ids)==len(slots)==18 and len(set(row_ids))==18
        assert ash(source_rgb)==refs[i][1]
        # Exact float source is checked against the old continuous reference SHA;
        # old BPG reference uses a different, explicitly retained RGB digest.
        assert all(bsha(source_rgb)==r['reference_sha256'] and r['source_id']==sid for r in bpg[i])
        images=[];bysha={};local=[]
        def add(method,snr,seed,a,expected,proof,gray,header,status):
            a=pixels(a);h=ash(a);assert h==expected
            if h not in bysha:bysha[h]=len(images);images.append(a.copy())
            local.append(dict(method=method,source_index=i,source_id=sid,N=1024,snr_db=snr,noise_seed=seed,image_key='images',image_slot=bysha[h],
                reconstruction_sha256=h,reference_rgb_sha256=refs[i][1],fixed_gray=gray,header_accepted=header,failure_status=status,original_evidence=proof))
        selected=[r for r in raw_rows if 'PARTIAL' in r['families'] and int(r['snr_db']) in SNRS]
        assert len(selected)==9 and {(int(r['snr_db']),int(r['noise_seed'])) for r in selected}=={(s,n) for s in SNRS for n in (6201,6202,6203)}
        for r in selected:
            assert r['source_index']==i and r['source_id']==sid and r['N']==1024
            arm=r['same_RX_images']['VAR_completion']
            proof=dict(source_metadata=verified[raw_row_old],source_archive=verified[raw_image_old],actual_received_state_sha256=r['received_sha256'],
                actual_state_sha256=r['actual_state_sha256'],noise_sha256=r['noise_sha256'],original_slot=arm['image_slot'],image_archive=arm['image_archive'],
                K=r['K'],m=r['m'],source_status=r['source_status'],header_ok=r['header_ok'],body_crc_accept=r['body_crc_accept'],population_role=r['population_role'],policy_selection=r['policy_selection'])
            add(METHODS[0],int(r['snr_db']),int(r['noise_seed']),raw_images[int(arm['image_slot'])],arm['image_sha256'],proof,r['gray'],r['header_ok'],r['status'])
        assert len(whole[i])==9
        loaded={raw_image_old:raw_images}
        for r in whole[i]:
            assert r['source_id']==sid and int(r['N'])==1024
            old=r['image_path'];p=original(old);assert verified[old]['sha256']==r['image_file_sha256']
            if old not in loaded:
                with np.load(p,allow_pickle=False) as z:loaded[old]=z[r['image_key']].copy()
            a=loaded[old] if r['image_key']=='image' else loaded[old][int(r['image_slot'])]
            proof=dict(csv='whole_frame_rows.csv',source_archive=verified[old],original_key=r['image_key'],original_slot=r['image_slot'],received_state_sha256=r['received_state_sha256'],
                source_status=r['source_status'],physical_status=r['physical_status'],point_id=r['point_id'],family=r['family'],diagnostic_truth_used_for_output=r['diagnostic_truth_used_for_output'])
            assert r['diagnostic_truth_used_for_output']=='False'
            add(METHODS[1],int(r['snr_db']),int(r['noise_seed']),a,r['image_sha256'],proof,r['gray'],r['header_ok'],r['physical_status'])
        assert len(bpg[i])==9
        for r in bpg[i]:
            j=row_ids.index(r['frame_id']);slot=int(slots[j]);a=pixels(bp_images[slot]);assert bsha(a)==r['image_sha256']
            proof=dict(csv='bpg_frame_rows.csv',source_archive=verified[bpg_old],frame_id=r['frame_id'],original_slot=slot,legacy_digest='sha256(float32:3,256,256:RGB\\0 + contiguous native bytes)',
                legacy_image_sha256=r['image_sha256'],legacy_reference_sha256=r['reference_sha256'],profile_id=r['profile_id'],selected_resolution=r['selected_resolution'],selected_qp=r['selected_qp'],
                header_rejected=r['header_rejected'],header_wrong_accept=r['header_wrong_accept'],undetected_payload_difference=r['undetected_payload_difference'])
            add(METHODS[2],int(r['snr_db']),int(r['noise_seed']),a,ash(a),proof,r['gray_substitution'],str(r['header_rejected']=='False'),r['failure_state'])
        assert len(local)==27 and len({(r['method'],r['snr_db'],r['noise_seed']) for r in local})==27
        path=DEST/'images'/f'{i:04d}.npz'
        with long(path).open('xb') as f:np.savez_compressed(f,images=np.stack(images),source_rgb=source_rgb);f.flush();os.fsync(f.fileno())
        archive=dict(path=(REMOTE/'images'/f'{i:04d}.npz').as_posix(),bytes=long(path).stat().st_size,sha256=sha(path),unique_images=len(images));archives.append(archive)
        for r in local:r['archive']=archive;frames.append(r)
        sources.append(dict(source_index=i,source_id=sid,reference_rgb_sha256=refs[i][1],original_reference=source['archive'],bridge_archive=archive))
        if (i+1)%25==0:print('BRIDGED',i+1,'logical',len(frames),'unique',sum(a['unique_images'] for a in archives),flush=True)
    assert len(frames)==13500 and collections.Counter(r['method'] for r in frames)==dict.fromkeys(METHODS,4500)
    save(DEST/'frame_index.json',frames);save(DEST/'sources.json',sources)
    value=dict(schema='COMMON500_SEALED_DIGITAL_PIXEL_BRIDGE_V1',status='PASS_BYTE_ONLY_NO_SCIENTIFIC_CALLS',logical_frames=13500,source_count=500,methods=list(METHODS),SNRS=list(SNRS),
        archive_root=REMOTE.as_posix(),archives=archives,frame_index=dict(path=(REMOTE/'frame_index.json').as_posix(),bytes=(DEST/'frame_index.json').stat().st_size,sha256=sha(DEST/'frame_index.json')),
        sources=dict(path=(REMOTE/'sources.json').as_posix(),bytes=(DEST/'sources.json').stat().st_size,sha256=sha(DEST/'sources.json')),
        original_input_files=list(verified.values()),original_context={n:pin(CONTEXT/n) for n in ('whole_frame_rows.csv','bpg_frame_rows.csv','source_records.json','P100000_frame_rows.csv')},
        restore_evidence={n:pin(DRIVE/n) for n in ('restore_plan.json','input_restore/completion.json','whole19_restore_v1/completion.json')},code=pin(__file__),
        all_original_file_SHA_PASS=True,all_original_pixel_SHA_PASS=True,all_500_reference_SHA_PASS=True,new_channel=0,new_reconstruction=0,new_training=0,new_metric_scores=0,seconds=time.monotonic()-started)
    save(DEST/'manifest.json',value)
    archive=DRIVE/'digital_input_bridge_v1.tar.gz';assert not archive.exists()
    with tarfile.open(archive,'w:gz',compresslevel=1) as tar:
        for p in sorted(DEST.rglob('*')):
            assert not p.is_symlink()
            if p.is_file():tar.add(p,arcname=p.relative_to(DEST).as_posix(),recursive=False)
    save(DEST/'transport_receipt.json',dict(archive=pin(archive),manifest=pin(DEST/'manifest.json'),archive_payload_files=503,raw_bytes=sum(a['bytes'] for a in archives),new_scientific_calls=0))
    print(json.dumps(read(DEST/'transport_receipt.json')),flush=True)
if __name__=='__main__':
    try:main()
    except BaseException:
        if DEST.exists() and not (DEST/'failure.json').exists():save(DEST/'failure.json',dict(status='FAILED_PRESERVE_NO_RETRY',traceback=traceback.format_exc()))
        raise

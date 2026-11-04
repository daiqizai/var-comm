"""Package closed evidence without rerunning models, metrics or statistics.

This reporting helper is separate from the immutable scientific registrations.
Raw arrays, checkpoints, source images and execution logs remain on PFS.
"""
from pathlib import Path
import argparse,csv,hashlib,json,os,re,shutil,subprocess,time


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def write_json(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2)


def write_csv(path,rows,fields=None):
    if not rows:return
    with Path(path).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields or list(rows[0]));w.writeheader();w.writerows(rows)


def require_closed(directory,completion=True):
    directory=Path(directory);wait=read(directory/'actual_wait.json')
    assert wait['returncode']==0 and wait.get('child_gone',False),directory
    if (directory/'child.json').exists():
        ident=read(directory/'child.json');p=Path('/proc',str(ident['pid']),'stat')
        assert not p.exists() or p.read_text().rsplit(')',1)[1].split()[19]!=ident['start_ticks'],directory
    if completion:assert read(directory/'complete.json')['passed'],directory
    return wait


def metadata_directories(root):
    """Do not enumerate the large image/array/cache trees."""
    prefixes=('p0_','p1_','p2_','training_qualification','microbatch','dynamic_phy',
              'cpu_codec','asset_','priority_eval','train_complement','data_union',
              'first_batch_continuation','repository_cpu')
    for top in sorted(root.iterdir()):
        if not top.is_dir() or not top.name.startswith(prefixes):continue
        if top.name in ('p0_public_inputs','p0_ready'):continue
        for base,dirs,files in os.walk(top):
            p=Path(base);depth=len(p.relative_to(top).parts)
            dirs[:]=[n for n in dirs if depth<4 and not re.fullmatch(r'\d{12}',n)
                     and n not in ('public','requests','__pycache__','rate_audits')
                     and not n.startswith('rate_audit_')]
            if 'actual_wait.json' in files:yield p


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--repo',required=True)
    ap.add_argument('--output',required=True);ap.add_argument('--include-p2',action='store_true');a=ap.parse_args()
    L=Path(a.root).resolve();R=Path(a.repo).resolve();O=Path(a.output).resolve();S=R/'experiments/rdvq-prefix-20261104'
    assert O.is_relative_to(R/'results/rdvq_prefix_20261104_v1') and not O.exists()
    assert read(L/'p0_gate.json')['passed']
    require_closed(L/'data_union_verified_v1');require_closed(L/'p1_source_chain_owner_control_v1',False)
    require_closed(L/'p1_source_analysis_v1');require_closed(L/'p1_control_analysis_v1')
    if a.include_p2:
        require_closed(L/'p2_owner_control_v1/study',False)
        require_closed(L/'p2_matched_analysis_v1')
    O.mkdir(parents=True);origin=[]
    def copy(src,dest=None):
        src=Path(src);dst=O/(dest or src.name);assert src.is_file() and src.stat().st_size<10_000_000
        assert src.suffix in ('.json','.csv','.md','.txt'),src
        dst.parent.mkdir(parents=True,exist_ok=True);assert not dst.exists();shutil.copyfile(src,dst)
        origin.append(dict(delivery_path=str(dst.relative_to(O)),source_path=str(src),sha256=sha(src),bytes=src.stat().st_size))
    for name in ['p0_gate.json','p1_source_gate.json','p1_p2_protocol.json','p1_control_protocol.json','dynamic_phy_scoped_gate.json','checkpoint.json','metric_assets.json','native_export_schema_check.json']:
        copy(L/name,'identities/'+name)
    for folder,files in {
        'p1_source_analysis_v1':['summary.csv','paired_deltas.csv','capacity.csv','calibrated_source_budget_targets.json','statistics_identity.json','complete.json','actual_wait.json'],
        'p1_control_analysis_v1':['reliability.csv','calibration_candidates.csv','frozen_reference_policies.csv','expected_summary.csv','expected_paired_deltas.csv','statistics_identity.json','unknown_received_packets.json','accepted_wrong_packets.json','complete.json','actual_wait.json'],
    }.items():
        for name in files:copy(L/folder/name,('source/' if folder=='p1_source_analysis_v1' else 'control/')+name)
    for name in ['request.json','launch.json']:
        copy(L/'first_batch_continuation_registration_v1'/name,'registration/continuation_'+name)
    for folder in ['p1_registration_v1','p2_registration_v1','p1_control_analysis_registration_v1']:
        p=L/folder
        if not p.exists():continue
        for item in sorted(p.glob('*.json')):
            # Large per-source path manifests have canonical identity CSVs and
            # PFS hashes; all compact launch/configuration requests are copied.
            if item.stat().st_size<100_000:
                copy(item,'registration/'+folder+'/'+item.name)
    rows=[];split_identities={}
    for split in ['train','development','calibration','test']:
        p=L/'data_union_verified_v1'/(split+'_manifest.json');data=read(p)['sources']
        split_identities[split]=dict(sources=len(data),path=str(p),sha256=sha(p),source_order=[v['source_id'] for v in data])
        for position,v in enumerate(data):
            rows.append(dict(split=split,position=position,source_id=v['source_id'],name=v['name'],jpeg_sha256=v['sha256'],native_rgb_sha256=v['native_rgb_sha256'],native_rgb_shape=json.dumps(v['native_rgb_shape'])))
    assert len(rows)==22500
    write_csv(O/'source_identities.csv',rows);write_json(O/'split_identity.json',split_identities)
    copy(R/'results/rdvq_prefix_20261104_v1/data_identity_clarification.json','data_identity_clarification.json')
    receipts=[];failures=[]
    for d in metadata_directories(L):
        w=read(d/'actual_wait.json');reg=read(d/'registration.json') if (d/'registration.json').exists() else {}
        argv=reg.get('argv',[]);joined=' '.join(argv);rel=str(d.relative_to(L))
        wrapper=any(v in rel for v in ('owner_control','first_batch_continuation','_chain'))
        wrapper=wrapper or any(v in joined for v in ('_owner.py','p1_coordinator.py','p2_study_controller.py','first_batch_continuation.py'))
        row=dict(role=rel,returncode=w['returncode'],started=w.get('started',''),finished=w.get('finished',''),elapsed_seconds=w.get('elapsed_seconds',''),
                 gpu_device_hours=w.get('charged_gpu_device_hours',0),inclusive_cpu_process_seconds=w.get('cpu_process_seconds',w.get('cpu_seconds','')),
                 accounting='wrapper_inclusive_cpu_do_not_add_to_leaves' if wrapper else 'leaf',actual_wait_sha256=sha(d/'actual_wait.json'))
        receipts.append(row)
        if w['returncode']!=0:
            diag=d/'failure.txt'
            if not diag.exists():diag=d/'stderr.log'
            text=diag.read_text(errors='replace') if diag.exists() else ''
            failures.append(dict(role=rel,returncode=w['returncode'],actual_wait_path=str(d/'actual_wait.json'),actual_wait_sha256=row['actual_wait_sha256'],
                                 diagnostic_path=str(diag) if diag.exists() else None,diagnostic_sha256=sha(diag) if diag.exists() else None,diagnostic_tail=text[-12000:]))
    write_csv(O/'resource_receipts.csv',receipts);write_json(O/'retained_failures.json',failures)
    raw_index=[]
    for p in [L/'p1_source_metrics_v1/scores.csv',L/'p1_source_analysis_v1/test_source_results.csv',L/'p1_source_analysis_v1/source_bootstrap_bank.npy',
              L/'p1_control_analysis_v1/frame_audit.csv',L/'p1_control_analysis_v1/test_expected_source_metrics.csv',L/'p1_r0_rate_proxy_v1/complete.json']:
        raw_index.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p),storage='PFS only; no new calls'))
    p2_status='NOT_STARTED' if not (L/'p2_launch.json').exists() else 'RUNNING_OR_PENDING_ACTUAL_CLOSURE'
    if a.include_p2:
        A=L/'p2_matched_analysis_v1'
        for name in ['summary.csv','paired_deltas.csv','capacity.csv','training_seed_variation.csv','statistics_identity.json','hypothesis_observations.json','complete.json','actual_wait.json']:
            copy(A/name,'matched/'+name)
        actual=list(csv.DictReader((A/'actual_compute.csv').open()));corrected=[]
        arms=read(L/'p2_registration_v1/queue_request.json')['arms'];by_name={v['name']:v for v in arms}
        for v in actual:
            original=float(v.pop('training_GPU_hours'));root=Path(by_name[v['name']]['output_root'])
            train=sum(read(root/f'train_to_{n:05d}/actual_wait.json')['charged_gpu_device_hours'] for n in [2000,4000])
            assert train<=original
            corrected.append(dict(**v,total_GPU_device_hours=original,training_GPU_device_hours=train,evaluation_GPU_device_hours=original-train))
        write_csv(O/'matched/actual_compute.csv',corrected)
        write_json(O/'matched/column_clarification.json',dict(raw_path=str(A/'actual_compute.csv'),raw_sha256=sha(A/'actual_compute.csv'),
                  explanation='Raw training_GPU_hours denotes the entire arm. Delivery renames it total_GPU_device_hours and separates the two actual training leaf receipts. No scientific result or statistic changed.'))
        p2_status='CLOSED_REQUIRES_SCIENTIFIC_INTERPRETATION'
    write_json(O/'raw_evidence_index.json',raw_index)
    write_json(O/'source_snapshot.json',{str(p.relative_to(R)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(S.rglob('*')) if p.is_file() and p.suffix in ['.py','.md']})
    write_json(O/'copy_provenance.json',origin)
    write_json(O/'delivery_status.json',dict(created=time.time(),base_head=subprocess.check_output(['git','-C',str(R),'rev-parse','HEAD'],text=True).strip(),
        P0='CLOSED',P1_source='CLOSED',P1_calibration_PHY='14400 actual events; native packet acceptance audit closed',
        P1_controller='conditional expected reference only; no actual test PHY',P2=p2_status,P3='NOT_RUN_NOT_AUTOMATIC',P4='NOT_RUN_NOT_AUTOMATIC',
        novel_model_calls=0,metric_calls=0,new_statistics=0,review='PENDING',publication='PENDING',
        ledger_scope='Registered process receipts. GPU wrappers charge zero. Inclusive wrapper CPU is retained separately and must not be added to child leaves. Historical CUDA visibility probes are documented separately in resource_snapshot_p0_v2.json.'))
    print(json.dumps(dict(output=str(O),copied=len(origin),source_identities=len(rows),resource_receipts=len(receipts),retained_failed_processes=len(failures),P2=p2_status)))


if __name__=='__main__':main()

"""Read-only R2/R3 complete-study routing for the one-job Phase2 recovery."""
from pathlib import Path
import importlib.util
from history_common import read, write, sha, verify, identity, row_ids, checkpoint_valid
import history_inheritance as r3
from audit_selected_runtime import ROSTER

NAME = 'HISTORICAL-PHASE2-RECOVERY-20261004-R1'
RESULT = 'historical_phase2_recovery_20261004_r1'
OLD = 'HISTORICAL-METRICS-R3-20261003'
OLD_RESULT = 'historical_metrics_r3_20261003'
STUDY = 'OPTIONAL_PHASE2_MAIN'
COMPLETED = {s:dict(adapter=a,frames=n,sources=100) for a,s,n in ROSTER[7:11]}
require = r3.require


def old_queue_path(root):
    return Path(root).resolve()/'outputs'/OLD/'queue_registration.json'


def validate_map(root, queue):
    r3.validate_map(root,queue)
    if 'phase2_recovery' not in queue:return
    original=read(old_queue_path(root))
    require(queue['jobs']==original['jobs'] and queue['coverage_manifest']==original['coverage_manifest']
        and queue['contrasts']==original['contrasts'] and queue['inherited_studies']==original['inherited_studies']
        and queue.get('execution_groups')==original.get('execution_groups'),'Historical selected scope changed')
    require(queue['phase2_recovery']['original_queue_sha256']==sha(old_queue_path(root))
        and queue.get('replay_adapter_overrides')=={STUDY:'historical_phase2_alias'},'Unexpected recovery adapter scope')
    require(set(queue.get('completed_r3_studies',{}))==set(COMPLETED),'Four completed R3 studies required')
    for study,item in queue['completed_r3_studies'].items():
        require(item['out']==str(Path(root).resolve()/'outputs'/OLD/study)
            and item['result']==str(Path(root).resolve()/'results'/OLD_RESULT/study)
            and item['receipt_path']==str(Path(root).resolve()/'outputs'/NAME/'inheritance'/(study+'.json'))
            and queue['input_proof_bindings'].get(item['receipt_path'])==item['receipt_sha256'],
            'Completed R3 admission path differs')
        verify({item['receipt_path']:item['receipt_sha256']})


def verify_inherited(root,queue,study):
    if study not in queue.get('completed_r3_studies',{}):return r3.verify_inherited(root,queue,study)
    item=queue['completed_r3_studies'][study];verify({item['receipt_path']:item['receipt_sha256']})
    proof=read(item['receipt_path']);spec=COMPLETED[study]
    require(proof.get('status')=='VERIFIED_COMPLETE_R3_STUDY' and proof.get('study')==study
        and all(proof.get(k)==v for k,v in spec.items()) and proof.get('source_checkpoints_verified')==100
        and proof.get('float_caches_verified')==100 and proof.get('original_identities_preserved') is True
        and proof.get('metrics_recomputed') is False and proof.get('training_updates')==0
        and proof.get('policy_selection_updates')==0,'Completed R3 admission identity differs')
    require(proof['original_queue']==dict(path=str(old_queue_path(root)),sha256=sha(old_queue_path(root))),
        'Completed R3 original queue differs')
    verify(proof['bindings']);return proof


def study_paths(root,queue,study):
    if study in queue.get('completed_r3_studies',{}):
        item=queue['completed_r3_studies'][study]
        return Path(item['out']),Path(item['result'])
    if study in queue.get('inherited_studies',{}):return r3.study_paths(root,queue,study)
    if 'phase2_recovery' in queue:require(study==STUDY,'Recovery must score only Phase2')
    return Path(root).resolve()/'outputs'/NAME/study,Path(root).resolve()/'results'/RESULT/study


def origin_queue(root,queue,queue_path,study):
    if study in queue.get('completed_r3_studies',{}):
        verify_inherited(root,queue,study);p=old_queue_path(root);return p,read(p)
    return r3.origin_queue(root,queue,queue_path,study)


def admit_completed(root,original):
    """Reuse the frozen, CPU-only 100-source admission algorithm, explicitly R3."""
    import importlib
    root=Path(root).resolve();module_path=Path(r3.__file__)
    spec=importlib.util.spec_from_file_location('_phase2_completed_r3_admission',module_path)
    audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
    audit.PREVIOUS=OLD;audit.PREVIOUS_RESULTS=OLD_RESULT
    audit.CURRENT=NAME;audit.CURRENT_RESULTS=RESULT;audit.INHERITED=COMPLETED
    predecessor=old_queue_path(root)
    pubpath,publication=audit.verify_origin(root,predecessor,original)
    result={}
    for study,specification in COMPLETED.items():
        adapter=importlib.import_module(specification['adapter']).create_adapter(root,study)
        receipt=audit.admit_study(root,study,adapter,predecessor,original,pubpath,publication)
        receipt['status']='VERIFIED_COMPLETE_R3_STUDY'
        path=root/'outputs'/NAME/'inheritance'/(study+'.json')
        require(not path.exists() or read(path)==receipt,'Earlier R3 admission changed')
        write(path,receipt)
        result[study]=dict(version='R3',out=str(root/'outputs'/OLD/study),
            result=str(root/'results'/OLD_RESULT/study),receipt_path=str(path),receipt_sha256=sha(path))
        print('COMPLETED_R3_VERIFIED',study,specification['frames'],flush=True)
    return result


def verify_complete_phase2(root,queue):
    """Final CPU gate: all original 81 payloads plus 19 new strict-grid payloads."""
    from phase2_inheritance import inherited_checkpoint
    from historical_phase2_alias import create_adapter, LIMITS
    from history_float_cache import verify_saved
    root=Path(root).resolve();out,result=study_paths(root,queue,STUDY)
    rp=result/'registration.json';reg=read(rp);donepath=out/'completion.json';done=read(donepath)
    require(done.get('status')=='HISTORICAL_STUDY_METRICS_COMPLETE' and done.get('frames')==4500
        and done.get('sources')==100 and done.get('inherited_sources')==81 and done.get('new_sources')==19
        and done.get('inherited_rows')==3645 and done.get('new_rows')==855
        and done.get('new_metric_offset_applied') is False and done.get('original_r3_files_written') is False,
        'Recovery Phase2 total/new/inherited coverage differs')
    inherited_path=Path(queue['phase2_recovery']['inheritance_path'])
    verify({str(inherited_path):queue['phase2_recovery']['inheritance_sha256']})
    inherited=read(inherited_path);verify(inherited['bindings'])
    aliaspath=result/'historical_numerical_alias.json';adapter=create_adapter(root,STUDY)
    require(read(aliaspath)==adapter.historical_alias_manifest() and sha(aliaspath)==done['historical_numerical_alias_sha256']
        ==reg['historical_numerical_alias_sha256'],'Complete 100-source historical alias disclosure differs')
    fields=('modelmanifest_sha256','evaluator_identity','numerical_runtime','metric_batch_size',
            'batch_qualification_sha256','metric_qualified_batch_sizes')
    evaluation={k:reg[k] for k in fields}
    expected_new={out/'source_checkpoints'/f'{i:04d}.json' for i in range(81,100)}
    require(set((out/'source_checkpoints').glob('*.json'))==expected_new,'Recovery must contain exactly 19 new checkpoints')
    bindings={str(p):sha(p) for p in (rp,donepath,inherited_path,aliaspath)}
    bindings.update(inherited['bindings']);fresh_pure=0
    for i in range(100):
        rows=list(adapter.expected_rows(i))
        if i<81:
            cp=inherited_checkpoint(inherited,i,rows)
        else:
            path=out/'source_checkpoints'/f'{i:04d}.json'
            cp=checkpoint_valid(read(path),identity(reg),row_ids(STUDY,i,rows),expected_rows=rows,
                study=STUDY,source_index=i,evaluation_identity=evaluation,
                qualified_batch_sizes=reg['metric_qualified_batch_sizes'])
            verify_saved(cp['float_reconstructions'],cp['rows'])
            proof=cp['float_reconstructions']
            require(proof['path']==str(out/'reconstructions'/f'{i:04d}.npz'),'New float cache path differs')
            bindings[str(path)]=sha(path);bindings[proof['path']]=proof['sha256']
            for row,parity in zip(cp['rows'],cp['parity']):
                if row['method']!='pure_continuous':continue
                native=parity.get('native_grid_replay_proof',{})
                require(parity.get('new_metric_offset_applied') is False
                    and parity.get('original_phase2_pixel_identity_claimed') is False
                    and parity.get('original_phase2_scalar_parity_claimed') is False
                    and native.get('passed') is True and native.get('replay_parity_passed') is True,
                    'New pure replay lacks explicit strict-grid/historical-alias proof')
                for metric,limit in LIMITS.items():
                    require(abs(float(parity['measured_grid_minus_original_grid'][metric]))<=limit,
                        'New pure grid metric exceeded its unchanged strict tolerance')
                fresh_pure+=1
    require(fresh_pure==285,'Fresh pure-grid proof coverage differs')
    # Bind the actual old qualification, without claiming it was recomputed.
    original_qualification=root/'outputs'/OLD/STUDY/'first_source_qualification.json'
    bindings[str(original_qualification)]=sha(original_qualification)
    return dict(status='PHASE2_RECOVERY_100_SOURCES_VERIFIED',sources=100,rows=4500,
        inherited_sources=81,inherited_rows=3645,new_sources=19,new_rows=855,
        new_pure_strict_grid_proofs=285,historical_alias_rows_disclosed=1500,
        new_metric_offset_applied=False,original_files_written=False,synthetic=False,
        bindings=bindings)

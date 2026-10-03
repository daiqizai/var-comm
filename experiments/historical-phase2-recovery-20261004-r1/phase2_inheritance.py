"""Read-only admission of exactly 81 already completed R3 source checkpoints."""
from pathlib import Path
from history_common import read, sha, verify, identity, row_ids, checkpoint_valid, source_bindings
from history_float_cache import verify_saved
from historical_phase2_alias import STUDY, REVISION, METRICS, LIMITS, pair_record

EVALUATION_FIELDS = ('modelmanifest_sha256','evaluator_identity','numerical_runtime',
    'metric_batch_size','batch_qualification_sha256','metric_qualified_batch_sizes')


def require(condition, message):
    if not condition: raise RuntimeError(message)


def validate_inherited_pure(row, proof, original, grid, float_proof, slot):
    """Preserve original evidence; never retag the checkpoint's old parity."""
    pair = pair_record(original,grid)
    native = proof.get('native_replay_proof', {})
    require(proof.get('replay_parity_passed') is True and proof.get('synthetic') is False,
            'Original actual Phase2 parity evidence required')
    require(proof.get('original_row_sha256') == identity(original), 'Old Phase2 row proof differs')
    require(native.get('passed') is True and native.get('replay_parity_passed') is True
            and native.get('synthetic') is False and native.get('original_row_sha256') == identity(grid),
            'Original strict native grid parity is missing')
    require({'waveform_sha256','observation_sha256'} <= set(native.get('exact_fields_checked',[])),
            'Original native grid waveform/observation checks missing')
    require(native.get('rgb_sha256') == proof.get('native_rgb_sha256') == float_proof['image_sha256'][slot],
            'Inherited actual scored float RGB differs from strict grid proof')
    require(native.get('target_sha256') == float_proof['reference_sha256'], 'Inherited grid target differs')
    for name in METRICS:
        require(abs(float(native['metric_deltas'][name])) <= LIMITS[name], 'Inherited grid strict metric bound failed')
        require(0 <= float(proof['tolerances'][name]) <= LIMITS[name]
                and float(proof['metric_differences'][name]) <= float(proof['tolerances'][name]),
                'Inherited original Phase2 parity bound failed')
    require(abs(float(native['metric_deltas']['mse'])) <= 1e-8 + 1e-6 * abs(float(grid['mse'])),
            'Inherited original grid MSE bound failed')
    return dict(history_row_id=row['history_row_id'],original_phase2_parity_preserved=True,
        original_parity_payload_sha256=identity(proof),native_grid_parity_payload_sha256=identity(native),
        inherited_scored_image_sha256=row['history_image_sha256'],historical_alias=pair,
        observed_metric_values_and_original_scores_rewritten=False,new_metric_offset_applied=False)


def admit_inherited(root, adapter, old_runtime):
    """Return a sealable receipt; caller must write it only in the NEW revision.

    This is intentionally not a fallback for arbitrary partial runs. Sources
    0..80 and their 3645 rows are the explicitly authorized inherited scope.
    """
    root, old_runtime = Path(root).resolve(), Path(old_runtime).resolve()
    out = root/'outputs/HISTORICAL-METRICS-R3-20261003'/STUDY
    result = root/'results/historical_metrics_r3_20261003'/STUDY
    regpath = result/'registration.json'; reg=read(regpath)
    own=source_bindings(old_runtime)
    require(reg.get('study')==STUDY and reg.get('adapter')=='historical_selected_optional'
            and reg.get('status')=='REGISTERED' and reg.get('synthetic') is False
            and reg.get('source_count')==100 and reg.get('frame_count')==4500
            and reg.get('source_bindings')==own, 'Original partial R3 registration differs')
    verify(own);verify(reg['original_bindings'])
    queue_path=root/'outputs/HISTORICAL-METRICS-R3-20261003/queue_registration.json'
    pubpath=root/'outputs/HISTORICAL-METRICS-R3-20261003/source_publication.json'
    require(sha(queue_path)==reg['queue_registration_sha256']
            and sha(pubpath)==reg['source_publication_sha256'], 'Old queue/publication differs')
    queue,pub=read(queue_path),read(pubpath)
    require(queue['source_bindings']==own and pub.get('runtime_source_bindings')==own
            and pub.get('status')=='PUSHED' and pub.get('checks')=='PASS'
            and pub.get('commit')==pub.get('remote_commit'), 'Original runtime publication not verified')
    for key in ('source_bindings','published_files','inputs'):verify(pub[key])
    batchpath=out/'metric_batch_qualification.json'
    require(sha(batchpath)==reg['batch_qualification_sha256'], 'Inherited metric batch qualification differs')
    files=sorted((out/'source_checkpoints').glob('*.json'))
    require([p.name for p in files]==[f'{i:04d}.json' for i in range(81)],
            'Recovery requires exactly the authorized 81 completed source checkpoints')
    expected=[adapter.expected_rows(i) for i in range(100)]
    ids=[row_ids(STUDY,i,rows) for i,rows in enumerate(expected)]
    require(all(len(rows)==45 for rows in expected) and identity(ids)==reg['expected_ids_sha256'],
            'New adapter changed the old registered scientific inventory')
    evaluation_identity={key:reg[key] for key in EVALUATION_FIELDS}
    bindings={str(p):sha(p) for p in (regpath,queue_path,pubpath,batchpath)}
    checkpoints=[];pure_audit=[]
    for index,path in enumerate(files):
        checkpoint=checkpoint_valid(read(path),identity(reg),ids[index],expected_rows=expected[index],study=STUDY,
            source_index=index,evaluation_identity=evaluation_identity,qualified_batch_sizes=reg['metric_qualified_batch_sizes'])
        verify_saved(checkpoint['float_reconstructions'],checkpoint['rows'])
        baseline=checkpoint['baseline']; source=reg['source_identity'][index]
        require(baseline['source_id']==source['image_id']==adapter.native.source_ids[index]
                and baseline['preprocessing_id']==source['preprocessing_id']
                and baseline['true_class_index']==source['class_index'], 'Inherited source/class identity differs')
        proof=checkpoint['float_reconstructions']; cache=Path(proof['path']).resolve()
        require(cache==out/'reconstructions'/f'{index:04d}.npz', 'Inherited cache path substituted')
        for row,parity,original,slot in zip(checkpoint['rows'],checkpoint['parity'],expected[index],proof['image_slots']):
            if original['method']=='pure_continuous':
                grid=adapter.native._main[index,'P4084',int(float(original['snr_db'])),int(original['seed'])]
                pure_audit.append(validate_inherited_pure(row,parity,original,grid,proof,slot))
        bindings[str(path)]=sha(path);bindings[str(cache)]=proof['sha256']
        checkpoints.append(dict(source_index=index,path=str(path),sha256=sha(path),
            float_path=str(cache),float_sha256=proof['sha256'],rows=45,binding=identity(reg),
            row_ids=ids[index],payload_sha256=checkpoint['payload_sha256']))
    require(len(pure_audit)==1215,'Inherited pure comparison coverage differs')
    return dict(status='PHASE2_R3_FIRST81_INHERITANCE_VERIFIED',revision=REVISION,study=STUDY,
        inherited_source_indices=list(range(81)),remaining_source_indices=list(range(81,100)),
        inherited_sources=81,inherited_rows=3645,remaining_sources=19,remaining_rows=855,
        original_registration_path=str(regpath),original_registration_sha256=sha(regpath),
        original_registration_binding=identity(reg),evaluation_identity=evaluation_identity,
        expected_ids_sha256=reg['expected_ids_sha256'],checkpoints=checkpoints,
        source_checkpoints={str(item['source_index']):item for item in checkpoints},
        floatcache_bindings={item['float_path']:item['float_sha256'] for item in checkpoints},
        pure_alias_audit=pure_audit,
        bindings=bindings,original_source_bindings=own,
        original_files_written=False,original_row_values_preserved=True,
        original_parity_evidence_preserved=True,actual_float_caches_verified=True,
        new_metric_offset_applied=False,synthetic=False,training_updates=0,policy_selection_updates=0)


def inherited_checkpoint(receipt, index, expected_rows):
    """Load and recheck the old payload as-is; never replace its binding."""
    require(receipt.get('status')=='PHASE2_R3_FIRST81_INHERITANCE_VERIFIED'
            and index in range(81), 'Only explicitly admitted original sources may be reused')
    path=Path(receipt['original_registration_path'])
    require(sha(path)==receipt['original_registration_sha256'], 'Inherited registration changed')
    item=receipt['source_checkpoints'][str(index)]
    require(sha(item['path'])==item['sha256'], 'Inherited checkpoint changed after admission')
    payload=checkpoint_valid(read(item['path']),receipt['original_registration_binding'],item['row_ids'],
        expected_rows=expected_rows,study=STUDY,source_index=index,
        evaluation_identity=receipt['evaluation_identity'],
        qualified_batch_sizes=receipt['evaluation_identity']['metric_qualified_batch_sizes'])
    verify_saved(payload['float_reconstructions'],payload['rows'])
    return payload

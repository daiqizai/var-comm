"""Read-only admission of completed R2 studies; never relabel old identities."""
from __future__ import annotations
import csv
from pathlib import Path

from audit_selected_runtime import ROSTER
from history_common import (read, write, sha, verify, identity, row_ids, checkpoint_valid,
                            original_fields)
from history_float_cache import verify_saved

PREVIOUS = 'HISTORICAL-METRICS-R2-20261003'
CURRENT = 'HISTORICAL-METRICS-R3-20261003'
PREVIOUS_RESULTS = 'historical_metrics_r2_20261003'
CURRENT_RESULTS = 'historical_metrics_r3_20261003'
INHERITED = {study:dict(adapter=adapter, frames=frames, sources=100)
             for adapter,study,frames in ROSTER[:7]}
EVALUATION_FIELDS = ('modelmanifest_sha256', 'evaluator_identity', 'numerical_runtime',
    'metric_batch_size', 'batch_qualification_sha256', 'metric_qualified_batch_sizes')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def paths(root, study, previous=False):
    root = Path(root).resolve()
    return (root/'outputs'/(PREVIOUS if previous else CURRENT)/study,
            root/'results'/(PREVIOUS_RESULTS if previous else CURRENT_RESULTS)/study)


def map_entry(root, study):
    out, result = paths(root, study, previous=True)
    receipt = Path(root).resolve()/'outputs'/CURRENT/'inheritance'/(study+'.json')
    return dict(version='R2', out=str(out), result=str(result), receipt_path=str(receipt))


def validate_map(root, queue):
    if 'inherited_studies' not in queue:
        return
    values = queue['inherited_studies']
    require(isinstance(values, dict) and set(values) == set(INHERITED), 'R3 inheritance must cover exactly seven completed R2 studies')
    require(queue['jobs'] == [dict(adapter=a, study=s) for a,s,_ in ROSTER], 'R3 selected roster changed')
    for study, item in values.items():
        wanted = map_entry(root, study)
        require(set(item) == set(wanted)|{'receipt_sha256'}
            and all(item[k] == v for k,v in wanted.items()), 'Inherited study location or version differs: '+study)
        require(queue['input_proof_bindings'].get(item['receipt_path']) == item['receipt_sha256']
            and sha(item['receipt_path']) == item['receipt_sha256'], 'Inherited admission is not frozen: '+study)


def study_paths(root, queue, study):
    if study in queue.get('inherited_studies', {}):
        item = queue['inherited_studies'][study]
        wanted = map_entry(root, study)
        require(all(item.get(k) == v for k,v in wanted.items()), 'Inherited study path substitution')
        return Path(item['out']), Path(item['result'])
    return paths(root, study)


def _public(value):
    require(value.get('status') == 'PUSHED' and value.get('checks') == 'PASS'
        and isinstance(value.get('commit'), str) and len(value['commit']) == 40
        and set(value['commit']) <= set('0123456789abcdef')
        and value.get('remote_commit') == value['commit'], 'Original source publication is not verified')


def verify_origin(root, predecessor_path, previous):
    root = Path(root).resolve()
    require(Path(predecessor_path).resolve() == root/'outputs'/PREVIOUS/'queue_registration.json', 'Unexpected predecessor queue')
    require(previous.get('status') == 'REGISTERED' and previous.get('training_updates') == 0
        and previous.get('policy_selection_updates') == 0, 'Original registered queue is required')
    for key in ('source_bindings', 'shared_metric_bindings', 'input_proof_bindings'):
        verify(previous[key])
    coverage = previous['coverage_manifest']; verify({coverage['path']:coverage['sha256']})
    publication_path = root/'outputs'/PREVIOUS/'source_publication.json'
    publication = read(publication_path); _public(publication)
    require(publication.get('runtime_source_bindings') == previous['source_bindings']
        and publication['inputs'].get(str(predecessor_path)) == sha(predecessor_path), 'R2 queue/source publication identity differs')
    for key in ('source_bindings', 'published_files', 'inputs'):
        verify(publication[key])
    return publication_path, publication


def _csv_row_matches(saved, row, columns):
    require(len(columns) == len(set(columns)) and set(row) <= set(columns), 'Inherited CSV columns are incomplete')
    wanted = {key:('' if row.get(key) is None else str(row.get(key, ''))) for key in columns}
    require(saved == wanted, 'Inherited exported row differs from its sealed source checkpoint')


def admit_study(root, study, adapter, predecessor_path, previous, publication_path, publication):
    """Validate all 100 checkpoints and actual float caches without model setup."""
    require(study in INHERITED, 'Unregistered inherited study')
    spec = INHERITED[study]; out, result = paths(root, study, previous=True)
    cp_path, reg_path = out/'completion.json', result/'registration.json'
    done, reg = read(cp_path), read(reg_path)
    require(done.get('status') == 'HISTORICAL_STUDY_METRICS_COMPLETE'
        and done.get('study') == study and done.get('parity_passed') is True
        and done.get('synthetic') is False and done.get('training_updates') == 0
        and done.get('policy_selection_updates') == 0
        and done.get('frames') == spec['frames'] and done.get('sources') == 100
        and done.get('source_bindings') == previous['source_bindings'], 'Inherited completion differs: '+study)
    verify(done['inputs']); verify(done['outputs'])
    require(reg.get('status') == 'REGISTERED' and reg.get('synthetic') is False
        and reg.get('study') == study and reg.get('adapter') == spec['adapter']
        and reg.get('source_bindings') == previous['source_bindings']
        and reg.get('queue_registration_sha256') == sha(predecessor_path)
        and reg.get('source_publication_sha256') == sha(publication_path)
        and reg.get('source_count') == 100 and reg.get('frame_count') == spec['frames']
        and done['outputs'].get(str(reg_path)) == sha(reg_path), 'Inherited score registration differs: '+study)
    require(done['inputs'].get(str(predecessor_path)) == sha(predecessor_path)
        and done['inputs'].get(str(publication_path)) == sha(publication_path), 'Inherited parent evidence differs')
    expected = [list(adapter.expected_rows(i)) for i in range(100)]
    ids = [row_ids(study, i, rows) for i,rows in enumerate(expected)]
    require(all(expected) and sum(map(len, expected)) == spec['frames']
        and reg['expected_ids_sha256'] == identity(ids), 'Inherited original row population differs')
    require(len(reg['source_identity']) == 100
        and reg['source_ids'] == [r['image_id'] for r in reg['source_identity']], 'Inherited source registration differs')
    checkpoints = [out/'source_checkpoints'/f'{i:04d}.json' for i in range(100)]
    require(set((out/'source_checkpoints').glob('*.json')) == set(checkpoints), 'Inherited source checkpoints are incomplete or duplicated')
    table, baseline = result/'metrics_per_frame.csv', result/'source_baseline.csv'
    for path in (table, baseline):
        require(done['outputs'].get(str(path)) == sha(path), 'Inherited table is not bound to completion')
    batch = out/'metric_batch_qualification.json'
    require(sha(batch) == reg['batch_qualification_sha256'], 'Inherited batch qualification changed')
    batch_value = read(batch)
    require(batch_value.get('status') == 'METRIC_BATCH_QUALIFICATION_PASS'
        and batch_value.get('payload_sha256') == identity({k:v for k,v in batch_value.items() if k != 'payload_sha256'})
        and batch_value.get('metric_evaluator_identity') == reg['evaluator_identity']
        and batch_value.get('modelmanifest_sha256') == reg['modelmanifest_sha256']
        and batch_value.get('chosen_batch_size') == reg['metric_batch_size']
        and batch_value.get('qualified_batch_sizes') == reg['metric_qualified_batch_sizes']
        and batch_value.get('rng_state_preserved') is True,
        'Inherited batch evaluator or numerical qualification differs')
    batch_context = batch_value['binding']['runtime_context']
    require(batch_context.get('study') == study and batch_context.get('native_bindings') == reg['original_bindings']
        and batch_context.get('numerical_runtime') == reg['numerical_runtime']
        and batch_context.get('native_numerical_runtime') == reg['native_numerical_runtime'],
        'Inherited batch context differs')
    metric_registration = Path(root)/'results/unified_metrics_20261002/metrics_registration.json'
    original_metric = read(metric_registration)
    require(sha(metric_registration) == reg['original_metric_registration_sha256']
        and original_metric['metric_evaluator_identity'] == reg['evaluator_identity']
        and original_metric['modelmanifest_sha256'] == reg['modelmanifest_sha256']
        and original_metric['numerical_runtime'] == reg['numerical_runtime'], 'Inherited shared metric identity differs')
    evaluation = {key:reg[key] for key in EVALUATION_FIELDS}
    binding = identity(reg); bound = {**done['inputs'], **done['outputs'],
        **previous['source_bindings'], **publication['source_bindings'], **publication['published_files'],
        str(predecessor_path):sha(predecessor_path), str(publication_path):sha(publication_path),
        str(cp_path):sha(cp_path), str(batch):sha(batch)}
    first = None
    with table.open(newline='', encoding='utf-8') as stream, baseline.open(newline='', encoding='utf-8') as base_stream:
        reader, base_reader = csv.DictReader(stream), csv.DictReader(base_stream)
        for i, path in enumerate(checkpoints):
            checkpoint = checkpoint_valid(read(path), binding, ids[i], expected_rows=expected[i], study=study,
                source_index=i, evaluation_identity=evaluation, qualified_batch_sizes=reg['metric_qualified_batch_sizes'])
            proof = checkpoint['float_reconstructions']
            require(proof['path'] == str(out/'reconstructions'/f'{i:04d}.npz'), 'Inherited float cache path differs')
            verify_saved(proof, checkpoint['rows'])
            source = reg['source_identity'][i]; base = checkpoint['baseline']
            require(base['source_id'] == source['image_id'] and base['preprocessing_id'] == source['preprocessing_id']
                and base['true_class_index'] == int(source['class_index']), 'Inherited source baseline identity differs')
            for row in checkpoint['rows']:
                require(row.get('history_source_id') == source['image_id']
                    and row.get('history_preprocessing_id') == source['preprocessing_id']
                    and row.get('history_true_class_index') == int(source['class_index'])
                    and row.get('history_reference_sha256') == base.get('reference_sha256'),
                    'Inherited row/source/reference identity differs')
                _csv_row_matches(next(reader, None), row, reader.fieldnames)
            _csv_row_matches(next(base_reader, None), base, base_reader.fieldnames)
            bound[str(path)] = sha(path); bound[proof['path']] = proof['sha256']
            if i == 0:
                first = checkpoint
        require(next(reader, None) is None and next(base_reader, None) is None, 'Inherited table has extra rows')
    qualification_path = out/'first_source_qualification.json'; qualification = read(qualification_path)
    require(qualification.get('status') == 'REAL_FIRST_SOURCE_PARITY_PASS'
        and qualification.get('synthetic') is False and qualification.get('parity_passed') is True
        and qualification.get('source_index') == 0 and qualification.get('rows') == len(expected[0])
        and qualification.get('checkpoint') == str(checkpoints[0])
        and qualification.get('checkpoint_sha256') == sha(checkpoints[0])
        and qualification.get('registration_sha256') == sha(reg_path)
        and qualification.get('original_values_preserved') is True
        and qualification.get('evaluation_identity') == evaluation
        and qualification.get('original_rows_sha256') == identity([r['history_original_row_sha256'] for r in first['rows']]),
        'Inherited real first-source qualification differs')
    bound[str(qualification_path)] = sha(qualification_path)
    return dict(status='VERIFIED_COMPLETE_R2_STUDY', study=study, adapter=spec['adapter'], frames=spec['frames'],
        sources=100, source_checkpoints_verified=100, float_caches_verified=100,
        original_rows_sha256=identity([r for rows in expected for r in rows]),
        original_queue=dict(path=str(predecessor_path), sha256=sha(predecessor_path)),
        source_publication=dict(path=str(publication_path), sha256=sha(publication_path)),
        evaluator_identity=reg['evaluator_identity'], modelmanifest_sha256=reg['modelmanifest_sha256'],
        numerical_runtime=reg['numerical_runtime'], bindings=bound,
        original_identities_preserved=True, metrics_recomputed=False, training_updates=0, policy_selection_updates=0)


def admit_all(root, predecessor_path, previous):
    import importlib
    publication_path, publication = verify_origin(root, predecessor_path, previous)
    result = {}; common = None
    for study, spec in INHERITED.items():
        adapter = importlib.import_module(spec['adapter']).create_adapter(root, study)
        receipt = admit_study(root, study, adapter, predecessor_path, previous, publication_path, publication)
        current = {key:receipt[key] for key in ('evaluator_identity', 'modelmanifest_sha256', 'numerical_runtime')}
        require(common is None or common == current, 'Inherited metric settings differ between studies')
        common = current
        entry = map_entry(root, study); destination = Path(entry['receipt_path'])
        require(not destination.exists() or read(destination) == receipt, 'Existing inherited admission differs')
        write(destination, receipt)
        entry['receipt_sha256'] = sha(destination); result[study] = entry
        print('INHERITED_R2_VERIFIED', study, spec['frames'], flush=True)
    return result


def verify_inherited(root, queue, study):
    item = queue.get('inherited_studies', {}).get(study)
    require(item is not None and study in INHERITED, 'Study has no explicit inherited admission')
    wanted = map_entry(root, study)
    require(all(item.get(k) == value for k,value in wanted.items())
        and queue['input_proof_bindings'].get(item['receipt_path']) == item['receipt_sha256']
        and sha(item['receipt_path']) == item['receipt_sha256'], 'Inherited receipt/path changed')
    receipt = read(item['receipt_path']); spec = INHERITED[study]
    require(receipt.get('status') == 'VERIFIED_COMPLETE_R2_STUDY' and receipt.get('study') == study
        and all(receipt.get(key) == value for key,value in spec.items())
        and receipt.get('source_checkpoints_verified') == 100 and receipt.get('float_caches_verified') == 100
        and receipt.get('original_identities_preserved') is True and receipt.get('metrics_recomputed') is False
        and receipt.get('training_updates') == 0 and receipt.get('policy_selection_updates') == 0,
        'Inherited admission scope differs')
    expected_queue = Path(root).resolve()/'outputs'/PREVIOUS/'queue_registration.json'
    require(receipt['original_queue']['path'] == str(expected_queue)
        and receipt['bindings'].get(str(expected_queue)) == receipt['original_queue']['sha256'], 'Inherited queue identity changed')
    verify(receipt['bindings'])
    return receipt


def origin_queue(root, queue, queue_path, study):
    if study not in queue.get('inherited_studies', {}):
        return Path(queue_path), queue
    receipt = verify_inherited(root, queue, study)
    path = Path(receipt['original_queue']['path'])
    return path, read(path)

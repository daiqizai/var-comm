"""Immutable scalar caches for sealed studies; no Torch dependency or model calls."""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path

STUDIES = ('N512', 'N1024', 'M1', 'M1_RATE')
FLOAT_FIELDS = ('clip_image_cosine', 'dists', 'dreamsim', 'ms_ssim', 'dinov2_vitl14_cosine')
EXACT_FIELDS = ('resnet50_prediction', 'resnet50_source_prediction', 'resnet50_top1_label',
                'resnet50_top1_source_prediction', 'resnet50_source_top1_label')
SCORE_FIELDS = FLOAT_FIELDS + EXACT_FIELDS


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def same_json(first, second):
    """Compare the immutable serialized contract, including exact scalar values.

    Python tuples are JSON arrays. A saved JSON array loads as a list, so direct
    Python container equality cannot validate an otherwise unchanged receipt.
    """
    return identity(first) == identity(second)


class JsonComparableRecord(dict):
    """A narrow read view for the frozen final runner's registration comparison."""
    def __eq__(self, other):
        return isinstance(other, dict) and same_json(self, other)

    def __ne__(self, other):
        return not self == other


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def verify(bindings):
    if not isinstance(bindings, dict) or not bindings:
        raise RuntimeError('Nonempty immutable bindings required')
    for path, digest in bindings.items():
        if sha(path) != digest:
            raise RuntimeError('Concurrent cache bound input changed: ' + path)


def seal(path, value):
    if Path(path).exists():
        if not same_json(read(path), value):
            raise RuntimeError('Immutable concurrent registration differs on resume')
    else:
        write(path, value)


def source_bindings():
    return {str(p.resolve()): sha(p) for p in sorted(Path(__file__).parent.iterdir()) if p.suffix in ('.py', '.md')}


def snapshot_receipt(directory, registration):
    directory = Path(directory)
    outputs = {}
    for path in sorted((directory / 'source_checkpoints').glob('*.json')):
        validate_checkpoint(read(path), registration, int(path.stem))
        outputs[str(path)] = sha(path)
    inputs = {str(directory / 'concurrent_registration.json'): sha(directory / 'concurrent_registration.json'),
              **registration['source_bindings'], **registration['frozen_inputs'], **registration['asset_inputs']}
    for name in ('runtime_registration.json', 'partial_launch.json'):
        path = directory / name
        inputs[str(path)] = sha(path)
    receipt = dict(status='SEALED_CACHE_SNAPSHOT_COMPLETE', synthetic=False, training_updates=0,
                   policy_selection_updates=0, parity_passed=True, sources_complete=len(outputs), total_sources=100,
                   inputs=inputs, outputs=outputs, source_bindings=registration['source_bindings'])
    write(directory / 'partial_completion.json', receipt)
    return receipt


def key_from_hashes(image_sha256, reference_sha256, evaluator_identity):
    h = hashlib.sha256()
    h.update(image_sha256.encode())
    h.update(reference_sha256.encode())
    h.update(json.dumps(evaluator_identity, sort_keys=True, separators=(',', ':')).encode())
    return h.hexdigest()


def validate_scores(value, truth, prediction):
    if set(value) != set(SCORE_FIELDS):
        raise RuntimeError('Concurrent metric fields differ')
    for name in FLOAT_FIELDS:
        if isinstance(value[name], bool) or not isinstance(value[name], (int, float)) or not math.isfinite(value[name]):
            raise RuntimeError('Invalid concurrent metric: ' + name)
    for name in EXACT_FIELDS:
        if type(value[name]) is not int:
            raise RuntimeError('Invalid classifier scalar type')
    if not 0 <= value['resnet50_prediction'] < 1000 or value['resnet50_source_prediction'] != prediction:
        raise RuntimeError('Classifier prediction identity differs')
    if (value['resnet50_top1_label'] != int(value['resnet50_prediction'] == truth)
            or value['resnet50_top1_source_prediction'] != int(value['resnet50_prediction'] == prediction)
            or value['resnet50_source_top1_label'] != int(prediction == truth)):
        raise RuntimeError('Classifier labels or denominators differ')
    return dict(value)


def row_inventory(engine, replay, index):
    return sorted([[study, replay.row_id(study, row), replay.source_row_hash(row)]
                   for study in STUDIES for row in engine.by_source[study].get(index, [])])


def validate_checkpoint(payload, registration, index):
    if index in registration.get('inherited_cache', {}).get('source_indexes', []) and 'inherited_origin' not in payload:
        raise RuntimeError('Registered inherited source lost its origin')
    if 'inherited_origin' in payload:
        validate_inherited_origin(payload, registration, index)
    if payload.get('payload_sha256') != identity({k: v for k, v in payload.items() if k != 'payload_sha256'}):
        raise RuntimeError('Concurrent source checksum differs')
    expected = registration['sources'][index]
    if (payload.get('registration_sha256') != identity(registration)
            or payload.get('source_index') != index or payload.get('source') != expected
            or payload.get('status') != 'SEALED_SOURCE_METRICS_COMPLETE'
            or payload.get('synthetic') is not False or payload.get('metric_batch_size') != 1):
        raise RuntimeError('Concurrent source registration differs')
    rows = payload.get('rows', [])
    inventory = sorted([[r['study'], r['row_id'], r['source_row_sha256']] for r in rows])
    if len(rows) != expected['frames'] or identity(inventory) != expected['row_inventory_sha256']:
        raise RuntimeError('Concurrent original row coverage/hash differs')
    ids = [row['row_id'] for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError('Duplicate concurrent source row')
    baseline = payload['baseline']
    for key in ('source_id', 'preprocessing_id', 'reference_sha256', 'true_class_index'):
        if baseline.get(key) != expected[key]:
            raise RuntimeError('Concurrent baseline differs: ' + key)
    prediction = baseline['resnet50_source_prediction']
    truth = expected['true_class_index']
    if (type(prediction) is not int or not 0 <= prediction < 1000
            or baseline.get('resnet50_source_top1_label') != int(prediction == truth)):
        raise RuntimeError('Concurrent original classifier baseline invalid')
    scores = payload['scores']
    for key, entry in scores.items():
        if key != key_from_hashes(entry['image_sha256'], expected['reference_sha256'], registration['metric_evaluator_identity']):
            raise RuntimeError('Concurrent exact RGB metric key differs')
        validate_scores(entry['metrics'], truth, prediction)
    for row in rows:
        parity = row['parity']
        if (parity.get('replay_parity_passed') is not True or parity.get('synthetic') is not False
                or parity.get('target_sha256') != expected['reference_sha256']
                or parity.get('original_row_sha256') != row['row_id']
                or row['metric_cache_key'] not in scores
                or parity.get('rgb_sha256') != scores[row['metric_cache_key']]['image_sha256']):
            raise RuntimeError('Concurrent original replay parity differs')
    if set(scores) != {row['metric_cache_key'] for row in rows}:
        raise RuntimeError('Concurrent cache contains unreferenced scores')
    return payload


def inheritance_contract(snapshot, registration, runtime_directory=None):
    """Allow a scheduling revision only; prove identical scientific inputs."""
    old = snapshot.registration
    if old.get('inherited_cache'):
        raise RuntimeError('Only the direct, original R3 snapshot can be inherited')
    fields = ('schema_version', 'studies', 'synthetic', 'training_updates', 'policy_selection_updates',
              'original_m2_read_or_changed', 'metric_batch_size', 'metric_evaluator_identity',
              'modelmanifest_sha256', 'numerical_runtime', 'sources', 'replay_compatibility',
              'original_replay_inventory', 'gpu_memory_fraction', 'frozen_inputs')
    for field in fields:
        if field not in old or field not in registration or not same_json(old[field], registration[field]):
            raise RuntimeError('Inherited scientific registration differs: ' + field)
    old_runtime = snapshot.directory / 'runtime'
    new_runtime = Path(runtime_directory or Path(__file__).parent).resolve()
    def partition(bindings, directory):
        local, scientific = {}, {}
        for path, digest in bindings.items():
            if Path(path).parent.resolve() == directory.resolve():
                local[Path(path).name] = digest
            else:
                scientific[path] = digest
        if not local or 'replay_compat.py' not in local:
            raise RuntimeError('Inherited runtime source inventory missing')
        return local, scientific
    old_local, old_science = partition(old['source_bindings'], old_runtime)
    new_local, new_science = partition(registration['source_bindings'], new_runtime)
    if not same_json(old_science, new_science) or old_local['replay_compat.py'] != new_local['replay_compat.py']:
        raise RuntimeError('Inherited original metric or compatibility source differs')
    for name in ('assets_complete.json', 'modelmanifest.json', 'models_qualification.json'):
        previous = {p: d for p, d in old['asset_inputs'].items() if Path(p).name == name}
        current = {p: d for p, d in registration['asset_inputs'].items() if Path(p).name == name}
        if len(previous) != 1 or previous != current:
            raise RuntimeError('Inherited verified model asset differs: ' + name)
    for candidate in (old, registration):
        paths = [p for p in candidate['asset_inputs'] if Path(p).name == 'concurrent_scalar_qualification.json']
        if len(paths) != 1:
            raise RuntimeError('Inherited scalar qualification missing or ambiguous')
        q = read(paths[0])
        if (q.get('status') != 'CONCURRENT_SCALAR_QUALIFICATION_PASS' or q.get('comparison', {}).get('passed') is not True
                or q.get('payload_sha256') != identity({k: v for k, v in q.items() if k != 'payload_sha256'})
                or q.get('synthetic_images') is not True or q.get('scientific_result') is not False):
            raise RuntimeError('Inherited scalar qualification invalid')
        for field in ('metric_evaluator_identity', 'modelmanifest_sha256', 'numerical_runtime', 'source_bindings', 'gpu_memory_fraction'):
            if not same_json(q['binding'][field], candidate[field]):
                raise RuntimeError('Inherited scalar qualification binding differs: ' + field)
    verify(snapshot.bindings)
    verify(registration['source_bindings']); verify(registration['frozen_inputs']); verify(registration['asset_inputs'])
    return snapshot.provenance()


def inherited_origin(snapshot, index):
    return dict(schema_version=1, source_index=index, checkpoint_path=str(snapshot.paths[index]),
                checkpoint_sha256=snapshot.bindings[str(snapshot.paths[index])],
                registration_path=str(snapshot.path), registration_sha256=sha(snapshot.path),
                snapshot_path=str(snapshot.directory / 'partial_completion.json'),
                snapshot_sha256=sha(snapshot.directory / 'partial_completion.json'),
                original_values_changed=False, original_parity_reused=True)


def validate_inherited_origin(payload, registration, index):
    proof = registration.get('inherited_cache')
    origin = payload.get('inherited_origin')
    if not proof or index not in proof.get('source_indexes', []) or origin.get('source_index') != index:
        raise RuntimeError('Inherited checkpoint lacks registered source provenance')
    if origin.get('original_values_changed') is not False or origin.get('original_parity_reused') is not True:
        raise RuntimeError('Inherited checkpoint changed its scientific values')
    for kind in ('checkpoint', 'registration', 'snapshot'):
        path, digest = origin[kind + '_path'], origin[kind + '_sha256']
        if proof['bindings'].get(path) != digest or registration['asset_inputs'].get(path) != digest or sha(path) != digest:
            raise RuntimeError('Inherited origin binding differs: ' + kind)
    old_reg = read(origin['registration_path'])
    old_payload = read(origin['checkpoint_path'])
    if old_reg.get('inherited_cache') or 'inherited_origin' in old_payload:
        raise RuntimeError('Recursive cache inheritance is not registered')
    validate_checkpoint(old_payload, old_reg, index)
    expected = dict(old_payload, registration_sha256=identity(registration), inherited_origin=origin)
    expected['payload_sha256'] = identity({k: v for k, v in expected.items() if k != 'payload_sha256'})
    if not same_json(payload, expected):
        raise RuntimeError('Inherited checkpoint rows, parity, baseline or scalar values changed')


def import_snapshot(snapshot, directory, registration):
    """Create attributable copies without modifying the sealed previous run."""
    if not same_json(registration.get('inherited_cache'), snapshot.provenance()):
        raise RuntimeError('Inherited snapshot differs from registered provenance')
    verify(snapshot.bindings)
    for index, path in sorted(snapshot.paths.items()):
        old = validate_checkpoint(read(path), snapshot.registration, index)
        value = dict(old, registration_sha256=identity(registration), inherited_origin=inherited_origin(snapshot, index))
        value['payload_sha256'] = identity({k: v for k, v in value.items() if k != 'payload_sha256'})
        validate_checkpoint(value, registration, index)
        seal(Path(directory) / 'source_checkpoints' / f'{index:03}.json', value)
    verify(snapshot.bindings)
    return sorted(snapshot.paths)


class CacheSnapshot:
    """A checked, immutable inventory; source payloads load only when needed."""
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.path = self.directory / 'concurrent_registration.json'
        self.registration = read(self.path)
        r = self.registration
        if (r.get('studies') != list(STUDIES) or r.get('synthetic') is not False
                or r.get('training_updates') != 0 or r.get('policy_selection_updates') != 0
                or r.get('metric_batch_size') != 1 or len(r.get('sources', [])) != 100):
            raise RuntimeError('Concurrent cache registration scope differs')
        self.bindings = {str(self.path): sha(self.path)}
        snapshot_path = self.directory / 'partial_completion.json'
        snapshot = read(snapshot_path)
        if (snapshot.get('status') != 'SEALED_CACHE_SNAPSHOT_COMPLETE'
                or snapshot.get('synthetic') is not False or snapshot.get('parity_passed') is not True
                or snapshot.get('training_updates') != 0 or snapshot.get('policy_selection_updates') != 0
                or snapshot.get('source_bindings') != r['source_bindings']):
            raise RuntimeError('Concurrent completed cache snapshot differs')
        verify(snapshot['inputs']); verify(snapshot['outputs'])
        if snapshot['inputs'].get(str(self.path)) != sha(self.path):
            raise RuntimeError('Concurrent snapshot did not bind its registration')
        self.bindings.update(snapshot['inputs'])
        self.bindings[str(snapshot_path)] = sha(snapshot_path)
        for field in ('source_bindings', 'frozen_inputs', 'asset_inputs'):
            verify(r[field])
            self.bindings.update(r[field])
        self.paths = {}
        actual_paths = sorted((self.directory / 'source_checkpoints').glob('*.json'))
        if set(snapshot['outputs']) != {str(p) for p in actual_paths} or snapshot.get('sources_complete') != len(actual_paths):
            raise RuntimeError('Concurrent checkpoint inventory differs from sealed snapshot')
        for path in actual_paths:
            if not path.stem.isdigit() or not 0 <= int(path.stem) < 100:
                raise RuntimeError('Unexpected concurrent checkpoint file')
            index = int(path.stem)
            if path.name != f'{index:03}.json' or index in self.paths:
                raise RuntimeError('Concurrent source checkpoint filename differs')
            validate_checkpoint(read(path), r, index)
            self.paths[index] = path
            self.bindings[str(path)] = sha(path)
        if not self.paths:
            raise RuntimeError('No completed concurrent source available')
        self.references = {}
        for index, source in enumerate(r['sources']):
            reference = source['reference_sha256']
            if reference in self.references:
                raise RuntimeError('Ambiguous duplicate source RGB')
            self.references[reference] = index

    def provenance(self):
        return dict(schema_version=1, scope=list(STUDIES), cached_metric_batch_size=1,
                    final_replay_and_old_metric_parity_required=True,
                    cache_values_are_scalars_only=True, original_row_annotations_recomputed=True,
                    source_indexes=sorted(self.paths), registration_sha256=sha(self.path),
                    metric_evaluator_identity=self.registration['metric_evaluator_identity'],
                    numerical_runtime=self.registration['numerical_runtime'], bindings=dict(self.bindings),
                    inherited_cache=self.registration.get('inherited_cache'))

    def scores_for(self, reference_sha256, evaluator_identity, truth, prediction, numerical_runtime):
        r = self.registration
        if evaluator_identity != r['metric_evaluator_identity'] or numerical_runtime != r['numerical_runtime']:
            raise RuntimeError('Final evaluator/numerical runtime differs from concurrent cache')
        index = self.references.get(reference_sha256)
        if index is None:
            raise RuntimeError('Final reference not in concurrent original source set')
        expected = r['sources'][index]
        if truth != expected['true_class_index']:
            raise RuntimeError('Final true label differs from concurrent cache')
        if index not in self.paths:
            return {}, index
        path = self.paths[index]
        if sha(path) != self.bindings[str(path)]:
            raise RuntimeError('Concurrent checkpoint changed after final registration')
        payload = validate_checkpoint(read(path), r, index)
        if payload['baseline']['resnet50_source_prediction'] != prediction:
            raise RuntimeError('Final original-image classifier prediction differs')
        return {key: validate_scores(entry['metrics'], truth, prediction)
                for key, entry in payload['scores'].items()}, index

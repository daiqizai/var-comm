"""Join verified historical scalar/parity evidence to unchanged original rows.

No image is reconstructed or impersonated here. RGB fingerprints name the
actual images measured by the sealed concurrent run; original table fields and
current metadata are joined only after exact complete-row identity checks.
"""
from __future__ import annotations
import json


HISTORICAL = 'historical_verified'
FRESH = 'fresh_replay'


def final_fields(value, parity, key, target_sha, truth, manifest_sha, origin):
    value.update(image_sha256=parity['rgb_sha256'], reference_sha256=target_sha,
        true_class_index=truth, modelmanifest_sha256=manifest_sha, replay_parity_passed=True,
        replay_metric_deltas=json.dumps(parity['metric_deltas'], sort_keys=True),
        metric_cache_key=key, parity_origin=origin)
    return value


class SourceAssembly:
    def __init__(self, snapshot, cache, replay, runner, engine, index, *, reference_sha,
                 evaluator_identity, numerical_runtime, truth, prediction, manifest_sha):
        self.snapshot, self.cache, self.replay, self.runner, self.engine = snapshot, cache, replay, runner, engine
        self.index, self.reference_sha, self.truth, self.manifest_sha = index, reference_sha, truth, manifest_sha
        registered = snapshot.registration
        if manifest_sha != registered['modelmanifest_sha256']:
            raise RuntimeError('Assembly model manifest differs from measured cache')
        record, expected = engine.records[index], registered['sources'][index]
        if (record['image_id'] != expected['source_id'] or record['preprocessing_id'] != expected['preprocessing_id']
                or int(record['class_index']) != expected['true_class_index'] or reference_sha != expected['reference_sha256']):
            raise RuntimeError('Assembly original source identity differs')
        self.scores, actual_index = snapshot.scores_for(reference_sha, evaluator_identity, truth, prediction, numerical_runtime)
        if actual_index != index:
            raise RuntimeError('Assembly source ordering changed')
        self.payload = None
        self.by_study = {}
        if index not in snapshot.paths:
            return
        path = snapshot.paths[index]
        if cache.sha(path) != snapshot.bindings[str(path)]:
            raise RuntimeError('Assembly checkpoint changed after registration')
        self.payload = cache.validate_checkpoint(cache.read(path), registered, index)
        inventory = cache.row_inventory(engine, replay, index)
        if cache.identity(inventory) != expected['row_inventory_sha256'] or len(inventory) != expected['frames']:
            raise RuntimeError('Assembly full original row inventory differs')
        for study in cache.STUDIES:
            originals = engine.by_source[study].get(index, [])
            mapping = {replay.row_id(study, row): row for row in originals}
            historical = [row for row in self.payload['rows'] if row['study'] == study]
            if len(mapping) != len(originals) or len(historical) != len(originals):
                raise RuntimeError('Assembly original row coverage differs')
            if {row['row_id'] for row in historical} != set(mapping):
                raise RuntimeError('Assembly historical row identities differ')
            for row in historical:
                if replay.source_row_hash(mapping[row['row_id']]) != row['source_row_sha256']:
                    raise RuntimeError('Assembly original complete-row hash differs')
            self.by_study[study] = (mapping, historical)

    def available(self, study):
        return study in self.by_study

    def assemble(self, study):
        if not self.available(study):
            raise RuntimeError('Study/source was never measured in the sealed cache')
        originals, historical = self.by_study[study]
        for evidence in historical:
            row = originals[evidence['row_id']]
            # Repeat the exact original augmentation and normalization path.
            meta = self.engine.metadata(row, study)
            augmented = dict(row, **meta)
            value = self.runner.normalized(augmented, study, self.engine.metadata(augmented, study))
            parity = dict(evidence['parity'], parity_origin=HISTORICAL)
            key = evidence['metric_cache_key']
            final_fields(value, parity, key, self.reference_sha, self.truth, self.manifest_sha, HISTORICAL)
            metrics = self.runner.PendingMetricScorer.metric_values(dict(self.scores[key], label_conditioned=False))
            value.update(metrics)
            yield value, parity


def origin_counts(rows, parity):
    if len(rows) != len(parity):
        raise RuntimeError('Assembly provenance row count differs')
    counts = {HISTORICAL: 0, FRESH: 0}
    for row, proof in zip(rows, parity):
        origin = row.get('parity_origin')
        if (origin not in counts or proof.get('parity_origin') != origin
                or proof.get('replay_parity_passed') is not True or proof.get('synthetic') is not False
                or proof.get('original_row_sha256') != row['replay_row_id']
                or proof.get('rgb_sha256') != row['image_sha256']
                or proof.get('target_sha256') != row['reference_sha256']):
            raise RuntimeError('Assembly parity origin/identity differs')
        counts[origin] += 1
    return counts


def execution_protocol(snapshot):
    return dict(schema_version=1, historical_studies=list(snapshot.registration['studies']),
        original_row_fields_retained=True, current_metadata_recomputed=True, exact_full_row_hash_join=True,
        historical_parity_origin=HISTORICAL, uncached_parity_origin=FRESH,
        historical_rgb_reconstructed_again=False, fake_rgb_created=False,
        cached_scalar_batch_size=1, uncached_scalar_batch='qualified_metric_batch_size',
        scientific_formulas_changed=False, training_updates=0, policy_selection_updates=0)

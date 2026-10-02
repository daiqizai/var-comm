"""Reuse verified scalar scores while the unmodified final runner replays all rows."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from pathlib import Path
import re
import cache_handoff as cache
from concurrent_runner import imports


def published_extension(root, directory, runtime_sources):
    """Require the reviewed runtime bytes to have been normally published."""
    path = directory / 'extension_source_publication.json'
    record = cache.read(path)
    if (record.get('status') != 'PUSHED' or record.get('checks') != 'PASS'
            or not re.fullmatch('[0-9a-f]{40}', str(record.get('commit', '')))
            or record.get('commit') != record.get('remote_commit')
            or record.get('runtime_source_bindings') != runtime_sources):
        raise RuntimeError('Concurrent extension source publication is not verified')
    canonical = root / 'experiments/metric-concurrent-20261002'
    expected = {str(canonical / Path(p).name): digest for p, digest in runtime_sources.items()}
    if record.get('source_bindings') != expected:
        raise RuntimeError('Published extension source differs from reviewed runtime')
    cache.verify(runtime_sources); cache.verify(expected)
    return {str(path): cache.sha(path), **expected, **runtime_sources}


@contextmanager
def bridge(runner, replay, torch, snapshot):
    """Only scalar cache seeding and explicit provenance extend the frozen loop."""
    base_scorer, base_write, base_read = runner.PendingMetricScorer, runner.write, runner.read
    base_sources = runner.source_bindings
    provenance = snapshot.provenance()
    provenance['replay_compatibility'] = snapshot.registration.get('replay_compatibility')
    registration_path = runner.RESULT / 'metrics_registration.json'
    completion_path = runner.OUT / 'scoring_completion.json'
    provenance_path = runner.RESULT / 'concurrent_cache_provenance.json'
    extension_sources = cache.source_bindings()
    reuse = {'cached_source_instances': 0, 'cache_hit_frames': 0, 'cache_hit_keys': set()}
    execution = dict(cached_scores_batch_size=1, final_uncached_scores_batch='metric_batch_size',
                     equivalence='final synthetic qualification compares selected batch against scalar at absolute2e-5; classification exact',
                     old_metrics_and_original_row_metadata='always recomputed by frozen final replay')

    class CachedScorer(base_scorer):
        def __init__(self, evaluator, tensor_module, reference, prepared, truth, batch_size, qualified_batch_sizes):
            super().__init__(evaluator, tensor_module, reference, prepared, truth, batch_size, qualified_batch_sizes)
            fingerprint = replay.rgb_fingerprint(reference[0])
            values, index = snapshot.scores_for(fingerprint, evaluator.identity(), truth,
                int(prepared['resnet50'][0]), runner.runtime_flags(torch))
            for key, value in values.items():
                self.cache[key] = self.metric_values(dict(value, label_conditioned=False))
            self.concurrent_keys = set(values)
            self.concurrent_index = index
            reuse['cached_source_instances'] += int(bool(values))

        def add(self, key, rgb, row):
            if key in self.concurrent_keys:
                reuse['cache_hit_frames'] += 1
                reuse['cache_hit_keys'].add((self.concurrent_index, key))
            return super().add(key, rgb, row)

    def sources():
        return dict(base_sources(), **extension_sources)

    def read(path):
        value = base_read(path)
        if Path(path).resolve() == registration_path.resolve():
            if not cache.same_json(value.get('concurrent_cache'), provenance) or not cache.same_json(value.get('metric_batch_execution'), execution):
                raise RuntimeError('Final concurrent cache provenance changed on resume')
            # The unchanged runner compares the pre-extension registration first.
            value = cache.JsonComparableRecord({k: v for k, v in value.items() if k not in ('concurrent_cache', 'metric_batch_execution')})
        return value

    def write(path, value):
        resolved = Path(path).resolve()
        if resolved == registration_path.resolve():
            r = snapshot.registration
            if (value['source_ids'] != [s['source_id'] for s in r['sources']]
                    or value['modelmanifest_sha256'] != r['modelmanifest_sha256']
                    or value['metric_evaluator_identity'] != r['metric_evaluator_identity']
                    or value['numerical_runtime'] != r['numerical_runtime']):
                raise RuntimeError('Final registered source/model/runtime differs from scalar cache')
            # Mutate the same dictionary: runner subsequently hashes it to bind
            # final source checkpoints. A copied augmented dictionary is unsafe.
            value['concurrent_cache'] = provenance
            value['metric_batch_execution'] = execution
            cache.seal(provenance_path, provenance)
        if resolved == completion_path.resolve():
            cache.verify(snapshot.bindings)
            cache.verify(extension_sources)
            value['inputs'].update(snapshot.bindings)
            value['inputs'][str(provenance_path)] = cache.sha(provenance_path)
            value['outputs'][str(provenance_path)] = cache.sha(provenance_path)
            value['concurrent_cache'] = provenance
            value['metric_batch_execution'] = execution
            value['current_invocation_cache_reuse'] = dict(cached_source_instances=reuse['cached_source_instances'],
                cache_hit_frames=reuse['cache_hit_frames'], unique_cache_hit_keys=len(reuse['cache_hit_keys']),
                resumed_final_checkpoints_may_already_contain_reused_scores=True)
        return base_write(path, value)

    runner.PendingMetricScorer, runner.write, runner.read, runner.source_bindings = CachedScorer, write, read, sources
    try:
        yield reuse
    finally:
        runner.PendingMetricScorer, runner.write, runner.read, runner.source_bindings = base_scorer, base_write, base_read, base_sources


def main():
    import fcntl
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--cache-dir', required=True)
    args = parser.parse_args()
    root, directory = Path(args.root).resolve(), Path(args.cache_dir).resolve()
    if directory != root / 'outputs/METRIC-CONCURRENT-R4-20261002':
        raise RuntimeError('Unexpected concurrent cache directory')
    lock = (directory / 'owner.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    snapshot = cache.CacheSnapshot(directory)
    snapshot.bindings.update(published_extension(root, directory, cache.source_bindings()))
    runner, replay, _, _ = imports(root)
    if snapshot.registration.get('replay_compatibility') != replay._historical_latent_alias_receipt:
        raise RuntimeError('Final replay compatibility differs from the concurrent cache')
    # Parent completion/source publication gates remain inside runner.main.
    import torch
    with bridge(runner, replay, torch, snapshot):
        runner.main()


if __name__ == '__main__':
    main()

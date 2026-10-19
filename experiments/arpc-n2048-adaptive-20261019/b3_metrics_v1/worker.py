"""Frozen seven metrics, actual missing pairs only, one source bootstrap."""
from common import *
import argparse, csv, gc, importlib.util, math, fcntl
import numpy as np
from statistical import METRICS, METHODS, source_means, summarize
import adapter


def install(seconds):
    adapter.install(ORIGINAL.read_bytes(), globals(), seconds)


def reuse_only(expected):
    r = check(expected)
    assert r['new_seven_metric_pairs'] == 0
    install(r['stage_seconds']['score'])
    pairs, logical, keys = [], [], set()
    for s in read(MOUT / 'pair_plan.json')['sources']:
        for p in s['pairs']:
            values = p['reuse']['scores']
            assert set(values) == set(METRICS) and all(math.isfinite(v) for v in values.values())
            key = p['reuse']['old_pair_key']
            assert isinstance(key, str) and len(key) == 64 and all(c in '0123456789abcdef' for c in key)
            assert key not in keys
            keys.add(key)
            pairs.append(dict(source_id=s['source_id'], source_index=s['source_index'], pair_key=key,
                              image_float_sha256=p['image_float_sha256'], reused=True, **values))
            logical.extend(dict(**f, pair_key=key, **values) for f in p['frames'])
    assert len(logical) == 1200 and len(pairs) == r['unique_pairs']
    # Preserve the exact prior numerical identities even when no model is
    # constructed. This keeps the zero-new cache schema identical to score().
    for name in ('four_metric_identity.json', 'supplemental_identity.json'):
        original = OLD_O/'a3_metrics_v1'/name
        d = next(d for d in r['inputs'] if Path(d['path']) == original)
        data = verify(d).read_bytes()
        with (MOUT/name).open('xb') as f:
            f.write(data); f.flush(); os.fsync(f.fileno())
        assert sha(MOUT/name) == d['sha256']
    csvwrite(MOUT/'logical_scores.csv', logical)
    csvwrite(MOUT/'actual_pair_scores.csv', pairs)
    put(MOUT/'score_completion.json', dict(status='N2048_SEVEN_METRICS_COMPLETE', logical_frames=1200,
        source_count=100, unique_pairs=len(pairs), new_pairs=0, reused_pairs=len(pairs),
        ledger=dict(counts=[], unresolved=0), bootstrap=0,
        outputs=[pin(MOUT/n) for n in ('logical_scores.csv','actual_pair_scores.csv',
                                     'four_metric_identity.json','supplemental_identity.json')]))


def statistics(expected):
    r = check(expected)
    closed(MRUN/'controls/score/actual_wait.json')
    done = read(MOUT/'score_completion.json')
    assert done['ledger']['unresolved'] == 0 and done['logical_frames'] == 1200
    for d in done['outputs']: verify(d)
    with (MOUT/'logical_scores.csv').open() as f: logical = list(csv.DictReader(f))
    ids = [s['source_id'] for s in read(MOUT/'pair_plan.json')['sources']]
    assert len(ids) == 100
    rows = source_means(logical, ids)
    spec = r['statistics']
    assert spec == dict(seed=2026101902, replicates=10000, draw_calls=1, unit='source after three-noise averaging')
    out = MRUN/'bootstrap'; out.mkdir()
    put(out/'reserved.json', dict(**spec, source_ids=ids))
    draws = np.random.default_rng(spec['seed']).integers(0, 100, size=(10000,100), dtype=np.int64)
    with (out/'source_indices.npy').open('xb') as f:
        np.save(f, draws, allow_pickle=False); f.flush(); os.fsync(f.fileno())
    put(out/'complete.json', dict(indices=pin(out/'source_indices.npy'), draw_calls=1))
    summary, paired = summarize(rows, ids, draws)
    install(r['stage_seconds']['score'])
    for name, values in [('source_noise_means.csv',rows), ('summary.csv',summary), ('paired_differences.csv',paired)]:
        csvwrite(MOUT/name, values)
    put(MOUT/'completion.json', dict(status='N2048_ADAPTIVE_TWO_METHOD_COMPARISON_COMPLETE',
        sources=100, logical_frames=1200, methods=2, snr_db=[4,10], metrics=list(METRICS),
        bootstrap=dict(**spec, indices=pin(out/'source_indices.npy'), interval='pointwise percentile95; direct paired differences; no multiplicity adjustment'),
        fixed_full_19dB_separate=True, no_further_candidates=True,
        outputs=[pin(MOUT/n) for n in ('score_completion.json','source_noise_means.csv','summary.csv','paired_differences.csv')]))


def run(role, expected):
    r = check(expected)
    assert role in ('score','statistics')
    if role == 'statistics': closed(MRUN/'controls/score/actual_wait.json')
    lock = (ADAPT_L/'execution.lock').open('a+'); fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    (MRUN/'controls').mkdir(exist_ok=True)
    out = MRUN/'controls'/role; out.mkdir()
    put(out/'supervisor.json', dict(process=identity(os.getpid()), parent_process=identity(os.getppid()), argv=sys.argv))
    gpu = role == 'score' and r['new_seven_metric_pairs'] > 0
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=r['resource']['visible_devices'] if gpu else '',
        CUDA_DEVICE_ORDER='PCI_BUS_ID', PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
        OMP_NUM_THREADS='6', MKL_NUM_THREADS='6', OPENBLAS_NUM_THREADS='6')
    env.pop('PYTHONPATH', None)
    try:
        wait_child(out, [str(PY),'-B',str(E/'worker.py'),'_'+role,'--expected',expected], env, r['stage_seconds'][role])
        put(out/'completion.json', dict(status='COMPLETE', actual_wait=pin(out/'actual_wait.json')))
    except BaseException:
        put(out/'failure.json', dict(error=traceback.format_exc(), automatic_retry=False)); raise


if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument('role', choices=['score','statistics','_score','_statistics']); a.add_argument('--expected', required=True)
    x = a.parse_args()
    if x.role == '_statistics': statistics(x.expected)
    elif x.role == '_score':
        r = check(x.expected)
        if r['new_seven_metric_pairs'] == 0: reuse_only(x.expected)
        else:
            install(r['stage_seconds']['score']); score(x.expected)
    else: run(x.role, x.expected)

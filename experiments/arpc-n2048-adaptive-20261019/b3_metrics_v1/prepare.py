"""Register only closed actual pixels, exact eligible seven-metric cache pairs."""
from common import *
from statistical import METRICS, METHODS, SNRS, SEEDS
import adapter
import argparse, csv, collections
import numpy as np


def image(p):
    if p['image'] is None:
        y = np.full((3,256,256), .5, np.float32)
        assert p['array_key'] is None
    else:
        with np.load(verify(p['image']), allow_pickle=False) as z:
            y = z[p['array_key']].copy()
    assert y.dtype == np.float32 and y.shape == (3,256,256)
    assert np.isfinite(y).all() and y.min() >= 0 and y.max() <= 1
    assert hashlib.sha256(y.tobytes()).hexdigest() == p['image_float_sha256']
    return y


def prior_pairs(root, original_four, original_supp):
    done = read(root/'score_completion.json')
    assert done.get('unresolved', done.get('ledger', {}).get('unresolved')) == 0
    for d in done['outputs']: verify(d)
    four, supp = read(root/'four_metric_identity.json'), read(root/'supplemental_identity.json')
    assert four['identity'] == original_four['identity']
    for key in ('protocol','model_metadata','versions','FP32','TF32'):
        assert supp[key] == original_supp[key]
    plans = {s['source_id']: s for s in read(root/'pair_plan.json')['sources']}
    with (root/'actual_pair_scores.csv').open() as f: rows = list(csv.DictReader(f))
    seen = set()
    for row in rows:
        key = (row['source_id'],row['image_float_sha256'])
        assert key not in seen; seen.add(key)
        s = plans[row['source_id']]
        p = next(p for p in s['pairs'] if p['image_float_sha256'] == row['image_float_sha256'])
        # Prior ARPC native source stores float_output; full N2048 explicitly
        # records each native array key. No conversion or rounding is permitted.
        p = dict(p)
        if 'array_key' not in p: p['array_key'] = 'float_output' if p['image'] else None
        yield key, dict(source=s['archive'], pair=p, scores={k:float(row[k]) for k in METRICS},
                        old_pair_key=row['pair_key'], evidence=pin(root/'actual_pair_scores.csv'))


def prepare(expected_native):
    assert not MOUT.exists() and not MRUN.exists()
    assert sha(NATIVE) == expected_native
    native, tr = read(NATIVE), read(TREG)
    assert native['status'] == 'COMPLETE'
    for d in native['inputs'] + native['outputs'] + native['native_actual_waits']: verify(d)
    assert native['process_closure_status'] == 'ALL_NATIVE_PROCESSES_GONE'
    # The native closure includes host-specific observed process identities.
    # Do not equate a PID on another host with the same integer on this host.
    for d in native['process_closure_evidence']: verify(d)
    closed(ADAPT_L/'b2b3_revision_v1/controls/B3_audit/actual_wait.json')
    assert tr['source_count'] == 100 and tr['caps']['logical_objects'] == 1200
    for d in tr['code']: verify(d)
    frames = read(verify(native['logical_pixels']))
    sources = read(verify(tr['private_sources']))
    assert len(sources) == 100 and len({s['source_id'] for s in sources}) == 100
    src = {s['source_id']: s for s in sources}
    groups = {}
    seen = set()
    logical = []
    for x in frames:
        method = METHODS[x['codec']]
        assert x.get('method',method) == method
        key = (x['source_id'],method,int(x['snr_db']),int(x['noise_seed']))
        assert key not in seen and key[0] in src and key[2] in SNRS and key[3] in SEEDS
        seen.add(key)
        assert x['source_index'] == src[x['source_id']]['source_index']
        p = {k:x[k] for k in ('image','array_key','image_float_sha256')}
        image(p)
        f = dict(key=x['key'], method=method, source_id=x['source_id'], source_index=x['source_index'],
                 snr_db=x['snr_db'], noise_seed=x['noise_seed'], receiver_status=x['receiver_status'],
                 accepted_wrong_post_exit=x['accepted_wrong_post_exit'])
        logical.append(dict(**f,**p))
        s = groups.setdefault(x['source_id'],dict(source_id=x['source_id'], source_index=x['source_index'],
             archive=src[x['source_id']]['source'], pairs={}))
        p0 = s['pairs'].setdefault(p['image_float_sha256'],dict(**p, frames=[]))
        assert np.array_equal(image(p),image(p0))
        p0['frames'].append(f)
    assert len(seen) == 1200 and len(groups) == 100
    priorreg = read(OLD_O/'a3_metrics_v1/registration.json')
    original_four = read(OLD_O/'a3_metrics_v1/four_metric_identity.json')
    original_supp = read(OLD_O/'a3_metrics_v1/supplemental_identity.json')
    caches = {}
    for root in (OLD_O/'a3_metrics_v1', PRIOR):
        for key, v in prior_pairs(root, original_four, original_supp):
            if key in caches:
                old = caches[key]
                assert old['source'] == v['source'] and old['scores'] == v['scores']
            else: caches[key] = v
    plan, reused = [], 0
    for rank, s in enumerate(sorted(groups.values(),key=lambda s:s['source_index'])):
        s['rank'] = rank
        verify(s['archive'])
        s['pairs'] = list(s['pairs'].values())
        assert sum(len(p['frames'])for p in s['pairs']) == 12
        for p in s['pairs']:
            old = caches.get((s['source_id'],p['image_float_sha256']))
            if old:
                assert old['source'] == s['archive']
                assert np.array_equal(image(p),image(old['pair']))
                p['reuse'] = dict(scores=old['scores'], old_pair_key=old['old_pair_key'],
                    old_image=old['pair']['image'], original_pair_scores=old['evidence'])
                reused += 1
        plan.append(s)
    total = sum(len(s['pairs'])for s in plan)
    assert 0 <= reused <= total <= 1200
    # Actual original CUDA identity is verified again by the score worker.
    # Same frozen four/supplemental implementations and numerical flags.
    resource = dict(priorreg['resource'])
    resource.update(visible_devices='0', allocator_gib=12)
    MOUT.mkdir(); MRUN.mkdir()
    put(MOUT/'actual_logical_pixels.json',logical)
    put(MOUT/'pair_plan.json',dict(sources=plan))
    auth = read(verify(tr['authorization']))
    assert auth['bootstrap']['seed'] == 2026101902 and auth['bootstrap']['replicates'] == 10000
    inputs = [pin(TREG),pin(NATIVE),native['logical_pixels'],pin(MOUT/'pair_plan.json'),
              tr['authorization'],tr['selection']]
    for root in (OLD_O/'a3_metrics_v1',PRIOR):
        inputs.extend(pin(root/n)for n in ('registration.json','score_completion.json','actual_pair_scores.csv',
                                         'pair_plan.json','four_metric_identity.json','supplemental_identity.json'))
    code = [pin(E/n)for n in ('common.py','adapter.py','statistical.py','prepare.py','worker.py','test_metrics.py','README.md')]
    code.append(pin(ADAPT_E/'shared.py'))
    adapter.selected_tree(ORIGINAL.read_bytes())
    code.append(pin(ORIGINAL))
    put(MREG,dict(schema='N2048_B3_FROZEN_SEVEN_METRICS_V1', inputs=inputs, code=code,
        original_metric_sources=priorreg['source_files'], original_four_metrics=priorreg['original_four_metrics'],
        metric_adapter=priorreg['metric_adapter'] if 'metric_adapter' in priorreg else pin(OLD_E/'a3_metrics_v1/metric_adapter.py'),
        metric_config=priorreg['metric_config'], supplemental_registration=priorreg['supplemental_registration'],
        supplemental_overlay_root=priorreg['supplemental_overlay_root'], resource=resource,
        source_count=100,logical_frames=1200, unique_pairs=total, exact_prior_seven_metric_pairs=reused,
        new_seven_metric_pairs=total-reused, statistics=auth['bootstrap'], stage_seconds=dict(score=14400,statistics=300),
        score_source_adapter=dict(original_sha256=adapter.ORIGINAL_SHA256,logical_denominator=[1800,1200],
            wall_guard_seconds=[3500,14340],metric_math_changes=0),
        policy_selection=False, no_19dB_recalibration=True, training=0, TX_probability=0))
    print(json.dumps(dict(registration=pin(MREG),unique_pairs=total,reused_pairs=reused,new_pairs=total-reused)))


if __name__ == '__main__':
    a=argparse.ArgumentParser();a.add_argument('--expected-native',required=True)
    x=a.parse_args();prepare(x.expected_native)

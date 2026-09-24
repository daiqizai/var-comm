"""CPU engineering checks; these synthetic arrays are not experiment metrics."""
from types import SimpleNamespace
import numpy as np
from tools.publish_qpsk_development import resource_bootstrap

def test_resource_bootstrap_keeps_negative_savings_and_unreached_samples():
    methods = ['P2048', 'P3060', 'P4084']
    context = {m: {'family': 'continuous', 'N': int(m[1:])} for m in methods}
    choices = []
    for family in ['raw', 'arithmetic']:
        for n in [2048, 3060, 4084]:
            m = f'{family}_{n}'
            context[m] = {'family': family, 'N': n}
            choices.append({'method': m, 'family': family, 'snr_db': 13})
    def values(method, snr):
        c = context[method]
        psnr = 19 if c['family']=='raw' and c['N']==2048 else 21
        return np.tile([.01, psnr, .1, .8], (100, 1))
    table = SimpleNamespace(snrs=[13], ids=list(range(100)), context=context, source_values=values, sha256='synthetic-unit-only')
    rows = resource_bootstrap(table, {'choices': choices}, {'targets': {
        'attainable': {'psnr_min': 20, 'lpips_max': .2},
        'unreachable': {'psnr_min': 30, 'lpips_max': .01}}}, repeats=251)
    raw = next(r for r in rows if r['target']=='attainable' and r['family']=='raw')
    assert raw['conditional_saving_ci95'] == [1-3060/2048, 1-3060/2048]
    assert raw['both_attained_repeats']==251 and raw['unreached_repeats']==0
    assert raw['digital_N_counts']['3060']==251
    for row in rows:
        if row['target']=='unreachable':
            assert row['conditional_saving_ci95'] is None
            assert row['unreached_repeats']==251
            assert row['continuous_unreached']==251 and row['digital_unreached']==251
        assert row['policy_refit'] is False
        assert row['training_seed_variation_included'] is False

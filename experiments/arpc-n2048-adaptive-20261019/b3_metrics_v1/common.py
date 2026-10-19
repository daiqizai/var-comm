from pathlib import Path
import hashlib, json, os, sys, time, traceback
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from shared import R, RT, BASE, E as ADAPT_E, O as ADAPT_O, L as ADAPT_L
from shared import OLD_E, OLD_O, OLD_L, CPY, PY, sha, pin, verify, read, put, identity, gone, cache, Ledger, wait_child
E = ADAPT_E / 'b3_metrics_v1'
MOUT = ADAPT_O / 'b3_metrics_v1'
MRUN = ADAPT_L / 'b3_metrics_v1'
MREG = MOUT / 'registration.json'
NATIVE = ADAPT_O / 'b2b3_analysis_v1/B3/native_completion.json'
TREG = ADAPT_O / 'b2b3_revision_v1/B3/registration.json'
TORUN = ADAPT_L / 'b2b3_revision_v1/B3'
ORIGINAL = R / 'experiments/arpc-n2048-validation-20261019/metrics.py'
PRIOR = R / 'results/arpc_n2048_validation_20261019_v1/metrics_v1'


def closed(path):
    w = read(path)
    assert type(w['returncode']) is int and w['returncode'] == 0
    assert not w.get('timed_out') and not w.get('terminated') and not w.get('killed')
    assert w.get('metadata_error') is None and gone(w['process'])
    return w


def check(expected):
    assert sha(MREG) == expected
    r = read(MREG)
    for d in r['inputs'] + r['code'] + r['original_metric_sources']:
        verify(d)
    assert r['logical_frames'] == 1200 and r['source_count'] == 100
    return r

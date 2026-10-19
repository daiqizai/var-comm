"""New control namespace; all original science implementations remain frozen."""
from pathlib import Path
import json
import os
import sys
import traceback

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import shared as old
from shared import R, RT, BASE, OLD_E, OLD_O, OLD_L, CPY, PY
from shared import sha, pin, verify, read, put, identity, gone, cache, Ledger, wait_child
from plan import CAPS

E = old.E / 'b2b3_revision_v1'
O = old.O / 'b2b3_revision_v1'
L = old.L / 'b2b3_revision_v1'
OLD_ADAPTIVE_E, OLD_ADAPTIVE_O, OLD_ADAPTIVE_L = old.E, old.O, old.L
SOURCE_NAMES = ('plan.py', 'common.py', 'prepare.py', 'transport.py', 'control.py', 'test_revision.py', 'README.md')
PUBLIC_FIELDS = {'schema', 'phase', 'caps', 'code', 'catalogue', 'jobs', 'registration_sha256', 'backend_identity'}


def closed_wait(path):
    w = read(path)
    assert type(w['returncode']) is int and w['returncode'] == 0
    assert w.get('timed_out', False) is False and w.get('terminated', False) is False
    assert w.get('killed', False) is False and w.get('metadata_error') is None
    assert gone(w['process'])
    launch = Path(path).with_name('launch.json')
    if launch.exists():
        assert read(launch)['process'] == w['process']
    return w


def registration(phase, expected):
    assert phase in CAPS
    p = O / phase / 'registration.json'
    assert sha(p) == expected
    r = read(p)
    assert r['phase'] == phase and r['caps'] == CAPS[phase]
    for d in r['code']:
        verify(d)
    return r


def require_contained(path, root):
    p, root = Path(path), Path(root)
    assert p.is_absolute() and '..' not in p.parts
    assert p.resolve().is_relative_to(root.resolve())
    for x in (p, *p.parents):
        assert not x.is_symlink()
    return p


def no_symlink_ancestors(path):
    p = Path(path)
    assert p.is_absolute() and '..' not in p.parts
    for x in (p, *p.parents):
        assert not x.is_symlink()
    return p


def public_configuration(path, expected, size):
    p = require_contained(path, L)
    assert p.stat().st_size == size and sha(p) == expected
    v = read(p)
    assert set(v) == PUBLIC_FIELDS and v['schema'] == 'N2048_B2B3_RX_PUBLIC_V1'
    phase = v['phase']
    assert phase in CAPS and v['caps'] == CAPS[phase]
    assert p == L / phase / 'received/public.json'
    for name in ('catalogue', 'jobs'):
        require_contained(v[name]['path'], p.parent)
    # Only implementation source/static descriptors, never source-image/object assets.
    for d in v['code']:
        q = require_contained(d['path'], R)
        assert q.is_relative_to(R / 'experiments') or q.is_relative_to(R / 'src')
    return v


def fail(path):
    p = Path(path)
    if p.exists() and not (p / 'failure.json').exists():
        put(p / 'failure.json', dict(error=traceback.format_exc(), automatic_retry=False))


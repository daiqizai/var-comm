"""Finite timing contract; importing this file performs no scientific work."""
import hashlib
import json
from pathlib import Path

R = Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
B = Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm')
RT = Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/var_comm_runtime_20261010')
E = R / 'experiments/five-method-timing-20261020'
T = B / 'five_method_timing_20261020_v1'
O = R / 'results/five_method_timing_20261020_v1'
METHODS = ('NEST', 'P100000', 'Swin200000', 'BPG', 'ARPC')
SNRS = (4, 10, 19)
INDICES = (0, 25, 50, 75, 4, 21, 24, 29, 33, 41, 52, 60, 64, 87, 92, 95)
PY = str(RT / 'envs/frozen_visual_py310/bin/python')
CPY = str(RT / 'envs/frozen_ldpc_py311/bin/python')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for v in iter(lambda: f.read(8 << 20), b''):
            h.update(v)
    return h.hexdigest()


def pin(path):
    p = Path(path)
    assert p.is_file() and not p.is_symlink(), p
    return dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p))


def checked(d):
    p = Path(d['path'])
    assert sha(p) == d['sha256'], p
    if 'bytes' in d:
        assert p.stat().st_size == d['bytes'], p
    return p


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    import os
    p = Path(path)
    with p.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())


def grid():
    """Warmups and measured calls are fresh; repeats share one noise realization."""
    rows = []
    for snr in SNRS:
        for slot, index in enumerate(INDICES):
            for repeat in range(6):
                rows.append(dict(key=f's{snr}_i{index:04d}_r{repeat}',
                                 snr_db=snr, source_index=index, source_slot=slot,
                                 repeat=repeat % 3,
                                 phase='warmup' if repeat < 3 else 'measure',
                                 counter=(SNRS.index(snr) * 100 + index) * 3))
    assert len(rows) == 288
    return rows


def process(pid):
    p = Path('/proc') / str(pid)
    if not p.exists():
        return None
    s = (p / 'stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=pid, start_ticks=int(s[19]), state=s[0])


def binding(expected):
    assert sha(O / 'registration.json') == expected
    r = read(O / 'registration.json')
    for d in r['new_sources']:
        checked(d)
    assert r['grid'] == grid() and r['methods'] == list(METHODS)
    return r

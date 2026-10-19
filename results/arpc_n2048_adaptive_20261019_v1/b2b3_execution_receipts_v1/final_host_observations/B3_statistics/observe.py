"""Read-only B3 statistics process closure; never invoke the statistics worker."""
from pathlib import Path
import hashlib
import json
import os
import stat
import time

R = Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
L = Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/arpc_n2048_adaptive_20261019_v1')
O = R/'results/arpc_n2048_adaptive_20261019_v1'
C = L/'b3_metrics_root_v1/statistics'
P = O/'b2b3_execution_receipts_v1/final_host_observations/B3_statistics'
HOST = 'aibox-rfbd2ce574fd-5df9766c45-vz6d9'
PINS = {
    C/'actual_wait.json': (283, '53f6afb2f852a1e03d1fedde30837fd8ffccfea07a8c76cca65dac0cacd54db8'),
    C/'launch.json': (719, '4b9302a018867cc9fe635b953ceeec8956d5717b0ccd3b59e5b99b931646a5ab'),
    C/'supervisor.json': (914, '5dc47a9a3565ba9e0293fab0938dc1f8fd13f7d34fec20b936049e4b8b43a8ef'),
    O/'b3_metrics_v1/completion.json': (1986, '1e9a004069717e005e64257d0ef7a9896db9aed328b5652b2dc5866f7475aa90'),
}


def pin(path):
    path = Path(path)
    assert path.is_absolute() and '..' not in path.parts
    assert all(not p.is_symlink() for p in (path, *path.parents))
    before = path.stat()
    assert stat.S_ISREG(before.st_mode) and before.st_size < 10000000
    data = path.read_bytes()
    after = path.stat()
    assert (before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_ino, after.st_size, after.st_mtime_ns)
    return dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def canonical(value):
    assert type(value['pid']) is int and type(value['start_ticks']) is int and value['host'] == HOST
    return {k: value[k] for k in ('pid', 'start_ticks', 'host')}


def is_gone(value):
    d = canonical(value)
    p = Path('/proc')/str(d['pid'])/'stat'
    if not p.exists():
        return True
    raw = p.read_text()
    fields = raw[raw.rindex(') ')+2:].split()
    return int(fields[19]) != d['start_ticks']


assert os.uname().nodename == HOST
assert not (P/'host_observation.json').exists()
for path, (n, h) in PINS.items():
    assert pin(path) == dict(path=str(path), bytes=n, sha256=h)
w = json.loads((C/'actual_wait.json').read_bytes())
launch = json.loads((C/'launch.json').read_bytes())
supervisor = json.loads((C/'supervisor.json').read_bytes())
assert type(w['returncode']) is int and w['returncode'] == 0 and w['metadata_error'] is None
assert all(w[k] is False for k in ('timed_out', 'terminated', 'killed'))
assert canonical(w['process']) == canonical(launch['process'])
assert canonical(supervisor['process']) == canonical(launch['parent_process'])
assert w['process']['pid'] == 4080590 and w['process']['start_ticks'] == 209908520
assert launch['parent_process']['pid'] == 4080589 and launch['parent_process']['start_ticks'] == 209908516
assert launch['argv'][-3:] == ['statistics', '--expected', '2d999a213719cae5a133816e9091707807f6e89b952ac94e925e4ee37b7cfb39']
assert all(is_gone(v) for v in (w['process'], launch['parent_process']))
assert not (C/'failure.json').exists()
done = json.loads((O/'b3_metrics_v1/completion.json').read_bytes())
assert done['status'] == 'N2048_ADAPTIVE_TWO_METHOD_COMPARISON_COMPLETE'
assert done['sources'] == 100 and done['logical_frames'] == 1200 and done['snr_db'] == [4, 10]
for d in done['outputs']:
    assert pin(d['path']) == d
now = time.time()
assert now >= launch['started_unix']+w['seconds']
observation = dict(schema='FINAL_PUBLICATION_PROCESS_OBSERVATION_V1', status='CLOSED_ON_EXECUTION_HOST',
    host=HOST, observer_host=HOST, execution_host=HOST, child=w['process'], parent=launch['parent_process'],
    processes=[canonical(w['process']), canonical(launch['parent_process'])], child_gone=True, parent_gone=True,
    all_process_generations_gone=True, actual_wait=pin(C/'actual_wait.json'), wait=pin(C/'actual_wait.json'),
    launch=pin(C/'launch.json'), argv=launch['argv'], supervisor=pin(C/'supervisor.json'),
    completion=pin(O/'b3_metrics_v1/completion.json'), output_files=done['outputs'],
    not_before_unix=launch['started_unix']+w['seconds'], observed_unix=now,
    observer_source=pin(__file__), observer_pid=os.getpid(), scientific_calls=0, statistic_calls=0,
    new_bootstrap_calls=0, observation_only=True,
    process_scope_note='The detached controller was reparented to PID 1. PID 1 is the system init, not an experiment process and is not asserted gone. Both actual worker and detached controller generations are gone.')
with (P/'host_observation.json').open('x', encoding='utf8', newline='\n') as f:
    json.dump(observation, f, ensure_ascii=False, indent=2)
    f.write('\n'); f.flush(); os.fsync(f.fileno())
print(json.dumps(dict(observation=pin(P/'host_observation.json'), content=observation)))

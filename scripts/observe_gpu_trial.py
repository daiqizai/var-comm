"""Sample GPU concurrency without changing any training or device settings."""

import argparse
import csv
from datetime import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import time


def stamp():
    return datetime.now().astimezone().isoformat()


def process_state(pid, process_root=Path('/proc')):
    try:
        process = process_root / str(pid)
        fields = (process / 'stat').read_text().rsplit(') ', 1)[1].split()
        return {'pid': pid, 'state': fields[0], 'parent_pid': int(fields[1]), 'start_ticks': fields[19]}
    except (FileNotFoundError, ProcessLookupError):
        return None


def owned_handle(pid, script, process_root=Path('/proc')):
    process = process_root / str(pid)
    if process.stat().st_uid != os.getuid() or str(script).encode() not in (process / 'cmdline').read_bytes().split(b'\0'):
        raise RuntimeError('observation target is not the expected owned process')
    state = process_state(pid, process_root)
    if state is None or state['state'] == 'Z':
        raise RuntimeError('observation target is already terminal')
    return state


def alive(handle, process_root=Path('/proc')):
    state = process_state(handle['pid'], process_root)
    return state is not None and state['state'] != 'Z' and state['start_ticks'] == handle['start_ticks']


def classify(pid, handles, process_root=Path('/proc')):
    roots = {handle['pid']: handle for handle in handles}
    seen = set()
    for depth in range(64):
        if pid <= 0 or pid in seen:
            return 'foreign'
        seen.add(pid)
        state = process_state(pid, process_root)
        if state is None:
            return 'unknown_process_exited_or_unreadable'
        if pid in roots and state['state'] != 'Z' and state['start_ticks'] == roots[pid]['start_ticks']:
            return 'owned_trial'
        pid = state['parent_pid']
    return 'unknown_ancestry_depth'


def parse_table(text, fields):
    rows = []
    for values in csv.reader(io.StringIO(text)):
        if not values:
            continue
        if len(values) != len(fields):
            raise RuntimeError('GPU observation fields are incomplete')
        row = {field: value.strip() for field, value in zip(fields, values)}
        if 'pid' in row:
            row['pid'] = int(row['pid'])
        rows.append(row)
    return rows


def query(kind, fields):
    text = subprocess.check_output(['nvidia-smi', '-i', '0', '--query-' + kind + '=' + ','.join(fields),
                                    '--format=csv,noheader,nounits'], text=True, timeout=15)
    return parse_table(text, fields)


def write_json(path, record):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(record, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--owner', action='append', required=True, help='PID=/absolute/script.py')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--interval', type=float, default=10)
    parser.add_argument('--execute', action='store_true')
    arguments = parser.parse_args()
    if arguments.interval < 5:
        raise ValueError('GPU observation interval must be at least five seconds')
    if not arguments.execute:
        print('PLAN ONLY: sample telemetry and process ancestry; no training, inference or device changes')
        return
    handles = []
    for specification in arguments.owner:
        pid, script = specification.split('=', 1)
        handles.append(owned_handle(int(pid), Path(script).resolve()))
    output = arguments.output.resolve()
    project = Path(__file__).resolve().parents[1]
    if not output.is_relative_to(project / 'outputs'):
        raise ValueError('observation output must remain inside project outputs')
    output.mkdir(parents=True, exist_ok=False)
    started = stamp()
    source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    metadata = {'started_local': started, 'pid': os.getpid(), 'owned_handles': handles,
        'interval_seconds': arguments.interval, 'source_sha256': source_sha,
        'passive_sampling_not_proof_of_continuous_exclusivity': True, 'research_goal_complete': False}
    write_json(output / 'metadata.json', metadata)
    counts = {'samples': 0, 'with_foreign_compute': 0, 'with_unknown_ownership': 0, 'query_errors': 0}
    gpu_fields = ('uuid', 'name', 'temperature.gpu', 'power.draw', 'clocks.current.sm',
                  'memory.used', 'utilization.gpu', 'clocks_event_reasons.sw_thermal_slowdown',
                  'clocks_event_reasons.hw_thermal_slowdown')
    with (output / 'samples.jsonl').open('a') as stream:
        while any(alive(handle) for handle in handles):
            tick = time.monotonic()
            sample = {'local_time': stamp(), 'monotonic_seconds': tick}
            try:
                sample['gpu'] = query('gpu', gpu_fields)
                sample['compute_processes'] = query('compute-apps', ('pid', 'process_name', 'used_gpu_memory'))
                if len(sample['gpu']) != 1:
                    raise RuntimeError('expected exactly one authorized GPU')
                for process in sample['compute_processes']:
                    process['ownership'] = classify(process['pid'], handles)
                counts['with_foreign_compute'] += any(process['ownership'] == 'foreign' for process in sample['compute_processes'])
                counts['with_unknown_ownership'] += any(process['ownership'].startswith('unknown') for process in sample['compute_processes'])
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
                sample['query_error'] = repr(error)
                counts['query_errors'] += 1
            sample['query_finished_local'] = stamp()
            counts['samples'] += 1
            stream.write(json.dumps(sample) + '\n')
            stream.flush()
            write_json(output / 'status.json', {**metadata, 'status': 'PASSIVELY_OBSERVING_LIVE_TRIAL',
                'last_sample_local': sample['local_time'], **counts})
            time.sleep(max(0, arguments.interval - (time.monotonic() - tick)))
    write_json(output / 'completion.json', {**metadata, 'completed_local': stamp(), 'status': 'OWNED_HANDLES_TERMINAL_OBSERVATION_ENDED',
        **counts, 'trial_success_not_inferred': True, 'samples_sha256': hashlib.sha256((output / 'samples.jsonl').read_bytes()).hexdigest()})


if __name__ == '__main__':
    main()

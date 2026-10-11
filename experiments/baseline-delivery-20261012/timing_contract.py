"""Finite timing admission and crash-safe case reservations; no model execution.

This is shared infrastructure for the pending six-method timing endpoint. It is
not a qualified timing runner and does not automatically launch experiments.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

FIXED16 = (0, 25, 50, 75, 4, 21, 24, 29, 33, 41, 52, 60, 64, 87, 92, 95)
SNRS = (4, 10, 19)
METHODS = ('RAW_WHOLE', 'RAW_PARTIAL', 'EC_VAR_WHOLE', 'EC_VAR_PARTIAL',
           'P100000', 'Swin200000')
WARMUPS = 3
REPEATS = 3


def fixed_cases():
    """Original fixed16, batch1, three warmups and three measured repetitions."""
    return [dict(method=m, source_index=i, snr_db=s, repeat=n,
                 phase='warmup' if n < WARMUPS else 'measurement')
            for m in METHODS for i in FIXED16 for s in SNRS
            for n in range(WARMUPS + REPEATS)]


def exclusive_snapshot(repository, allowed_observer=None):
    """Preserve the original all-device/process admission; never stop other jobs."""
    output = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                      '--format=csv,noheader,nounits'], text=True)
    gpu_pids = {int(s.strip()) for s in output.splitlines() if s.strip().isdigit()}
    allowed = {os.getpid()}
    if allowed_observer is not None:
        # The caller must first verify its registered observer identity and role.
        assert allowed_observer['identity_verified'] is True
        assert allowed_observer['wait_only'] is True
        allowed.add(allowed_observer['pid'])
    process_text = subprocess.check_output(['ps', '-eo', 'pid=,comm=,args='], text=True)
    conflicts = []
    for line in process_text.splitlines():
        parts = line.strip().split(None, 2)
        if (len(parts) == 3 and int(parts[0]) not in allowed and
                parts[1].startswith('python') and str(repository) in parts[2]):
            conflicts.append(int(parts[0]))
    return dict(admitted=not (gpu_pids - {os.getpid()}) and not conflicts,
                checked_unix=time.time(), gpu_pids=sorted(gpu_pids),
                conflicting_repository_python_pids=conflicts,
                scope='original all-device idle and repository-process guard',
                jobs_stopped=0, global_gpu_settings_changed=False)


class CaseReservations:
    """Persist before entering measured code, flush after it, block crash replay.

    File I/O is excluded from timed regions. Packet callbacks count in memory
    only. The original four-digital-arm cap is preserved at 2,304 decodes; the
    two continuous baselines do not call this digital LDPC receiver ledger.
    """
    def __init__(self, path):
        self.path = Path(path)
        self.plan = fixed_cases()
        self.total_packet_cap = 4 * len(FIXED16) * len(SNRS) * (WARMUPS + REPEATS) * 2
        self.current = None
        if self.path.exists():
            raise RuntimeError('Existing timing attempt: inspect its journal; never replay automatically')
        self.state = dict(schema='FROZEN_SIX_METHOD_TIMING_CASES_V1',
                          status='REGISTERED_NOT_EXECUTED', cases=[], packet_decodes=0,
                          case_cap=len(self.plan), measured_case_cap=len(self.plan) // 2,
                          packet_cap=self.total_packet_cap)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open('x') as handle:
            json.dump(self.state, handle, indent=2)

    def flush(self):
        temporary = self.path.with_suffix('.tmp')
        with temporary.open('w') as handle:
            json.dump(self.state, handle, indent=2)
            handle.flush(); os.fsync(handle.fileno())
        temporary.replace(self.path)

    def begin(self, case):
        assert self.current is None, 'Previous case unresolved'
        index = len(self.state['cases'])
        assert index < len(self.plan) and case == self.plan[index], 'Frozen case order changed'
        row = dict(case=case, state='RESERVED', actual_packet_decodes=0)
        self.state['cases'].append(row); self.state['status'] = 'RUNNING'
        self.flush(); self.current = row

    def packet_call(self, callback):
        assert self.current is not None, 'No paid case reservation'
        assert self.current['case']['method'] in METHODS[:4]
        assert self.current['actual_packet_decodes'] < 2, 'Header/body cap exceeded'
        assert self.state['packet_decodes'] < self.total_packet_cap
        # Charge before executing, including failures. No disk I/O in this call.
        self.current['actual_packet_decodes'] += 1
        self.state['packet_decodes'] += 1
        return callback()

    def finish(self, identity, timings):
        assert self.current is not None
        assert identity['fresh_visual_encoder'] and not identity['cached_tokens_or_streams']
        assert identity['batch_size'] == 1
        assert all(identity[name] for name in ('wave_sha256', 'observation_sha256', 'reconstruction_sha256'))
        assert all(timings[name] >= 0 for name in ('tx_seconds', 'rx_seconds', 'channel_seconds', 'outer_seconds'))
        assert timings['outer_seconds'] >= max(timings['tx_seconds'], timings['rx_seconds'])
        timings = dict(timings)
        timings['software_e2e_excluding_channel_seconds'] = timings['outer_seconds'] - timings['channel_seconds']
        assert timings['software_e2e_excluding_channel_seconds'] >= 0
        self.current.update(state='COMPLETE', identity=identity, timings=timings)
        self.current = None
        if len(self.state['cases']) == len(self.plan):
            self.state['status'] = 'COMPLETE_ALL_FIXED_TIMING_CASES'
        self.flush()


def binding(path):
    p = Path(path)
    return dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())

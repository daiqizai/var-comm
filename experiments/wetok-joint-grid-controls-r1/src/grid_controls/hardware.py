"""Memory-guarded quality execution; this does not provide compute or whole-process isolation."""

import csv
import io
import os
import subprocess

import torch

from innovation_comm.hardware import gpu_telemetry
from wetok_comm.common import now


def gpu_memory():
    output = subprocess.check_output(['nvidia-smi', '-i', '0', '--query-gpu=memory.free,memory.total',
        '--format=csv,noheader,nounits'], text=True, timeout=15)
    rows = list(csv.reader(io.StringIO(output)))
    if len(rows) != 1 or len(rows[0]) != 2:
        raise RuntimeError('cannot identify the authorized GPU memory')
    free, total = (int(value.strip()) * 1024 ** 2 for value in rows[0])
    if free < 0 or total <= 0 or free > total:
        raise RuntimeError('invalid GPU memory observation')
    return free, total


def process_ids():
    output = subprocess.check_output(['nvidia-smi', '-i', '0', '--query-compute-apps=pid',
        '--format=csv,noheader'], text=True, timeout=15)
    rows = [value.strip() for value in output.splitlines() if value.strip()]
    if any(not value.isdigit() for value in rows):
        raise RuntimeError('GPU process ownership cannot be observed')
    return sorted({int(value) for value in rows})


def admit_quality(config, shared):
    foreign = [pid for pid in process_ids() if pid != os.getpid()]
    if foreign and not shared:
        raise RuntimeError('quality sharing needs explicit --shared-gpu when another GPU task is present')
    free, total = gpu_memory()
    requested = int(config['shared_start_free_GiB'] * 1024 ** 3)
    if free < requested:
        raise RuntimeError('insufficient free GPU memory for the registered quality admission')
    device_total = torch.cuda.get_device_properties(0).total_memory
    limit = int(config['pytorch_allocator_limit_GiB'] * 1024 ** 3)
    if limit > device_total:
        raise RuntimeError('registered allocator limit exceeds visible device memory')
    torch.cuda.memory.set_per_process_memory_fraction(limit / device_total, device=0)
    return {'local_time': now(), 'shared_start_requested': bool(shared), 'foreign_compute_pids': foreign,
        'free_bytes_before_models': free, 'reported_total_bytes': total, 'allocator_limit_bytes': limit,
        'not_a_whole_process_or_compute_isolation_guarantee': True}


def quality_telemetry(config):
    free, total = gpu_memory()
    if free < int(config['shared_minimum_free_GiB_between_sources'] * 1024 ** 3):
        raise RuntimeError('insufficient free GPU memory between quality source images')
    return {**gpu_telemetry(), 'compute_pids': process_ids(), 'observer_pid': os.getpid(),
        'free_bytes': free, 'reported_total_bytes': total, 'quality_phase_not_a_latency_measurement': True}

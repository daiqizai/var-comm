"""Read-only timing context and balanced receiver measurement order."""

import csv
import io
import subprocess

from wetok_comm.common import now


FIELDS = ('uuid', 'name', 'temperature.gpu', 'fan.speed', 'power.draw', 'clocks.current.sm',
          'clocks_event_reasons.sw_thermal_slowdown', 'clocks_event_reasons.hw_thermal_slowdown')


def gpu_telemetry():
    output = subprocess.check_output(['nvidia-smi', '-i', '0', '--query-gpu=' + ','.join(FIELDS),
                                      '--format=csv,noheader,nounits'], text=True, timeout=15)
    rows = list(csv.reader(io.StringIO(output)))
    if len(rows) != 1 or len(rows[0]) != len(FIELDS):
        raise RuntimeError('cannot attribute receiver timing to exactly one GPU')
    return {'local_time': now(), **{key: value.strip() for key, value in zip(FIELDS, rows[0])}}


def receiver_order(names, frame_index):
    names = list(names)
    if len(names) != 6 or len(set(names)) != 6 or frame_index < 0:
        raise ValueError('receiver timing order requires six distinct registered methods')
    shift = frame_index % len(names)
    return names[shift:] + names[:shift]

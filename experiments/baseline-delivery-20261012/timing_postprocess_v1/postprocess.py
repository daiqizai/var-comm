"""Read-only CPU audit and publication tables after the real v5 child exits.

Never imports an experiment owner, Torch, models, PHY or a bootstrap function.
Only an independently created sibling delivery directory is written by `run`.
"""
import argparse
import csv
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
RUNTIME = Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/baseline_strength_20261011/six_method_timing_v5')
EXPECTED_REGISTRATION = '44b14186a878621ea15e18da8cbf50e185927e6390826d019cff0865afc666f6'
METHODS = ('RAW_WHOLE', 'RAW_PARTIAL', 'EC_VAR_WHOLE', 'EC_VAR_PARTIAL', 'P100000', 'Swin200000')
SOURCES = (0, 25, 50, 75, 4, 21, 24, 29, 33, 41, 52, 60, 64, 87, 92, 95)
SNRS = (4, 10, 19)
LABELS = dict(zip(METHODS, ('Whole-scale + VAR (raw)', 'NeST-Com (raw)',
    'Whole-scale + VAR (entropy-coded)', 'NeST-Com (entropy-coded)',
    'Latent-space JSCC (100k)', 'SwinJSCC (adapted, 200k)')))
CASE_KEYS = ('method', 'source_index', 'snr_db', 'repeat', 'phase')
COMPONENTS = ('tx_seconds', 'rx_seconds', 'software_e2e_excluding_channel_seconds', 'channel_seconds')
RAW_FIELDS = ('raw_wall_tx_seconds', 'raw_wall_rx_seconds', 'raw_wall_outer_seconds',
              'audit_excluded_tx_seconds', 'audit_excluded_rx_seconds', 'audit_excluded_seconds')
BOUNDARIES = {
    'scope': 'shared-environment latency on the same H800 and software',
    'TX': 'fresh image encoding and communication encoding; digital serialization or all actual entropy fallback attempts, paid header, LDPC and modulation included',
    'RX': 'actual received fields, header/body decoding and failure rule; independent entropy decoding where applicable; VAR completion and frozen Dc or the baseline decoder',
    'software_e2e': 'audit-adjusted outer wall time minus separately measured channel-noise generation time',
    'channel': 'software channel-noise generation; neither radio airtime nor propagation latency',
    'audit': 'only named pure provenance hash costs explicitly measured and subtracted; original raw wall values retained',
    'warmup': 'three fresh warmups followed by three fresh measured repeats per method/source/SNR; repeated wave/RX/output identities must match',
    'units': 'CSV stores original seconds and derived milliseconds; manuscript table rounds milliseconds only',
}
LIMITATIONS = [
    'Shared resident workloads were present. These are shared-environment latency measurements, not uncontended latency.',
    'Resource observations occur at case boundaries, not continuously inside each timed window.',
    'Methods execute in a fixed sequential order; shared load and temporal drift can confound cross-method speed comparisons.',
    'The three repetitions reuse the registered noise realization with fresh computation; they are timing repetitions, not independent quality-noise trials.',
    'Qualification establishes within-run repeatability and source/model/code bindings; no new cross-host numerical-equivalence claim is made.',
    'Software timing excludes model construction, disk persistence, evaluation networks and radio airtime. Energy consumption was not measured.',
    'No same-hardware HiFi or BPG timing is supplied. No ratio to historical 4090D measurements is reported.',
    'Parameter counts describe loaded components. P communication parameters alone are not the total system parameter count.',
    'The worker receipt has no start tick; postprocessing conservatively requires its PID to be absent, even if that PID might have been reused.',
]


class AuditError(RuntimeError):
    pass


class Pending(RuntimeError):
    pass


def require(value, message):
    if not value:
        raise AuditError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def pin(path):
    p = Path(path)
    return dict(path=str(p), sha256=sha(p), bytes=p.stat().st_size)


def verify_pin(item):
    require(Path(item['path']).is_file(), 'Missing bound file: ' + item['path'])
    require(sha(item['path']) == item['sha256'], 'SHA mismatch: ' + item['path'])
    if 'bytes' in item:
        require(Path(item['path']).stat().st_size == item['bytes'], 'Byte count mismatch: ' + item['path'])
    return item


def planned_cases():
    return [dict(method=m, source_index=i, snr_db=s, repeat=r,
                 phase='warmup' if r < 3 else 'measurement')
            for m in METHODS for i in SOURCES for s in SNRS for r in range(6)]


def proc_identity(pid, proc_root=Path('/proc')):
    p = Path(proc_root) / str(pid)
    try:
        fields = (p / 'stat').read_text().rsplit(') ', 1)[1].split()
        argv = (p / 'cmdline').read_bytes().decode().split('\0')
        return dict(pid=pid, start_ticks=fields[19], state=fields[0], argv=[a for a in argv if a])
    except FileNotFoundError:
        return None


def completion_gate(folder, process_reader=proc_identity):
    """Read only receipts first. Missing success is WAIT; a failure is not retryable."""
    folder = Path(folder)
    wait_path = folder / 'actual_wait.json'
    if not wait_path.is_file():
        raise Pending('WAIT_ACTUAL_CHILD_EXIT: no actual_wait.json')
    wait = read(wait_path)
    require(wait.get('actual_child_waited') is True and wait.get('exit_code') == 0
            and wait.get('timed_out') is False, 'Actual child did not exit successfully; preserve scene, no retry')
    launch, worker = read(folder / 'launch.json'), read(folder / 'worker_launch.json')
    require(worker['owner_pid'] == launch['owner']['pid'], 'Worker/owner identity receipt differs')
    observed = process_reader(worker['pid'])
    if observed is not None:
        raise Pending('WAIT_WORKER_PID_ABSENT: cannot prove recorded child PID is gone')
    owner = launch['owner']; observed_owner = process_reader(owner['pid'])
    if observed_owner is not None and observed_owner['start_ticks'] == str(owner['start_ticks']):
        raise Pending('WAIT_OWNER_EXIT: original owner is still present')
    if not (folder / 'completion.json').is_file():
        raise Pending('WAIT_COMPLETION_RECEIPT')
    done = read(folder / 'completion.json')
    require(done.get('status') == 'COMPLETE_FIXED16_SIX_METHOD_TIMING', 'Incomplete timing status')
    require(done.get('shared_environment') is True and done.get('exclusive_or_uncontended_claimed') is False,
            'Shared wrapper did not complete')
    for key in ('case_journal', 'raw_wall_and_audit', 'per_boundary_resource_history', 'pure_provenance_exclusions'):
        require(key in done, 'Missing final wrapper evidence: ' + key)
    return dict(wait=wait, launch=launch, worker=worker, completion=done,
                child_pid_absent=True, original_owner_exited=True,
                observed_reused_owner_pid=observed_owner)


def failed_exit_gate(folder, process_reader=proc_identity):
    folder = Path(folder)
    if not (folder / 'actual_wait.json').is_file():
        raise Pending('WAIT_ACTUAL_CHILD_EXIT')
    wait, launch, worker = read(folder / 'actual_wait.json'), read(folder / 'launch.json'), read(folder / 'worker_launch.json')
    require(wait['actual_child_waited'] is True and wait['exit_code'] != 0, 'Explicit partial audit requires a real failed exit')
    require(worker['owner_pid'] == launch['owner']['pid'], 'Owner mismatch')
    if process_reader(worker['pid']) is not None:
        raise Pending('WAIT_WORKER_PID_ABSENT')
    owner = process_reader(launch['owner']['pid'])
    if owner is not None and owner['start_ticks'] == str(launch['owner']['start_ticks']):
        raise Pending('WAIT_OWNER_EXIT')
    return dict(wait=wait, launch=launch, worker=worker, child_pid_absent=True, original_owner_exited=True)


def finite(value, name):
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value >= 0, 'Invalid duration/number: ' + name)
    return float(value)


def close(a, b, name, absolute=1e-9, relative=1e-10):
    require(math.isclose(float(a), float(b), abs_tol=absolute, rel_tol=relative), 'Numeric mismatch: ' + name)


def validate_registration(reg):
    for name, expected in [('methods', list(METHODS)), ('SNRs', list(SNRS)), ('fixed_indices', list(SOURCES)),
                           ('warmups', 3), ('repeats', 3), ('total_frames', 1728), ('measured_frames', 864),
                           ('digital_packet_cap', 2304), ('batch_size', 1), ('threads', 6), ('interop_threads', 2)]:
        require(reg[name] == expected, 'Frozen registration changed: ' + name)
    require(reg['schema'] in ('FROZEN_SIX_METHOD_FIXED16_TIMING_V5_RECOVERY',
                              'FROZEN_SIX_METHOD_FIXED16_TIMING_V6_PARTIAL_RECOVERY'), 'Wrong revision')
    require(reg['automatic_retries'] is False, 'Retry policy changed')


def validate_grid(journal, reg, done, partial=False):
    plan = planned_cases(); rows = journal['cases']; records = {r['source_index']: r for r in reg['records']}
    require(len(records) == 16, 'Incomplete source binding grid')
    unresolved = []
    if partial:
        require(0 < len(rows) <= 1728 and journal['status'] != 'COMPLETE_ALL_FIXED_TIMING_CASES', 'Not a partial stopped journal')
        require(all(r['case'] == plan[n] for n, r in enumerate(rows)), 'Partial grid order or duplicate differs')
        unfinished = [n for n, r in enumerate(rows) if r['state'] != 'COMPLETE']
        require(not unfinished or unfinished == [len(rows) - 1], 'Nontrailing unresolved case')
        if unfinished:
            unresolved = [rows[-1]]
            require(rows[-1]['state'] == 'RESERVED', 'Unknown unresolved state')
            count = rows[-1]['actual_packet_decodes']
            require(type(count) is int and 0 <= count <= (2 if rows[-1]['case']['method'] in METHODS[:4] else 0), 'Unresolved packet count invalid')
            rows = rows[:-1]
    else:
        require(len(rows) == 1728, 'Incomplete case grid')
        require(journal['status'] == 'COMPLETE_ALL_FIXED_TIMING_CASES', 'Unresolved timing journal')
    require((journal['case_cap'], journal['measured_case_cap'], journal['packet_cap']) == (1728, 864, 2304), 'Budget drift')
    first = {}; packets = 0; measured = 0; qualification = []; outcomes = Counter(); residuals = []; group_counts = Counter()
    for n, (row, expected) in enumerate(zip(rows, plan)):
        require(row['case'] == expected and row['state'] == 'COMPLETE', 'Order/duplicate/unresolved case: ' + str(n))
        c, a, t = row['case'], row['identity'], row['timings']; m = c['method']; source = records[c['source_index']]
        count = row['actual_packet_decodes']
        require(type(count) is int and 0 <= count <= (2 if m in METHODS[:4] else 0), 'Packet budget violation')
        packets += count; measured += c['phase'] == 'measurement'
        require(a['fresh_visual_encoder'] is True and a['cached_tokens_or_streams'] is False and a['batch_size'] == 1,
                'Nonfresh or batched timing case')
        require(a['source_id'] == source['source_id'], 'Source identity mismatch')
        for k in ('wave_sha256', 'observation_sha256', 'reconstruction_sha256'):
            require(isinstance(a[k], str) and len(a[k]) == 64 and all(v in '0123456789abcdef' for v in a[k]), 'Invalid ' + k)
        if m in METHODS[:4]:
            require(a['fresh_tokens_sha256'] == source['tokens_sha256'], 'Fresh tokenizer/source token mismatch')
        finite(a['actual_energy'], 'actual_energy')
        if m == 'P100000':
            require(abs(a['actual_energy'] - 2048) < .02, 'Continuous baseline energy mismatch')
        key = (m, c['source_index'], c['snr_db'])
        group_counts[key] += 1
        if key in first:
            require(first[key] == a, 'Fresh repetitions differ in wave, RX, output or receiver status: ' + str(key))
        else:
            first[key] = a
            qualification.append(dict(method=m, source_index=c['source_index'], source_id=a['source_id'], snr_db=c['snr_db'],
                identical_fresh_calls=6, wave_sha256=a['wave_sha256'], observation_sha256=a['observation_sha256'],
                reconstruction_sha256=a['reconstruction_sha256'], fresh_tokens_sha256=a.get('fresh_tokens_sha256', 'not applicable')))
        require(type(a['header_accepted']) is bool and a['body_crc_accepted'] in (None, True, False), 'Invalid failure flags')
        require(isinstance(a['status'], str) and a['status'], 'Missing receiver status')
        for k in ('tx_seconds', 'rx_seconds', 'channel_seconds', 'outer_seconds', *RAW_FIELDS,
                  'software_e2e_excluding_channel_seconds'):
            finite(t[k], k)
        close(t['raw_wall_tx_seconds'] - t['audit_excluded_tx_seconds'], t['tx_seconds'], 'wall/audit TX', absolute=1e-7)
        close(t['raw_wall_rx_seconds'] - t['audit_excluded_rx_seconds'], t['rx_seconds'], 'wall/audit RX', absolute=1e-7)
        close(t['audit_excluded_seconds'], t['audit_excluded_tx_seconds'] + t['audit_excluded_rx_seconds'], 'Total audit exclusions', absolute=1e-7)
        close(t['software_e2e_excluding_channel_seconds'], t['outer_seconds'] - t['channel_seconds'], 'E2E identity')
        require(t['outer_seconds'] + 1e-6 >= t['tx_seconds'] + t['rx_seconds'] + t['channel_seconds'], 'Outer timer truncates stages')
        # Outer raw/corrected clocks are sampled separately, so exact equality is not expected.
        residual = t['raw_wall_outer_seconds'] - t['audit_excluded_seconds'] - t['outer_seconds']
        residuals.append(residual)
        for side in ('TX', 'RX'):
            require(all(math.isfinite(v) and v >= 0 for v in t[side + '_components'].values()), 'Invalid component timing')
            close(t[side + '_components'][side + '_total'], t[side.lower() + '_seconds'], 'Nested total')
        events = t['audit_events']
        require(all(type(e['calls']) is int and e['calls'] >= 0 and e['seconds'] >= 0 for e in events.values()), 'Invalid audit events')
        close(sum(e['seconds'] for e in events.values()), t['audit_excluded_seconds'], 'Audit event sum', absolute=1e-7)
        outcomes[(m, c['snr_db'], c['phase'], a['status'], a['header_accepted'], a['body_crc_accepted'])] += 1
    for q in qualification:
        q['identical_fresh_calls'] = group_counts[q['method'], q['source_index'], q['snr_db']]
    recorded_packets = packets + sum(r['actual_packet_decodes'] for r in unresolved)
    require(recorded_packets == journal['packet_decodes'] and recorded_packets <= 2304, 'Packet totals differ')
    if not partial:
        require(measured == 864 and len(first) == 288 and all(n == 6 for n in group_counts.values()), 'Measured grid incomplete')
        require(recorded_packets == done['packet_calls'], 'Completion packet count differs')
        require(done['frames'] == 1728 and done['measured_frames'] == 864, 'Completion count differs')
        require(done['training_updates'] == 0 and done['original_intervals_recomputed'] == 0, 'Forbidden extra science')
    outcome_rows = [dict(zip(('method', 'snr_db', 'phase', 'receiver_status', 'header_accepted', 'body_crc_accepted', 'calls'), (*k, v)))
                    for k, v in outcomes.items()]
    return dict(frames=len(rows), measured_frames=measured, repeated_identity_groups=len(first), actual_packet_calls=recorded_packets,
                unresolved_cases=len(unresolved), unresolved_case_records=unresolved, all_outcomes_retained=True,
                outer_clock_residual_min_seconds=min(residuals) if residuals else None,
                outer_clock_residual_max_seconds=max(residuals) if residuals else None,
                outer_clock_residuals_seconds=residuals,
                outer_clock_diagnostic_flag_threshold_seconds=.001,
                outer_clock_flagged_case_indices=[n for n, value in enumerate(residuals) if abs(value) > .001],
                outer_clock_residual_note='Two independently sampled outer timestamps can be separated by CPU scheduling. All residuals are retained; 1 ms is an explanatory flag only, not a protocol gate or retry trigger.'), qualification, outcome_rows


def stats(values):
    import numpy as np
    a = np.asarray(values, dtype=np.float64)
    require(len(a) == 48 and np.isfinite(a).all() and (a >= 0).all(), 'Expected 48 finite measured calls')
    return dict(mean_seconds=float(a.mean()), median_seconds=float(np.median(a)), p95_seconds=float(np.quantile(a, .95)),
                min_seconds=float(a.min()), max_seconds=float(a.max()))


def summaries(journal):
    rows = []
    for m in METHODS:
        for s in SNRS:
            selected = [r for r in journal['cases'] if r['case']['method'] == m and r['case']['snr_db'] == s and r['case']['phase'] == 'measurement']
            for component in (*COMPONENTS, *RAW_FIELDS):
                result = stats([r['timings'][component] for r in selected])
                rows.append(dict(method=m, method_label=LABELS[m], snr_db=s, component=component,
                    source_count=16, repeats=3, calls=48, **result,
                    **{k.replace('_seconds', '_ms'): 1000 * v for k, v in result.items()}))
    return rows


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def csv_case(row):
    return dict(method=row['method'], source_index=int(row['source_index']), snr_db=int(row['snr_db']),
                repeat=int(row['repeat']), phase=row['phase'])


def validate_csvs(per_call, wall, original_summary, journal, computed):
    require(len(per_call) == len(wall) == 1728, 'Per-call CSV incomplete')
    aliases = dict(TX_seconds='tx_seconds', RX_seconds='rx_seconds', channel_seconds='channel_seconds',
                   software_e2e_excluding_channel_seconds='software_e2e_excluding_channel_seconds')
    wall_fields = ('tx_seconds', 'rx_seconds', 'outer_seconds', 'channel_seconds', *RAW_FIELDS)
    for n, (a, b, item) in enumerate(zip(per_call, wall, journal['cases'])):
        require(csv_case(a) == csv_case(b) == item['case'], 'CSV order/duplicate differs: ' + str(n))
        for col, key in aliases.items():
            require(float(a[col]) == item['timings'][key], 'Per-call CSV altered: ' + col)
        for key in wall_fields:
            require(float(b[key]) == item['timings'][key], 'Wall CSV altered: ' + key)
        require(a['receiver_status'] == item['identity']['status'] and a['header_accepted'] == str(item['identity']['header_accepted']), 'Outcome CSV mismatch')
    expected = {(r['method'], r['snr_db'], r['component']): r for r in computed if r['component'] in COMPONENTS[:3]}
    require(len(original_summary) == len(expected) == 54, 'Summary rows missing or duplicated')
    seen = set()
    for row in original_summary:
        key = (row['method'], int(row['snr_db']), row['component'])
        require(key in expected and key not in seen, 'Unexpected/duplicate summary row'); seen.add(key)
        for k in ('source_count', 'repeats', 'calls'):
            require(int(row[k]) == expected[key][k], 'Summary count mismatch')
        for k in ('mean_seconds', 'median_seconds', 'p95_seconds', 'min_seconds', 'max_seconds'):
            close(float(row[k]), expected[key][k], 'Summary/' + k, absolute=1e-12, relative=1e-12)
    return dict(per_call_rows=1728, wall_rows=1728, original_summary_rows=54,
                channel_summary_new_from_existing_durations=True, bootstrap_calls=0)


def validate_resources(history, numerical, device, reg, admission, worker_pid, expected_boundaries=1730):
    require(len(history) == expected_boundaries, 'Missing/extra before/after or per-case resource snapshots')
    require(admission['admitted'] is True and admission['selected_GPU']['memory_free_MiB'] >= 18432
            and admission['selected_GPU']['utilization_percent'] <= 25, 'Original low-load admission missing')
    require(numerical['FP32'] is True and numerical['TF32'] is False and numerical['deterministic'] is True,
            'Numeric precision changed')
    require(numerical['shared_GPU'] is True and numerical['timing_measurement'] is True, 'Wrong runtime scope')
    require(numerical['threads'] == 6 and numerical['interop_threads'] == 2
            and sorted(numerical['affinity']) == sorted(reg['CPU_slots']), 'Thread/affinity changed')
    require(numerical['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8' and numerical['allocator_cap_bytes'] == 16 * 1024**3,
            'Numeric/runtime cap differs')
    binding = device['configured_device_binding']
    require(device['same_physical_PCI'] is True and numerical['device_identity']['same_physical_PCI'] is True, 'Device PCI identity not established')
    require(device['actual_pci_bus_id'] == binding['pci_bus_id'] == numerical['device_identity']['actual_pci_bus_id'], 'Wrong physical PCI device')
    require(device['actual_cuda_uuid'] == binding['cuda_uuid'] == numerical['device_identity']['actual_cuda_uuid'], 'CUDA identity mismatch')
    require(device['configured_nvml_uuid'] == binding['nvml_uuid'] == numerical['device_identity']['configured_nvml_uuid'], 'NVML identity mismatch')
    require(device['configured_nvml_index'] == reg['GPU_index'] == binding['nvml_gpu_index'], 'GPU index changed')
    times = []; free = []; util = []; foreign = []; releases = set()
    for row in history:
        require(row['admitted'] is True and row['shared_environment'] is True and row['exclusive_or_uncontended_claimed'] is False, 'Resource window violated')
        require(row['jobs_stopped'] == 0 and row['GPU_global_settings_changed'] is False
                and not row['conflicting_repository_python_pids'], 'Interference/setting guard violated')
        gpu = row['selected_GPU']
        require(gpu['index'] == reg['GPU_index'] and gpu['uuid'] == device['configured_nvml_uuid'] and gpu['name'] == numerical['device'], 'Resource device differs')
        require(gpu['memory_free_MiB'] >= 2048 and row['memory_admitted'] is True, 'Free-memory floor violated')
        require(row['coordination']['released'] is True, 'Quality/timing coordination was not released')
        releases.add((row['coordination']['release']['path'], row['coordination']['release']['sha256']))
        times.append(finite(row['checked_unix'], 'resource timestamp')); free.append(gpu['memory_free_MiB']); util.append(gpu['utilization_percent'])
        foreign.append(sum(int(p[1]) != worker_pid for p in row['resident_compute_processes']))
    require(times == sorted(times), 'Resource snapshots reordered')
    require(len(releases) == 1, 'Coordination release changed')
    return dict(boundary_samples=len(history), minimum_free_MiB=min(free), maximum_GPU_utilization_percent=max(util),
                minimum_foreign_resident_processes=min(foreign), maximum_foreign_resident_processes=max(foreign),
                first_sample_unix=times[0], last_sample_unix=times[-1], continuous_resource_monitoring=False,
                CUDA_and_NVML_UUID_namespaces_differ=device['actual_cuda_uuid'] != device['configured_nvml_uuid'],
                same_physical_PCI=True, pci_bus_id=device['actual_pci_bus_id'], numerical_runtime=numerical), releases


def validate_inheritance(reg, done, merged, prior, authorization):
    """Prove the two segments cover the original grid without replaying the prefix."""
    expected = dict(inherited_complete_frames=576, inherited_measured_frames=288, inherited_packet_calls=1152,
        remaining_complete_frames=1152, remaining_measured_frames=576, remaining_packet_cap=1152,
        incomplete_prior_fresh_visual_encoder_attempts=1, known_fresh_visual_attempts_at_success=1729,
        physical_endpoint_entries_at_success=1729, logical_grid_slots=1728, complete_frame_cap=1728, measured_frame_cap=864)
    for key, value in expected.items():
        require(reg[key] == value, 'Recovery budget differs: ' + key)
    require(reg['recovery_is_not_a_repeated_scientific_case'] is False, 'Failed warmup repetition must be disclosed')
    require(authorization['status'] == 'REVIEWED_V6_RECOVERY_AUTHORIZED'
            and authorization['failed_slot_repeat_authorized'] is True and authorization['automatic_retries'] is False,
            'Missing explicit recovery authorization')
    for key, value in dict(inherited_complete_frames=576, remaining_complete_frames=1152, prior_failed_visual_encoder_attempts=1,
                           complete_frame_cap=1728, measured_frame_cap=864, packet_cap=2304, known_fresh_visual_attempts_at_success=1729).items():
        require(authorization[key] == value, 'Recovery authorization budget differs: ' + key)
    require(bool(authorization['authority_evidence']) and bool(authorization['review_evidence']), 'Missing recovery review/authority evidence')
    require(authorization['prior_journal'] == reg['prior_journal'], 'Authorization journal differs')
    require(len(prior['cases']) == 577 and len(merged['cases']) == 1728, 'Segment grid length differs')
    require(merged['cases'][:576] == prior['cases'][:576], 'Inherited raw calls altered or replayed')
    require(prior['cases'][-1] == dict(case=planned_cases()[576], state='RESERVED', actual_packet_decodes=0), 'Original failed slot differs')
    require(reg['archived_failed_slot'] == prior['cases'][-1], 'Failed reservation not preserved')
    require(prior['packet_decodes'] == 1152 and sum(r['actual_packet_decodes'] for r in prior['cases'][:576]) == 1152, 'Inherited packet charges differ')
    require(all(r['case']['method'] in METHODS[2:] for r in merged['cases'][576:]), 'Unexpected new raw endpoint calls')
    require(sum(r['actual_packet_decodes'] for r in merged['cases'][576:]) <= 1152, 'Remaining packet cap exceeded')
    recovery = merged['recovery']
    require(recovery['inherited_journal'] == reg['prior_journal'] and recovery['archived_failed_slot'] == prior['cases'][-1], 'Recovery archive binding differs')
    for key, value in dict(inherited_complete_frames=576, inherited_packet_decodes=1152, failed_fresh_visual_encoder_attempts=1,
                           failed_slot_is_not_a_complete_warmup=True, remaining_complete_frames=1152,
                           remaining_measured_frames=576, remaining_packet_cap=1152, automatic_retry=False).items():
        require(recovery[key] == value, 'Recovery journal budget differs: ' + key)
    for key, value in dict(inherited_complete_frames=576, newly_completed_frames=1152, completed_frames_across_attempts=1728,
                           failed_partial_visual_attempts=1, known_fresh_visual_attempts_across_attempts=1729,
                           physical_endpoint_entries_across_attempts=1729, prior_failure_preserved=True,
                           prior_failed_slot_not_counted_as_warmup=True).items():
        require(done[key] == value, 'Recovery completion differs: ' + key)
    require(done['inherited_journal'] == reg['prior_journal'] and done['recovery_authorization'] == reg['recovery_authorization'], 'Completion segment binding differs')
    return dict(inherited_v5_complete_calls=576, new_v6_complete_calls=1152, complete_calls=1728,
                measurement_calls=864, preserved_failed_fresh_visual_attempts=1,
                known_fresh_visual_attempts_across_attempts=1729, physical_endpoint_entries_across_attempts=1729,
                prior_failed_warmup_counted_as_complete=False, inherited_rows_identical=True,
                new_call_scope='Only original slots 576 through 1727; the first entropy warmup was repeated after its diagnosed failure.',
                timing_scope='Two shared-environment segments on the same GPU/software; resident load may differ.'), [
                    dict(logical_slot=n, **r['case'], segment='v5' if n < 576 else 'v6', inherited=n < 576,
                         original_row_sha256=hashlib.sha256(json.dumps(r, sort_keys=True, separators=(',', ':')).encode()).hexdigest())
                    for n, r in enumerate(merged['cases'])]


def audit(folder, expected_registration=None):
    """All expensive reads occur only after the actual child and owner have exited."""
    folder = Path(folder); gate = completion_gate(folder); done = gate['completion']
    reg_path = folder / 'registration.json'
    if folder.name == 'six_method_v5':
        expected_registration = EXPECTED_REGISTRATION
    else:
        require(folder.name == 'six_method_v6' and expected_registration is not None, 'v6 needs an explicitly pinned registration SHA')
    require(sha(reg_path) == gate['launch']['registration_sha256'] == expected_registration, 'Unexpected frozen registration')
    reg = read(reg_path); validate_registration(reg)
    evidence = {str(reg_path): pin(reg_path)}
    def evidence_pin(item):
        verify_pin(item); evidence[item['path']] = pin(item['path']); return read(item['path']) if item['path'].endswith('.json') else item
    for name, values in reg.items():
        if name.endswith('_sha256') and isinstance(values, dict):
            for p, h in values.items():
                evidence_pin(dict(path=p, sha256=h))
    # Additional pinned references are verified without importing any historical owner.
    for name in ('previous_registration', 'prelaunch_revision0', 'recovery_of', 'recovery_of_v3', 'unlaunched_v4_registration',
                 'P_checkpoint', 'Swin_checkpoint', 'visual_request', 'phy_request'):
        evidence_pin(reg[name])
    visual = read(reg['visual_request']['path'])
    model_files = [v for v in visual['spec']['files'] if Path(v['path']).suffix in ('.pt', '.pth', '.bin', '.safetensors')]
    require(len(model_files) >= 3, 'Missing frozen visual model bindings')
    for item in model_files:
        evidence_pin(item)
    import numpy as np
    require([r['source_index'] for r in reg['records']] == list(SOURCES), 'Source order differs')
    for row in reg['records']:
        evidence_pin(row['archive']); cp = evidence_pin(row['checkpoint'])
        for key in ('source_index', 'source_id', 'preprocessing_id', 'tokens_sha256'):
            require(cp[key] == row[key], 'Source checkpoint mismatch: ' + key)
        with np.load(row['archive']['path'], allow_pickle=False) as z:
            pixels, tokens = z['pixels'], z['tokens']
            require(pixels.dtype == np.uint8 and pixels.shape == (3, 256, 256), 'Source pixels differ')
            require(tokens.dtype == np.int64 and tokens.shape == (680,), 'Source token format differs')
            require(hashlib.sha256(pixels.tobytes()).hexdigest() == row['preprocessing_id'], 'Pixel hash mismatch')
            require(hashlib.sha256(b'int64:680\0' + tokens.astype('<i8').tobytes()).hexdigest() == row['tokens_sha256'], 'Token hash mismatch')
    for k in ('case_journal', 'raw_wall_and_audit', 'per_boundary_resource_history', 'pure_provenance_exclusions'):
        evidence_pin(done[k])
    for name, expected in done['output_sha256'].items():
        require(Path(name).name == name, 'Unexpected CSV output path')
        evidence_pin(dict(path=str(folder / name), sha256=expected))
    for name in ('timing_per_call.csv', 'timing_summary.csv'):
        require(name in done['output_sha256'], 'Missing pinned final CSV: ' + name)
    run = Path(done['case_journal']['path']).parent
    require(not (run / 'failure.json').exists() and not (run / 'worker_failure.json').exists(), 'v5 failure scene exists')
    journal = read(done['case_journal']['path']); grid, qualification, outcomes = validate_grid(journal, reg, done)
    computed = summaries(journal)
    csv_audit = validate_csvs(read_csv(folder / 'timing_per_call.csv'), read_csv(done['raw_wall_and_audit']['path']),
                             read_csv(folder / 'timing_summary.csv'), journal, computed)
    numerical, device = read(run / 'numerical_runtime.json'), read(run / 'cuda_device_identity.json')
    history = [json.loads(line) for line in Path(done['per_boundary_resource_history']['path']).read_text().splitlines() if line]
    is_recovery = reg['schema'] == 'FROZEN_SIX_METHOD_FIXED16_TIMING_V6_PARTIAL_RECOVERY'
    resources, releases = validate_resources(history, numerical, device, reg, read(folder / 'resource_wait.json'),
                                             gate['worker']['pid'], expected_boundaries=1154 if is_recovery else 1730)
    segments, origins = None, []
    if is_recovery:
        previous_folder = folder.parent / 'six_method_v5'; previous_gate = failed_exit_gate(previous_folder)
        require(previous_gate['wait'] == dict(actual_child_waited=True, exit_code=1, timed_out=False), 'v5 failure receipt differs')
        require(sha(previous_folder / 'registration.json') == EXPECTED_REGISTRATION, 'Wrong v5 predecessor')
        require(not (previous_folder / 'completion.json').exists(), 'Unexpected v5 successful completion')
        evidence_pin(reg['recovery_of_v5']); prior = evidence_pin(reg['prior_journal'])
        authorization = evidence_pin(reg['recovery_authorization'])
        prior_reg = read(previous_folder / 'registration.json')
        for key in ('records', 'policies', 'P_checkpoint', 'Swin_checkpoint', 'visual_request', 'phy_request',
                    'raw_noise', 'entropy_noise', 'continuous_noise_seed', 'CPU_slots', 'threads', 'interop_threads', 'precision'):
            require(reg[key] == prior_reg[key], 'Frozen scientific input differs across segments: ' + key)
        prior_grid, _, _ = validate_grid(prior, prior_reg, {}, partial=True)
        segments, origins = validate_inheritance(reg, done, journal, prior, authorization)
        previous_run = Path(reg['prior_journal']['path']).parent
        trace = read(previous_run / 'worker_failure.json')['traceback']
        require(trace.endswith("KeyError: 'IndependentProvider'\n") and 'fresh_entropy_choice' in trace, 'Diagnosed failure changed')
        prior_history_path = done['inherited_resource_history']; evidence_pin(prior_history_path)
        prior_history = [json.loads(line) for line in Path(prior_history_path['path']).read_text().splitlines() if line]
        prior_numerical = evidence_pin(done['inherited_numerical_runtime'])
        prior_device = read(previous_run / 'cuda_device_identity.json')
        previous_resources, previous_releases = validate_resources(prior_history, prior_numerical, prior_device, prior_reg,
            read(previous_folder / 'resource_wait.json'), previous_gate['worker']['pid'], expected_boundaries=578)
        for key in ('torch', 'cuda', 'cudnn', 'python', 'numpy', 'device', 'device_identity', 'FP32', 'TF32',
                    'deterministic', 'threads', 'interop_threads', 'affinity', 'CUBLAS_WORKSPACE_CONFIG', 'allocator_cap_bytes'):
            require(prior_numerical[key] == numerical[key], 'Segment numerical/hardware identity differs: ' + key)
        require(previous_resources['last_sample_unix'] <= resources['first_sample_unix'], 'Timing segments overlap or reorder')
        require(previous_releases == releases, 'Coordination release changed across segments')
        resources = dict(v5=previous_resources, v6=resources, same_GPU_and_software=True, shared_load_may_differ=True)
        segments['v5_failed_exit_gate'] = previous_gate
        segments['v5_partial_grid_audit'] = prior_grid
        for p in [previous_folder / 'actual_wait.json', previous_folder / 'launch.json', previous_folder / 'worker_launch.json',
                  previous_folder / 'resource_wait.json', previous_run / 'cuda_device_identity.json', folder / 'actual_resume_adapter.json']:
            evidence[str(p)] = pin(p)
        resume = read(folder / 'actual_resume_adapter.json')
        require(resume['inverse_AST_equal'] is True and resume['completed_prefix_does_not_enter_endpoint'] is True,
                'No evidence that inherited prefix bypasses endpoint execution')
    for p, h in releases:
        release = evidence_pin(dict(path=p, sha256=h))
        require(release['quality_jobs_stopped'] is True and release['no_quality_science_until_timing_actual_exit'] is True, 'Coordination contract differs')
        for item in release['evidence']:
            evidence_pin(item)
    counts = read(folder / 'actual_model_parameter_counts.json'); evidence_pin(counts['runtime'])
    require(counts['all_models_frozen'] is True and counts['P_communication_parameters_are_not_the_total_system'] is True, 'Model inventory claim differs')
    audit_adapter = read(done['pure_provenance_exclusions']['path'])
    require(audit_adapter['clock_adapter']['inverse_AST_equal'] is True and audit_adapter['protocol_checks_excluded'] is False
            and audit_adapter['actual_hashes_executed'] is True and audit_adapter['global_clock_or_hashlib_modified'] is False, 'Audit exclusion boundary differs')
    for p in [folder / n for n in ('actual_wait.json', 'launch.json', 'worker_launch.json', 'completion.json', 'resource_wait.json',
                                  'actual_model_parameter_counts.json', 'actual_constructor_compatibility.json', 'actual_module_aliases.json')]:
        evidence[str(p)] = pin(p)
    for n in ('numerical_runtime.json', 'cuda_device_identity.json', 'visual_resource_adapter.json', 'raw_resource_adapter.json'):
        p = run / n; evidence[str(p)] = pin(p)
    # Detect an input mutation while reading; only the explicit pinned inventory is traversed.
    for item in evidence.values():
        verify_pin(item)
    completion_gate(folder)
    report = dict(status='PASS_COMPLETED_SHARED_ENVIRONMENT_TIMING_AUDIT', audited_at_utc=datetime.now(timezone.utc).isoformat(),
        registration_sha256=expected_registration, gate=gate, grid=grid, CSV_audit=csv_audit, resource_audit=resources,
        segment_audit=segments, call_origins=origins,
        model_parameter_components=counts, model_checkpoint_bindings=[reg['P_checkpoint'], reg['Swin_checkpoint'], *model_files],
        source_count=16, SNRs=list(SNRS), timing_boundaries=BOUNDARIES, limitations=LIMITATIONS,
        original_outputs_modified=False, new_model_calls=0, channel_calls=0, packet_calls=0, bootstrap_calls=0,
        input_inventory=list(evidence.values()))
    return report, computed, qualification, outcomes


def write_csv(path, rows):
    require(bool(rows), 'Empty output table')
    with Path(path).open('x', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False); handle.write('\n')


def partial_audit(folder, process_reader=proc_identity):
    """Failure evidence only. Never turns a partial attempt into a timing result."""
    folder = Path(folder)
    gate = failed_exit_gate(folder, process_reader)
    wait, launch = gate['wait'], gate['launch']
    require(sha(folder / 'registration.json') == launch['registration_sha256'] == EXPECTED_REGISTRATION, 'Wrong failed attempt')
    reg = read(folder / 'registration.json'); validate_registration(reg)
    # This one named runtime is the published v5 output location; no directory discovery.
    run = RUNTIME
    journal_path = run / 'case_reservations.json'; journal = read(journal_path)
    grid, qualification, outcomes = validate_grid(journal, reg, {}, partial=True)
    evidence = {}
    for name, values in reg.items():
        if name.endswith('_sha256') and isinstance(values, dict):
            for p, h in values.items():
                verify_pin(dict(path=p, sha256=h)); evidence[p] = pin(p)
    failure_paths = [p for p in (run / 'worker_failure.json', run / 'failure.json') if p.exists()]
    require(bool(failure_paths), 'Failed exit lacks recorded failure scene')
    for p in [folder / n for n in ('actual_wait.json', 'launch.json', 'worker_launch.json', 'registration.json')]:
        evidence[str(p)] = pin(p)
    for p in [journal_path, *failure_paths]:
        evidence[str(p)] = pin(p)
    complete = [r for r in journal['cases'] if r['state'] == 'COMPLETE']
    saved = [dict(**r['case'], receiver_status=r['identity']['status'],
                 wave_sha256=r['identity']['wave_sha256'], observation_sha256=r['identity']['observation_sha256'],
                 reconstruction_sha256=r['identity']['reconstruction_sha256'],
                 persisted_packet_calls=r['actual_packet_decodes'],
                 **{k: r['timings'][k] for k in (*COMPONENTS, *RAW_FIELDS)}) for r in complete]
    grid['persisted_packet_calls'] = grid.pop('actual_packet_calls')
    report = dict(status='PARTIAL_STOP_NOT_A_SIX_METHOD_TIMING_RESULT', audited_at_utc=datetime.now(timezone.utc).isoformat(),
        registration_sha256=EXPECTED_REGISTRATION, actual_wait=wait, child_pid_absent=True, grid=grid,
        completed_by_method=dict(Counter(r['case']['method'] for r in complete)),
        failure_evidence=[dict(path=str(p), content=read(p)) for p in failure_paths],
        completed_calls_must_not_be_repeated=True, partial_six_method_means_emitted=False,
        unresolved_activity='The RESERVED case may have executed model work; persisted packet count is not proof of no in-flight work. No automatic retry or budget refill is authorized.',
        complete_delivery_gate_passed=False, no_new_scientific_calls=True, bootstrap_calls=0,
        input_inventory=list(evidence.values()))
    for item in evidence.values():
        verify_pin(item)
    return report, saved, qualification, outcomes


def partial_deliver(folder):
    folder = Path(folder).resolve(); output = folder.parent / 'six_method_v5_partial_stop_v1'
    require(not output.exists(), 'Partial failure audit already exists; inspect it, do not overwrite')
    report, saved, qualification, outcomes = partial_audit(folder)
    output.mkdir(exist_ok=False)
    write_json(output / 'audit.json', report)
    if saved:
        write_csv(output / 'saved_complete_calls.csv', saved)
        write_csv(output / 'saved_qualification_groups.csv', qualification)
        write_csv(output / 'saved_receiver_outcomes.csv', outcomes)
    (output / 'REPORT.md').write_text('# Partial stopped timing attempt; no six-method result\n\n'
        f"Actual child exit: {report['actual_wait']['exit_code']}. Saved completed calls: {report['grid']['frames']}; "
        f"unresolved reserved calls: {report['grid']['unresolved_cases']}; persisted packet calls: {report['grid']['persisted_packet_calls']}.\n\n"
        'Completed calls and identities are retained. No six-method latency table or aggregate claim is emitted. '
        'The reserved case may already have executed model work, and a nonpersisted in-flight packet call cannot '
        'be ruled out merely from the reservation count. Independent failure diagnosis and original-budget '
        'accounting are required before any successor. This entry does not retry or resume a scientific case.\n', encoding='utf-8')
    write_json(output / 'completion.json', dict(status='COMPLETE_PARTIAL_FAILURE_AUDIT_ONLY',
        full_timing_complete=False, new_scientific_calls=0, bootstrap_calls=0,
        output_sha256={p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()}))
    return output


def deliver(folder, output, expected_registration=None):
    folder, output = Path(folder).resolve(), Path(output).resolve()
    require(folder.name in ('six_method_v5', 'six_method_v6'), 'Unknown attempt folder')
    require(output == folder.parent / (folder.name + '_delivery_v1'), 'Only the independent sibling delivery directory is writable')
    require(not output.exists(), 'Delivery already exists; inspect it, never overwrite')
    report, rows, qualification, outcomes = audit(folder, expected_registration)
    output.mkdir(exist_ok=False)
    write_json(output / 'audit.json', report)
    write_csv(output / 'timing_summary_all_components.csv', rows)
    write_csv(output / 'qualification.csv', qualification)
    write_csv(output / 'receiver_outcomes.csv', outcomes)
    if report['call_origins']:
        write_csv(output / 'call_origins.csv', report['call_origins'])
    selected = {(r['method'], r['snr_db'], r['component']): r for r in rows}
    paper = []
    for m in METHODS:
        for s in SNRS:
            item = dict(method=m, method_label=LABELS[m], snr_db=s, source_count=16, repeats=3, measured_calls=48)
            for label, component in zip(('TX', 'RX', 'software_e2e', 'channel'), COMPONENTS):
                r = selected[m, s, component]
                for statistic in ('mean', 'median', 'p95'):
                    item[label + '_' + statistic + '_ms'] = r[statistic + '_ms']
            matching = [r for r in outcomes if r['method'] == m and r['snr_db'] == s and r['phase'] == 'measurement']
            item['header_reject_calls'] = sum(r['calls'] for r in matching if not r['header_accepted'])
            item['body_crc_reject_calls'] = sum(r['calls'] for r in matching if r['body_crc_accepted'] is False)
            item['receiver_status_counts_json'] = json.dumps({r['receiver_status']: sum(x['calls'] for x in matching if x['receiver_status'] == r['receiver_status']) for r in matching}, sort_keys=True)
            paper.append(item)
    write_csv(output / 'paper_timing_table.csv', paper)
    tex = [r'% Fragment: shared-environment latency, milliseconds; no bootstrap intervals.',
           r'\begin{tabular}{llrrrr}', r'\hline',
           r'Method & SNR (dB) & TX & RX & Software E2E & Channel \\', r'\hline']
    for row in paper:
        tex.append(row['method_label'] + ' & ' + str(row['snr_db']) + ' & ' +
                   ' & '.join(f"{row[k + '_mean_ms']:.3f}" for k in ('TX', 'RX', 'software_e2e', 'channel')) + r' \\')
    tex += [r'\hline', r'\end{tabular}', '% Means over 16 sources x 3 measured repetitions; all receiver outcomes included.',
            '% Channel is software noise generation, excluded from Software E2E. Full CSV includes median and p95.']
    (output / 'paper_timing_table.tex').write_text('\n'.join(tex) + '\n', encoding='utf-8')
    description = ('Shared-environment latency was measured on the same NVIDIA H800 with batch size one, '
        'the fixed 16 development sources, and SNRs 4, 10, and 19 dB. Each method/source/SNR condition used '
        'three fresh warmups followed by three fresh measured repetitions. All 1,728 calls completed, '
        'including 864 measurements; all receiver outcomes were retained. Repeated transmitted waveforms, '
        'received observations and reconstructions matched within every condition. TX includes fresh source '
        'encoding and all actual entropy fallback attempts where applicable. RX includes the actual receive '
        'path and reconstruction. Software E2E excludes separately reported software noise-generation time '
        'and explicitly measured pure-provenance audit costs. Raw wall and excluded durations are reported '
        'separately. These measurements reflect shared resident workloads and fixed method order; they do '
        'not establish uncontended latency or a cross-hardware speedup. No latency confidence interval was bootstrapped.')
    if report['segment_audit']:
        description += (' The complete grid combines 576 unchanged successful calls from the stopped v5 attempt '
            'and 1,152 successful calls from the reviewed v6 recovery. The original failed first entropy warmup '
            'is archived and is not counted as a completed warmup. It adds one known fresh visual-encoder attempt: '
            'the two attempts have 1,729 known fresh visual attempts/endpoint entries for 1,728 completed '
            'end-to-end calls. The shared GPU/software match, but resident workload conditions can differ between segments.')
    table = '| Method | SNR | TX (ms) | RX (ms) | E2E (ms) | Channel (ms) |\n|---|---:|---:|---:|---:|---:|\n'
    for row in paper:
        table += '| ' + row['method_label'] + ' | ' + str(row['snr_db']) + ' | ' + ' | '.join(f"{row[k + '_mean_ms']:.3f}" for k in ('TX', 'RX', 'software_e2e', 'channel')) + ' |\n'
    (output / 'REPORT.md').write_text('# Six-method shared-environment latency\n\n' + description + '\n\n' + table +
        '\nThe complete CSV additionally reports medians and p95, raw wall durations, provenance exclusions and all receiver outcomes.\n\n' +
        '\n'.join('- ' + item for item in LIMITATIONS) + '\n', encoding='utf-8')
    segment_zh = ('\n\n完整网格由 v5 已完成且逐行原样保留的 576 次调用，以及 v6 完成的 1,152 次调用组成。'
        '原首个熵编码预热失败现场单独归档，不计为完成的预热；另有 1 次已发生的视觉编码尝试。'
        '因此两段已知 fresh 视觉尝试/端点进入数为 1,729，完整端到端调用为 1,728。两段 GPU/软件相同，但共享负载可能不同。'
        if report['segment_audit'] else '')
    (output / 'REPORT_zh.md').write_text('# 六方法共享环境延迟\n\n'
        '同一 H800、同一软件环境、batch 1，固定 16 张 development 来源，SNR 为 4/10/19 dB。'
        '每个条件实际执行 3 次全新预热和 3 次全新测量，共 1,728 次调用，其中 864 次计时测量。'
        '接收失败结果不剔除。逐条件六次波形、接收观测和重建哈希一致，fresh token 与原冻结来源一致。\n\n'
        '下表为均值毫秒；完整 CSV 同时给出中位数、p95、原始 wall 与审计扣除时间、接收状态。'
        '信道一列仅为软件噪声生成时间，已从软件端到端时间中扣除，不是无线空口时延。\n\n' + table +
        '\n测量时有其他驻留作业，因此只能称共享环境延迟。资源证据是每次调用边界的快照，'
        '不能证明测量窗内持续空闲；方法固定顺序和共享负载变化仍可能影响对比。'
        '本入口没有运行模型、信道、译码或 bootstrap；不报告相对旧 4090D、HiFi 或 BPG 的加速比。'
        '数值资格限于各条件的重复性及冻结身份，不能外推为跨主机逐位等价。' + segment_zh + '\n', encoding='utf-8')
    (output / 'manuscript_timing.tex').write_text('% Manuscript fragment; shared-environment latency only.\n' + description.replace('_', r'\_') + '\n', encoding='utf-8')
    write_json(output / 'timing_boundaries.json', BOUNDARIES)
    write_json(output / 'completion.json', dict(status='COMPLETE_CPU_TIMING_AUDIT_AND_TABLES',
        input_registration_sha256=report['registration_sha256'], measured_calls=864, timing_scope='shared-environment latency',
        output_sha256={p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()},
        source_entry=pin(Path(__file__)), new_scientific_calls=0, bootstrap_calls=0))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('check', 'run', 'partial-check', 'partial-report'))
    parser.add_argument('--input', type=Path, default=ROOT / 'results/baseline_strength_20261011/timing/six_method_v5')
    parser.add_argument('--expected-registration', help='Required pinned SHA for a reviewed v6 successor; v5 always uses its fixed SHA')
    args = parser.parse_args()
    try:
        if args.command == 'partial-check':
            report, _, _, _ = partial_audit(args.input)
            print(json.dumps(dict(status=report['status'], grid=report['grid'], writes=0), indent=2))
        elif args.command == 'partial-report':
            out = partial_deliver(args.input)
            print(json.dumps(dict(status='COMPLETE_PARTIAL_FAILURE_AUDIT_ONLY', path=str(out), full_timing_complete=False)))
        elif args.command == 'check':
            report, _, _, _ = audit(args.input, args.expected_registration)
            print(json.dumps(dict(status=report['status'], grid=report['grid'], writes=0), indent=2))
        else:
            out = deliver(args.input, args.input.parent / (args.input.name + '_delivery_v1'), args.expected_registration)
            print(json.dumps(dict(status='COMPLETE_CPU_DELIVERY', path=str(out), new_scientific_calls=0)))
    except Pending as error:
        print(json.dumps(dict(status='WAIT_ACTUAL_COMPLETION', reason=str(error), writes=0)))
        return 2
    except (AuditError, KeyError, OSError, ValueError) as error:
        print(json.dumps(dict(status='STOP_AUDIT_FAILURE_NO_RETRY', reason=str(error), new_scientific_calls=0)), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

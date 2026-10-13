"""Synthetic ledger/receipt tests only; no model, channel or packet execution."""
import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import postprocess as p


def fixture():
    records = [dict(source_index=i, source_id='source-' + str(i), tokens_sha256=hashlib.sha256(str(i).encode()).hexdigest()) for i in p.SOURCES]
    reg = dict(schema='FROZEN_SIX_METHOD_FIXED16_TIMING_V5_RECOVERY', methods=list(p.METHODS), SNRs=list(p.SNRS),
        fixed_indices=list(p.SOURCES), warmups=3, repeats=3, total_frames=1728, measured_frames=864,
        digital_packet_cap=2304, batch_size=1, threads=6, interop_threads=2, automatic_retries=False,
        records=records, CPU_slots=[25, 29, 31, 35, 39, 43], GPU_index=0)
    rows = []
    for case in p.planned_cases():
        m, i, snr = case['method'], case['source_index'], case['snr_db']
        h = hashlib.sha256(f'{m}/{i}/{snr}'.encode()).hexdigest()
        identity = dict(fresh_visual_encoder=True, cached_tokens_or_streams=False, batch_size=1,
            wave_sha256=h, observation_sha256=h, reconstruction_sha256=h, source_id='source-' + str(i),
            status='HEADER_REJECT_GRAY' if i == 0 and snr == 4 else 'DECODED', header_accepted=not(i == 0 and snr == 4),
            body_crc_accepted=None if m in p.METHODS[4:] else i != 25, actual_energy=2048., source_attempts=[])
        if m in p.METHODS[:4]:
            identity['fresh_tokens_sha256'] = next(r['tokens_sha256'] for r in records if r['source_index'] == i)
        x = .001 * (case['repeat'] + 1)
        t = dict(tx_seconds=x, rx_seconds=2*x, channel_seconds=.01, outer_seconds=3*x+.011,
                 software_e2e_excluding_channel_seconds=3*x+.001, TX_components={'TX_total': x}, RX_components={'RX_total': 2*x},
                 raw_wall_tx_seconds=x+.0001, raw_wall_rx_seconds=2*x+.0002, raw_wall_outer_seconds=3*x+.011+.0003-.000004,
                 audit_excluded_tx_seconds=.0001, audit_excluded_rx_seconds=.0002, audit_excluded_seconds=.0003,
                 audit_events={'test_pure_provenance': dict(calls=2, seconds=.0003)})
        rows.append(dict(case=case, state='COMPLETE', actual_packet_decodes=2 if m in p.METHODS[:4] else 0,
                         identity=identity, timings=t))
    journal = dict(status='COMPLETE_ALL_FIXED_TIMING_CASES', case_cap=1728, measured_case_cap=864, packet_cap=2304,
                   packet_decodes=2304, cases=rows)
    done = dict(frames=1728, measured_frames=864, packet_calls=2304, training_updates=0, original_intervals_recomputed=0)
    return reg, journal, done


def csv_rows(journal):
    per_call, wall = [], []
    for r in journal['cases']:
        c, t, a = r['case'], r['timings'], r['identity']
        per_call.append(dict(**c, TX_seconds=t['tx_seconds'], RX_seconds=t['rx_seconds'], channel_seconds=t['channel_seconds'],
                             software_e2e_excluding_channel_seconds=t['software_e2e_excluding_channel_seconds'],
                             receiver_status=a['status'], header_accepted=a['header_accepted']))
        wall.append(dict(**c, **{k: t[k] for k in ('tx_seconds', 'rx_seconds', 'outer_seconds', 'channel_seconds', *p.RAW_FIELDS)}))
    computed = p.summaries(journal)
    summary = [{k: r[k] for k in ('method', 'snr_db', 'component', 'source_count', 'repeats', 'calls',
                                  'mean_seconds', 'median_seconds', 'p95_seconds', 'min_seconds', 'max_seconds')}
               for r in computed if r['component'] in p.COMPONENTS[:3]]
    def roundtrip(rows):
        f = io.StringIO(); w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows); f.seek(0)
        return list(csv.DictReader(f))
    return roundtrip(per_call), roundtrip(wall), roundtrip(summary), computed


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.reg, self.journal, self.done = fixture()

    def test_complete_grid_and_failures_retained(self):
        p.validate_registration(self.reg)
        result, qualification, outcomes = p.validate_grid(self.journal, self.reg, self.done)
        self.assertEqual((result['frames'], result['measured_frames'], result['actual_packet_calls']), (1728, 864, 2304))
        self.assertEqual(len(qualification), 288)
        self.assertTrue(any(not r['header_accepted'] for r in outcomes))
        self.assertEqual(sum(r['calls'] for r in outcomes), 1728)

    def test_duplicate_or_reordered_case_rejected(self):
        self.journal['cases'][1]['case'] = copy.deepcopy(self.journal['cases'][0]['case'])
        with self.assertRaisesRegex(p.AuditError, 'Order/duplicate'):
            p.validate_grid(self.journal, self.reg, self.done)

    def test_unresolved_case_rejects_full_claim(self):
        self.journal['cases'][-1]['state'] = 'RESERVED'
        with self.assertRaisesRegex(p.AuditError, 'unresolved'):
            p.validate_grid(self.journal, self.reg, self.done)

    def test_changed_repeat_wave_rejected(self):
        self.journal['cases'][3]['identity']['wave_sha256'] = 'f' * 64
        with self.assertRaisesRegex(p.AuditError, 'Fresh repetitions differ'):
            p.validate_grid(self.journal, self.reg, self.done)

    def test_fresh_token_hash_mismatch_rejected(self):
        self.journal['cases'][0]['identity']['fresh_tokens_sha256'] = 'a' * 64
        with self.assertRaisesRegex(p.AuditError, 'tokenizer'):
            p.validate_grid(self.journal, self.reg, self.done)

    def test_packet_budget_and_baseline_packet_charge_rejected(self):
        self.journal['cases'][0]['actual_packet_decodes'] = 3
        with self.assertRaisesRegex(p.AuditError, 'Packet budget'):
            p.validate_grid(self.journal, self.reg, self.done)
        self.journal['cases'][0]['actual_packet_decodes'] = 2
        self.journal['cases'][-1]['actual_packet_decodes'] = 1
        with self.assertRaisesRegex(p.AuditError, 'Packet budget'):
            p.validate_grid(self.journal, self.reg, self.done)

    def test_partial_failure_keeps_reservation_without_full_aggregate(self):
        self.journal['cases'] = self.journal['cases'][:577]
        self.journal['cases'][-1] = dict(case=p.planned_cases()[576], state='RESERVED', actual_packet_decodes=0)
        self.journal.update(status='RUNNING', packet_decodes=1152)
        result, qualification, outcomes = p.validate_grid(self.journal, self.reg, {}, partial=True)
        self.assertEqual((result['frames'], result['measured_frames'], result['unresolved_cases'], result['actual_packet_calls']), (576, 288, 1, 1152))
        self.assertEqual(len(qualification), 96)
        self.assertEqual(sum(r['calls'] for r in outcomes), 576)
        with self.assertRaises(p.AuditError):
            p.validate_grid(self.journal, self.reg, self.done)

    def test_csv_exact_roundtrip_and_channel_summaries(self):
        a, b, s, computed = csv_rows(self.journal)
        result = p.validate_csvs(a, b, s, self.journal, computed)
        self.assertEqual(result['original_summary_rows'], 54)
        self.assertEqual(len(computed), 180)
        tx = next(r for r in computed if r['component'] == 'tx_seconds')
        self.assertAlmostEqual(tx['mean_seconds'], .005)
        self.assertEqual(len([r for r in computed if r['component'] == 'channel_seconds']), 18)

    def test_csv_duration_tampering_rejected(self):
        a, b, s, computed = csv_rows(self.journal); a[0]['channel_seconds'] = '99'
        with self.assertRaisesRegex(p.AuditError, 'CSV altered'):
            p.validate_csvs(a, b, s, self.journal, computed)

    def test_warmups_cannot_enter_paper_summary(self):
        a, b, s, computed = csv_rows(self.journal); s[0]['mean_seconds'] = '.0035'
        with self.assertRaisesRegex(p.AuditError, 'Summary/'):
            p.validate_csvs(a, b, s, self.journal, computed)

    def test_nan_or_audit_subtraction_tampering_rejected(self):
        self.journal['cases'][0]['timings']['tx_seconds'] = float('nan')
        with self.assertRaises(p.AuditError):
            p.validate_grid(self.journal, self.reg, self.done)
        self.journal['cases'][0]['timings']['tx_seconds'] = .001
        self.journal['cases'][0]['timings']['audit_excluded_tx_seconds'] = .1
        with self.assertRaisesRegex(p.AuditError, 'wall/audit'):
            p.validate_grid(self.journal, self.reg, self.done)

    def test_outer_clock_scheduling_residual_is_retained_flag_not_rejection(self):
        self.journal['cases'][0]['timings']['raw_wall_outer_seconds'] += .01
        result, _, _ = p.validate_grid(self.journal, self.reg, self.done)
        self.assertEqual(result['outer_clock_flagged_case_indices'], [0])
        self.assertAlmostEqual(result['outer_clock_residuals_seconds'][0], .009996)


class GateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup); self.root = Path(self.temp.name)
    def save(self, name, data):
        (self.root / name).write_text(json.dumps(data))
    def receipts(self):
        self.save('actual_wait.json', dict(actual_child_waited=True, exit_code=0, timed_out=False))
        self.save('launch.json', dict(owner=dict(pid=100, start_ticks='123', argv=['owner'])))
        self.save('worker_launch.json', dict(pid=101, owner_pid=100))
        self.save('completion.json', dict(status='COMPLETE_FIXED16_SIX_METHOD_TIMING', shared_environment=True,
            exclusive_or_uncontended_claimed=False, case_journal={}, raw_wall_and_audit={},
            per_boundary_resource_history={}, pure_provenance_exclusions={}))
    def test_missing_actual_wait_does_not_create_output(self):
        with self.assertRaises(p.Pending):
            p.completion_gate(self.root, lambda pid: None)
        self.assertEqual(list(self.root.iterdir()), [])
    def test_real_failure_blocks_default_complete_gate(self):
        self.receipts(); self.save('actual_wait.json', dict(actual_child_waited=True, exit_code=1, timed_out=False))
        with self.assertRaisesRegex(p.AuditError, 'did not exit'):
            p.completion_gate(self.root, lambda pid: None)
    def test_live_worker_and_original_owner_block(self):
        self.receipts()
        with self.assertRaisesRegex(p.Pending, 'WORKER'):
            p.completion_gate(self.root, lambda pid: {'start_ticks':'1'} if pid == 101 else None)
        with self.assertRaisesRegex(p.Pending, 'OWNER'):
            p.completion_gate(self.root, lambda pid: {'start_ticks':'123'} if pid == 100 else None)
    def test_absent_child_and_reused_owner_pid_can_pass_receipt_gate(self):
        self.receipts()
        result = p.completion_gate(self.root, lambda pid: {'start_ticks':'999'} if pid == 100 else None)
        self.assertTrue(result['child_pid_absent'])
    def test_missing_shared_wrapper_completion_blocks(self):
        self.receipts(); self.save('completion.json', dict(status='COMPLETE_FIXED16_SIX_METHOD_TIMING'))
        with self.assertRaisesRegex(p.AuditError, 'Shared wrapper'):
            p.completion_gate(self.root, lambda pid: None)


class SegmentTests(unittest.TestCase):
    def setUp(self):
        self.reg, self.merged, self.done = fixture()
        self.prior = copy.deepcopy(self.merged)
        self.prior['cases'] = self.prior['cases'][:576] + [dict(case=p.planned_cases()[576], state='RESERVED', actual_packet_decodes=0)]
        self.prior.update(status='RUNNING', packet_decodes=1152)
        journal_pin = dict(path='/v5/case_reservations.json', sha256='a'*64)
        auth_pin = dict(path='/v6/recovery_authorization.json', sha256='b'*64)
        self.reg.update(schema='FROZEN_SIX_METHOD_FIXED16_TIMING_V6_PARTIAL_RECOVERY',
            inherited_complete_frames=576, inherited_measured_frames=288, inherited_packet_calls=1152,
            remaining_complete_frames=1152, remaining_measured_frames=576, remaining_packet_cap=1152,
            incomplete_prior_fresh_visual_encoder_attempts=1, known_fresh_visual_attempts_at_success=1729,
            physical_endpoint_entries_at_success=1729, logical_grid_slots=1728, complete_frame_cap=1728, measured_frame_cap=864,
            recovery_is_not_a_repeated_scientific_case=False, archived_failed_slot=copy.deepcopy(self.prior['cases'][-1]),
            prior_journal=journal_pin, recovery_authorization=auth_pin)
        self.auth = dict(status='REVIEWED_V6_RECOVERY_AUTHORIZED', failed_slot_repeat_authorized=True, automatic_retries=False,
            inherited_complete_frames=576, remaining_complete_frames=1152, prior_failed_visual_encoder_attempts=1,
            complete_frame_cap=1728, measured_frame_cap=864, packet_cap=2304, known_fresh_visual_attempts_at_success=1729,
            authority_evidence=['registered protocol'], review_evidence=['CPU reviews'], prior_journal=journal_pin)
        self.merged['recovery'] = dict(inherited_journal=journal_pin, archived_failed_slot=copy.deepcopy(self.prior['cases'][-1]),
            inherited_complete_frames=576, inherited_packet_decodes=1152, failed_fresh_visual_encoder_attempts=1,
            failed_slot_is_not_a_complete_warmup=True, remaining_complete_frames=1152,
            remaining_measured_frames=576, remaining_packet_cap=1152, automatic_retry=False)
        self.done.update(inherited_complete_frames=576, newly_completed_frames=1152, completed_frames_across_attempts=1728,
            failed_partial_visual_attempts=1, known_fresh_visual_attempts_across_attempts=1729,
            physical_endpoint_entries_across_attempts=1729, prior_failure_preserved=True,
            prior_failed_slot_not_counted_as_warmup=True, inherited_journal=journal_pin, recovery_authorization=auth_pin)
    def validate(self):
        return p.validate_inheritance(self.reg, self.done, self.merged, self.prior, self.auth)
    def test_exact_prefix_and_origin_split_with_failed_attempt_disclosed(self):
        report, origins = self.validate()
        self.assertEqual(report['known_fresh_visual_attempts_across_attempts'], 1729)
        self.assertEqual(sum(r['segment']=='v5' for r in origins), 576)
        self.assertEqual(sum(r['segment']=='v6' for r in origins), 1152)
        p.validate_grid(self.merged, self.reg, self.done)
    def test_inherited_timing_change_rejected_even_if_grid_intact(self):
        self.merged['cases'][0]['timings']['tx_seconds'] += .001
        with self.assertRaisesRegex(p.AuditError, 'Inherited raw calls'):
            self.validate()
    def test_archived_failed_slot_cannot_be_relabelled_complete(self):
        self.reg['archived_failed_slot']['state']='COMPLETE'
        with self.assertRaisesRegex(p.AuditError, 'Failed reservation'):
            self.validate()
    def test_failed_visual_attempt_cannot_disappear_from_counts(self):
        self.done['known_fresh_visual_attempts_across_attempts']=1728
        with self.assertRaisesRegex(p.AuditError, 'Recovery completion'):
            self.validate()
    def test_recovery_needs_reviewed_failed_slot_authorization(self):
        self.auth['failed_slot_repeat_authorized']=False
        with self.assertRaisesRegex(p.AuditError, 'authorization'):
            self.validate()
    def test_new_segment_must_not_replay_a_raw_method(self):
        self.merged['cases'][576]['case']['method']='RAW_WHOLE'
        with self.assertRaisesRegex(p.AuditError, 'new raw'):
            self.validate()
    def test_separate_resource_counts_accept_only_actual_segment_boundaries(self):
        resource = ResourcesTests();resource.setUp()
        history=resource.history[:578]
        result,_=p.validate_resources(history,resource.numerical,resource.device,resource.reg,resource.admission,101,expected_boundaries=578)
        self.assertEqual(result['boundary_samples'],578)
        with self.assertRaisesRegex(p.AuditError,'snapshots'):
            p.validate_resources(history,resource.numerical,resource.device,resource.reg,resource.admission,101,expected_boundaries=1154)


class ResourcesTests(unittest.TestCase):
    def setUp(self):
        self.reg, _, _ = fixture()
        self.device = dict(same_physical_PCI=True, configured_nvml_index=0, configured_nvml_uuid='NVML', actual_cuda_uuid='CUDA',
            actual_pci_bus_id='0000:63:00.0', configured_device_binding=dict(pci_bus_id='0000:63:00.0', cuda_uuid='CUDA', nvml_uuid='NVML', nvml_gpu_index=0))
        self.numerical = dict(FP32=True, TF32=False, deterministic=True, shared_GPU=True, timing_measurement=True,
            threads=6, interop_threads=2, affinity=self.reg['CPU_slots'], CUBLAS_WORKSPACE_CONFIG=':4096:8',
            allocator_cap_bytes=16*1024**3, device_identity={k:self.device[k] for k in ('same_physical_PCI', 'actual_pci_bus_id', 'actual_cuda_uuid', 'configured_nvml_uuid')}, device='NVIDIA H800')
        template = dict(admitted=True, shared_environment=True, exclusive_or_uncontended_claimed=False, jobs_stopped=0,
            GPU_global_settings_changed=False, conflicting_repository_python_pids=[], selected_GPU=dict(index=0, uuid='NVML', name='NVIDIA H800', memory_free_MiB=3000, utilization_percent=80),
            memory_admitted=True, coordination=dict(released=True, release=dict(path='/release', sha256='a'*64)),
            resident_compute_processes=[['NVML', '200', 'other', '1024'], ['NVML','101','worker','2048']])
        self.history = [dict(copy.deepcopy(template), checked_unix=1000+n) for n in range(1730)]
        self.admission = dict(admitted=True, selected_GPU=dict(memory_free_MiB=19000, utilization_percent=0))
    def validate(self):
        return p.validate_resources(self.history, self.numerical, self.device, self.reg, self.admission, 101)
    def test_different_uuid_namespaces_same_pci_and_busy_shared_gpu_are_valid(self):
        result, _ = self.validate(); self.assertTrue(result['same_physical_PCI']); self.assertEqual(result['maximum_GPU_utilization_percent'],80)
        self.assertFalse(result['continuous_resource_monitoring'])
    def test_wrong_pci_rejected(self):
        self.device['actual_pci_bus_id'] = '0000:01:00.0'
        with self.assertRaisesRegex(p.AuditError, 'PCI'):
            self.validate()
    def test_one_resource_guard_violation_rejected(self):
        self.history[10]['selected_GPU']['memory_free_MiB'] = 100
        with self.assertRaisesRegex(p.AuditError, 'Free-memory'):
            self.validate()
    def test_missing_boundary_observations_rejected(self):
        self.history = self.history[:10]
        with self.assertRaisesRegex(p.AuditError, 'snapshots'):
            self.validate()


class OutputTests(unittest.TestCase):
    def test_independent_delivery_preserves_segments_and_never_overwrites(self):
        reg, journal, done = fixture()
        _, qualification, outcomes = p.validate_grid(journal, reg, done)
        computed = p.summaries(journal)
        origins = [dict(logical_slot=n, **r['case'], segment='v5' if n < 576 else 'v6', inherited=n < 576,
                        original_row_sha256='a'*64) for n,r in enumerate(journal['cases'])]
        report = dict(status='SYNTHETIC_TEST_ONLY', registration_sha256='b'*64, call_origins=origins,
                      segment_audit={'known_fresh_visual_attempts_across_attempts':1729})
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'six_method_v6';folder.mkdir();sentinel=folder/'sentinel.txt';sentinel.write_text('unchanged')
            output=folder.parent/'six_method_v6_delivery_v1'
            with patch.object(p,'audit',return_value=(report,computed,qualification,outcomes)):
                p.deliver(folder,output,'b'*64)
            self.assertEqual(sentinel.read_text(),'unchanged')
            paper=p.read_csv(output/'paper_timing_table.csv')
            self.assertEqual(len(paper),18)
            self.assertAlmostEqual(float(paper[0]['TX_mean_ms']),5.)
            self.assertEqual(len(p.read_csv(output/'call_origins.csv')),1728)
            text=(output/'REPORT.md').read_text()
            self.assertIn('1,729',text);self.assertIn('shared-environment',text)
            self.assertEqual(p.read(output/'completion.json')['bootstrap_calls'],0)
            with self.assertRaisesRegex(p.AuditError,'already exists'):
                p.deliver(folder,output,'b'*64)


if __name__ == '__main__':
    unittest.main()

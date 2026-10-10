"""CPU-only full1000 scope/finalist/reuse and owner guard checks; no PHY run."""
import copy
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import h800_ep_full_phy_core_v1 as c
import h800_ep_full_phy_v1 as h
import ep_plan as plan


def policy():
    return dict(status='T1_POLICIES_FROZEN_CALIBRATION_ONLY_V1', holdout_used_for_selection=False,
        source_count=1000, calibration_source_ids=[f'original{i}' for i in range(1000)],
        policies={'EC_VAR_WHOLE': {'4': {'candidate_id': 'm7_q2_r2-3'},
            '10': {'candidate_id': 'm9_q4_r2-3'}, '19': {'candidate_id': 'm9_q6_r5-6'}}})


def records():
    return [dict(source_index=i, source_id=f'original{i}') for i in range(1000)]


def metrics():
    return [dict(r, psnr_db=20., lpips_alex=.2, dinov2_vitl14_cosine=.5,
        convnext_top1_source_prediction=1)
        for r in c.pilot.frames(records()[:100], policy())]


def finalists():
    return c.pilot.finalists(metrics(), policy())


class FullScopeTests(unittest.TestCase):
    def test_complete_actual_metric_grid_is_required_and_whole_winner_retained(self):
        rows = metrics(); result = c.shortlist(rows, c.pilot.finalists(rows, policy()), policy())
        chosen = c.admitted_finalists(result, policy())
        for snr, ids in chosen.items():
            self.assertEqual(ids[:3], sorted(x['candidate_id'] for x in plan.candidates())[:3])
            self.assertIn(policy()['policies']['EC_VAR_WHOLE'][str(snr)]['candidate_id'], ids)
            self.assertLessEqual(len(ids), 4)

    def test_top3_only_or_missing_fourth_metric_cannot_enter_full(self):
        rows = metrics(); result = c.pilot.finalists(rows, policy())
        with self.assertRaises(RuntimeError): c.shortlist(rows[:3], result, policy())
        rows[0].pop('convnext_top1_source_prediction')
        with self.assertRaises((RuntimeError, KeyError)): c.shortlist(rows, result, policy())

    def test_wrong_source_or_wrong_dino_field_or_changed_finalist_is_refused(self):
        rows = metrics(); result = c.pilot.finalists(rows, policy())
        rows[0]['source_id'] = 'new100'
        with self.assertRaises(RuntimeError): c.shortlist(rows, result, policy())
        rows = metrics(); rows[0]['dino_cosine'] = rows[0].pop('dinov2_vitl14_cosine')
        with self.assertRaises((RuntimeError, KeyError)): c.shortlist(rows, result, policy())
        changed = copy.deepcopy(result); changed['snrs']['19']['finalists'].pop()
        with self.assertRaises(RuntimeError): c.shortlist(metrics(), changed, policy())

    def test_full_grid_uses_three_seeds_and_original1000_counter(self):
        result = finalists(); rows = c.frames(records(), result, policy())
        self.assertLessEqual(len(rows), 36000)
        self.assertEqual(len(rows), 3000 * sum(len(x['finalists']) for x in result['snrs'].values()))
        self.assertEqual({r['noise_seed'] for r in rows}, {4101, 4102, 4103})
        self.assertEqual(c.counter(999, 19, 4103), 17999)
        self.assertEqual(c.counter(99, 4, 4101), c.pilot.counter(99, 4))
        self.assertEqual(c.PACKET_CAP, 2 * c.LOGICAL_CAP)

    def test_no_omitted_reordered_or_holdout_source(self):
        result = finalists(); original = records(); rows = c.frames(original, result, policy())
        for changed in (rows[:-1], list(reversed(rows))):
            with self.assertRaises(RuntimeError): c.complete_rows(changed, original, result, policy())
        original[999]['source_id'] = 'new100'
        with self.assertRaises(RuntimeError): c.frames(original, result, policy())

    def test_all_partial_top3_still_must_retain_original_whole(self):
        result = finalists()
        ranked = [x['candidate_id'] for x in plan.candidates() if x['target_K'] > 0]
        ranked += [x['candidate_id'] for x in plan.candidates() if x['target_K'] == 0]
        for row in result['snrs'].values():
            row['ranking'] = ranked
            row['finalists'] = list(plan.full_shortlist(ranked, row['original_final_whole_winner']))
        self.assertTrue(all(len(v) == 4 for v in c.admitted_finalists(result, policy()).values()))
        result['snrs']['4']['finalists'].pop()
        with self.assertRaises(RuntimeError): c.admitted_finalists(result, policy())


class ExactReuseTests(unittest.TestCase):
    def packet(self):
        return dict(logical_event=dict(source_id='original0', snr_db=4, noise_seed=4101, public_frame_counter=3000,
            candidate_id='any_legal_target'), transmission=dict(profile_id=144,
            public_frame_counter=3000, session='frozen', transmitted_frame_sha256='wave'),
            payload_sha256='payload', noise_sha256='noise', observation_sha256='observed',
            full_public_receive_catalogue=True, source_truth_supplied_to_RX=False)

    def test_different_candidate_label_does_not_create_different_actual_event(self):
        first = self.packet(); second = copy.deepcopy(first)
        second['logical_event']['candidate_id'] = 'another_target'
        identity = dict(profile_count=360, numeric='sameCPU', catalogue_sha256='same')
        self.assertEqual(c.pilot.physical_key(first, identity), c.pilot.physical_key(second, identity))

    def test_crc_or_candidate_name_cannot_replace_exact_observation_identity(self):
        first = self.packet(); identity = dict(profile_count=360, numeric='sameCPU', catalogue_sha256='same')
        key = c.pilot.physical_key(first, identity)
        for field in ('payload_sha256', 'noise_sha256', 'observation_sha256'):
            second = copy.deepcopy(first); second[field] = 'changed'
            self.assertNotEqual(key, c.pilot.physical_key(second, identity))
        second = copy.deepcopy(first); second['transmission']['public_frame_counter'] += 1
        self.assertNotEqual(key, c.pilot.physical_key(second, identity))
        second = copy.deepcopy(first); second['logical_event']['noise_seed'] = 4102
        self.assertNotEqual(key, c.pilot.physical_key(second, identity))

    def test_old144_profile_or_unclosed_certificate_refused(self):
        with self.assertRaises(RuntimeError):
            c.pilot.physical_key(self.packet(), dict(profile_count=144))
        with self.assertRaises(RuntimeError):
            c.pilot_cache(dict(schema='OLD_UNVERIFIED'), {}, lambda _: None)

    def cache_fixture(self):
        identity = dict(profile_count=360, numeric='sameCPU', catalogue_sha256='same')
        event = c.pilot.frames(records()[:100], policy())[0]
        packet = dict(self.packet(), schema=c.pilot.SCHEMA, external_packet_reuse=False,
            logical_event=event, actual_RX=dict(status='HEADER_REJECT', header=dict(header_ok=False),
            body=None, rx_profile=None))
        packet['physical_key'] = c.pilot.physical_key(packet, identity)
        pin = lambda path: dict(path=path, sha256='a'*64)
        request = dict(schema='H800_EP_ORIGINAL100_PILOT_PHY_V1', phy_identity=identity,
            frames=c.pilot.frames(records()[:100], policy()))
        worker = dict(status='PASS_H800_ORIGINAL100_PILOT_PHY_ONLY', request_sha256='a'*64,
            CUDA_initialized=False, results=dict(logical_frames=[None]*43200,
            packet_ledger=dict(unresolved=0,total=1), actual_new_physical=[pin('packet')]))
        owner = dict(status=worker['status'], worker_completion=pin('worker'), request_sha256='a'*64,
            actual_wait=dict(success=True), actual_children_waited=True, worker_exit_codes=[0])
        values = dict(request=request, worker=worker, owner=owner, packet=packet,
            wait=dict(actual_wait=True,returncode=0))
        certificate = dict(schema='CLOSED_ORIGINAL100_PILOT_PHY_REUSE_V1', phy_identity=identity,
            previous_call_counts_preserved=True, owner_completion=pin('owner'), owner_actual_wait=pin('wait'),
            request=pin('request'), closed48={})
        return certificate, identity, values

    def test_actual_failed_pilot_observation_is_preserved_and_reusable(self):
        cert, identity, values = self.cache_fixture()
        with mock.patch.object(c.pilot, 'external_packets', return_value={}):
            cache = c.pilot_cache(cert, identity, lambda p: values[p['path']])
        self.assertEqual(len(cache), 1)
        row = next(iter(cache.values()))
        self.assertEqual(row['actual_RX']['status'], 'HEADER_REJECT')
        self.assertEqual(row['origin'], 'CLOSED_PILOT')

    def test_unwaited_owner_or_changed_packet_key_cannot_supply_reuse(self):
        cert, identity, values = self.cache_fixture(); values['wait']['returncode'] = 1
        with self.assertRaises(RuntimeError): c.pilot_cache(cert, identity, lambda p: values[p['path']])
        cert, identity, values = self.cache_fixture(); values['packet']['physical_key'] = 'changed'
        with mock.patch.object(c.pilot, 'external_packets', return_value={}), self.assertRaises(RuntimeError):
            c.pilot_cache(cert, identity, lambda p: values[p['path']])


class OwnerTests(unittest.TestCase):
    def test_bound_caps_admit_only_two_cpu_phy_and_finite_window(self):
        caps = h.scientific_caps()
        self.assertEqual(caps['actual_new_packet_decodes_cap'], 72000)
        self.assertEqual(caps['logical_frames_cap'], 36000)
        for k in ('source_TX_calls', 'encoder_calls', 'model_calls', 'RX_VAR_calls',
                  'render_calls', 'metric_calls', 'new_confirmation_source_reads'):
            self.assertEqual(caps[k], 0)
        self.assertFalse(h.execution(time.time()+21600, 21600)['CUDA_visible'])
        with self.assertRaises(RuntimeError): h.execution(time.time()+30000, 21601)

    def test_engine_preserves_earlier_owner_and_private_worker_identity(self):
        before = h.source.engine().worker_identity.__globals__['__file__']
        runner = h.engine()
        self.assertEqual(h.source.engine().worker_identity.__globals__['__file__'], before)
        self.assertEqual(runner.worker_identity.__globals__['__file__'], h.__file__)
        with mock.patch.object(h, 'FULL_CORE_SHA', '0'*64), self.assertRaises(RuntimeError): h.engine()

    def test_missing_future_source_or_metric_closure_refused_without_loading(self):
        with self.assertRaises(RuntimeError): h.source_closure({})
        with self.assertRaises(RuntimeError): h.metric_closure({}, policy())

    def test_owner_replay_refused_before_child_launch(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); (root/'run').mkdir(); path=root/'request.json'; path.write_text('{}')
            class Context:
                def __enter__(self): return self
                def __exit__(self,*a): pass
            shared=mock.Mock(); shared.owner_lock.return_value=Context()
            args=mock.Mock(request=str(path),request_sha256='f'*64)
            with mock.patch.object(h.g.sys,'platform','linux'), \
                 mock.patch.object(h,'registration',return_value=(dict(execution={}),{})), \
                 mock.patch.object(h.g,'inside',side_effect=Path), \
                 mock.patch.object(h.g,'helper',return_value=shared), \
                 mock.patch.object(h.subprocess,'Popen') as popen, self.assertRaisesRegex(RuntimeError,'already claimed'):
                h.run(args)
            popen.assert_not_called()


if __name__ == '__main__': unittest.main()

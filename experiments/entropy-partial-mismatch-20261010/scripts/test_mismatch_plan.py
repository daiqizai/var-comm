"""Pure metadata tests: no NumPy/model/PHY/metric/statistic calls."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import mismatch_plan as m

BASE = Path(__file__).resolve().parents[1]/'tests/fixtures/common500_v1'


class PlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old = m.pinned_json(BASE/'plan.json', m.ORIGINAL_PLAN_SHA)
        cls.manifest = m.pinned_json(BASE/'manifest.json', m.SOURCE_MANIFEST_SHA)
        cls.sources, cls.schedule = m.validate_inputs(cls.old, cls.manifest)
        cls.rows = list(m.logical_frames(cls.sources, cls.schedule))

    def test_exact_grid_and_physical_dedup(self):
        plan = m.make_plan(self.old, self.manifest)
        self.assertEqual(len(self.rows), 5400)
        self.assertEqual(len({m.physical_key(r) for r in self.rows}), 4500)
        self.assertEqual(plan['accounting']['matched_unique_physical_frames'], 1500)
        self.assertEqual(plan['accounting']['maximum_new_packet_calls_if_all_matched_physical_frames_admitted'], 6000)
        self.assertEqual(plan['accounting']['historical_reuse_actually_admitted'], 0)

    def test_counter_matches_original_code_all_900_channel_cases(self):
        path = BASE/'raw64_unified500_plan.py'
        data = path.read_bytes()
        self.assertEqual(hashlib.sha256(data).hexdigest(), m.ORIGINAL_NOISE_CODE_SHA)
        namespace = {}; exec(compile(data, str(path), 'exec'), namespace)
        for source in self.sources:
            for actual in m.ACTUAL_SNRS:
                for seed in m.SEEDS:
                    c = m.channel_identity(source, actual, seed)
                    self.assertEqual(c['public_counter'], namespace['frame_counter'](source['source_index'], actual, seed))
                    material = namespace['canonical']([m.PROTOCOL, 'holdout', source['source_id'], actual, seed]).encode()
                    self.assertEqual(c['pcg64_seed'], int.from_bytes(hashlib.sha256(material).digest()[:16], 'little'))
        # standard_noise is deliberately NEVER called.
        self.assertEqual(m.channel_identity(self.sources[0], 4, 6201)['public_counter'], 1500)
        self.assertEqual(m.channel_identity(self.sources[99], 10, 6203)['public_counter'], 4799)

    def test_lookup_changes_only_policy_not_channel(self):
        selected = [r for r in self.rows if r['source']['source_index'] == 0 and r['actual_snr_db'] == 4
                    and r['channel']['noise_seed'] == 6201]
        self.assertEqual(len(selected), 6)
        self.assertEqual(len({m.digest(r['channel']) for r in selected}), 1)
        self.assertEqual(len({r['profile_id'] for r in selected}), 5)
        at7 = [r for r in selected if r['config_snr_db'] == 7]
        self.assertEqual(m.physical_key(at7[0]), m.physical_key(at7[1]))
        changed = copy.deepcopy(at7[0]); changed['channel']['receiver_snr_db'] = 7
        with self.assertRaises(ValueError): m.physical_key(changed)

    def test_invalid_source_order_seed_and_lookup_rejected(self):
        changed = copy.deepcopy(self.manifest); changed['records'][0]['source_index'] = 100
        with self.assertRaises(ValueError): m.validate_inputs(self.old, changed)
        changed = copy.deepcopy(self.manifest); changed['records'][0]['source_id'] = 'different'
        with self.assertRaises(ValueError): m.validate_inputs(self.old, changed)
        with self.assertRaises(ValueError): m.frame(self.sources[0], 4, 10, 'WHOLE', self.schedule)
        with self.assertRaises(ValueError): m.channel_identity(self.sources[0], 19, 6201)
        with self.assertRaises(ValueError): m.channel_identity(self.sources[0], 4, 2001)
        with self.assertRaises(ValueError): m.channel_identity(dict(self.sources[0], source_index=100), 4, 6201)

    def test_changed_frozen_noise_and_policy_rejected(self):
        changed = copy.deepcopy(self.old); changed['noise']['source_count'] = 100
        with self.assertRaises(ValueError): m.validate_inputs(changed, self.manifest)
        changed = copy.deepcopy(self.old); changed['schedule'][0]['profile_id'] = 1
        with self.assertRaises(ValueError): m.validate_inputs(changed, self.manifest)

    def test_same_bytes_hash_pin(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'metadata.json'; data = b'{"x":1}'; path.write_bytes(data)
            self.assertEqual(m.pinned_json(path, hashlib.sha256(data).hexdigest()), {'x': 1})
            path.write_bytes(b'{"x":2}')
            with self.assertRaises(ValueError): m.pinned_json(path, hashlib.sha256(data).hexdigest())

    def test_every_phy_reuse_binding_must_match(self):
        expected = {k: 'same' for k in m.PHY_IDENTITY_KEYS}
        self.assertEqual(m.check_identity_bindings(expected, expected, 'phy')['status'], 'METADATA_IDENTITY_MATCH_ONLY')
        self.assertFalse(m.check_identity_bindings(expected, expected, 'phy')['scientific_reuse_admitted'])
        for key in m.PHY_IDENTITY_KEYS:
            changed = dict(expected); changed[key] = 'different'
            self.assertEqual(m.check_identity_bindings(expected, changed, 'phy')['mismatched'], [key])
        incomplete = dict(expected); incomplete.pop('received_sha256')
        self.assertEqual(m.check_identity_bindings(expected, incomplete, 'phy')['status'], 'IDENTITY_REJECTED')

    def test_image_and_metric_pins_separate_from_phy(self):
        for stage, keys in [('image', m.IMAGE_IDENTITY_KEYS), ('metrics', m.METRIC_IDENTITY_KEYS)]:
            expected = {k: 'same' for k in keys}
            for key in keys:
                observed = dict(expected); observed[key] = 'changed'
                self.assertEqual(m.check_identity_bindings(expected, observed, stage)['mismatched'], [key])

    @staticmethod
    def rx(accepted=True, crc=False):
        return dict(header_ok=accepted, body_attempted=accepted, gray=not accepted,
            source_decode_complete=True, execution_status='complete', rule='KEEP',
            rx_profile_id=387 if accepted else None, rx_profile_legal=accepted,
            tx_profile_id=152, body_crc_accept=crc if accepted else False,
            source_status='KEEP_ACTUAL_HARD_TOKENS' if accepted else 'HEADER_REJECT_GRAY',
            packet_call_count=2 if accepted else 1)

    def test_accepted_wrong_legal_profile_and_bad_crc_keep(self):
        self.assertTrue(m.validate_receiver_summary(self.rx()))
        wrong = self.rx(); wrong['gray'] = True
        with self.assertRaises(ValueError): m.validate_receiver_summary(wrong)
        wrong = self.rx(); wrong['rx_profile_legal'] = False
        with self.assertRaises(ValueError): m.validate_receiver_summary(wrong)

    def test_header_failure_no_body_not_body_failure_denominator(self):
        self.assertTrue(m.validate_receiver_summary(self.rx(False)))
        wrong = self.rx(False); wrong['body_attempted'] = True
        with self.assertRaises(ValueError): m.validate_receiver_summary(wrong)

    def test_actual_call_count_duplicates_and_unresolved(self):
        rows = [dict(physical_key='A', state='complete', mode='new', actual_new_packet_calls=2, receiver=self.rx()),
                dict(physical_key='B', state='complete', mode='new', actual_new_packet_calls=1, receiver=self.rx(False))]
        self.assertEqual(m.call_accounting(rows)['actual_new_packet_calls'], 3)
        with self.assertRaises(ValueError): m.call_accounting(rows + [rows[0]])
        wrong = copy.deepcopy(rows); wrong[1]['state'] = 'reserved'
        with self.assertRaises(ValueError): m.call_accounting(wrong)
        wrong = copy.deepcopy(rows); wrong[1]['actual_new_packet_calls'] = 0
        with self.assertRaises(ValueError): m.call_accounting(wrong)

    def test_cache_path_alone_does_not_count_as_reuse(self):
        row = dict(physical_key='A', state='complete', mode='historical_reuse', actual_new_packet_calls=0, receiver=self.rx())
        with self.assertRaises(ValueError): m.call_accounting([row])
        row['reuse_admission'] = dict(path='independently-verified-admission.json', sha256='1'*64)
        self.assertTrue(m.call_accounting([row])['independent_admission_check_required'])


if __name__ == '__main__':
    unittest.main(verbosity=2)

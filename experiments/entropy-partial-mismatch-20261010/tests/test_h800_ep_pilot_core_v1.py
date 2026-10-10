"""Focused CPU-only pilot scope, actual-input cache and selection tests."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import h800_ep_pilot_core_v1 as c
import ep_plan as plan
import ep_source_codec as partial
import leo_whole_gate_v1 as g


def policy():
    return dict(status='T1_POLICIES_FROZEN_CALIBRATION_ONLY_V1', holdout_used_for_selection=False,
        source_count=1000, calibration_source_ids=[f'source{i}' for i in range(1000)],
        policies={'EC_VAR_WHOLE':{'4':{'candidate_id':'m7_q2_r2-3'},
            '10':{'candidate_id':'m9_q4_r2-3'}, '19':{'candidate_id':'m9_q6_r5-6'}}})


def rows():
    return c.frames([dict(source_index=i, source_id=f'source{i}') for i in range(100)], policy())


def actual_rx():
    return dict(status='PAYLOAD_PARSED', header=dict(header_ok=True, profile_id=144),
        body=dict(crc_accepted=True, parser_accepted=True, payload=[1, 0, 1, 0]),
        rx_profile=dict(profile_id=144, family='EC_VAR_PARTIAL', m=4, K=6, q=2, nominal_rate='1/2'))


def packet():
    return dict(logical_event=dict(source_id='source0', noise_seed=4101, snr_db=4, public_frame_counter=3000,
                                  candidate_id='target1'),
        transmission=dict(profile_id=144, session='session', codeword_sha256='code', transmitted_frame_sha256='wave'),
        payload_sha256='bits', noise_sha256='noise', observation_sha256='observed',
        full_public_receive_catalogue=True, source_truth_supplied_to_RX=False, actual_RX=actual_rx())


class PlanTests(unittest.TestCase):
    def test_complete_exact_grid_and_finite_budgets(self):
        value = rows(); c.complete_rows(value)
        self.assertEqual(len(value), 43200)
        self.assertEqual({r['noise_seed'] for r in value}, {4101})
        self.assertEqual(c.PACKET_CAP, 2 * len(value))
        self.assertEqual(c.SOURCE_CAPS['prior_scale'], 68 * 10)
        self.assertEqual(c.VISUAL_CAPS['prior_scale'], 43200 * 20)
        self.assertEqual(c.SOURCE_CAPS['encoder'] + c.VISUAL_CAPS['encoder'], 0)

    def test_missing_duplicate_or_changed_candidate_refused(self):
        original = rows()
        for change in ('missing', 'duplicate', 'capacity', 'noise'):
            value = copy.deepcopy(original)
            if change == 'missing': value.pop()
            elif change == 'duplicate': value[-1] = value[0]
            elif change == 'capacity': value[0]['source_capacity_bits'] += 1
            else: value[0]['noise_seed'] = 4102
            with self.subTest(change=change), self.assertRaises(RuntimeError): c.complete_rows(value)

    def test_holdout_or_reordered_source_cannot_enter(self):
        records = [dict(source_index=i, source_id=f'source{i}') for i in range(100)]
        records[0]['source_id'] = 'new_confirmation0'
        with self.assertRaisesRegex(RuntimeError, 'original calibration'): c.frames(records, policy())

    def test_counter_is_original_numeric_counter(self):
        self.assertEqual(c.counter(99, 19), 15297)
        with self.assertRaises(RuntimeError): c.counter(100, 19)
        with self.assertRaises(RuntimeError): c.counter(0, 13)


class ReuseTests(unittest.TestCase):
    def test_identical_actual_wave_not_target_label_defines_physical_reuse(self):
        first = packet(); second = copy.deepcopy(first)
        second['logical_event']['candidate_id'] = 'another_target'
        identity = dict(profile_count=360, catalogue_sha256='cat', numeric='FP32')
        self.assertEqual(c.physical_key(first, identity), c.physical_key(second, identity))
        second['observation_sha256'] = 'different_noise_or_observation'
        self.assertNotEqual(c.physical_key(first, identity), c.physical_key(second, identity))

    def test_changed_backend_catalogue_session_or_counter_prevents_reuse(self):
        first = packet(); identity = dict(profile_count=360, catalogue_sha256='cat', numeric='FP32')
        original = c.physical_key(first, identity)
        for key in ('catalogue_sha256', 'numeric'):
            other = dict(identity); other[key] = 'changed'
            self.assertNotEqual(original, c.physical_key(first, other))
        for field in ('session', 'codeword_sha256'):
            other = copy.deepcopy(first); other['transmission'][field] = 'changed'
            self.assertNotEqual(original, c.physical_key(other, identity))
        other = copy.deepcopy(first); other['logical_event']['public_frame_counter'] += 1
        self.assertNotEqual(original, c.physical_key(other, identity))
        with self.assertRaises(RuntimeError): c.physical_key(first, dict(identity, profile_count=144))

    def test_source_reuse_uses_received_family_mK_bits_not_transport_q(self):
        first = actual_rx(); second = copy.deepcopy(first)
        second['rx_profile'].update(q=6, nominal_rate='5/6', profile_id=155)
        self.assertEqual(c.source_input(first, 'frozen_visual'), c.source_input(second, 'frozen_visual'))
        for field, value in [('m', 5), ('K', 12), ('family', 'EC_VAR_WHOLE')]:
            other = copy.deepcopy(first); other['rx_profile'][field] = value
            self.assertNotEqual(c.source_input(first, 'frozen_visual'), c.source_input(other, 'frozen_visual'))
        other = copy.deepcopy(first); other['body']['payload'][0] ^= 1
        self.assertNotEqual(c.source_input(first, 'frozen_visual'), c.source_input(other, 'frozen_visual'))
        self.assertNotEqual(c.source_input(first, 'GPU0'), c.source_input(first, 'GPU2'))

    def test_rejected_bits_do_not_become_source_oracle_inputs(self):
        first = actual_rx(); first['body']['crc_accepted'] = False
        second = actual_rx(); second.update(body=None, rx_profile=None)
        second['header']['header_ok'] = False
        self.assertEqual(c.source_input(first, 'same'), c.source_input(second, 'same'))
        self.assertEqual(c.source_input(first, 'same')['kind'], 'FIXED_GRAY')

    def test_external_cache_requires_actual_closed48_and_no_old144(self):
        certificate = dict(schema='OLD_T1', phy_identity={}, already_consumed_counts_preserved=True)
        with self.assertRaises(RuntimeError): c.external_packets(certificate, {})
        certificate['schema'] = 'CLOSED_EP48_EXACT_EVENT_REUSE_V1'
        certificate['phy_identity'] = {'profile_count':144}
        with self.assertRaises(RuntimeError): c.external_packets(certificate, {'profile_count':360})


class SourceTests(unittest.TestCase):
    def test_old32_cannot_be_reencoded_by_new68_stage(self):
        with self.assertRaisesRegex(RuntimeError, '0032..0099'):
            c.source68(dict(source_index=31), None, None, None, None, None, None)

    def test_new68_actual_dynamic_bits_use_exactly10TXpriors_and_no_RX(self):
        with tempfile.TemporaryDirectory() as td:
            ledger = g.Ledger(Path(td) / 'calls', lambda:None, c.SOURCE_CAPS)
            backend = types.SimpleNamespace(g=g, boundary=lambda:None)
            class Codec:
                def encode_endpoints(self, tokens, endpoints):
                    self.received = tokens.copy()
                    for scale in range(10):
                        ledger.call('prior_scale', lambda:None)
                        backend.traces.append(dict(source=32, role='TX', m=10, scale=scale, cdf_sha256='f'*64))
                    return {(m, K):dict(m=m, K=K, arithmetic_bits=200+m+K,
                        bits=np.zeros(200+m+K, dtype=np.uint8)) for m, K in endpoints}
            codec = Codec(); read = mock.Mock(return_value=np.arange(680, dtype=np.int64))
            result = c.source68(dict(source_index=32, source_id='source32'), backend, codec, partial,
                                ledger, read, Path(td) / 'source')
            read.assert_called_once()
            decoded = c.link.load_streams(result['archive'], c.checked(result['metadata']), partial)
            self.assertEqual(len(decoded[4, 0]), 204)
            self.assertEqual(ledger.completed['source_tx'], 1)
            self.assertEqual(ledger.completed['prior_scale'], 10)
            self.assertEqual(ledger.completed['source_rx'], 0)
            self.assertEqual(ledger.completed['var_render'], 0)


class SelectionTests(unittest.TestCase):
    def metric_rows(self):
        return [dict(r, **dict(zip(c.METRICS, (20., .2, .5, 1.)))) for r in rows()]

    def test_full_whole_winner_retained_with_lexical_ties(self):
        values = self.metric_rows(); result = c.finalists(values, policy())
        legal = sorted(x['candidate_id'] for x in plan.candidates())
        for snr in plan.SNRS:
            row = result['snrs'][str(snr)]
            self.assertEqual(row['ranking'], legal)
            self.assertEqual(row['finalists'][:3], legal[:3])
            self.assertIn(policy()['policies']['EC_VAR_WHOLE'][str(snr)]['candidate_id'], row['finalists'])
        self.assertFalse(result['final_strategy_frozen'])

    def test_wrong_DINO_or_missing_fourth_metric_blocks_selection(self):
        values = self.metric_rows()
        values[0]['dino_cosine'] = values[0].pop('dinov2_vitl14_cosine')
        with self.assertRaises((RuntimeError, KeyError)): c.finalists(values, policy())
        values = self.metric_rows(); values[0].pop('convnext_top1_source_prediction')
        with self.assertRaises((RuntimeError, KeyError)): c.finalists(values, policy())

    def test_missing_cell_or_percent_agreement_blocks_selection(self):
        values = self.metric_rows()
        with self.assertRaises(RuntimeError): c.finalists(values[:-1], policy())
        values[0]['convnext_top1_source_prediction'] = 100.
        with self.assertRaises(RuntimeError): c.finalists(values, policy())


if __name__ == '__main__': unittest.main()

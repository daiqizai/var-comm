"""Pure CPU engineering tests; every header/LDPC decoder below is a test double."""
from pathlib import Path
import copy
import hashlib
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

HERE = Path(__file__).resolve().parent
if (HERE.parent/'phy_codec').exists():
    sys.path.insert(0, str(HERE.parent/'phy_codec'))
sys.path.insert(0, str(HERE))
import h_payload_cpu as core
from h64_catalog import all_buckets, catalogue, SIZES, digest, token_count
import h64_phy as phy


class FakeBackend:
    """No LDPC implementation: scripted decoder results and transparent encoder."""
    def __init__(self):
        self.encoded = None
        self.decoded_override = None
        self.decode_calls = []
        self.raise_decode = False

    def plan(self, k, n, q):
        return dict(k=k, n=n, q=q, layout_id=f'TEST_ONLY_{k}_{n}_{q}')

    def encode(self, bits, n, q):
        self.encoded = np.asarray(bits).copy()
        return np.pad(self.encoded, ((0, 0), (0, n-self.encoded.shape[1])))

    def decode(self, logits, k, n, q):
        self.decode_calls.append((k, n, q))
        if self.raise_decode:
            raise RuntimeError('scripted CPU decoder failure')
        bits = self.decoded_override if self.decoded_override is not None else self.encoded[0]
        if len(bits) != k:
            bits = np.zeros(k, dtype=np.uint8)  # test wrong layout -> CRC rejection
        return np.asarray(bits, dtype=np.uint8)[None]


class FakeHeader:
    def __init__(self, force_id=None, reject=False):
        self.force_id, self.reject, self.sent_ids, self.received = force_id, reject, [], []

    def transmit(self, profile_id):
        self.sent_ids.append(profile_id)
        return np.ones((68, 2), dtype=np.float32)

    def receive(self, received, snr, codebook):
        self.received.append(received.copy())
        pid = self.force_id if self.force_id is not None else self.sent_ids[-1]
        ok = not self.reject and str(pid) in codebook
        return dict(header_ok=ok, profile_id=pid if ok else None, header_crc_ok=not self.reject)


class FakeLedger:
    branch = 'H'
    limits = {core.PHASE: 20000, 'development': 13200}

    def __init__(self):
        self.events, self.results, self.configurations = [], {}, {}

    def register_configuration(self, key, definition):
        if key in self.configurations:
            assert self.configurations[key] == definition
        self.configurations[key] = copy.deepcopy(definition)

    def decode_once(self, phase, event_id, kind, key, request, decode):
        binding = digest([phase, kind, key, request])
        if event_id in self.results:
            expected, result = self.results[event_id]
            assert expected == binding
            return copy.deepcopy(result)
        row = dict(phase=phase, event_id=event_id, kind=kind, request=copy.deepcopy(request), status='RESERVED')
        self.events.append(row)
        try:
            result = decode()
        except BaseException:
            row['status'] = 'FAILED'
            raise
        row['status'] = 'COMPLETE'
        self.results[event_id] = (binding, copy.deepcopy(result))
        return result


class PayloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cat = catalogue(all_buckets(FakeBackend()))
        cls.protocol = dict(schema='H_CODEC_PROTOCOL_V1', N=1024, header_symbols=68, body_symbols=956, Es=2, channel='AWGN')

    def setUp(self):
        self.scales = [(np.arange(s*s, dtype=np.int64)+31*i) % 4096 for i, s in enumerate(SIZES)]
        self.streams = {m: np.zeros(12*token_count(m)+2, dtype=np.uint8) for m in (6, 7, 8, 9)}
        self.backend, self.header, self.ledger = FakeBackend(), FakeHeader(), FakeLedger()

    def candidate(self, arm='H64-R', target=7, rate='1/2', slot=0, snr=13):
        identity = dict(arm=arm, target_m=target, nominal_rate=rate)
        return dict(**identity, candidate_id=digest(identity), q=4 if arm.startswith('H16-') else 6,
                    K=0, snr_db=snr, slot=slot, selection_rank=0)

    def context(self, candidates):
        shortlist = dict(status='H_EXPECTED_PSNR_SHORTLIST_FROZEN', ready_for_real_calibration=True,
                         source_ids=[f'source_{i:04d}' for i in range(200)], whole_candidates=candidates)
        contract = dict(status='H_PAYLOAD_CPU_ENGINEERING_SEALED', execution_registration_sha256='a'*64,
                        core_source_sha256=core.file_sha(core.__file__), protocol_canonical_sha256=digest(self.protocol),
                        catalogue_canonical_sha256=digest(self.cat), shortlist_canonical_sha256=digest(shortlist),
                        engineering_choices_sha256=digest(core.ENGINEERING_CHOICES))
        return shortlist, contract

    def run_frame(self, candidate, others=(), **changes):
        shortlist, contract = self.context([candidate, *others])
        args = dict(protocol=self.protocol, catalogue=self.cat, shortlist=shortlist, contract=contract,
                    candidate=candidate, source_id='source_0000', source_index=0, noise_seed=6101,
                    scales=self.scales, arithmetic_bits=self.streams, backend=self.backend,
                    header=self.header, ledger=self.ledger)
        args.update(changes)
        return core.run_frame(**args)

    def test_strict_arithmetic_shorter_and_raw_tie(self):
        c = self.candidate('H64-A')
        raw = phy.raw_payload(self.scales, 7)
        self.streams[7] = raw.copy()
        self.assertEqual(core.select_payload(c, self.cat, self.scales, self.streams)['profile']['mode'], 'raw')
        self.streams[7] = raw[:-1]
        selected = core.select_payload(c, self.cat, self.scales, self.streams)
        self.assertEqual(selected['profile']['mode'], 'arithmetic')
        np.testing.assert_array_equal(selected['payload'], raw[:-1])

    def test_fallback_descends_every_m_and_public_id_is_actual(self):
        c = self.candidate('H16-A', 9)
        self.streams[8] = np.zeros(2000, dtype=np.uint8)
        result = self.run_frame(c)
        self.assertEqual([r['m'] for r in result['tx']['attempts']], [9, 8, 7])
        self.assertEqual(result['tx']['actual_m'], 7)
        self.assertEqual(result['tx']['mode'], 'raw')
        self.assertEqual(self.header.sent_ids, [result['tx']['profile_id']])
        self.assertEqual(core.public_codebook(self.cat)[str(result['tx']['profile_id'])]['m'], 7)

    def test_raw_arm_does_not_use_short_arithmetic_stream(self):
        self.streams[9] = np.zeros(2, dtype=np.uint8)
        c = self.candidate('H64-R', 9)
        p = core.select_payload(c, self.cat, self.scales, self.streams)
        self.assertEqual(p['actual_m'], 7)
        self.assertEqual(p['profile']['mode'], 'raw')
        self.assertTrue(all(a['arithmetic_bits'] is None for a in p['attempts']))

    def test_noise_is_paired_and_schedule_has_no_collisions(self):
        c1, c2 = self.candidate('H16-R', slot=0), self.candidate('H64-R', slot=1)
        r1 = self.run_frame(c1, [c2])
        r2 = self.run_frame(c2, [c1])
        self.assertEqual(r1['noise_sha256'], r2['noise_sha256'])
        self.assertNotEqual(r1['transmitted_sha256'], r2['transmitted_sha256'])
        self.assertNotEqual(r1['received_sha256'], r2['received_sha256'])
        self.assertNotEqual(core.standard_noise('source_0000', 13, 6101).tobytes(),
                            core.standard_noise('source_0000', 13, 6102).tobytes())
        counters = {core.frame_counter(s, i, n) for s in range(16) for i in range(200) for n in core.SEEDS}
        self.assertEqual(counters, set(range(9600)))

    def test_exact_frame_noise_and_no_energy_normalization(self):
        c = self.candidate()
        captured = {}
        original = phy.receive_frame
        def watch(*args):
            captured['y'] = args[2].copy()
            return original(*args)
        with patch.object(phy, 'receive_frame', watch):
            r = self.run_frame(c)
        tx = core.select_payload(c, self.cat, self.scales, self.streams)
        body, _ = phy.transmit_body(self.backend, tx['payload'], tx['profile'], r['public_frame_counter'], core.SESSION)
        sent = np.concatenate((np.ones((68, 2)), body)).astype(np.float64)
        expected = (sent+10**(-13/20)*core.standard_noise('source_0000', 13, 6101)).astype(np.float32)
        np.testing.assert_array_equal(captured['y'], expected)
        self.assertEqual(r['total_energy'], float(np.square(sent).sum()))
        self.assertFalse(r['frame_normalized'])

    def test_header_reject_drops_body_without_free_fallback(self):
        self.header.reject = True
        r = self.run_frame(self.candidate())
        self.assertEqual(r['rx']['status'], 'HEADER_REJECT')
        self.assertEqual([e['kind'] for e in self.ledger.events], ['header'])
        self.assertEqual(self.backend.decode_calls, [])
        self.assertTrue(r['gray'])

    def test_body_crc_reject_no_second_attempt_and_preserves_decoded_bits(self):
        c = self.candidate()
        p = core.select_payload(c, self.cat, self.scales, self.streams)['profile']
        bad = phy.pack_body(phy.raw_payload(self.scales, p['m']), p)
        bad[-1] ^= 1
        self.backend.decoded_override = bad
        r = self.run_frame(c)
        self.assertEqual(r['rx']['body']['status'], 'CRC_REJECT')
        self.assertEqual(r['rx']['body']['decoded_bits'], bad.tolist())
        self.assertEqual(len(self.backend.decode_calls), 1)
        self.assertTrue(all(e['phase'] == 'initial_true200' for e in self.ledger.events))

    def test_crc_accepted_invalid_padding_keeps_bits_and_turns_gray(self):
        c = self.candidate()
        p = core.select_payload(c, self.cat, self.scales, self.streams)['profile']
        info = phy.pack_body(phy.raw_payload(self.scales, p['m']), p)[:-16]
        info[-1] = 1
        self.backend.decoded_override = phy.append_crc(info)
        r = self.run_frame(c)
        self.assertTrue(r['rx']['body']['crc_accepted'])
        self.assertEqual(r['rx']['body']['invalid_reason'], 'NONZERO_KNOWN_PADDING')
        self.assertTrue(r['gray'])
        self.assertEqual(r['rx']['body']['decoded_bits'], self.backend.decoded_override.tolist())
        self.assertEqual(len(self.backend.decode_calls), 1)

    def test_unknown_header_id_is_rejected_not_repaired_with_tx_id(self):
        self.header.force_id = 4095
        r = self.run_frame(self.candidate())
        self.assertFalse(r['rx']['header']['header_ok'])
        self.assertIsNone(r['rx_profile'])
        self.assertEqual([e['kind'] for e in self.ledger.events], ['header'])
        self.assertEqual(self.backend.decode_calls, [])

    def test_wrong_accepted_known_header_controls_real_rx_layout(self):
        c = self.candidate()
        p = next(p for p in self.cat['profiles'] if p['q'] == 4 and p['m'] == 6 and p['K'] == 0
                 and p['mode'] == 'raw' and p['nominal_rate'] == '1/2')
        actual = phy.raw_payload(self.scales, 6)
        actual[0] ^= 1
        self.header.force_id = p['profile_id']
        self.backend.decoded_override = phy.pack_body(actual, p)
        r = self.run_frame(c)
        self.assertEqual(self.backend.decode_calls, [(p['k'], p['n'], p['q'])])
        self.assertEqual(r['rx']['body']['payload'], actual.tolist())
        self.assertEqual(r['received_raw_tokens'], phy.raw_tokens(actual, p).tolist())
        self.assertTrue(r['evaluation_only']['accepted_wire_mismatch'])
        self.assertFalse(r['evaluation_only']['header_correct'])
        self.assertFalse(r['gray'])

    def test_arithmetic_is_pending_canonical_not_clean_quality_or_success(self):
        self.streams[7] = np.zeros(1000, dtype=np.uint8)
        r = self.run_frame(self.candidate('H64-A'))
        self.assertEqual(r['rx_source_status'], 'ARITHMETIC_CANONICAL_RX_REQUIRED')
        self.assertIsNone(r['gray'])
        self.assertFalse(r['arithmetic_source_decode_run'])
        self.assertFalse(r['neural_metrics_run'])

    def test_decoder_failure_is_precharged_and_not_retried(self):
        self.backend.raise_decode = True
        with self.assertRaisesRegex(RuntimeError, 'scripted CPU'):
            self.run_frame(self.candidate())
        self.assertEqual(len(self.backend.decode_calls), 1)
        self.assertEqual([r['status'] for r in self.ledger.events], ['COMPLETE', 'FAILED'])

    def test_same_event_reuse_is_deterministic_not_new_decoding(self):
        c = self.candidate()
        initial_charged = len(self.ledger.events)
        first = self.run_frame(c)
        charged_after_first = len(self.ledger.events)
        body_calls_after_first = len(self.backend.decode_calls)
        second = self.run_frame(c)
        charged_after_reuse = len(self.ledger.events)
        self.assertEqual(first, second)
        self.assertNotIn('actual_packet_attempts', first)
        self.assertEqual(first['logical_packet_events'], 2)
        self.assertEqual(first['packet_event_ids'], [first['event_id']+':header', first['event_id']+':body'])
        self.assertEqual(second['packet_event_ids'], first['packet_event_ids'])
        self.assertEqual([e['event_id'] for e in self.ledger.events], first['packet_event_ids'])
        self.assertEqual(charged_after_first-initial_charged, 2)
        self.assertEqual(charged_after_reuse-charged_after_first, 0)
        self.assertEqual(len(self.backend.decode_calls)-body_calls_after_first, 0)
        self.assertEqual(body_calls_after_first, 1)

    def test_unsealed_changed_or_nonshortlisted_inputs_stop_before_encode(self):
        c = self.candidate()
        shortlist, contract = self.context([c])
        for changed in [dict(contract, status='DRAFT'), dict(contract, engineering_choices_sha256='0'*64),
                        dict(contract, core_source_sha256='0'*64)]:
            with self.assertRaises(ValueError):
                self.run_frame(c, contract=changed)
        with self.assertRaisesRegex(ValueError, 'Hash200'):
            self.run_frame(c, source_id='unregistered_source')
        with self.assertRaisesRegex(ValueError, 'exact frozen'):
            self.run_frame(dict(c, expected_psnr_db=999), shortlist=shortlist, contract=contract)
        self.assertEqual(self.header.sent_ids, [])
        self.assertEqual(self.ledger.events, [])

    def test_missing_arithmetic_or_unavailable_bucket_does_not_fallback_silently(self):
        c = self.candidate('H64-A')
        with self.assertRaisesRegex(ValueError, 'Missing frozen arithmetic'):
            core.select_payload(c, self.cat, self.scales, {})
        bad = copy.deepcopy(self.cat)
        next(b for b in bad['buckets'] if b['q'] == 6 and b['nominal_rate'] == '1/2')['admission'] = 'UNAVAILABLE'
        with self.assertRaisesRegex(ValueError, 'bucket is unavailable'):
            core.select_payload(c, bad, self.scales, self.streams)

    def test_asset_loader_hash_identity_and_npz_key_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            streams = p/'source.npz'
            np.savez(streams, **{f'm{m}_bits': v for m, v in self.streams.items()},
                     unused_image=np.array([object()], dtype=object))
            old = p/'s1.npz'
            np.savez(old, tokens=np.concatenate(self.scales), unused_pixels=np.array([object()], dtype=object))
            cp = dict(source_id='source_0000', source_index=0, registration_sha256='a'*64,
                      independent_roundtrip=True, outputs={str(streams): core.file_sha(streams)},
                      lengths=[dict(m=m, raw_bits=12*token_count(m), arithmetic_bits=len(v), zero_extension_reads=30)
                               for m, v in self.streams.items()])
            s1 = dict(source_id='source_0000', source_index=0, archive=str(old), outputs={str(old): core.file_sha(old)})
            (p/'cp.json').write_text(json.dumps(cp), encoding='utf8')
            (p/'s1.json').write_text(json.dumps(s1), encoding='utf8')
            args = dict(source_checkpoint_sha256=core.file_sha(p/'cp.json'), s1_checkpoint_sha256=core.file_sha(p/'s1.json'),
                        source_id='source_0000', source_index=0, source_registration_sha256='a'*64)
            result = core.load_source_assets(p/'cp.json', p/'s1.json', **args)
            np.testing.assert_array_equal(np.concatenate(result['scales']), np.concatenate(self.scales))
            with self.assertRaisesRegex(ValueError, 'Source identity'):
                core.load_source_assets(p/'cp.json', p/'s1.json', **dict(args, source_id='other'))
            streams.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'asset changed'):
                core.load_source_assets(p/'cp.json', p/'s1.json', **args)


if __name__ == '__main__':
    unittest.main()

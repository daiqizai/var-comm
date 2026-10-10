"""CPU fake-CDF engineering only; never real VAR, PHY or scientific evidence."""
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np

import ep_source_codec as ep
import ep_plan as plan

WORKSPACE = Path(__file__).resolve().parents[3]
BOUND_NUMPY_SOURCE = (WORKSPACE if all((WORKSPACE / 'src/var_comm' / name).is_file()
                                     for name in ('entropy.py', 'whole_entropy.py'))
                      else WORKSPACE / '.research/wcl_evidence_closure_20261009/remote_snapshot/source')


class FakeProvider:
    """A positive integer CDF depending only on explicit recovered coarse tokens."""
    def __init__(self):
        self.scale = 0
        self.history = []
        self.cdf_calls = 0
        self.closed = False

    def __enter__(self):
        return self

    def cdf(self):
        self.cdf_calls += 1
        counts = np.full(4096, (1 << 24) // 4096, dtype=np.int64)
        key = (self.scale + sum(int(a.sum()) for a in self.history)) % 4096
        counts[key] += 2048
        counts[(key + 1) % 4096] -= 2048
        cdf = np.concatenate((np.zeros(1, np.int64), counts.cumsum()))
        return np.broadcast_to(cdf, (ep.SIZES[self.scale] ** 2, 4097))

    def advance(self, values):
        if np.asarray(values).shape != (ep.SIZES[self.scale] ** 2,):
            raise AssertionError('Incomplete scale advanced')
        self.history.append(np.asarray(values).copy())
        self.scale += 1

    def __exit__(self, *_):
        self.closed = True


def codec(cls=ep.PartialSourceCodec):
    # Original SHA-bound integer classes only; native=None does not load a model.
    obj = cls(BOUND_NUMPY_SOURCE)
    obj.native = SimpleNamespace(torch=SimpleNamespace(no_grad=nullcontext))
    providers = []
    def factory():
        p = FakeProvider()
        providers.append(p)
        return p
    obj.provider_factory = factory
    return obj, providers


class SourceEngineeringTests(unittest.TestCase):
    def setUp(self):
        self.tokens = (np.arange(680, dtype=np.int64) * 61 + 17) % 4096
        self.codec, self.providers = codec()

    def test_registered_floor_quarters_and_all_wire_endpoints(self):
        self.assertEqual([ep.quarter_counts(m) for m in range(4, 10)],
                         [(6, 12, 18), (9, 18, 27), (16, 32, 48),
                          (25, 50, 75), (42, 84, 126), (64, 128, 192)])
        self.assertEqual(len(ep.all_endpoints()), 24)
        self.assertEqual(sum(K > 0 for m, K in ep.all_endpoints()), 18)
        counts = [ep.token_count(*p) for p in ep.all_endpoints()]
        self.assertEqual(counts, sorted(set(counts)))
        self.assertEqual(ep.token_count(9, 192), 616)

    def test_preregistered_plan_and_codec_metadata_agree_at_every_endpoint(self):
        for m, K in ep.all_endpoints():
            self.assertEqual(ep.quarter_counts(m), plan.ks(m))
            self.assertEqual(ep.token_count(m, K), plan.token_count(m, K))
            self.assertEqual(ep.fallback_endpoints(m, K), plan.fallback_endpoints(m, K))
        for candidate in plan.candidates():
            chain = ep.fallback_endpoints(candidate['target_m'], candidate['target_K'])
            lengths = {p: candidate['source_capacity_bits'] + 1 for p in chain}
            lengths[chain[-1]] = 722
            actual = ep.choose_encoded_length(lengths, candidate['target_m'],
                                              candidate['target_K'], candidate['source_capacity_bits'])
            planned = plan.select_stream(lengths, candidate)
            self.assertEqual((actual['actual_m'], actual['actual_K']),
                             (planned['m'], planned['K']))
            self.assertEqual(actual['arithmetic_bits'], planned['bits'])

    def test_one_tx_pass_flushes_all_endpoints_without_changing_old_whole_bits(self):
        streams = self.codec.encode_endpoints(self.tokens, ep.all_endpoints())
        self.assertEqual(len(self.providers), 1)
        self.assertEqual(self.providers[0].cdf_calls, 10)
        self.assertEqual(self.providers[0].scale, 9)
        self.assertTrue(self.providers[0].closed)
        original, _ = codec(ep.SourceCodec)
        old = original.encode(ep.WHOLE_FAMILY, self.tokens, tuple(range(4, 10)))
        for m in range(4, 10):
            self.assertTrue(np.array_equal(streams[m, 0]['bits'], old[m]['bits']))
            self.assertEqual(streams[m, 0]['flush_bits'], old[m]['flush_bits'])

    def test_all_18_partial_streams_independent_rx_roundtrip_and_no_partial_advance(self):
        wanted = tuple(p for p in ep.all_endpoints() if p[1])
        streams = self.codec.encode_endpoints(self.tokens, wanted)
        for m, K in wanted:
            before = len(self.providers)
            received = self.codec.decode(ep.PARTIAL_FAMILY, streams[m, K]['bits'], m, K)
            self.assertEqual(len(self.providers), before + 1)
            p = self.providers[-1]
            self.assertEqual(p.scale, m)
            self.assertEqual(len(p.history), m)
            self.assertEqual(p.cdf_calls, m + 1)
            self.assertTrue(p.closed)
            self.assertTrue(np.array_equal(received['received_tokens'], self.tokens[:ep.token_count(m, K)]))
            self.assertEqual(received['zero_extension_reads'], 30)
            self.assertFalse(received['receiver_truth_used'])
            self.assertFalse(received['transmitter_probability_table_used'])
            self.assertFalse(received['same_scale_partial_conditioning'])

    def test_each_partial_endpoint_flush_matches_an_independent_tx_pass(self):
        wanted = ((4, 6), (7, 25), (8, 84), (9, 192))
        together = self.codec.encode_endpoints(self.tokens, wanted)
        for p in wanted:
            alone = self.codec.encode_endpoints(self.tokens, (p,))[p]
            self.assertTrue(np.array_equal(together[p]['bits'], alone['bits']))

    def test_legacy_whole_decode_keeps_original_wire_result(self):
        stream = self.codec.encode_endpoints(self.tokens, ((9, 0),))[9, 0]
        value = self.codec.decode(ep.WHOLE_FAMILY, stream['bits'], 9, 0)
        self.assertEqual(value['K'], 0)
        self.assertEqual(value['source_family'], ep.WHOLE_FAMILY)
        self.assertTrue(np.array_equal(value['received_tokens'], self.tokens[:ep.OFFSETS[9]]))

    def test_legacy_static_misreceived_profile_remains_available(self):
        cdf = np.arange(4097, dtype=np.int64) * ((1 << 24) // 4096)
        self.codec.static_cdf = np.broadcast_to(cdf, (10, 4097))
        stream = self.codec.encode('EC_STATIC_WHOLE', self.tokens, (4,))[4]
        value = self.codec.decode('EC_STATIC_WHOLE', stream['bits'], 4)
        self.assertTrue(np.array_equal(value['received_tokens'], self.tokens[:ep.OFFSETS[4]]))

    def test_other_canonical_payload_is_decoded_without_truth_rejection(self):
        other = (self.tokens + 73) % 4096
        encoded = self.codec.encode_endpoints(other, ((8, 42),))[8, 42]
        decoded = self.codec.decode(ep.PARTIAL_FAMILY, encoded['bits'], 8, 42)
        self.assertTrue(np.array_equal(decoded['received_tokens'], other[:ep.token_count(8, 42)]))
        self.assertFalse(np.array_equal(decoded['received_tokens'], self.tokens[:ep.token_count(8, 42)]))

    def test_noncanonical_extra_bit_rejected(self):
        stream = self.codec.encode_endpoints(self.tokens, ((7, 25),))[7, 25]['bits']
        with self.assertRaises(ep.InvalidSourceStream):
            self.codec.decode(ep.PARTIAL_FAMILY, np.append(stream, 0), 7, 25)

    def test_invalid_payload_vector_or_terminal_budget_rejected(self):
        for value in [[], [0], [0, 2], np.zeros((2, 2)), [0, 0]]:
            with self.subTest(value=str(value)[:20]), self.assertRaises(ep.InvalidSourceStream):
                self.codec.decode(ep.PARTIAL_FAMILY, value, 9, 192)

    def test_illegal_paid_metadata_not_silently_normalized(self):
        for m, K in [(3, 0), (10, 0), (8, 85), (8, 169), (True, 0), (7, True)]:
            with self.subTest(m=m, K=K), self.assertRaises(ValueError):
                ep.endpoint(m, K)
        with self.assertRaises(ValueError):
            self.codec.decode(ep.PARTIAL_FAMILY, [0, 0], 7, 0)
        with self.assertRaises(ValueError):
            self.codec.decode(ep.WHOLE_FAMILY, [0, 0], 7, 25)

    def test_bad_source_or_duplicate_endpoint_is_an_engineering_error(self):
        with self.assertRaises(ValueError):
            self.codec.encode_endpoints(self.tokens[:-1], ((7, 25),))
        with self.assertRaises(ValueError):
            self.codec.encode_endpoints(self.tokens, ((7, 25), (7, 25)))
        with self.assertRaises(ValueError):
            self.codec.encode_endpoints(self.tokens, ())

    def test_provider_failure_propagates_not_channel_gray_or_parse_error(self):
        class Broken(FakeProvider):
            def cdf(self):
                raise RuntimeError('Synthetic model failure')
        self.codec.provider_factory = Broken
        with self.assertRaisesRegex(RuntimeError, 'Synthetic model failure'):
            self.codec.decode(ep.PARTIAL_FAMILY, [0, 0] * 100, 7, 25)

    def test_partial_fallback_contains_earlier_partials_and_whole_in_exact_order(self):
        values = ep.fallback_endpoints(7, 25)
        self.assertEqual(values[:5], ((7, 25), (7, 0), (6, 48), (6, 32), (6, 16)))
        self.assertEqual(values[-1], (4, 0))
        self.assertEqual(ep.fallback_endpoints(9, 0), ((9, 0), (8, 0), (7, 0), (6, 0), (5, 0), (4, 0)))

    def test_length_only_selection_stops_at_first_fit_or_precise_missing_asset(self):
        chain = ep.fallback_endpoints(7, 25)
        lengths = {p: 1000 for p in chain}
        lengths[(7, 0)] = 800
        value = ep.choose_encoded_length(lengths, 7, 25, 927)
        self.assertEqual((value['actual_m'], value['actual_K']), (7, 0))
        self.assertEqual(len(value['attempts']), 2)
        del lengths[(7, 0)]
        missing = ep.choose_encoded_length(lengths, 7, 25, 927)
        self.assertEqual(missing['status'], 'BLOCKED_MISSING_PREFIX')
        self.assertEqual(missing['missing_endpoint'], [7, 0])
        self.assertNotEqual(missing['status'], 'TX_UNENCODABLE')

    def test_whole_policy_never_inherits_partial_fallback_implicitly(self):
        lengths = {p: 1000 for p in ep.all_endpoints()}
        lengths[(8, 126)] = 100
        lengths[(8, 0)] = 900
        value = ep.choose_encoded_length(lengths, 9, 0, 927)
        self.assertEqual((value['actual_m'], value['actual_K']), (8, 0))

    def test_source_model_identity_is_fixed_complete_and_no_class_argument(self):
        self.assertEqual(len(ep.partial_source_model_id()), 64)
        self.assertEqual(ep.partial_source_model_id(), ep.partial_source_model_id())
        self.assertNotIn('label', ep.PartialSourceCodec.decode.__code__.co_varnames)
        self.assertNotIn('tokens', ep.PartialSourceCodec.decode.__code__.co_varnames)


if __name__ == '__main__':
    print('CPU_FAKE_CDF_ENGINEERING_ONLY; real_VAR_qualification=false; PHY_calls=0; training_updates=0')
    unittest.main(verbosity=2)

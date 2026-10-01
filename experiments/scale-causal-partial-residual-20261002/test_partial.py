"""CPU protocol/receiver tests; production native MAP by default.

Run from the repository: python experiments/.../test_partial.py.
--portable uses an independent NumPy Viterbi reference on hosts without the
native compiler; its receipt must not be described as a native-decoder test.
The real frozen VAR parity/dual-end checks live in qualification.py.
"""
import argparse
import inspect
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'src'))
import partial_phy as phy
try:
    import torch
    import partial_receiver as rx
except ModuleNotFoundError as error:
    if error.name != 'torch':
        raise
    torch = rx = None


def reference_map(evidence):
    """Independent terminated memory-6 Viterbi, solely for portable CPU QA."""
    evidence = np.asarray(evidence).reshape(-1, 2)
    states = np.arange(64)
    incoming_bit = states & 1
    predecessors = np.stack((states >> 1, (states >> 1) + 32), axis=1)
    register = (predecessors << 1) | incoming_bit[:, None]
    signs = np.stack([1 - 2*np.vectorize(lambda x: int(x).bit_count() & 1)(register & generator)
                      for generator in (0o171, 0o133)], axis=-1)
    scores = np.full(64, -np.inf); scores[0] = 0
    paths = np.empty((len(evidence), 64), dtype=np.uint8)
    for k, obs in enumerate(evidence):
        alternatives = scores[predecessors] + (signs * obs).sum(-1)
        chosen = alternatives.argmax(1)
        paths[k] = predecessors[states, chosen]
        scores = alternatives[states, chosen]
    state = 0; decoded = np.empty(len(evidence), dtype=np.uint8)
    for k in range(len(evidence)-1, -1, -1):
        decoded[k] = state & 1; state = int(paths[k, state])
    if state != 0:
        raise AssertionError('Reference traceback did not reach the fixed initial state')
    return decoded, float(scores[0])


def source(seed=917):
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 4096, p*p, dtype=np.int64) for p in phy.SIZES]


def semantic_event(action, scales, *, prefix_ok=True, partial_ok=True, fields_ok=True,
                   header_ok=True, values=None, positions=None):
    usable = bool(action.q and prefix_ok and partial_ok and fields_ok)
    return dict(header_ok=header_ok, action=action if header_ok else None,
        prefix=[x.copy() for x in scales[:action.m]] if header_ok else [],
        prefix_crc_ok=prefix_ok, partial_crc_ok=partial_ok if action.q else None,
        partial_fields_legal=fields_ok, partial_usable=usable,
        partial_values=np.full(action.q, 4000, np.int64) if values is None else values,
        partial_positions=positions,
        partial_discard_reason='prefix_crc_failure' if not prefix_ok else
            'partial_crc_failure' if action.q and not partial_ok else '',
        source_error='' if prefix_ok else 'unverified_hard_prefix; partial_discarded')


class ProtocolTests(unittest.TestCase):
    def test_finite_grid_rates_strong_whole_and_mask_cost(self):
        counts = []
        for N in (512, 1024):
            for family in phy.PHY_FAMILIES:
                grid = phy.action_grid(N, family); counts.append(len(grid))
                self.assertEqual(len(grid), len({a.name for a in grid}))
                self.assertTrue(any(a.q == 0 for a in grid))
                for a in grid:
                    r = phy.action_record(a)
                    self.assertEqual(r['N_header'] + r['N_prefix'] + r['N_partial'], N)
                    self.assertEqual(r['header_source_bits'], 12)
                    self.assertEqual(r['header_mother_bits'], 68)
                    self.assertEqual(r['header_coded_bits'], 136)
                    self.assertEqual(r['body_crc_bits'], 16*(1+bool(a.q)))
                    self.assertEqual(r['body_tail_bits'], 6*(1+bool(a.q)))
                    self.assertLessEqual(r['prefix_effective_rate'], .9)
                    if a.q:
                        self.assertLessEqual(r['partial_effective_rate'], .9)
                        self.assertLess(a.q, phy.next_length(a.m))
                        self.assertEqual(r['mask_bits'], phy.next_length(a.m) if a.order == 'oracle' else 0)
                    else:
                        self.assertEqual(a.order, 'whole'); self.assertEqual(r['N_partial'], 0)
                for m in range(4, 8):
                    for order in phy.ORDERS:
                        maximum = phy.max_legal_q(N, family, m, order)
                        registered = {a.q for a in grid if a.m == m and a.order == order}
                        if maximum:
                            self.assertIn(maximum, registered)
                            phy.Action(N, family, m, maximum, order)
                        if maximum + 1 < phy.next_length(m):
                            with self.assertRaises(ValueError):
                                phy.Action(N, family, m, maximum + 1, order)
        strong = phy.Action(1024, '16QAM', 8, 0, 'whole')
        self.assertEqual(phy.action_record(strong)['prefix_source_bits'], 3060)
        self.assertAlmostEqual(phy.action_record(strong)['prefix_effective_rate'], 3082/3824)
        self.assertGreater(sum(counts), 100)

    def test_noiseless_all_legal_grid_actual_fec(self):
        scales = source()
        energies = []
        for N in (512, 1024):
            for family in phy.PHY_FAMILIES:
                for a in phy.action_grid(N, family):
                    positions = np.arange(a.q)[::-1] if a.q else None
                    wave, ledger = phy.transmit(scales, a, positions)
                    event = phy.receive(wave, 60., N, family)
                    self.assertTrue(event['header_ok'], a.name)
                    self.assertEqual(event['action'], a)
                    self.assertTrue(event['prefix_crc_ok'], a.name)
                    self.assertTrue(all(np.array_equal(x, y) for x, y in zip(event['prefix'], scales[:a.m])))
                    self.assertEqual(wave.shape, (N, 2))
                    self.assertEqual(ledger['E'], float(np.square(wave).sum()))
                    if family == 'QPSK': self.assertEqual(ledger['E'], 2*N)
                    else: energies.append(ledger['E'])
                    if a.q:
                        self.assertTrue(event['partial_crc_ok']); self.assertTrue(event['partial_usable'])
                        wire_positions = np.sort(positions) if a.order == 'oracle' else positions
                        np.testing.assert_array_equal(event['partial_values'], scales[a.m][wire_positions])
                        if a.order == 'oracle': np.testing.assert_array_equal(event['partial_positions'], wire_positions)
                        else: self.assertIsNone(event['partial_positions'])
                    else:
                        self.assertIsNone(event['partial_crc_ok']); self.assertFalse(event['partial_usable'])
        self.assertGreater(np.ptp(energies), 1.)

    def test_rx_only_received_action_and_crc_failure_decisions(self):
        self.assertEqual(list(inspect.signature(phy.receive).parameters), ['y', 'snr', 'N', 'phy'])
        scales = source()
        for order in phy.ORDERS:
            a = phy.Action(1024, '16QAM', 4, 6, order)
            wave, _ = phy.transmit(scales, a, np.arange(a.q))
            head = np.r_[phy.indices_to_bits([0], 3), phy.indices_to_bits([6], 7),
                         phy.indices_to_bits([phy.ORDERS.index(order)], 2)]
            pref = phy.indices_to_bits(np.concatenate(scales[:4]))
            mask = np.r_[np.ones(6, np.uint8), np.zeros(19, np.uint8)] if order == 'oracle' else []
            partial = np.r_[mask, phy.indices_to_bits(scales[4][:6])].astype(np.uint8)
            with patch.object(phy, 'decode_packet', side_effect=[(head, True, 1.), (pref, False, 2.), (partial, True, 3.)]):
                e = phy.receive(wave, 7., 1024, '16QAM')
            self.assertFalse(e['partial_usable']); self.assertTrue(e['partial_crc_ok'])
            self.assertEqual(e['partial_discard_reason'], 'prefix_crc_failure')
            self.assertTrue(e['raw_candidate_used_after_crc_failure'])
            self.assertTrue(all(np.array_equal(x,y) for x,y in zip(e['prefix'],scales[:4])))
            with patch.object(phy, 'decode_packet', side_effect=[(head, True, 1.), (pref, True, 2.), (partial, False, 3.)]):
                e = phy.receive(wave, 7., 1024, '16QAM')
            self.assertFalse(e['partial_usable']); self.assertEqual(e['partial_discard_reason'], 'partial_crc_failure')
        # A CRC-valid but reserved header and an invalid paid bitmap are both rejected.
        bad = np.r_[phy.indices_to_bits([7],3), np.zeros(9,np.uint8)]
        with patch.object(phy, 'decode_packet', return_value=(bad, True, 1.)) as mocked:
            e = phy.receive(np.ones((1024,2)), 7., 1024, 'QPSK')
        self.assertFalse(e['header_ok']); self.assertEqual(mocked.call_count,1)
        a=phy.Action(1024,'16QAM',4,6,'oracle');wave,_=phy.transmit(scales,a,np.arange(6))
        head=np.r_[np.zeros(3,np.uint8),phy.indices_to_bits([6],7),phy.indices_to_bits([3],2)]
        partial=np.r_[np.zeros(25,np.uint8),phy.indices_to_bits(scales[4][:6])]
        with patch.object(phy,'decode_packet',side_effect=[(head,True,1.),(pref,True,2.),(partial,True,3.)]):
            e=phy.receive(wave,7.,1024,'16QAM')
        self.assertFalse(e['partial_usable']);self.assertEqual(e['partial_discard_reason'],'partial_fields_illegal')

    def test_header_erasure_stops_body_and_channel_is_paired(self):
        a=phy.Action(512,'QPSK',4,0,'whole');wave,_=phy.transmit(source(),a)
        with patch.object(phy,'decode_packet',return_value=(np.zeros(12,np.uint8),False,0.)) as mocked:
            event=phy.receive(wave,1.,512,'QPSK')
        self.assertFalse(event['header_ok']);self.assertEqual(event['prefix'],[])
        self.assertEqual(mocked.call_count,1)
        noise=np.random.default_rng(123).normal(size=wave.shape)
        np.testing.assert_array_equal(phy.apply_channel(wave,7,noise),wave+noise*10**(-7/20))
        with self.assertRaises(ValueError):phy.Action(1024,'16QAM',7,100,'entropy')


if torch is not None:
    class FakePrior:
        def __init__(self, vae, var, device=None):
            self.history=[];self.fhat=torch.zeros((1,32,16,16));self.closed=False
        def logits(self,k):
            length=phy.SIZES[k]**2
            base=(sum(int(x.sum()) for x in self.history)+np.arange(length))%4096
            z=torch.full((1,length,4096),-8.,dtype=torch.float32)
            z[0,torch.arange(length),torch.as_tensor(base)]=5.
            z[0,torch.arange(length),torch.as_tensor((base+1)%4096)]=torch.linspace(-2,4,length)
            return z
        def advance(self,tokens,k):
            self.history.append(np.asarray(tokens).copy());self.fhat.add_(float(np.asarray(tokens).sum())/10000)
        def close(self):self.closed=True


@unittest.skipIf(torch is None, 'Torch unavailable; receiver checks require CPU Torch')
class ReceiverTests(unittest.TestCase):
    def test_order_determinism_ties_truth_boundary_and_oracle(self):
        prefix=source()[:4];length=25
        tied=np.zeros((length,4096),np.float32)
        np.testing.assert_array_equal(rx.order_positions(prefix,'entropy',6,tied),np.arange(6))
        one=rx.order_positions(prefix,'random',24)
        np.testing.assert_array_equal(one,rx.order_positions([x.copy() for x in prefix],'random',24))
        self.assertEqual(len(set(one.tolist())),24)
        changed=[x.copy() for x in prefix];changed[0][0]^=1
        self.assertFalse(np.array_equal(one,rx.order_positions(changed,'random',24)))
        with self.assertRaises(ValueError):rx.order_positions(prefix,'entropy',6,tied,np.ones(length,np.int64))
        truth=np.zeros(length,np.int64);truth[[2,7,20]]=5
        np.testing.assert_array_equal(rx.order_positions(prefix,'oracle',3,tied,truth),[2,7,20])
        with self.assertRaises(ValueError):rx.entropy_values(tied.astype(np.float64))
        z=tied.copy();z[0,0]=30
        self.assertNotIn(0,rx.order_positions(prefix,'entropy',6,z))

    def test_clamp_fixed_missing_scale_unchanged_finer_changes_and_failures(self):
        scales=source();a=phy.Action(1024,'16QAM',4,6,'entropy')
        whole=semantic_event(phy.Action(1024,'16QAM',4,0,'whole'),scales)
        with patch.object(rx,'_Prior',FakePrior):
            baseline=rx.complete_partial(None,None,whole,return_logits=True)
            event=semantic_event(a,scales)
            result=rx.complete_partial(None,None,event,return_logits=True)
            pos=np.asarray(result['diagnostics']['known_positions'])
            missing=np.setdiff1d(np.arange(25),pos)
            np.testing.assert_array_equal(result['tokens'][4][pos],event['partial_values'])
            np.testing.assert_array_equal(result['tokens'][4][missing],baseline['tokens'][4][missing])
            np.testing.assert_array_equal(result['next_prior_tokens'],baseline['tokens'][4])
            self.assertTrue(all(np.array_equal(result['tokens'][k],scales[k]) for k in range(4)))
            self.assertFalse(np.array_equal(result['tokens'][5],baseline['tokens'][5]))
            self.assertTrue(result['diagnostics']['known_tokens_fixed'])
            for order in phy.ORDERS:
                aa=phy.Action(1024,'16QAM',4,6,order)
                for failure in [dict(prefix_ok=False),dict(partial_ok=False)]:
                    ev=semantic_event(aa,scales,**failure)
                    failed=rx.complete_partial(None,None,ev)
                    self.assertFalse(failed['diagnostics']['partial_used'])
                    self.assertTrue(torch.equal(failed['fhat'],baseline['fhat']))
                    self.assertTrue(all(np.array_equal(x,y) for x,y in zip(failed['tokens'],baseline['tokens'])))
            erased=rx.complete_partial(None,None,semantic_event(a,scales,header_ok=False))
            self.assertIsNone(erased['fhat']);self.assertEqual(erased['tokens'],[])

    def test_cache_is_actual_received_content_and_interfaces_have_no_truth(self):
        a=phy.Action(1024,'16QAM',4,6,'raster');scales=source()
        e=semantic_event(a,scales);changed=semantic_event(a,scales,values=np.arange(6))
        self.assertNotEqual(rx.received_cache_key(e),rx.received_cache_key(changed))
        failed=semantic_event(a,scales,prefix_ok=False)
        whole=semantic_event(phy.Action(1024,'16QAM',4,0,'whole'),scales)
        self.assertEqual(rx.received_cache_key(failed),rx.received_cache_key(whole))
        changed=semantic_event(a,source(918),prefix_ok=False)
        self.assertNotEqual(rx.received_cache_key(failed),rx.received_cache_key(changed))
        for f in [rx.complete_partial,rx.prefix_logits,rx.received_cache_key]:
            self.assertFalse({'truth','truth_next','clean','source_id'} & set(inspect.signature(f).parameters))


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--portable',action='store_true')
    args,remaining=parser.parse_known_args()
    if args.portable:phy.decode_map=reference_map
    print('decoder_test_backend=' + ('independent_numpy_reference' if args.portable else 'production_native_MAP'))
    unittest.main(argv=[sys.argv[0]]+remaining)

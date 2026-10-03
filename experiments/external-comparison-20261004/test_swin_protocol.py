"""CPU contract tests; native trellis roundtrip additionally runs on Linux."""
import copy
import math
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest import mock
import numpy as np
import swin_protocol as p
from swin_state import SourceOrder, identity, register, write, restore_training_log
from swin_data import validate_population


class FakeDecodedPHY:
    """Only for rejection-path tests; never used for physical performance."""
    def __init__(self, N, power=1., rank=0, crc=True):
        spec = p.layout(N)
        bits = int.from_bytes(struct.pack('>f', power), 'big')
        self.decoded = np.concatenate((p._bits(bits,32),p._bits(rank,spec['mask_rank_bits']),np.zeros(22,dtype=np.uint8)))
        self.crc = crc
    def rate_match_indices(self,*_): return None
    def channel_evidence(self,*_): return None
    def decode_map(self,*_): return self.decoded,0
    def crc_accepts(self,*_): return self.crc


def history(step, values=(1.,1.,1.)):
    return [dict(step=s,mean_mse_by_N={str(N):value for N in p.RATES},
        mean_mse_by_cell={f'N{N}_snr{snr}':value for N in p.RATES for snr in p.CAL_SNRS})
        for s,value in zip((step-20000,step-10000,step),values)]


class ProtocolTests(unittest.TestCase):
    def test_exact_paid_layout(self):
        for N,C,data,head,bits in ((1024,6,768,256,41),(2048,13,1664,384,76)):
            spec = p.layout(N)
            self.assertEqual((spec['channels'],spec['data_N'],spec['header_N'],spec['mask_rank_bits']),(C,data,head,bits))
            self.assertEqual(spec['data_N']+spec['header_N'],N)
            self.assertEqual(spec['E'],2*N)
            self.assertEqual(spec['transmitted_header_bits'],2*head)
            self.assertEqual(spec['information_bits'],bits+32+16+6)
            self.assertEqual(spec['no_information_padding_N'],0)

    def test_unsupported_budget_rejected(self):
        for N in (512,3060,1024.,True):
            with self.assertRaises(ValueError): p.layout(N)

    def test_subset_roundtrip_edges_and_random(self):
        rng = np.random.default_rng(52)
        for C in (6,13):
            candidates = [tuple(range(C)),tuple(range(320-C,320))]
            candidates += [tuple(sorted(rng.choice(320,C,replace=False))) for _ in range(100)]
            for indices in candidates:
                self.assertEqual(p.unrank_subset(p.rank_subset(indices,C),C),indices)

    def test_illegal_subset_rejected(self):
        for bad in ((1,1,2,3,4,5),(0,1,2,3,4,320),(0,1,2,3,4), (0,2,1,3,4,5)):
            with self.assertRaises(ValueError): p.rank_subset(bad,6)
        with self.assertRaises(ValueError): p.unrank_subset(math.comb(320,6),6)

    def test_crc_failure_erases_context(self):
        with mock.patch.object(p,'_PHY',FakeDecodedPHY(1024,power=.125,crc=False)):
            rx = p.decode_header(np.zeros((256,2)),7,1024)
        self.assertFalse(rx.accepted); self.assertFalse(rx.crc_accepted)
        self.assertIsNone(rx.power); self.assertIsNone(rx.indices)

    def test_legal_fields_cannot_override_crc(self):
        with mock.patch.object(p,'_PHY',FakeDecodedPHY(2048,power=1.25,crc=True)):
            rx=p.decode_header(np.zeros((384,2)),7,2048)
        self.assertTrue(rx.accepted); self.assertEqual(rx.power,1.25)
        self.assertEqual(rx.indices,tuple(range(13)))

    def test_illegal_crc_accepted_fields_erased(self):
        for power,rank in ((float('nan'),0),(0.,0),(-1.,0),(1.,math.comb(320,6))):
            with mock.patch.object(p,'_PHY',FakeDecodedPHY(1024,power,rank,True)):
                rx=p.decode_header(np.zeros((256,2)),1,1024)
            self.assertTrue(rx.crc_accepted); self.assertFalse(rx.accepted)
            self.assertIsNone(rx.indices); self.assertIsNone(rx.power)

    def test_noise_is_identity_bound(self):
        a=p.standard_noise('source17',4101,1024,7)
        self.assertTrue(np.array_equal(a,p.standard_noise('source17',4101,1024,7)))
        self.assertFalse(np.array_equal(a,p.standard_noise('source18',4101,1024,7)))
        self.assertFalse(np.array_equal(a,p.standard_noise('source17',4102,1024,7)))

    def test_awgn_real_variance_contract(self):
        with mock.patch.object(p,'decode_header',return_value='rx'):
            body,rx=p.observe_frame(np.zeros((1024,2)),np.ones((1024,2)),10,1024)
        self.assertEqual(rx,'rx'); self.assertEqual(body.shape,(768,2))
        np.testing.assert_allclose(body,1/math.sqrt(10),rtol=0,atol=1e-16)

    def test_20k_diagnostic_only(self):
        decision=p.plateau_decision(history(20000),20000)
        self.assertTrue(decision['extend']); self.assertFalse(decision['lower_learning_rate'])

    def test_two_reductions_then_full_final_lr_windows(self):
        first=p.plateau_decision(history(40000),40000)
        self.assertEqual(first['next_learning_rate'],3e-5); self.assertTrue(first['extend'])
        second=p.plateau_decision(history(60000),60000,learning_rate=3e-5,last_lr_step=40000)
        self.assertEqual(second['next_learning_rate'],1e-5); self.assertTrue(second['extend'])
        final=p.plateau_decision(history(80000),80000,learning_rate=1e-5,last_lr_step=60000)
        self.assertFalse(final['extend']); self.assertTrue(final['final_lr_plateau_met'])
        self.assertFalse(final['scientific_convergence_proven'])

    def test_single_cell_improving_blocks_plateau(self):
        rows=history(80000)
        rows[-1]['mean_mse_by_cell']['N1024_snr1']=.99
        decision=p.plateau_decision(rows,80000,learning_rate=1e-5,last_lr_step=60000)
        self.assertTrue(decision['extend']); self.assertFalse(decision['plateau_rule_met'])

    def test_rate_improving_in_either_window_blocks_plateau(self):
        for values in ((1.,.99,.99),(1.,1.,.99)):
            self.assertTrue(p.plateau_decision(history(80000,values),80000,learning_rate=1e-5,last_lr_step=60000)['extend'])

    def test_cap_marks_unfinished_schedule_or_improvement(self):
        for lr,values in ((1e-4,(1.,1.,1.)),(1e-5,(1.,.9,.8))):
            decision=p.plateau_decision(history(240000,values),240000,learning_rate=lr,last_lr_step=200000)
            self.assertFalse(decision['extend']); self.assertTrue(decision['budget_truncated'])

    def test_missing_calibration_cannot_stop(self):
        with self.assertRaises(RuntimeError): p.plateau_decision(history(40000)[1:],40000)


class StateTests(unittest.TestCase):
    def test_exact_order_resume_across_epochs(self):
        a=SourceOrder(29,7); a.next(17); saved=copy.deepcopy(a.state_dict())
        expected=[a.next(16) for _ in range(8)]
        b=SourceOrder(29,100); b.load_state_dict(saved)
        self.assertEqual(expected,[b.next(16) for _ in range(8)])

    def test_every_source_before_new_epoch(self):
        a=SourceOrder(29,7)
        self.assertEqual(sorted(a.next(29)),list(range(29)))

    def test_invalid_order_rejected(self):
        a=SourceOrder(29,7); saved=a.state_dict(); saved['order'][0]=saved['order'][1]
        with self.assertRaises(RuntimeError): a.load_state_dict(saved)

    def test_registration_immutable(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'registration.json'
            register(path,{'foo':1}); register(path,{'foo':1})
            with self.assertRaises(RuntimeError): register(path,{'foo':2})

    def test_log_resume_removes_only_uncommitted_rows_and_keeps_receipt(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'training_log.jsonl'
            path.write_text('{"step": 500}\n{"step": 525}\n')
            restore_training_log(path,500); restore_training_log(path,500)
            self.assertEqual(path.read_text(),'{"step": 500}\n')
            self.assertEqual(len(list(Path(folder).glob('rollback_*.json'))),1)

    def test_ids_and_mapping_strict(self):
        ids=[f't{i}' for i in range(20000)]; rows=[{'image_id':s} for s in ids]
        ref=dict(train_ids=ids,calibration_ids=[f'c{i}' for i in range(1000)],source_image_bindings={'train':rows})
        validate_population('train',ids,rows,ref)
        with self.assertRaises(RuntimeError): validate_population('train',ids[::-1],rows,ref)
        with self.assertRaises(RuntimeError): validate_population('train',ids,rows[::-1],ref)
        with self.assertRaises(RuntimeError): validate_population('development',ids,rows,ref)


class NativePHYTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root=os.environ.get('SWIN_TEST_ROOT')
        if root is None:
            root=Path(__file__).resolve().parents[2]/'historical_eval_20261003/source'
        if not (Path(root)/'src/var_comm/scale_channel.py').is_file():
            raise unittest.SkipTest('Set SWIN_TEST_ROOT to the original registered repository')
        p.configure_phy(root)

    def test_all_header_uses_are_real_rate_matched_symbols(self):
        for N,C in p.RATES.items():
            head=p.encode_header(.7,tuple(range(C)),N)
            self.assertEqual(head.shape,(p.layout(N)['header_N'],2))
            self.assertEqual(float(np.square(head).sum()),2*p.layout(N)['header_N'])
            self.assertTrue(np.isin(head,[-1.,1.]).all())

    def test_frame_energy_and_shape(self):
        for N,C in p.RATES.items():
            spec=p.layout(N)
            signal=p.transmit_frame(np.ones((spec['data_N'],2)),1.,tuple(range(C)),N)
            self.assertEqual(signal.shape,(N,2)); self.assertEqual(float(np.square(signal).sum()),2*N)
            with self.assertRaises(RuntimeError): p.transmit_frame(np.zeros((spec['data_N'],2)),1.,tuple(range(C)),N)

    @unittest.skipUnless(sys.platform.startswith('linux'),'native trellis compiled on registered Linux host')
    def test_actual_noiseless_trellis_roundtrip(self):
        for N,C in p.RATES.items():
            indices=tuple(range(320-C,320)); value=float(np.float32(.017))
            received=p.decode_header(p.encode_header(value,indices,N),1,N)
            self.assertTrue(received.accepted); self.assertEqual(received.indices,indices)
            self.assertEqual(received.power,value)


if __name__=='__main__': unittest.main()

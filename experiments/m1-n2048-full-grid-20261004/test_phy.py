"""Independent CPU accounting tests for the full original N2048 action grid."""
import hashlib
import os
from pathlib import Path
import sys
import unittest
from unittest import mock
import numpy as np

HERE=Path(__file__).resolve().parent
LOCAL=HERE.parents[1]
ROOT=Path(os.environ['M1_PHY_TEST_ROOT']) if 'M1_PHY_TEST_ROOT' in os.environ else LOCAL/'historical_eval_20261003/source'
sys.path.insert(0,str(ROOT/'src'))
import m1_phy as phy

class PhyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.actions=[a for f in phy.PHY_FAMILIES for a in phy.action_grid(2048,f)]
        rng=np.random.default_rng(143)
        cls.scales=[rng.integers(0,4096,size=s*s,dtype=np.int64) for s in phy.SIZES]
    def test_exact_derivation_keeps_original_lower_budgets_untouched(self):
        original=(ROOT/'experiments/scale-causal-partial-residual-20261002/partial_phy.py') if 'M1_PHY_TEST_ROOT' in os.environ else LOCAL/'methods_20261002/source/partial_phy.py'
        raw=original.read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(),'d5b26b35b1ce5ba8c147998e2ec6aa88615a2c84a0ab13d59026e27094b7d756')
        for before,after,count in ((b'raw-partial-token-20261002-v1',b'm1-full-grid-raw-partial-N2048-20261004-v1',1),
            (b'N not in (512, 1024)',b'N not in (2048,)',2),(b'budgets=[512, 1024]',b'budgets=[2048]',1)):
            self.assertEqual(raw.count(before),count);raw=raw.replace(before,after)
        self.assertEqual((HERE/'m1_phy.py').read_bytes(),raw)
    def test_full138_grid_exact_k_and_rate_caps(self):
        self.assertEqual(len(self.actions),138);self.assertEqual(len({a.name for a in self.actions}),138)
        grid={4:{6,12,18,24},5:{9,18,27,35},6:{16,32,48,63},7:{25,50,75,99}}
        for family in phy.PHY_FAMILIES:
            group=[a for a in self.actions if a.phy==family]
            self.assertEqual(len(group),69);self.assertEqual({a.m for a in group if a.q==0},{4,5,6,7,8})
            for order in phy.ORDERS:
                for m,counts in grid.items():self.assertEqual({a.q for a in group if a.order==order and a.m==m},counts)
        for a in self.actions:
            r=phy.action_record(a)
            self.assertEqual(r['N_prefix']+r['N_partial']+68,2048)
            self.assertEqual(r['body_crc_bits'],16*(1+bool(a.q)))
            self.assertEqual(r['body_tail_bits'],6*(1+bool(a.q)))
            self.assertLessEqual(10*r['prefix_information_bits'],9*r['prefix_coded_bits'])
            self.assertLessEqual(10*r['partial_information_bits'],9*r['partial_coded_bits'])
            self.assertEqual(r['total_coded_bits'],4096 if a.phy=='QPSK' else 8056)
    def test_every_transmitted_waveform_has_paid_allocation_and_real_energy(self):
        sixteen=[]
        for a in self.actions:
            positions=np.arange(a.q,dtype=np.int64) if a.q else None
            wave,r=phy.transmit(self.scales,a,positions)
            self.assertEqual(wave.shape,(2048,2));self.assertEqual(r['E'],float(np.square(wave).sum()))
            self.assertEqual(r['header_coded_bits'],136)
            if a.phy=='QPSK':self.assertEqual(r['E'],4096)
            else:sixteen.append(r['E'])
        self.assertGreater(len(set(sixteen)),1)
        self.assertTrue(any(abs(e-4096)>1e-5 for e in sixteen))
    def test_oracle_pays_full_mask_inside_partial_crc_and_sorts_positions(self):
        a=phy.Action(2048,'QPSK',7,25,'oracle');positions=np.arange(99,74,-1,dtype=np.int64)
        with mock.patch.object(phy,'packet',wraps=phy.packet) as packet:
            _,r=phy.transmit(self.scales,a,positions)
        payload=packet.call_args_list[1].args[0]
        self.assertEqual(r['mask_bits'],100);self.assertEqual(r['partial_information_bits'],100+12*25+22)
        np.testing.assert_array_equal(np.flatnonzero(payload[:100]),np.sort(positions))
        np.testing.assert_array_equal(phy.bits_to_indices(payload[100:]),self.scales[7][np.sort(positions)])
        self.assertEqual(len(payload),400)
    def test_scope_rejects_old_budgets_and_new_unregistered_scales(self):
        for N,m,q in ((512,4,0),(1024,4,0),(2048,8,1),(2048,9,0)):
            with self.assertRaises(ValueError):phy.Action(N,'QPSK',m,q,'entropy' if q else 'whole')
        for order in phy.ORDERS:self.assertEqual(phy.Action(2048,'QPSK',8,0,order).order,'whole')
    def test_noise_scaling_and_header_failure_keep_receiver_information_boundary(self):
        np.testing.assert_array_equal(phy.apply_channel(np.zeros((2048,2)),10,np.ones((2048,2))),np.full((2048,2),10**(-10/20)))
        with mock.patch.object(phy,'decode_packet',return_value=(np.zeros(12,dtype=np.uint8),False,0.)):
            event=phy.receive(np.zeros((2048,2)),7,2048,'QPSK')
        self.assertFalse(event['header_ok']);self.assertIsNone(event['action']);self.assertFalse(event['source_complete'])

if __name__=='__main__':unittest.main()

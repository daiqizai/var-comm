"""Checks for the byte-derived N2048 PHY; never edits the frozen M1 source."""
import hashlib
import json
import os
from pathlib import Path
import sys
import unittest
from unittest import mock
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=Path(os.environ.get('OWN_CONTROLS_TEST_ROOT',str(HERE.parents[1]/'historical_eval_20261003/source'))).resolve()
sys.path.insert(0,str(ROOT/'src'))
import own_controls_phy as phy

ORIGINAL_SHA='d5b26b35b1ce5ba8c147998e2ec6aa88615a2c84a0ab13d59026e27094b7d756'


class ProvenanceTests(unittest.TestCase):
    def test_exact_byte_derivation_and_original_publication_binding(self):
        source=ROOT/'experiments/scale-causal-partial-residual-20261002/partial_phy.py'
        publication=ROOT/'results/scale_causal_partial_residual_20261002/provenance/m1_publication.json'
        raw=source.read_bytes();receipt=json.loads(publication.read_text())
        self.assertEqual(hashlib.sha256(raw).hexdigest(),ORIGINAL_SHA)
        self.assertEqual(receipt['status'],'PUSHED');self.assertEqual(receipt['checks'],'PASS')
        matches=[v for k,v in receipt['source_bindings'].items() if k.endswith('/experiments/scale-causal-partial-residual-20261002/partial_phy.py')]
        self.assertEqual(matches,[ORIGINAL_SHA])
        substitutions=((b'raw-partial-token-20261002-v1',b'own-controls-raw-partial-N2048-20261004-v1',1),
            (b'N not in (512, 1024)',b'N not in (2048,)',2),
            (b'budgets=[512, 1024]',b'budgets=[2048]',1))
        for old,new,count in substitutions:
            self.assertEqual(raw.count(old),count);raw=raw.replace(old,new)
        self.assertEqual((HERE/'own_controls_phy.py').read_bytes(),raw)


class PhysicalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.actions=phy.action_grid(2048,'QPSK',orders=('entropy',),include_whole=True)
        rng=np.random.default_rng(143)
        cls.scales=[rng.integers(0,4096,size=size*size,dtype=np.int64) for size in phy.SIZES]

    def test_registered_scope_and_exact_formal_grid(self):
        registration=phy.registration()
        self.assertEqual(registration['protocol'],'own-controls-raw-partial-N2048-20261004-v1')
        self.assertEqual(registration['budgets'],[2048]);self.assertEqual(registration['class_bits'],0)
        self.assertEqual(len(self.actions),21)
        self.assertEqual(sum(action.q==0 for action in self.actions),5)
        self.assertEqual(sum(action.q>0 for action in self.actions),16)
        self.assertEqual({a.order for a in self.actions},{'whole','entropy'})
        self.assertEqual({a.phy for a in self.actions},{'QPSK'})
        self.assertEqual({a.m for a in self.actions if a.q==0},{4,5,6,7,8})

    def test_all_21_actual_qpsk_waveforms_and_paid_ledgers(self):
        for action in self.actions:
            positions=np.arange(action.q,dtype=np.int64) if action.q else None
            signal,ledger=phy.transmit(self.scales,action,positions)
            self.assertEqual(signal.shape,(2048,2));self.assertEqual(float(np.square(signal).sum()),4096)
            self.assertEqual(ledger['N_header'],68)
            self.assertEqual(ledger['N_prefix']+ledger['N_partial']+68,2048)
            self.assertEqual(ledger['header_class_bits'],0)
            self.assertEqual(ledger['body_crc_bits'],16*(1+bool(action.q)))
            self.assertEqual(ledger['body_tail_bits'],6*(1+bool(action.q)))
            self.assertLessEqual(ledger['prefix_effective_rate'],.9)
            if action.q: self.assertLessEqual(ledger['partial_effective_rate'],.9)

    def test_m8_whole_allowed_m8_partial_and_old_budgets_rejected(self):
        self.assertEqual(phy.Action(2048,'QPSK',8,0,'whole').m,8)
        for N,m,q in ((512,4,0),(1024,4,0),(2048,8,1)):
            with self.assertRaises(ValueError): phy.Action(N,'QPSK',m,q,'entropy' if q else 'whole')

    def test_short_or_illegal_received_frame_rejected(self):
        with self.assertRaises(ValueError): phy.receive(np.zeros((1024,2)),7,2048,'QPSK')
        with self.assertRaises(ValueError): phy.receive(np.zeros((2048,2)),7,1024,'QPSK')

    def test_actual_noise_uses_original_real_variance(self):
        received=phy.apply_channel(np.zeros((2048,2)),10,np.ones((2048,2)))
        np.testing.assert_allclose(received,10**(-10/20),rtol=0,atol=0)

    def test_crc_failed_header_exposes_no_action(self):
        with mock.patch.object(phy,'decode_packet',return_value=(np.zeros(12,dtype=np.uint8),False,0.)):
            event=phy.receive(np.zeros((2048,2)),7,2048,'QPSK')
        self.assertFalse(event['header_ok']);self.assertIsNone(event['action']);self.assertFalse(event['source_complete'])

    @unittest.skipUnless(sys.platform.startswith('linux'),'actual compiled trellis is qualified on Linux')
    def test_full_native_noiseless_receive_all21(self):
        for action in self.actions:
            positions=np.arange(action.q,dtype=np.int64) if action.q else None
            signal,_=phy.transmit(self.scales,action,positions)
            event=phy.receive(signal,13,2048,'QPSK')
            self.assertTrue(event['header_ok']);self.assertTrue(event['prefix_crc_ok']);self.assertTrue(event['body_crc_ok'])
            self.assertEqual(event['action'],action)
            for actual,expected in zip(event['prefix'],self.scales[:action.m]): np.testing.assert_array_equal(actual,expected)
            if action.q:
                self.assertTrue(event['partial_usable']);np.testing.assert_array_equal(event['partial_values'],self.scales[action.m][positions])


if __name__=='__main__': unittest.main()

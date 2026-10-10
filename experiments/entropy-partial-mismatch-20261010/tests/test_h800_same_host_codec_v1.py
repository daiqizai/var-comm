"""CPU fake codec and tiny archive checks; no models, GPU, channel or images."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import h800_same_host_codec_v1 as h

TOKENS=np.arange(680,dtype=np.int64)
BITS=np.arange(313,dtype=np.uint8)%2
HASHES={i:hashlib.sha256(('fresh H800 scale '+str(i)).encode()).hexdigest() for i in range(9)}


class FakeBackend:
    def __init__(self,ledger,events,mode=None):
        self.ledger=ledger;self.events=events;self.mode=mode;self.traces=[];self.tx_cdfs={}
    def source_tx(self,tokens):
        assert np.array_equal(tokens,TOKENS)
        self.events.append('TX')
        if self.mode=='TX_failure':raise RuntimeError('fake TX failure')
        for scale in range(9):
            self.ledger.call('prior_scale',lambda:None)
            self.traces.append(dict(source=0,role='TX',m=9,scale=scale,cdf_sha256=HASHES[scale]))
        self.tx_cdfs=dict(HASHES)
        bits=BITS.copy() if self.mode!='bad_bits' else BITS.astype(np.int64)
        return {m:dict(bits=bits.copy()) for m in range(4,10)}
    def source_rx(self,bits,m):
        assert m==4 and np.array_equal(bits,BITS)
        assert self.traces==[] and self.tx_cdfs=={i:HASHES[i] for i in range(4)}
        self.events.append('RX')
        for scale in range(4):
            self.ledger.call('prior_scale',lambda:None)
            self.traces.append(dict(source=0,role='RX',m=4,scale=scale,cdf_sha256=HASHES[scale]))
            if self.mode=='RX_failure':raise RuntimeError('fake independent RX failure')
        return dict(canonical=True,zero_extension_reads=30,received_tokens=TOKENS[:30].copy())


class CodecTests(unittest.TestCase):
    def case(self,mode=None):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);out=Path(temp.name)
        ledger=h.g.Ledger(out/'calls',lambda:None,h.CAPS);events=[]
        backend=ledger.call('model_load',lambda:FakeBackend(ledger,events,mode))
        return out,ledger,events,backend

    def test_actual_new_TX_dynamic_bits_and_independent_RX_exact_budget(self):
        out,ledger,events,backend=self.case();reads=[]
        def expected():
            self.assertEqual(events,['TX','RX']);reads.append('late');return TOKENS[:30].copy()
        result=h.exercise(TOKENS,backend,ledger,expected,out/'checks',h.engine().exercise)
        self.assertEqual(reads,['late']);self.assertEqual(ledger.completed,h.CAPS)
        self.assertEqual(ledger.summary()['unresolved'],0)
        self.assertEqual(result['actual_TX']['payload_bits'],313)
        self.assertFalse(result['actual_TX']['old_TX_CDF_hashes_used'])
        with np.load(out/'checks/actual_h800_tx_m4.npz',allow_pickle=False) as archive:
            self.assertEqual(archive.files,['bits']);self.assertTrue(np.array_equal(archive['bits'],BITS))
        cdf=json.loads((out/'checks/independent_rx/cdf_hash_assertions.json').read_text())
        self.assertEqual(cdf['expected_new_TX_hashes'],{str(i):HASHES[i] for i in range(4)})
        self.assertFalse(cdf['probabilities_shared_with_RX']);self.assertFalse(cdf['comparison_tokens_supplied_to_RX'])
        with self.assertRaisesRegex(RuntimeError,'cap exhausted'):ledger.call('prior_scale',lambda:None)

    def test_TX_failure_reservation_preserved_and_no_RX_or_expected_reader(self):
        out,ledger,events,backend=self.case('TX_failure');reader=mock.Mock()
        with self.assertRaisesRegex(RuntimeError,'fake TX failure'):
            h.exercise(TOKENS,backend,ledger,reader,out/'checks',h.engine().exercise)
        reader.assert_not_called();self.assertEqual(events,['TX'])
        self.assertEqual(ledger.reserved['source_tx'],1);self.assertEqual(ledger.completed['source_tx'],0)
        self.assertEqual(ledger.reserved['source_rx'],0);self.assertEqual(ledger.summary()['unresolved'],1)
        with self.assertRaises(RuntimeError):ledger.call('source_tx',lambda:None)

    def test_RX_failure_preserves_actual_new_TX_and_never_reads_comparison(self):
        out,ledger,events,backend=self.case('RX_failure');reader=mock.Mock()
        with self.assertRaisesRegex(RuntimeError,'fake independent RX failure'):
            h.exercise(TOKENS,backend,ledger,reader,out/'checks',h.engine().exercise)
        reader.assert_not_called();self.assertEqual(events,['TX','RX'])
        self.assertTrue((out/'checks/actual_h800_tx_m4.npz').is_file())
        self.assertEqual(ledger.completed['source_tx'],1);self.assertEqual(ledger.completed['prior_scale'],10)
        self.assertEqual(ledger.reserved['source_rx'],1);self.assertEqual(ledger.completed['source_rx'],0)
        self.assertEqual(ledger.summary()['unresolved'],1)

    def test_bad_TX_bits_stop_before_RX(self):
        out,ledger,events,backend=self.case('bad_bits');reader=mock.Mock()
        with self.assertRaisesRegex(RuntimeError,'bit vector malformed'):
            h.exercise(TOKENS,backend,ledger,reader,out/'checks',h.engine().exercise)
        reader.assert_not_called();self.assertEqual(events,['TX']);self.assertEqual(ledger.reserved['source_rx'],0)

    def test_no_encoder_render_or_decoder_budget_is_added(self):
        out,ledger,events,backend=self.case();called=[]
        for name in ('encoder','var_render','decoder_forward'):
            with self.subTest(name=name),self.assertRaises(RuntimeError):ledger.call(name,lambda:called.append(name))
        self.assertEqual(called,[])


class SourceAndBindingTests(unittest.TestCase):
    def test_TX_archive_only_tokens_with_original_hash_and_size(self):
        with tempfile.TemporaryDirectory() as td:
            archive=Path(td)/'source.npz';np.savez(archive,tokens=TOKENS)
            record=dict(asset=dict(archive=str(archive),tokens_sha256=hashlib.sha256(b'int64:680\0'+TOKENS.astype('<i8').tobytes()).hexdigest()))
            class Resolver:
                def path(self,name):assert name==str(archive);return archive
            resolver=Resolver();pin=h.source_pin(record,resolver)
            self.assertTrue(np.array_equal(h.load_tx_tokens(record,resolver,pin),TOKENS))
            bad=copy.deepcopy(record);bad['asset']['tokens_sha256']='0'*64
            with self.assertRaisesRegex(RuntimeError,'Frozen TX tokens differ'):
                h.load_tx_tokens(bad,resolver,h.source_pin(bad,resolver))
            np.savez(archive,tokens=TOKENS[:-1])
            with self.assertRaisesRegex(RuntimeError,'source pin differs'):h.load_tx_tokens(record,resolver,pin)

    def test_new_engine_reuses_bounded_owner_without_changing_frozen_host_engine(self):
        before=h.g.sha(h.host.__file__);old=h.host.engine();fresh=h.engine()
        self.assertIs(fresh.run.__code__,old.run.__code__)
        self.assertEqual(old.CAPS['prior_scale'],4);self.assertEqual(fresh.CAPS['prior_scale'],13)
        self.assertEqual(fresh.run.__globals__['__file__'],h.__file__)
        self.assertIs(fresh.run.__globals__['registration'],h.registration)
        self.assertIn('h800_single_rx_v1.py',fresh.TOOLS)
        self.assertEqual(before,h.g.sha(h.host.__file__))

    def test_modified_frozen_host_adapter_cannot_start_engine(self):
        with mock.patch.object(h.g,'sha',return_value='0'*64),self.assertRaisesRegex(RuntimeError,'host adapter changed'):
            h.engine()


if __name__=='__main__':unittest.main(verbosity=2)

"""Focused CPU owner/closure tests; fake prior only, never a real model."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest import mock
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import h800_ep32_source_gate_v1 as h
import ep_source_codec as partial

TOKENS=np.arange(680,dtype=np.int64)
HASHES={i:hashlib.sha256(('new H800 '+str(i)).encode()).hexdigest() for i in range(10)}


class FakeCodec:
    def __init__(self,backend,ledger,events,mode):
        self.backend=backend;self.ledger=ledger;self.events=events;self.mode=mode;self.bits={}
    def trace(self,count):
        b=self.backend;role,m=b.current
        for scale in range(count):
            self.ledger.call('prior_scale',lambda:None,source=b.source_index,scale=scale)
            b.traces.append(dict(source=b.source_index,role=role,m=m,scale=scale,cdf_sha256=HASHES[scale]))
            if role=='TX':b.tx_cdfs[scale]=HASHES[scale]
    def encode_endpoints(self,tokens,endpoints):
        self.events.append('TX');assert np.array_equal(tokens,TOKENS)
        if self.mode=='TX_failure':raise RuntimeError('fake TX failure')
        self.trace(10);result={}
        for m,K in endpoints:
            # Dynamic endpoint-specific lengths; no reuse of a 310/311-bit witness.
            bits=np.arange(301+m+K,dtype=np.uint16).astype(np.uint8)%2
            self.bits[m,K]=bits.copy()
            result[m,K]=dict(bits=bits,arithmetic_bits=len(bits),m=m,K=K)
        if self.mode=='missing':result.pop((9,192))
        if self.mode=='bad_bits':result[4,0]['bits']=result[4,0]['bits'].astype(np.int64)
        return result
    def decode(self,family,bits,m,K):
        assert family==(partial.PARTIAL_FAMILY if K else partial.WHOLE_FAMILY)
        assert np.array_equal(bits,self.bits[m,K]);self.events.append(('RX',m,K))
        self.trace(m+int(K>0))
        if self.mode=='RX_failure':raise RuntimeError('fake RX failure')
        if self.mode=='bad_CDF':self.backend.traces[-1]['cdf_sha256']='0'*64
        tokens=TOKENS[:partial.token_count(m,K)].copy()
        if self.mode=='bad_tokens':tokens[0]+=1
        return dict(canonical=True,zero_extension_reads=30,received_tokens=tokens)


class GateTests(unittest.TestCase):
    def case(self,mode=None):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);out=Path(temp.name)
        ledger=h.g.Ledger(out/'calls',lambda:None,h.CAPS);events=[]
        class Backend:
            def boundary(inner):events.append('boundary')
        backend=ledger.call('model_load',Backend)
        codec=FakeCodec(backend,ledger,events,mode);record=dict(source_index=0,source_id='fixed0')
        reads=[]
        def load(actual):
            assert actual==record
            if reads:
                self.assertIsInstance(events[-1],tuple);self.assertEqual(events[-1][0],'RX')
            reads.append(len(events));return TOKENS.copy()
        return out,ledger,events,backend,codec,record,reads,load

    def invoke(self,c):
        out,ledger,events,backend,codec,record,reads,load=c
        with mock.patch.object(h,'load_tokens',side_effect=load):
            return h.exercise_source(record,backend,codec,partial,ledger,out/'source')

    def test_one_source_has_exact_18partial_2shortwhole_154priors(self):
        c=self.case();result=self.invoke(c);out,ledger,events,backend,codec,record,reads,load=c
        self.assertEqual(ledger.completed,dict(h.CAPS,source_tx=1,source_rx=20,prior_scale=154))
        self.assertEqual(ledger.summary()['unresolved'],0);self.assertEqual(len(reads),21)
        actual=[e for e in events if isinstance(e,tuple)]
        self.assertEqual(actual[-2:],[('RX',4,0),('RX',5,0)])
        self.assertEqual(sum(K>0 for _,m,K in actual),18)
        self.assertEqual(events.count('boundary'),22)
        with np.load(out/'source/actual_streams.npz',allow_pickle=False) as archive:
            self.assertEqual(len(archive.files),24);self.assertEqual(len(archive['m9_K192']),502)
        self.assertTrue((out/'source/completion.json').is_file())
        self.assertEqual(result['sha256'],h.g.sha(out/'source/completion.json'))

    def test_partial_scope_cannot_silently_include_all_whole_RX(self):
        tx,rx=h.endpoint_scope(partial)
        self.assertEqual([p for p in rx if p[1]==0],[(4,0),(5,0)])
        self.assertEqual(32*(10+sum(m+int(K>0) for m,K in rx)),4928)
        self.assertEqual(32*len(rx),640);self.assertEqual(32*sum(K>0 for m,K in rx),576)

    def test_TX_failure_has_no_RX_and_unresolved_reservation(self):
        c=self.case('TX_failure')
        with self.assertRaisesRegex(RuntimeError,'fake TX failure'):self.invoke(c)
        ledger=c[1];self.assertEqual(ledger.reserved['source_tx'],1);self.assertEqual(ledger.completed['source_tx'],0)
        self.assertEqual(ledger.reserved['source_rx'],0);self.assertEqual(ledger.summary()['unresolved'],1)
        self.assertEqual(len(c[6]),1)

    def test_RX_failure_keeps_actual_streams_without_comparison_read(self):
        c=self.case('RX_failure')
        with self.assertRaisesRegex(RuntimeError,'fake RX failure'):self.invoke(c)
        self.assertEqual(len(c[6]),1);self.assertEqual(c[1].reserved['source_rx'],1)
        self.assertEqual(c[1].completed['source_rx'],0);self.assertEqual(c[1].summary()['unresolved'],1)
        self.assertTrue((c[0]/'source/actual_streams.npz').is_file())
        self.assertFalse((c[0]/'source/completion.json').exists())

    def test_missing_endpoint_fails_before_RX(self):
        c=self.case('missing')
        with self.assertRaisesRegex(RuntimeError,'Missing or extra'):self.invoke(c)
        self.assertEqual(c[1].reserved['source_rx'],0)

    def test_bad_bits_fail_before_RX(self):
        c=self.case('bad_bits')
        with self.assertRaisesRegex(RuntimeError,'bits malformed'):self.invoke(c)
        self.assertEqual(c[1].reserved['source_rx'],0)

    def test_independent_CDF_mismatch_stops_before_truth_read(self):
        c=self.case('bad_CDF')
        with self.assertRaisesRegex(RuntimeError,'CDF witnesses differ'):self.invoke(c)
        self.assertEqual(len(c[6]),1);self.assertFalse((c[0]/'source/completion.json').exists())

    def test_actual_decode_token_mismatch_never_marks_source_complete(self):
        c=self.case('bad_tokens')
        with self.assertRaises(h.g.GateMismatch):self.invoke(c)
        self.assertEqual(len(c[6]),2);self.assertFalse((c[0]/'source/completion.json').exists())

    def test_frozen_no_encoder_render_decoder_or_extra_prior_permission(self):
        c=self.case();ledger=c[1];called=[]
        for name in ('encoder','var_render','decoder_forward'):
            with self.assertRaisesRegex(RuntimeError,'cap exhausted'):ledger.call(name,lambda:called.append(name))
        ledger.reserved['prior_scale']=4928
        with self.assertRaisesRegex(RuntimeError,'cap exhausted'):ledger.call('prior_scale',lambda:called.append('prior'))
        self.assertEqual(called,[])


class InputTests(unittest.TestCase):
    def test_token_archive_hash_shape_and_tokens_only(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'a.npz';np.savez(path,tokens=TOKENS)
            record=dict(archive=dict(path=str(path),sha256=h.g.sha(path),bytes=path.stat().st_size),
                tokens_sha256=hashlib.sha256(b'int64:680\0'+TOKENS.astype('<i8').tobytes()).hexdigest())
            with mock.patch.object(h.g,'inside',side_effect=Path):
                self.assertTrue(np.array_equal(h.load_tokens(record),TOKENS))
                bad=copy.deepcopy(record);bad['tokens_sha256']='0'*64
                with self.assertRaisesRegex(RuntimeError,'token hash differs'):h.load_tokens(bad)
                np.savez(path,tokens=TOKENS[:-1])
                with self.assertRaisesRegex(RuntimeError,'Source bytes changed'):h.load_tokens(record)

    def test_new_execution_window_is_separate_from_historical_input_spec(self):
        deadline=time.time()+4000;e=h.execution(deadline,3600)
        self.assertEqual(e['max_seconds'],3600);self.assertTrue(e['input_contract_is_not_execution_budget'])
        historical=h.engine().inherited_spec(dict(spec=dict(marker='unchanged')),deadline,900)
        self.assertEqual(historical,dict(marker='unchanged',max_seconds=900,deadline_unix=deadline))
        for seconds in (0,3601,3600.0,True):
            with self.subTest(seconds=seconds),self.assertRaises(RuntimeError):h.execution(deadline,seconds)
        with self.assertRaises(RuntimeError):h.execution(time.time()-1,3600)

    def test_engine_preserves_original_globals_and_frozen_source_pins(self):
        before=h.same.engine();current=h.engine()
        self.assertEqual(before.CAPS['prior_scale'],13);self.assertEqual(current.CAPS['prior_scale'],4928)
        self.assertIs(current.worker_identity.__code__,before.worker_identity.__code__)
        self.assertEqual(current.worker_identity.__globals__['__file__'],h.__file__)
        with mock.patch.object(h.g,'sha',return_value='0'*64),self.assertRaisesRegex(RuntimeError,'Frozen source changed'):h.engine()


if __name__=='__main__':unittest.main(verbosity=2)

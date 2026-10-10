"""CPU-only fake arrays/callbacks; no torch, models, channels or source images."""
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
import leo_single_rx_diagnostic_v1 as d

HASHES={i:hashlib.sha256(str(i).encode()).hexdigest() for i in range(4)}
BITS=np.arange(311,dtype=np.uint8)%2
TOKENS=np.arange(30,dtype=np.int64)

class Fake:
    def __init__(self,ledger,events,mode=None):
        self.ledger=ledger;self.events=events;self.mode=mode;self.traces=[];self.tx_cdfs={}
    def source_rx(self,bits,m):
        self.events.append('decode_start')
        assert m==4 and np.array_equal(bits,BITS)
        assert self.tx_cdfs==HASHES and all(isinstance(x,str) for x in self.tx_cdfs.values())
        if self.mode=='OOM':raise MemoryError('fake OOM')
        for k in range(4):
            self.ledger.call('prior_scale',lambda:None,scale=k)
            self.traces.append(dict(source=0,role='RX',m=4,scale=k,cdf_sha256=HASHES[k]))
        if self.mode=='cdf':self.traces[1]['cdf_sha256']='f'*64
        self.events.append('decode_return')
        result=TOKENS.copy()
        if self.mode=='tokens':result[0]+=1
        return dict(canonical=self.mode!='canonical',zero_extension_reads=30,received_tokens=result)

class Tests(unittest.TestCase):
    def test_exact_budget_and_expected_tokens_read_only_after_rx(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td);ledger=d.g.Ledger(out/'calls',lambda:None,d.CAPS);events=[]
            ledger.call('model_load',lambda:None)
            backend=Fake(ledger,events)
            def expected():
                self.assertEqual(events,['decode_start','decode_return']);events.append('truth_assertion');return TOKENS.copy()
            result=d.exercise(BITS,HASHES,backend,ledger,expected,out/'checks')
            self.assertTrue(result['exact']);self.assertEqual(ledger.completed,d.CAPS)
            self.assertEqual(ledger.summary()['unresolved'],0)
            self.assertEqual(events,['decode_start','decode_return','truth_assertion'])
            self.assertFalse(json.loads((out/'checks/cdf_hash_assertions.json').read_text())['probabilities_shared_with_RX'])

    def test_forbidden_calls_rejected_before_callback(self):
        with tempfile.TemporaryDirectory() as td:
            ledger=d.g.Ledger(Path(td)/'calls',lambda:None,d.CAPS);called=[]
            for kind in ('encoder','source_tx','var_render','decoder_forward'):
                with self.subTest(kind=kind),self.assertRaisesRegex(RuntimeError,'Call cap exhausted'):
                    ledger.call(kind,lambda:called.append(kind))
            self.assertEqual(called,[])
            for k in range(4):ledger.call('prior_scale',lambda:None)
            with self.assertRaises(RuntimeError):ledger.call('prior_scale',lambda:called.append('extra'))
            self.assertEqual(called,[])

    def test_canonical_or_cdf_failure_does_not_read_comparison_tokens(self):
        for mode in ('canonical','cdf'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as td:
                ledger=d.g.Ledger(Path(td)/'calls',lambda:None,d.CAPS);reader=mock.Mock(return_value=TOKENS)
                with self.assertRaises(RuntimeError):d.exercise(BITS,HASHES,Fake(ledger,[],mode),ledger,reader,Path(td)/'checks')
                reader.assert_not_called();self.assertEqual(ledger.completed['source_rx'],1)

    def test_token_mismatch_preserved_without_acceptance(self):
        with tempfile.TemporaryDirectory() as td:
            ledger=d.g.Ledger(Path(td)/'calls',lambda:None,d.CAPS)
            with self.assertRaisesRegex(RuntimeError,'Exact witness mismatch'):
                d.exercise(BITS,HASHES,Fake(ledger,[],'tokens'),ledger,lambda:TOKENS,Path(td)/'checks')
            r=json.loads((Path(td)/'checks/s0_rx_m4.json').read_text())
            self.assertFalse(r['exact']);self.assertTrue((Path(td)/'checks/s0_rx_m4.npz').is_file())

    def test_failed_rx_leaves_reserved_call_and_cannot_replay(self):
        with tempfile.TemporaryDirectory() as td:
            calls=Path(td)/'calls';ledger=d.g.Ledger(calls,lambda:None,d.CAPS);reader=mock.Mock()
            with self.assertRaises(MemoryError):d.exercise(BITS,HASHES,Fake(ledger,[],'OOM'),ledger,reader,Path(td)/'checks')
            reader.assert_not_called();self.assertEqual(ledger.summary()['unresolved'],1)
            with self.assertRaises(RuntimeError):ledger.call('source_rx',lambda:None)
            with self.assertRaises(FileExistsError):d.g.Ledger(calls,lambda:None,d.CAPS)

    def test_witness_bits_require_exact_saved_actual_bytes(self):
        digest=d.g.image_sha(BITS);d.validate_bits(BITS,digest)
        bad=BITS.copy();bad[33]^=1
        for arr in (bad,BITS.astype(np.int64),BITS[:-1],np.full(311,2,np.uint8)):
            with self.subTest(dtype=arr.dtype,shape=arr.shape),self.assertRaises(RuntimeError):d.validate_bits(arr,digest)

    def test_actual_npz_never_reads_expected_member(self):
        class Archive:
            def __enter__(self):return self
            def __exit__(self,*a):pass
            def __getitem__(self,key):
                if key!='actual':raise AssertionError('Expected reference member reached decoder loader')
                return BITS
        with mock.patch.object(d.g,'inside',return_value=Path('fake')),mock.patch.object(d.g,'sha',return_value='a'*64),mock.patch.object(np,'load',return_value=Archive()):
            r=d.actual_bits(dict(pins={'checks/s0_tx_m4.npz':dict(path='fake',sha256='a'*64)},actual_bit_array_sha256=d.g.image_sha(BITS)))
            self.assertTrue(np.array_equal(r,BITS));self.assertIsNot(r,BITS)

    def test_cdf_witness_is_exact_source0_tx9_with_no_missing_scales(self):
        rows=[dict(scale=i,source=0,role='TX',m=9,cdf_sha256=hashlib.sha256(str(i).encode()).hexdigest()) for i in range(9)]
        trace=dict(source_index=0,current_endpoint=['TX',9],new_host_TX_RX_traces=rows)
        self.assertEqual(d.cdf_witness(trace),HASHES)
        for key,value in [('source',1),('role','RX'),('m',4),('scale',0),('cdf_sha256','not-sha')]:
            bad=copy.deepcopy(trace);bad['new_host_TX_RX_traces'][1][key]=value
            with self.subTest(key=key),self.assertRaises(RuntimeError):d.cdf_witness(bad)

    def test_only_window_changes_and_parent_is_never_mutated(self):
        parent=dict(spec=dict(caps=copy.deepcopy(d.g.CAPS),files=[{'immutable':'pin'}],deadline_unix=1,max_seconds=1800))
        old=copy.deepcopy(parent);result=d.inherited_spec(parent,time.time()+500,400)
        self.assertEqual(parent,old);self.assertEqual(result['caps'],d.g.CAPS)
        self.assertEqual(result['files'],old['spec']['files']);self.assertEqual(result['max_seconds'],400)
        with self.assertRaises(RuntimeError):d.inherited_spec(parent,time.time()-1,400)
        with self.assertRaises(RuntimeError):d.inherited_spec(parent,time.time()+500,901)

    def test_registration_tool_closure_does_not_accept_partial_or_changed_sources(self):
        with self.assertRaises(RuntimeError):d.check_tools({})
        bindings={n:'a'*64 for n in (*d.TOOLS,Path(d.__file__).name)}
        with mock.patch.object(d.g,'sha',return_value='a'*64):d.check_tools(bindings)
        with mock.patch.object(d.g,'sha',return_value='b'*64),self.assertRaises(RuntimeError):d.check_tools(bindings)

    def test_witness_metadata_binds_failed_parent_actual_wait_and_actual_stream(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td);run=base/'run';(run/'checks').mkdir(parents=True)
            counts=dict(model_load=1,encoder=1,source_tx=1,source_rx=0,var_render=0,prior_scale=9,decoder_forward=0)
            data={
                'worker_failure.json':dict(request_sha256=d.PARENT_SHA,mismatch_layer='bitstream',status='STOPPED_NO_RETRY',
                    counts=dict(completed=counts,reserved=counts,unresolved=0)),
                'actual_child_wait.json':dict(actual_child_waited=True,child_exit_code=1,automatic_retry=False),
                'checks/s0_encoder.json':dict(exact=True,actual_shape=[680],expected_shape=[680],actual_sha256='a'*64,expected_sha256='a'*64),
                'checks/s0_tx_m4.json':dict(name='s0_tx_m4',exact=False,actual_dtype='uint8',actual_shape=[311],actual_sha256=d.g.image_sha(BITS)),
                'cdf_trace_at_failure.json':dict(source_index=0,current_endpoint=['TX',9],new_host_TX_RX_traces=[dict(
                    source=0,role='TX',m=9,scale=i,cdf_sha256=hashlib.sha256(str(i).encode()).hexdigest()) for i in range(9)])}
            for name,value in data.items():(run/name).write_text(json.dumps(value))
            (run/'checks/s0_tx_m4.npz').write_bytes(b'Only metadata hashes this opaque fixture')
            pins={name:d.g.sha(run/name) for name in d.WITNESS_SHA}
            with mock.patch.object(d,'WITNESS_SHA',pins),mock.patch.object(d.g,'inside',side_effect=Path):
                result=d.witness_metadata(base/'request.json');self.assertEqual(result['expected_new_TX_CDF_hashes'],HASHES)
                (run/'checks/s0_tx_m4.npz').write_bytes(b'changed')
                with self.assertRaisesRegex(RuntimeError,'witness bytes changed'):d.witness_metadata(base/'request.json')
            # Even a freshly hashed metadata set cannot change failure into success.
            data['actual_child_wait.json']['child_exit_code']=0
            (run/'actual_child_wait.json').write_text(json.dumps(data['actual_child_wait.json']))
            pins={name:d.g.sha(run/name) for name in d.WITNESS_SHA}
            with mock.patch.object(d,'WITNESS_SHA',pins),mock.patch.object(d.g,'inside',side_effect=Path),self.assertRaisesRegex(RuntimeError,'actual failed wait'):
                d.witness_metadata(base/'request.json')

    def test_no_expected_array_read_on_invalid_external_hash_scope(self):
        with tempfile.TemporaryDirectory() as td:
            ledger=d.g.Ledger(Path(td)/'calls',lambda:None,d.CAPS);backend=Fake(ledger,[]);reader=mock.Mock()
            with self.assertRaisesRegex(RuntimeError,'Four external CDF hash'):
                d.exercise(BITS,{0:HASHES[0]},backend,ledger,reader,Path(td)/'checks')
            reader.assert_not_called();self.assertEqual(ledger.completed['source_rx'],0)

if __name__=='__main__':unittest.main(verbosity=2)

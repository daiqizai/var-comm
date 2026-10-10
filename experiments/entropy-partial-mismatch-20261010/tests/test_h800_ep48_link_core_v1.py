"""CPU-only actual-RX and finite-budget checks; no PHY or model execution."""
from contextlib import nullcontext
import copy
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import h800_ep48_link_core_v1 as c
import ep_source_codec as partial
import leo_whole_gate_v1 as g


def received(family='EC_VAR_PARTIAL',m=4,K=6):
    return dict(status='PAYLOAD_PARSED',header=dict(header_ok=True,profile_id=144),
        rx_profile=dict(family=family,m=m,K=K,profile_id=144),
        body=dict(crc_accepted=True,parser_accepted=True,payload=[1,0,1,1,0]))


class ReceivedTests(unittest.TestCase):
    def setup_case(self,mode=None):
        td=tempfile.TemporaryDirectory();self.addCleanup(td.cleanup)
        ledger=g.Ledger(Path(td.name)/'calls',lambda:None,c.CAPS);events=[]
        backend=types.SimpleNamespace(source_index=3,traces=[],current=None)
        backend.native=types.SimpleNamespace(torch=types.SimpleNamespace(no_grad=nullcontext))
        def render(native,tokens,m,K):
            events.append(('render',m,K,tokens.copy()))
            for scale in range(10):ledger.call('prior_scale',lambda:None)
            return ledger.call('decoder_forward',lambda:np.zeros((3,256,256),dtype=np.float32))
        backend.render_function=render
        class Codec:
            def decode(self,family,bits,m,K):
                events.append(('decode',family,bits.copy(),m,K))
                if family!='EC_STATIC_WHOLE':
                    for scale in range(m+int(K>0)):ledger.call('prior_scale',lambda:None)
                if mode=='parse':raise partial.InvalidSourceStream('synthetic noncanonical payload')
                if mode=='software':raise RuntimeError('synthetic software fault')
                return dict(canonical=True,zero_extension_reads=30,
                    received_tokens=np.full(partial.token_count(m,K),7,dtype=np.int64))
        return Codec(),backend,ledger,events

    def test_received_partial_mK_and_actual_payload_are_only_decode_inputs(self):
        codec,backend,ledger,events=self.setup_case();rx=received(m=5,K=9)
        image,e=c.recover_and_render(rx,codec,backend,partial,ledger,lambda:None)
        self.assertEqual(events[0][1],'EC_VAR_PARTIAL');self.assertEqual(events[0][3:],(5,9))
        self.assertTrue(np.array_equal(events[0][2],rx['body']['payload']))
        self.assertEqual(events[1][1:3],(5,9));self.assertEqual(len(events[1][3]),64)
        self.assertTrue(e['render_called']);self.assertFalse(e['comparison_truth_used'])
        self.assertEqual(ledger.completed['prior_scale'],16);self.assertEqual(ledger.completed['source_rx'],1)
        self.assertEqual(ledger.completed['var_render'],ledger.completed['decoder_forward'])

    def test_legally_received_static_family_uses_static_decode(self):
        codec,backend,ledger,events=self.setup_case();rx=received('EC_STATIC_WHOLE',4,0)
        _,e=c.recover_and_render(rx,codec,backend,partial,ledger,lambda:None)
        self.assertEqual(events[0][1],'EC_STATIC_WHOLE');self.assertEqual(ledger.completed['prior_scale'],10)
        self.assertTrue(e['render_called'])

    def test_header_reject_is_exact_gray_without_source_or_render_calls(self):
        codec,backend,ledger,events=self.setup_case();rx=received();rx.update(status='HEADER_REJECT',body=None,rx_profile=None)
        rx['header']['header_ok']=False
        image,e=c.recover_and_render(rx,codec,backend,partial,ledger,lambda:None)
        self.assertTrue(np.array_equal(image,np.full((3,256,256),.5,dtype=np.float32)))
        self.assertEqual(events,[]);self.assertEqual(sum(ledger.completed.values()),0)

    def test_crc_and_parser_rejections_do_not_decode_source(self):
        for key in ('crc_accepted','parser_accepted'):
            with self.subTest(key=key):
                codec,backend,ledger,events=self.setup_case();rx=received();rx['body'][key]=False
                image,e=c.recover_and_render(rx,codec,backend,partial,ledger,lambda:None)
                self.assertEqual(events,[]);self.assertEqual(e['kind'],'gray')

    def test_canonical_rejection_completes_RX_call_with_gray_and_no_render(self):
        codec,backend,ledger,events=self.setup_case('parse')
        image,e=c.recover_and_render(received(),codec,backend,partial,ledger,lambda:None)
        self.assertEqual(e['source_status'],'SOURCE_PARSE_REJECT');self.assertTrue((image==.5).all())
        self.assertEqual(ledger.completed['source_rx'],1);self.assertEqual(ledger.summary()['unresolved'],0)
        self.assertEqual(ledger.completed['var_render'],0)

    def test_software_fault_is_not_reclassified_as_channel_rejection(self):
        codec,backend,ledger,events=self.setup_case('software')
        with self.assertRaisesRegex(RuntimeError,'software fault'):
            c.recover_and_render(received(),codec,backend,partial,ledger,lambda:None)
        self.assertEqual(ledger.completed['source_rx'],0);self.assertEqual(ledger.summary()['unresolved'],1)
        self.assertEqual(ledger.completed['var_render'],0)


class PlanTests(unittest.TestCase):
    def test_fixed48_cpu_orchestration_retains_actual_error_accepted_profile(self):
        import ep_phy as phy
        from t2_ledger import Ledger
        root=Path(__file__).resolve().parents[3]
        policy=json.loads((root/'results/wcl_evidence_closure_20261009/T1_entropy_whole/v3/frozen_policy.json').read_text())
        records=[dict(source_index=i,source_id=s) for i,s in enumerate(policy['calibration_source_ids'][:4])]
        rows=c.plan(policy,records,partial);events=[]
        class Runtime:
            def __init__(self):self.profiles={str(p['profile_id']):p for p in phy.profile_templates()}
            def transmit(self,pid,bits,counter):
                events.append((pid,len(bits),counter));return np.zeros((1024,2),dtype=np.float64),dict(profile_id=pid)
            def receive(self,observed,snr,counter,codebook,ledger,event,phase):
                assert codebook is self.profiles and len(codebook)==360 and observed.shape==(1024,2)
                for kind in ('header','body'):
                    ledger.call(dict(event_id=event+':'+kind),dict(snr=snr,counter=counter),lambda:dict(complete=True))
                # A legal but incorrect received static profile must remain visible.
                return received('EC_STATIC_WHOLE',4,0)
        streams={i:{p:np.array([0,1]*11,dtype=np.uint8) for p in partial.all_endpoints()} for i in range(4)}
        with tempfile.TemporaryDirectory() as td:
            ledger=Ledger(Path(td)/'calls.sqlite','a'*64,96)
            try:
                pins=c.cpu_frames(Runtime(),rows,streams,partial,phy,ledger,lambda:None,Path(td)/'frames')
                self.assertEqual(len(events),48);self.assertEqual(len(pins),48)
                self.assertEqual(ledger.snapshot(),dict(total=96,unresolved=0,cap=96))
                packet=json.loads(Path(pins[1]['path']).read_text())
                self.assertGreater(packet['fallback']['actual_K'],0)
                self.assertEqual(packet['actual_RX']['rx_profile']['family'],'EC_STATIC_WHOLE')
            finally:ledger.close()

    def test_original_independent_provider_is_observed_without_TX_CDF_oracle(self):
        class Original:
            def __init__(self,native,primitives):self.scale=0;self.native=native;self.primitives=primitives
            def cdf(self):return np.arange(12,dtype=np.int64).reshape(3,4)
        def definitions(path,names,namespace):
            self.assertEqual(names,{'IndependentProvider'});namespace['IndependentProvider']=Original
        backend=types.SimpleNamespace(native=object(),source_index=2,current=('RX',4),traces=[],tx_cdfs={0:'differentTXhash'})
        codec=types.SimpleNamespace(entropy=types.SimpleNamespace(probability_cdf=object()))
        with mock.patch.object(c,'sha',return_value=c.SOURCE_DRIVER_SHA):
            c.bind_independent_provider(codec,backend,types.SimpleNamespace(definitions=definitions),'frozen_driver',g)
        first=codec.provider_factory();second=codec.provider_factory()
        self.assertIsNot(first,second);self.assertIsInstance(first,Original)
        self.assertTrue(np.array_equal(first.cdf(),np.arange(12).reshape(3,4)))
        self.assertNotEqual(backend.traces[0]['cdf_sha256'],backend.tx_cdfs[0])

    def test_actual_published_winners_define_exact48_and_original1000_counter(self):
        root=Path(__file__).resolve().parents[3]
        path=root/'results/wcl_evidence_closure_20261009/T1_entropy_whole/v3/frozen_policy.json'
        self.assertEqual(c.sha(path),c.POLICY_SHA)
        policy=json.loads(path.read_text());records=[dict(source_index=i,source_id=s) for i,s in enumerate(policy['calibration_source_ids'][:4])]
        rows=c.plan(policy,records,partial);self.assertEqual(len(rows),48)
        self.assertEqual(rows[0]['public_frame_counter'],3000)
        self.assertEqual(rows[4]['public_frame_counter'],9000)
        self.assertEqual(rows[8]['public_frame_counter'],15000)
        self.assertEqual(rows[-1]['public_frame_counter'],15009)
        self.assertEqual([x['target_K'] for x in rows[:4]],[0,25,50,75])
        bad=copy.deepcopy(records);bad[1]['source_id']=bad[0]['source_id']
        with self.assertRaisesRegex(RuntimeError,'identity/order'):c.plan(policy,bad,partial)

    def test_caps_require_one_model_and_no_unresolved_or_extra_work(self):
        counts=dict(completed=dict(c.CAPS),reserved=dict(c.CAPS),unresolved=0)
        self.assertEqual(c.close_counts(counts),counts)
        for name in ('encoder','source_tx','source_rx','prior_scale','var_render','decoder_forward'):
            bad=copy.deepcopy(counts);bad['completed'][name]+=1;bad['reserved']=dict(bad['completed'])
            with self.subTest(name=name),self.assertRaises(RuntimeError):c.close_counts(bad)
        bad=copy.deepcopy(counts);bad['unresolved']=1
        with self.assertRaisesRegex(RuntimeError,'Unresolved'):c.close_counts(bad)


if __name__=='__main__':unittest.main(verbosity=2)

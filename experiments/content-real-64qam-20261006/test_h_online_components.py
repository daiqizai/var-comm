"""Pure frozen integer coding/fake visual tests; no physical decoder or model."""
import ast
from contextlib import closing
import copy
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

HERE=Path(__file__).absolute().parent
for path in (HERE,HERE.parent,HERE.parent/'phy_codec',HERE.parent/'payload',
             HERE.parent/'full_calibration',HERE.parent/'development_prep'):
    sys.path.insert(0,str(path))
import h_online_components as timing
import h64_phy as phy
import h64_source as codec
import h_source_driver as source_api
import h_payload_rx as rx
import h_development_cpu as dev
from test_h_development_cpu import fixture,seal
from test_h64_core import Provider,load_primitives
import test_h_payload_rx as rx_fixture


class TimedProvider(Provider):
    def __init__(self,registry):
        super().__init__(registry);self.model_seconds=0.;self.cdf_seconds=0.


class OnlineComponentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.args=fixture();cls.ctx=dev.prepare(**cls.args,contract=seal(cls.args))
        cls.primitives=load_primitives();rx_fixture.ReceiverTests.setUpClass()

    def setUp(self):
        self.f=rx_fixture.ReceiverTests();self.f.setUp();self.providers=[]
        self.pixels=np.zeros((3,256,256),np.uint8)
        self.flat=np.concatenate(self.f.scales)
        self.encodes=[]

    def tx(self,entry):
        def encode(pixels):
            self.assertEqual(pixels.dtype,np.uint8);self.encodes.append(1);return self.flat.copy()
        return timing.tx_once(self.pixels,entry,self.ctx,encoder=encode,source_api=source_api,
            development_core=dev,codec=codec,primitives=self.primitives,
            provider_factory=lambda:TimedProvider(self.providers),synchronize=lambda:None)

    def rx(self,trace,render=None):
        return timing.rx_once(rx.receiver_view(trace),self.f.cat,receiver_api=rx,codec=codec,
            primitives=self.primitives,provider_factory=lambda:TimedProvider(self.providers),
            render_tokens=render or self.f.receiver.render_tokens,synchronize=lambda:None)

    def test_raw_tx_reencodes_visual_and_does_not_run_VAR_probability(self):
        entry=next(r for r in self.ctx['schedule'] if r['candidate']['arm']=='H64-R')
        a,b=self.tx(entry),self.tx(entry)
        self.assertEqual(a['fingerprint'],b['fingerprint']);self.assertEqual(len(self.encodes),2)
        self.assertEqual(a['probability']['provider_calls'],0)
        self.assertEqual(a['attempted_arithmetic_prefixes'],[])
        expected=dev.select_payload(self.ctx,entry,self.f.scales,{})
        np.testing.assert_array_equal(a['tx']['payload'],expected['payload'])

    def test_arithmetic_fallback_is_fresh_and_matches_original_payload_selector(self):
        entry=copy.deepcopy(next(r for r in self.ctx['schedule'] if r['candidate']['arm']=='H64-A'))
        # This changes only the fake test row: uniform CDF makes arithmetic
        # slightly longer than raw, forcing registered m9->m8->m7 fallback.
        entry['candidate'].update(target_m=9,nominal_rate='1/2')
        a=self.tx(entry)
        self.assertEqual(a['attempted_arithmetic_prefixes'],[9,8,7])
        self.assertEqual(a['probability']['provider_calls'],3)
        self.assertEqual(a['probability']['advanced_scales'],24)
        self.assertEqual(a['tx']['actual_m'],7);self.assertEqual(a['tx']['profile']['mode'],'raw')
        plain=codec.encode_prefixes(self.f.scales,lambda:Provider([]),self.primitives)
        expected=dev.select_payload(self.ctx,entry,self.f.scales,{m:r['bits'] for m,r in plain.items()})
        self.assertEqual(a['tx']['attempts'],expected['attempts'])
        np.testing.assert_array_equal(a['tx']['payload'],expected['payload'])

    def test_wrapped_integer_operations_preserve_bits_and_canonical_RX(self):
        bits=codec.encode_prefixes(self.f.scales,lambda:Provider([]),self.primitives,(6,))[6]['bits']
        trace=self.f.trace(self.f.profile('arithmetic'),bits)
        output=self.rx(trace)
        self.assertEqual(output['receiver_summary']['source_status'],'ARITHMETIC_SOURCE_DECODED')
        self.assertEqual(output['probability']['provider_calls'],1)
        self.assertEqual(output['component_calls']['RX_integer_decode'],6)
        self.assertEqual(output['component_calls']['RX_integer_encode'],6)
        self.assertEqual(output['component_calls']['RX_suffix_VAR_and_Dc'],1)
        self.assertEqual(output['new_packet_decodes'],0)

    def test_actual_wrong_rx_only_and_failure_gray_have_no_truth_correction(self):
        bits=phy.raw_payload(self.f.scales,6);bits[0]^=1
        trace=self.f.trace(self.f.profile(),bits)
        first=self.rx(trace)
        trace.update(tx={'arbitrary':'never read'},evaluation_only={'arbitrary':'never read'})
        self.assertEqual(self.rx(trace)['fingerprint'],first['fingerprint'])
        bad=dict(rx=dict(header=dict(header_ok=False,profile_id=None),body=None,status='HEADER_REJECT'),rx_profile=None)
        gray=self.rx(bad)
        self.assertTrue(gray['receiver_summary']['gray'])
        self.assertNotIn('RX_suffix_VAR_and_Dc',gray['seconds'])
        self.assertEqual(gray['probability']['provider_calls'],0)
        with self.assertRaisesRegex(RuntimeError,'RX view'):
            timing.rx_once(trace,self.f.cat,receiver_api=rx,codec=codec,primitives=self.primitives,
                provider_factory=lambda:TimedProvider([]),render_tokens=lambda *a:None,synchronize=lambda:None)

    def test_repetitions_no_output_cache_warmup_separate_and_numeric_drift_stops(self):
        trace=self.f.trace(self.f.profile(),phy.raw_payload(self.f.scales,6))
        calls=[]
        result=timing.repetitions(lambda:self.rx(trace),synchronize=lambda:None,guard=lambda:calls.append(1))
        self.assertEqual(len(self.f.rendered),4);self.assertEqual(len(calls),4)
        self.assertEqual(len(result['warmups']),1);self.assertEqual(len(result['measured']),3)
        self.assertNotIn('image',result['measured'][0]);self.assertEqual(result['new_packet_decodes'],0)
        self.assertEqual(len(timing.summarize_repetitions(result)['RX_suffix_VAR_and_Dc']['seconds']),3)
        changed=iter(['a','b'])
        with self.assertRaisesRegex(RuntimeError,'changed output'):
            timing.repetitions(lambda:dict(new_packet_decodes=0,fingerprint=next(changed)),
                               synchronize=lambda:None,guard=lambda:None)

    def test_software_failure_is_not_a_timed_success_or_gray(self):
        trace=self.f.trace(self.f.profile(),phy.raw_payload(self.f.scales,6))
        def fail(*_):raise RuntimeError('Synthetic resource failure')
        with self.assertRaisesRegex(RuntimeError,'resource failure'):self.rx(trace,fail)
        provider=TimedProvider([]);meter=timing.Meter(lambda:None)
        factory=timing.Providers(lambda:provider,meter,'RX')
        with factory():pass
        with self.assertRaisesRegex(RuntimeError,'reused'):factory()

    def test_historical_ledger_read_is_only_bound_event_interval_not_kernel_latency(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'ledger.sqlite';value={'header_ok':False};checksum=timing.digest(value)
            with closing(sqlite3.connect(path)) as db:
                db.execute('CREATE TABLE events (event_id TEXT,phase TEXT,kind TEXT,status TEXT,reserved_at REAL,completed_at REAL,result TEXT,result_sha TEXT,worker TEXT)')
                db.execute('INSERT INTO events VALUES (?,?,?,?,?,?,?,?,?)',('event','development','header','COMPLETE',10,10.2,
                           json.dumps(value),checksum,json.dumps({'pid':123})))
                db.commit()
            before=path.read_bytes()
            result=timing.historical_phy_windows(path,{'event':{'kind':'header','result_sha256':checksum}})
            self.assertAlmostEqual(result['rows'][0]['wall_seconds'],.2)
            self.assertFalse(result['standalone_PHY_measured']);self.assertFalse(result['compatible_for_end_to_end_sum'])
            self.assertEqual(before,path.read_bytes());self.assertEqual(result['new_packet_decodes'],0)
            with self.assertRaisesRegex(RuntimeError,'differs'):
                timing.historical_phy_windows(path,{'event':{'kind':'header','result_sha256':'f'*64}})

    def test_post_timing_TX_parity_cannot_change_selection(self):
        entry=next(r for r in self.ctx['schedule'] if r['candidate']['arm']=='H64-R');a=self.tx(entry);tx=a['tx'];p=tx['profile']
        trace={'tx':dict(profile_id=p['profile_id'],profile_key=p['profile_key'],target_m=tx['target_m'],
            actual_m=tx['actual_m'],K=p['K'],mode=p['mode'],fell_back=tx['fell_back'],attempts=tx['attempts'],
            payload_sha256=phy.array_sha(tx['payload']))}
        self.assertTrue(timing.verify_tx_against_trace(a,trace,phy))
        trace['tx']['actual_m']=9
        with self.assertRaisesRegex(RuntimeError,'differs'):timing.verify_tx_against_trace(a,trace,phy)

    def test_design_uses_original16_and_no_packet_budget_or_hidden_decoder(self):
        proto=json.loads((HERE.parent/'H_CODEC_PROTOCOL.json').read_text())
        budget=json.loads((HERE.parent/'resource_budget.json').read_text())
        plan=timing.design(proto,budget,self.args['development_registration'],self.ctx['schedule'])
        self.assertEqual(plan['source_indices'],[0,25,50,75,4,21,24,29,33,41,52,60,64,87,92,95])
        self.assertEqual(plan['new_packet_decodes'],0);self.assertFalse(plan['reserve_spending'])
        self.assertEqual(plan['noise_seed'],6201);self.assertEqual(plan['component_cases'],288)
        tree=ast.parse(Path(timing.__file__).read_text())
        attrs={n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)}
        self.assertFalse({'decode_once','receive_frame','receive_body','receive_header','BudgetLedger'}&attrs)
        self.assertFalse(any(isinstance(n,ast.Import) and any(a.name=='torch' for a in n.names) for n in tree.body))


if __name__=='__main__':unittest.main()

"""Pure timing-driver tests. No Torch, actual model, physical decoder or launch."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np

HERE=Path(__file__).absolute().parent
for p in (HERE,HERE.parent/'phy_codec',HERE.parent/'development_prep'):
    sys.path.insert(0,str(p))
import h_online_components as timing
import h_online_components_driver as d
import h64_phy as phy


def fixture():
    image=np.zeros((3,256,256),np.float32);tokens=np.zeros(680,np.int64);payload=np.array([0,1],np.uint8)
    profile=dict(profile_id=7,profile_key='actual-profile',K=0,mode='raw')
    tx=dict(profile=profile,target_m=7,actual_m=7,fell_back=False,attempts=[{'m':7}],payload=payload)
    trace_tx=dict(profile_id=7,profile_key='actual-profile',K=0,mode='raw',target_m=7,actual_m=7,
                  fell_back=False,attempts=[{'m':7}],payload_sha256=phy.array_sha(payload))
    summary={k:None for k in d.PARITY_FIELDS}
    summary.update(source_status='RAW_SOURCE_DECODED',gray=False,source_decode_complete=True,image_sha256=phy.array_sha(image),
        actual_received_tokens_sha256=phy.array_sha(tokens[:155]),receiver_view_sha256='v'*64,new_packet_decodes=0,
        target_image_used_for_reconstruction=False,truth_correction=False,cached_clean_image_used=False)
    trace=dict(development_slot=0,source_index=0,source_id='source0',noise_seed=6201,snr_db=13,
               role='H_WHOLE_SYSTEM',candidate_id='fixed-candidate',arm='H64-R',tx=trace_tx,
               rx={'received':'only'},rx_profile={'paid':'profile'})
    expected=dict(source_status='RAW_SOURCE_DECODED',gray=False,image_sha256=phy.array_sha(image),
                  receiver_view_sha256=summary['receiver_view_sha256'],rx_summary=summary)
    prob=dict(provider_calls=0,advanced_scales=0,probability_seconds=0.,cdf_seconds=0.)
    tv=dict(tx=tx,tokens=tokens,seconds={'TX_visual_encoding':.1,'TX_source_total':.2},
            component_calls={'TX_visual_encoding':1,'TX_source_total':1},probability=prob,
            attempted_arithmetic_prefixes=[],fingerprint='T'*64,new_packet_decodes=0,cross_frame_cache=False)
    rv=dict(image=image,receiver_summary=summary,seconds={'RX_source_and_image_total':.3,'RX_suffix_VAR_and_Dc':.25},
            component_calls={'RX_source_and_image_total':1,'RX_suffix_VAR_and_Dc':1},probability=prob,
            fingerprint='R'*64,new_packet_decodes=0,final_output_cache=False)
    token_sha=hashlib.sha256(b'int64:680\0'+tokens.astype('<i8').tobytes()).hexdigest()
    return trace,expected,tv,rv,token_sha


class TimingDriverTests(unittest.TestCase):
    def setUp(self):
        self.trace,self.expected,self.tx,self.rx,self.token_sha=fixture();self.calls=[]
        self.components={'tx':lambda *args:self.tx_call(*args),'rx':lambda view:self.rx_call(view)}
        self.receiver=SimpleNamespace(receiver_view=lambda f:dict(rx=copy.deepcopy(f['rx']),recorded_rx_profile=copy.deepcopy(f['rx_profile'])))
    def tx_call(self,*args):self.calls.append('TX');return copy.deepcopy(self.tx)
    def rx_call(self,view):
        self.assertEqual(set(view),{'rx','recorded_rx_profile'});self.calls.append('RX');return copy.deepcopy(self.rx)
    def measured(self):
        return d.measure_case(timing,self.components,np.zeros((3,256,256),np.uint8),{'frozen':'entry'},object(),
            self.trace,self.expected,self.token_sha,self.receiver,phy,synchronize=lambda:None,guard=lambda:None)

    def test_all_eight_fresh_computations_match_actual_tx_and_rx(self):
        value=self.measured()
        self.assertEqual(self.calls,['TX']*4+['RX']*4)
        self.assertEqual(value['TX_exact_repetitions'],4);self.assertEqual(value['RX_exact_repetitions'],4)
        self.assertEqual(len(value['TX']['warmups']),1);self.assertEqual(len(value['TX']['measured']),3)
        self.assertNotIn('tokens',value['TX']['measured'][0]);self.assertNotIn('image',value['RX']['measured'][0])
        self.assertEqual(value['TX_probability_summary']['probability_seconds']['samples'],[0.]*3)
        self.assertIs(value['quality_sample_added'],False)

    def test_wrong_source_tokens_payload_or_output_stops_not_corrects(self):
        self.tx['tokens'][679]=1
        with self.assertRaisesRegex(RuntimeError,'full680-token'):self.measured()
        self.setUp();self.tx['tx']['payload'][0]^=1
        with self.assertRaisesRegex(RuntimeError,'Fresh online TX differs'):self.measured()
        self.setUp();self.rx['image'][0,0,0]=.25
        with self.assertRaisesRegex(RuntimeError,'Fresh actual RX RGB'):self.measured()
        self.assertEqual(self.calls.count('RX'),1)

    def test_accepted_wrong_header_tokens_and_gray_keep_exact_recorded_rx(self):
        for status,gray in (('RAW_SOURCE_DECODED',False),('ARITHMETIC_SOURCE_INVALID_GRAY',True),('WIRE_REJECT_GRAY',True)):
            self.setUp()
            self.rx['receiver_summary'].update(source_status=status,gray=gray,received_profile_id=999,received_m=6)
            self.expected['rx_summary']=copy.deepcopy(self.rx['receiver_summary'])
            self.expected.update(source_status=status,gray=gray)
            value=self.measured();self.assertEqual(value['rx_source_status'],status)
            self.assertEqual(value['RX']['measured'][0]['receiver_summary']['received_profile_id'],999)
        self.rx['receiver_summary']['received_m']=7
        with self.assertRaisesRegex(RuntimeError,'source parse differs'):self.measured()

    def test_second_repeat_drift_does_not_hide_behind_first_result(self):
        def rx(view):
            value=self.rx_call(view)
            if self.calls.count('RX')==2:value['receiver_summary']['receiver_view_sha256']='changed'
            return value
        self.components['rx']=rx
        with self.assertRaisesRegex(RuntimeError,'source parse differs'):self.measured()
        self.assertEqual(self.calls.count('RX'),2)

    def test_fixed_grid_summary_uses_source_means_warmup_excluded_and_no_end_to_end(self):
        row=self.measured();rows=[]
        for index in timing.FIXED:
            for slot in range(18):
                value=copy.deepcopy(row);value.update(source_index=index,development_slot=slot)
                rows.append(value)
        result=d.summarize(rows,dict(source_indices=list(timing.FIXED)))
        self.assertEqual(result['component_case_count'],288)
        self.assertIsNone(result['exclusive_PHY_decode']);self.assertIsNone(result['end_to_end_latency'])
        self.assertFalse(result['historical_PHY_can_be_added_to_GPU_totals'])
        probability=[x for x in result['components'] if x['component']=='TX_VAR_probability']
        self.assertEqual(len(probability),18);self.assertEqual(probability[0]['mean_seconds'],0.)
        rows[0]['source_index']=1
        with self.assertRaisesRegex(RuntimeError,'coverage incomplete'):d.summarize(rows,dict(source_indices=list(timing.FIXED)))

    def test_owner_empty_output_is_safe_but_previous_attempt_cannot_restart(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/'precreated';out.mkdir();d.claim_output(out,'a'*64)
            self.assertEqual(d.read(out/'attempt.json')['new_packet_decodes'],0)
            with self.assertRaisesRegex(RuntimeError,'no retry'):d.claim_output(out,'a'*64)

    def test_frozen_core_unchanged_and_no_runtime_physical_or_quality_functions(self):
        self.assertEqual(d.sha(timing.__file__),d.CORE_SHA)
        tree=ast.parse(Path(d.__file__).read_text())
        calls={n.func.attr if isinstance(n.func,ast.Attribute) else n.func.id
               for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,(ast.Attribute,ast.Name))}
        self.assertFalse(calls & {'decode_once','receive_frame','receive_body','transmit_body','BudgetLedger','randn','randn_like',
                                 'score_reconstruction','load_suite','ConvNeXtValidation'})
        funcs={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
        self.assertNotIn('build_components(',ast.unparse(funcs['inspect_inputs']))
        self.assertNotIn('save(',ast.unparse(funcs['inspect_inputs']))


if __name__=='__main__':unittest.main()

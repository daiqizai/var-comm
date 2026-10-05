"""Small actual-RX schema tests with synthetic bits/CDFs and a fake renderer."""
import copy
import hashlib
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np

HERE=Path(__file__).resolve().parent
for path in (HERE,HERE.parent,HERE.parent/'phy_codec',HERE.parent/'payload'):sys.path.insert(0,str(path))
import h_full_payload_rx as full
import h_full_payload_cpu as core
import h_payload_rx as rx
import h_payload_render_driver as cache_api
import h64_phy as phy
import h64_source as codec
from test_h_payload_rx import ReceiverTests
from test_h64_core import Provider


class FullRXTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):ReceiverTests.setUpClass()
    def setUp(self):
        self.f=ReceiverTests();self.f.setUp();self.receiver=self.f.receiver
        self.target=np.zeros((3,256,256),np.uint8);self.targetsha=hashlib.sha256(self.target.tobytes()).hexdigest()
        self.schedule=[];self.sources={p:dict(phase=p,source_id='synthetic0',source_index=0,registration_sha256=('a' if p==full.PHASES[0] else 'b')*64,frames=[]) for p in full.PHASES}
        p=self.f.profile(q=6);wire=self.f.trace(p,phy.raw_payload(self.f.scales,6))
        for slot in range(14):
            phase=full.PHASES[slot>=8];local=slot if slot<8 else slot-8
            c=dict(slot=local,candidate_id=f'fixture{slot}',arm='H64-R' if slot<8 else core.PARTIAL_ARM,
                target_m=6 if slot<8 else 8,K=0 if slot<8 else 140,q=6,nominal_rate='5/6',snr_db=13 if local%2==0 else 19)
            self.schedule.append(dict(full_slot=slot,phase=phase,candidate=c))
            for seed in full.SEEDS:
                frame=copy.deepcopy(wire);frame.update(status='H_FULL_CPU_FRAME_TRACE',stage=phase,source_id='synthetic0',source_index=0,
                    candidate_id=c['candidate_id'],candidate_slot=local,full_slot=slot,arm=c['arm'],snr_db=c['snr_db'],noise_seed=seed,
                    public_frame_counter=core.frame_counter(slot,0,seed),scrambling_session=core.SESSION,noise_stage=core.NOISE_STAGE,
                    execution_registration_sha256=self.sources[phase]['registration_sha256'],event_id=f'synthetic:{slot}:{seed}',
                    evaluation_only=dict(header_correct=False,parsed_wire_matches_transmission=False,accepted_wire_mismatch=True))
                self.sources[phase]['frames'].append(frame)
    def render(self,sources=None,target=None,**kwargs):
        target=self.target if target is None else target
        return full.render_source(self.sources if sources is None else sources,target,hashlib.sha256(target.tobytes()).hexdigest(),
            self.schedule,core,self.receiver,rx,cache_api,{'model':'test-frozen'},kwargs.get('boundary',lambda:None))
    def replace_wire(self,frame,wire):
        frame['rx']=copy.deepcopy(wire['rx']);frame['rx_profile']=copy.deepcopy(wire['rx_profile'])

    def test_all42_actual_frames_share_rx_cache_across_phases_seeds_without_truth(self):
        rows,images,cost=self.render()
        self.assertEqual(len(rows),42);self.assertEqual(cost['receiver_cache_misses'],1);self.assertEqual(cost['receiver_cache_hits'],41)
        self.assertEqual(len(self.f.rendered),1);self.assertEqual(len(images),1)
        self.assertEqual({r['phase'] for r in rows},set(full.PHASES));self.assertEqual(rows[-1]['K'],140)
        self.assertTrue(all(r['received_K']==0 and r['received_m']==6 for r in rows))
        self.assertTrue(all(r['rx_summary']['truth_correction'] is False for r in rows))
        self.assertEqual(len({r['rx_summary']['receiver_cache_origin_event'] for r in rows}),1)

    def test_truth_intended_payload_and_target_cannot_change_reconstruction(self):
        first,images,_=self.render();source=copy.deepcopy(self.sources)
        for value in source.values():
            for frame in value['frames']:
                frame['tx']={'profile_id':4095,'target_m':9,'actual_m':1,'K':0,'payload':[1,0]}
                frame['received_raw_tokens']=[4095]*680;frame['gray']=True
                frame['evaluation_only']={'header_correct':True,'parsed_wire_matches_transmission':True,'accepted_wire_mismatch':False}
        target=np.full_like(self.target,255);second,images2,_=self.render(source,target)
        self.assertEqual([r['image_sha256'] for r in first],[r['image_sha256'] for r in second])
        self.assertNotEqual(first[0]['psnr_db'],second[0]['psnr_db'])
        for k in images:np.testing.assert_array_equal(images[k],images2[k])

    def test_wrong_accepted_partial_header_uses_complete_paid_catalogue(self):
        p=next(p for p in self.f.cat['profiles'] if (p['mode'],p['m'],p['K'],p['q'],p['nominal_rate'])==('raw',8,140,6,'5/6'))
        wire=self.f.trace(p,phy.raw_payload(self.f.scales,8,140));source=copy.deepcopy(self.sources)
        frame=source[full.PHASES[0]]['frames'][0];self.replace_wire(frame,wire)
        rows,_,cost=self.render(source)
        self.assertEqual(rows[0]['target_m'],6);self.assertEqual(rows[0]['received_m'],8);self.assertEqual(rows[0]['received_K'],140)
        self.assertEqual(len(self.f.rendered[0][0]),395);self.assertEqual(cost['receiver_cache_misses'],2)
        bad=copy.deepcopy(source);bad[full.PHASES[0]]['frames'][0]['rx_profile']['K']=141
        with self.assertRaisesRegex(rx.TraceIntegrityError,'differs from actual'):self.render(bad)

    def test_each_cache_hit_revalidates_decoded_bits_header_and_profile(self):
        source=copy.deepcopy(self.sources);bad=source[full.PHASES[1]]['frames'][0]
        bad['rx']['body']['decoded_bits'][-1]^=1
        with self.assertRaisesRegex(rx.TraceIntegrityError,'parse differs'):self.render(source)
        source=copy.deepcopy(self.sources);source[full.PHASES[1]]['frames'][0]['rx']['header']['profile_id']=4095
        with self.assertRaisesRegex(rx.TraceIntegrityError,'ID unavailable'):self.render(source)

    def test_real_crc_rejection_stays_gray_and_does_not_use_cached_success(self):
        source=copy.deepcopy(self.sources);p=self.f.profile(q=6);valid=phy.pack_body(phy.raw_payload(self.f.scales,6),p)
        valid[-1]^=1;self.replace_wire(source[full.PHASES[1]]['frames'][0],self.f.trace(p,decoded=valid))
        rows,images,cost=self.render(source);gray=[r for r in rows if r['gray']]
        self.assertEqual(len(gray),1);self.assertEqual(gray[0]['source_status'],'WIRE_REJECT_GRAY')
        self.assertFalse(gray[0]['rx_summary']['receiver_result_cache_hit']);self.assertEqual(cost['suffix_renderer_calls'],1)
        np.testing.assert_array_equal(images[gray[0]['image_key']],np.full((3,256,256),.5,np.float32))

    def test_actual_arithmetic_canonical_path_independent_and_errors_propagate(self):
        encoded=codec.encode_prefixes(self.f.scales,lambda:Provider([]),self.f.primitives,(6,))
        wire=self.f.trace(self.f.profile('arithmetic'),encoded[6]['bits']);source=copy.deepcopy(self.sources)
        for phase in full.PHASES:
            for frame in source[phase]['frames']:self.replace_wire(frame,wire)
        rows,_,cost=self.render(source)
        self.assertEqual(cost['arithmetic_canonical_calls'],1);self.assertEqual(len(self.f.registry),1)
        self.assertTrue(all(r['source_status']=='ARITHMETIC_SOURCE_DECODED' for r in rows))
        malformed=self.f.trace(self.f.profile('arithmetic'),np.append(encoded[6]['bits'],0).astype(np.uint8))
        self.replace_wire(source[full.PHASES[0]]['frames'][0],malformed)
        rows,_,cost=self.render(source);self.assertEqual(rows[0]['source_status'],'ARITHMETIC_SOURCE_INVALID_GRAY')
        with patch.object(codec,'decode_prefix',side_effect=RuntimeError('resource/model failure')):
            with self.assertRaisesRegex(RuntimeError,'resource/model'):self.render(source)

    def test_missing_duplicate_wrong_counter_and_one_unclosed_group_rejected(self):
        bad=copy.deepcopy(self.sources);bad[full.PHASES[1]]['frames'].pop()
        with self.assertRaisesRegex(RuntimeError,'Incomplete'):self.render(bad)
        bad=copy.deepcopy(self.sources);bad[full.PHASES[1]]['frames'].append(copy.deepcopy(bad[full.PHASES[1]]['frames'][0]))
        with self.assertRaisesRegex(RuntimeError,'duplicate'):self.render(bad)
        bad=copy.deepcopy(self.sources);bad[full.PHASES[0]]['frames'][0]['public_frame_counter']+=1
        with self.assertRaisesRegex(RuntimeError,'scheduling'):self.render(bad)
        with self.assertRaisesRegex(RuntimeError,'Both whole'):self.render({full.PHASES[0]:self.sources[full.PHASES[0]]})
        with self.assertRaisesRegex(RuntimeError,'source STOP'):
            self.render(boundary=lambda:(_ for _ in ()).throw(RuntimeError('source STOP')))


if __name__=='__main__':unittest.main()

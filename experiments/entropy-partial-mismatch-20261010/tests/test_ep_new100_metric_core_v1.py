"""Synthetic CPU checks only: no real pixels, models, scoring or bootstrap."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import ep_new100_metric_core_v1 as c
from test_ep_new100_confirmation_core_v1 import fixture,grid


class Local:
    inside=staticmethod(Path)
    @staticmethod
    def write(p,v):Path(p).write_text(json.dumps(v),encoding='utf-8')


class Ledger:
    def __init__(self):self.counts={k:0 for k in c.CAPS}
    def add(self,k,n=1):
        self.counts[k]+=n
        if self.counts[k]>c.CAPS[k]:raise RuntimeError('cap exceeded')
    def summary(self):return dict(caps=c.CAPS,reserved=dict(self.counts),completed=dict(self.counts),unresolved=0)


class Evaluator:
    identity='a'*64
    def __init__(self,ledger):self.ledger=ledger;ledger.add('model_constructions',3)
    def prepare(self,target):
        for k in ('reference_preparations','dinov2_vitl14_reference','convnext_reference'):self.ledger.add(k)
        return dict(reference_sha256=c.array_sha(target))
    def score(self,target,image,class_index,prepared):
        assert prepared['reference_sha256']==c.array_sha(target)
        for k in ('image_scores','dinov2_vitl14_reconstruction','convnext_reconstruction','lpips_pair'):self.ledger.add(k)
        self.ledger.add('lpips_alexnet_backbone_forward',2)
        return dict(psnr_db=-1.,lpips_alex=.7,dinov2_vitl14_cosine=-.2,convnext_top1_source_prediction=0)


def fake():
    events=grid(fixture());records=[dict(source_index=i,source_id=events[i*36]['source_id'],evaluation_class_index=0,
        pixels_archive=dict(path='/synthetic/reference'+str(i),sha256='a'*64)) for i in range(100)]
    image=np.zeros((3,256,256),dtype=np.float32);image_sha=c.array_sha(image);docs={};pins=[]
    for i,event in enumerate(events):
        pin=dict(path='/synthetic/logical/'+str(i),sha256='b'*64);pins.append(pin)
        docs[pin['path']]=dict(frame_index=i,logical_event=event,physical_frame=dict(path='/synthetic/phy/'+str(i),sha256='c'*64),
            reconstruction=dict(image_archive=dict(path='/synthetic/image',sha256='d'*64),image_sha256=image_sha),
            actual_RX_status='GRAY_HEADER_FAILURE' if i%2 else 'KEEP_ACTUAL_HARD_TOKENS',actual_received_profile=None if i%2 else {'m':4,'K':0},
            actual_header_ok=i%2==0,actual_crc_accepted=False,actual_parser_accepted=None,
            actual_source_status='NO_SOURCE_HEADER_REJECT' if i%2 else 'RAW_KEEP',body_attempted=i%2==0,fixed_gray=i%2==1)
    request=dict(records=records,logical_events=events,reconstruction_results=pins,historical_score_reuse_allowed=False)
    return request,docs,image


class MetricCoreTests(unittest.TestCase):
    def test_budgets_and_new_population(self):
        self.assertEqual(c.CAPS,c.plan.scientific_caps()['metrics']);self.assertEqual(c.CAPS['model_constructions'],3)
        self.assertEqual(c.CAPS['reference_preparations'],100);self.assertEqual(c.CAPS['image_scores'],3600)
        self.assertEqual(c.CAPS['lpips_alexnet_backbone_forward'],7200)

    def test_private_provider_retains_every_scientific_code_object_and_old_globals(self):
        before=dict(vars(c.original));provider,proof=c.private_provider()
        for name,same in proof['methods_same_code_object'].items():self.assertTrue(same);self.assertIs(getattr(provider,name).__code__,getattr(c.original.FourMetrics,name).__code__)
        self.assertTrue(all(proof['functions_same_code_object'].values()))
        self.assertEqual(before,dict(vars(c.original)));self.assertEqual(c.original.POPULATION,'original_calibration_pilot')

    def test_no_old_score_reuse_or_wrong_population(self):
        for config in (dict(confirmation_population='original_holdout500',policy_selection=False,historical_score_reuse_allowed=False),
            dict(confirmation_population=c.POPULATION,policy_selection=True,historical_score_reuse_allowed=False),
            dict(confirmation_population=c.POPULATION,policy_selection=False,historical_score_reuse_allowed=True)):
            with self.assertRaises(ValueError):c.bound_assets(config)

    def test_same_image_different_references_count_independently_and_all_failures_survive(self):
        r,docs,image=fake();ledger=Ledger();ev=Evaluator(ledger)
        with tempfile.TemporaryDirectory() as tmp,mock.patch.object(c,'reference_pixels',side_effect=lambda record:np.full((3,256,256),record['source_index']/100,dtype=np.float32)),mock.patch.object(c,'reconstruction_pixels',return_value=image):
            answer=c.scores(r,ev,ledger,lambda:None,Path(tmp)/'scores',Local,lambda p:docs[p['path']])
            rows=json.loads(Path(answer['metric_rows']['path']).read_bytes())
            self.assertEqual(answer['actual_unique_pairs'],100);self.assertEqual(answer['reused_pairs'],3500)
            self.assertEqual(answer['logical_frames'],3600);self.assertEqual(answer['historical_score_reuse'],0)
            self.assertEqual(answer['bootstrap_calls'],0);self.assertFalse(answer['policy_selection'])
            self.assertEqual(sum(not x['actual_header_ok'] for x in rows),1800)
            self.assertEqual(sum(x['fixed_gray'] for x in rows),1800)
            self.assertNotEqual(rows[0]['actual_source_status'],rows[1]['actual_source_status'])
            self.assertTrue(all(x['actual_crc_accepted'] is False for x in rows))
            self.assertEqual(rows[0]['psnr_db'],-1.);self.assertEqual(rows[1]['dinov2_vitl14_cosine'],-.2)
            self.assertNotEqual(rows[0]['actual_RX_status'],rows[1]['actual_RX_status'])
            self.assertEqual(rows[0]['metric_pair_key'],rows[1]['metric_pair_key'])
            c.validate_completion(answer,rows,r)
            changed=copy.deepcopy(answer);changed['actual_unique_pairs']+=1;changed['reused_pairs']-=1
            with self.assertRaises(ValueError):c.validate_completion(changed,rows,r)
            changed=copy.deepcopy(answer);changed['counts']['completed']['model_constructions']=2
            changed['counts']['reserved']['model_constructions']=2
            with self.assertRaises(ValueError):c.validate_completion(changed,rows,r)

    def test_incomplete_grid_or_foreign_reference_stops(self):
        r,docs,image=fake();r['reconstruction_results'].pop()
        with tempfile.TemporaryDirectory() as tmp,self.assertRaises(ValueError):c.scores(r,Evaluator(Ledger()),Ledger(),lambda:None,Path(tmp)/'scores',Local,lambda p:docs[p['path']])
        r,docs,image=fake();r['records'][0]['source_id']='foreign'
        with tempfile.TemporaryDirectory() as tmp,self.assertRaises(ValueError):c.scores(r,Evaluator(Ledger()),Ledger(),lambda:None,Path(tmp)/'scores',Local,lambda p:docs[p['path']])

    def test_foreign_reconstruction_event_stops_before_score(self):
        r,docs,image=fake();docs[r['reconstruction_results'][0]['path']]['logical_event']=dict(r['logical_events'][0],noise_seed=6201)
        with tempfile.TemporaryDirectory() as tmp,self.assertRaises(ValueError):c.scores(r,Evaluator(Ledger()),Ledger(),lambda:None,Path(tmp)/'scores',Local,lambda p:docs[p['path']])

    def test_source_reference_uses_original_conversion_function(self):
        record=dict(preprocessing_id='x',pixels_archive={'path':'p','sha256':'s'})
        with mock.patch.object(c.original.core,'reference_pixels',return_value='sentinel') as method:
            self.assertEqual(c.reference_pixels(record),'sentinel');method.assert_called_once_with(dict(record,archive=record['pixels_archive']))

    def test_quantized_or_changed_reconstruction_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'image.npz'
            np.savez(path,image=np.zeros((3,256,256),dtype=np.uint8));pin=c.original.core.descriptor(path)
            with self.assertRaises(ValueError):c.reconstruction_pixels(pin,'a'*64,Path)
            image=np.zeros((3,256,256),dtype=np.float32);np.savez(path,image=image);pin=c.original.core.descriptor(path)
            np.testing.assert_array_equal(c.reconstruction_pixels(pin,c.array_sha(image),Path),image)
            with self.assertRaises(ValueError):c.reconstruction_pixels(pin,'a'*64,Path)

    def test_cache_key_has_both_pixels_and_provider(self):
        a=np.zeros((3,256,256),dtype=np.float32);b=np.ones_like(a)
        self.assertNotEqual(c.pair_cache_key(a,b,'a'*64),c.pair_cache_key(b,b,'a'*64))
        self.assertNotEqual(c.pair_cache_key(a,b,'a'*64),c.pair_cache_key(a,a,'a'*64))
        self.assertNotEqual(c.pair_cache_key(a,b,'a'*64),c.pair_cache_key(a,b,'b'*64))

    def test_nonbinary_agreement_and_policy_change_rejected(self):
        r,docs,_=fake();rows=[dict(e,psnr_db=1.,lpips_alex=.2,dinov2_vitl14_cosine=.3,convnext_top1_source_prediction=0,
            **{k:docs[r['reconstruction_results'][i]['path']][k] for k in c.FAILURE_FIELDS}) for i,e in enumerate(r['logical_events'])]
        rows[0]['convnext_top1_source_prediction']=.5
        with self.assertRaises(ValueError):c.validate_rows(rows,r['logical_events'],r['records'])
        rows[0]['convnext_top1_source_prediction']=0;rows[0]['policy']={'foreign':'policy'}
        with self.assertRaises(ValueError):c.validate_rows(rows,r['logical_events'],r['records'])

if __name__=='__main__':unittest.main()

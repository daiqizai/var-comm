"""Synthetic CPU checks only; no original pixels, weights, PHY or model calls."""
import copy
import hashlib
import importlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import h800_ep_full_evaluation_core_v1 as c
import h800_ep_full_visual_v1 as v
import h800_ep_full_metric_owner_v1 as m
import h800_ep_full_phy_v1 as phy
from test_h800_ep_full_phy_v1 import policy,records,finalists


class GridTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy=policy();cls.records=records();cls.finalists=finalists()
        cls.events=c.full.frames(cls.records,cls.finalists,cls.policy)
        cls.rows=[dict(e,psnr_db=0.,lpips_alex=.8,dinov2_vitl14_cosine=.1,
            convnext_top1_source_prediction=0) for e in cls.events]

    def test_complete1000_three_noise_means_and_lexical_tie(self):
        result=c.winners(self.rows,self.records,self.finalists,self.policy)
        self.assertEqual(result['population'],'original_calibration_full1000')
        self.assertTrue(result['final_strategy_frozen']);self.assertFalse(result['new100_admitted'])
        for row in result['snrs'].values():
            self.assertEqual(row['winner'],min(row['ranking']))
            self.assertIn(row['original_final_whole_winner'],row['ranking'])
            self.assertEqual(row['source_count'],1000);self.assertEqual(row['noise_count'],3)

    def test_last_source_and_all_three_noises_affect_winner(self):
        rows=copy.deepcopy(self.rows);cid=self.finalists['snrs']['19']['finalists'][-1]
        for row in rows:
            if row['snr_db']==19 and row['candidate_id']==cid and row['source_index']==999:
                row['dinov2_vitl14_cosine']=.9 if row['noise_seed']==4103 else .4
        result=c.winners(rows,self.records,self.finalists,self.policy)
        self.assertEqual(result['snrs']['19']['winner'],cid)
        self.assertAlmostEqual(result['snrs']['19']['means'][cid]['dinov2_vitl14_cosine'],(.1*999+1.7/3)/1000)

    def test_subset_duplicate_reorder_wrong_source_rejected(self):
        for rows in (self.rows[:100],self.rows[:-1],self.rows[:-1]+[self.rows[0]],list(reversed(self.rows))):
            with self.assertRaises(RuntimeError):c.winners(rows,self.records,self.finalists,self.policy)
        changed=copy.deepcopy(self.rows);changed[-1]['source_id']='new100'
        with self.assertRaises(RuntimeError):c.winners(changed,self.records,self.finalists,self.policy)

    def test_missing_metric_nan_and_nonbinary_agreement_rejected(self):
        for key,value in [('dinov2_vitl14_cosine',float('nan')),('convnext_top1_source_prediction',.5)]:
            changed=copy.deepcopy(self.rows);changed[0][key]=value
            with self.assertRaises(RuntimeError):c.winners(changed,self.records,self.finalists,self.policy)
        changed=copy.deepcopy(self.rows);changed[0].pop('lpips_alex')
        with self.assertRaises(KeyError):c.winners(changed,self.records,self.finalists,self.policy)

    def test_failure_and_negative_values_retained(self):
        changed=copy.deepcopy(self.rows);changed[0]['dinov2_vitl14_cosine']=-.8;changed[0]['psnr_db']=-1.
        result=c.winners(changed,self.records,self.finalists,self.policy)
        row=changed[0];self.assertLess(result['snrs'][str(row['snr_db'])]['means'][row['candidate_id']]['psnr_db'],0)


class Local:
    inside=staticmethod(Path)
    sha=staticmethod(lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest())
    image_sha=staticmethod(c.metric.array_sha)
    @staticmethod
    def write(path,value):Path(path).write_text(json.dumps(value))


class Ledger:
    def __init__(self,caps):self.caps=caps;self.values={k:0 for k in caps}
    def add(self,key,n=1):
        self.values[key]+=n
        if self.values[key]>self.caps[key]:raise RuntimeError('over cap')
    def summary(self):return dict(caps=self.caps,reserved=dict(self.values),completed=dict(self.values),unresolved=0)


class Evaluator:
    identity='a'*64
    def __init__(self,ledger):self.ledger=ledger;ledger.add('model_constructions',3)
    def prepare(self,target):
        for k in ('reference_preparations','dinov2_vitl14_reference','convnext_reference'):self.ledger.add(k)
        return {'fake':True}
    def score(self,target,image,class_index,prepared):
        for k in ('image_scores','dinov2_vitl14_reconstruction','convnext_reconstruction','lpips_pair'):self.ledger.add(k)
        self.ledger.add('lpips_alexnet_backbone_forward',2)
        return dict(psnr_db=2.,lpips_alex=.8,dinov2_vitl14_cosine=.1,convnext_top1_source_prediction=0)


class CacheTests(unittest.TestCase):
    def setup_scores(self,root,indices):
        image=np.full((3,256,256),.5,dtype=np.float32);path=root/'image.npz';np.savez(path,image=image)
        archive=c.core.descriptor(path);ish=c.metric.array_sha(image);data={};events=[];pins=[]
        for i,index in enumerate(indices):
            event=dict(source_index=index,source_id=f'original{index}',snr_db=4,noise_seed=4101,candidate_id='cid')
            pin={'path':str(i)};events.append(event);pins.append(pin)
            data[str(i)]=dict(frame_index=i,logical_event=event,reconstruction=dict(image_archive=archive,image_sha256=ish),
                actual_RX_status='HEADER_FAILURE' if i%2 else 'BODY_FAILURE',actual_received_profile=None,
                actual_header_ok=bool(i%2),actual_crc_accepted=False,actual_parser_accepted=False)
        rec=[dict(source_index=i,source_id=f'original{i}',archive={'path':str(i)},evaluation_class_index=0) for i in range(max(indices)+1)]
        request=dict(records=rec,logical_events=events,reconstruction_results=pins,finalists={},policy={'path':'policy'})
        data['policy']={};return request,data,image

    def test_closed_pair_reuse_does_not_construct_reference_features(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);r,data,image=self.setup_scores(root,[0,0]);ledger=Ledger(c.METRIC_CAPS);e=Evaluator(ledger)
            key=c.metric.pair_cache_key(image,image,e.identity);values=dict(psnr_db=1.,lpips_alex=.7,dinov2_vitl14_cosine=.2,convnext_top1_source_prediction=0)
            cache={key:(values,{'path':'oldpair'},'CLOSED_PILOT')}
            with mock.patch.object(c.full,'complete_rows'),mock.patch.object(c,'winners',return_value={}),mock.patch.object(c,'pilot_pairs',return_value=cache),mock.patch.object(c.core,'reference_pixels',return_value=image):
                result=c.scores(r,e,ledger,lambda:None,root/'scores',Local,lambda pin:data[pin['path']])
            self.assertEqual(result['actual_reference_preparations'],0)
            self.assertEqual(result['reuse'],dict(closed_pilot=2,same_run=0,new_actual_pairs=0))
            rows=json.loads(Path(result['metric_rows']['path']).read_text())
            self.assertNotEqual(rows[0]['actual_RX_status'],rows[1]['actual_RX_status'])

    def test_only_new_pairs_prepare_sources_and_same_run_reuses(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);r,data,image=self.setup_scores(root,[0,0,1]);ledger=Ledger(c.METRIC_CAPS);e=Evaluator(ledger)
            def target(record):return np.full_like(image,.1+record['source_index']*.1)
            with mock.patch.object(c.full,'complete_rows'),mock.patch.object(c,'winners',return_value={}),mock.patch.object(c,'pilot_pairs',return_value={}),mock.patch.object(c.core,'reference_pixels',side_effect=target):
                result=c.scores(r,e,ledger,lambda:None,root/'scores',Local,lambda pin:data[pin['path']])
            self.assertEqual(result['actual_reference_preparations'],2)
            self.assertEqual(result['reuse'],dict(closed_pilot=0,same_run=1,new_actual_pairs=2))

    def test_cross_identity_pilot_cache_is_empty_without_pixel_reads(self):
        with mock.patch.object(c.core,'reference_pixels',side_effect=AssertionError('no pixels')):
            self.assertEqual(c.pilot_pairs({'reuse_pilot_pairs':{'metric_identity':{}}},types.SimpleNamespace(identity='new'),Local,
                lambda _:dict(identity='old'),lambda:None),{})

    def test_pair_key_changes_for_either_pixel_or_provider(self):
        a=np.zeros((3,256,256),dtype=np.float32);b=a.copy();b[0,0,0]=.1
        keys=[c.metric.pair_cache_key(x,y,z) for x,y,z in [(a,a,'a'*64),(a,b,'a'*64),(b,a,'a'*64),(a,a,'b'*64)]]
        self.assertEqual(len(set(keys)),4)

    def test_archive_tamper_and_unresolved_counts_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'x.npz';image=np.zeros((3,256,256),dtype=np.float32);np.savez(path,image=image)
            pin=c.core.descriptor(path);path.write_bytes(b'changed')
            with self.assertRaises(RuntimeError):c.image_bytes(pin,c.metric.array_sha(image),Local)
        counts=Ledger(c.METRIC_CAPS).summary();counts['unresolved']=1
        with self.assertRaises(RuntimeError):c.check_counts(counts,c.METRIC_CAPS)

    def test_reused_gray_keeps_each_actual_header_and_body_failure(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);image=np.full((3,256,256),.5,dtype=np.float32)
            ip=root/'old.npz';np.savez(ip,image=image);archive=c.core.descriptor(ip);ish=c.metric.array_sha(image)
            identity={'host':'same'}
            rx1=dict(header={'header_ok':False},body=None,rx_profile=None,status='HEADER_FAILURE')
            rx2=dict(header={'header_ok':True},body={'crc_accepted':False,'parser_accepted':False},rx_profile=None,status='CRC_FAILURE')
            key=c.core.digest(c.core.source_input(rx1,identity));events=[{'source_index':0},{'source_index':1}]
            data={'policy':{},'oldrx':{'actual_RX':rx1},'packet0':{'actual_RX':rx1},'packet1':{'actual_RX':rx2},
                'old':dict(physical_frame={'path':'oldrx'},reconstruction=dict(image_archive=archive,image_sha256=ish,
                    actual_source_input_key=key,evidence={},origin='THIS_PILOT'))}
            for i in range(2):data[f'row{i}']=dict(frame_index=i,logical_event=events[i],physical_frame={'path':f'packet{i}'})
            r=dict(policy={'path':'policy'},records=[],finalists={},logical_events=events,visual_identity=identity,
                physical_frames=[{'path':'row0'},{'path':'row1'}],reuse_pilot_images=[{'path':'old'}])
            ledger=Ledger(c.VISUAL_CAPS);ledger.add('model_load')
            with mock.patch.object(c.full,'complete_rows'),mock.patch.object(c.core.link,'recover_and_render',side_effect=AssertionError('no new render')):
                result=c.images(r,None,None,None,ledger,lambda:None,root/'recon',Local,lambda pin:data[pin['path']])
            self.assertEqual(result['reuse'],dict(closed_pilot=2,same_run=0,new_actual_source_inputs=0))
            rows=[json.loads(Path(pin['path']).read_text()) for pin in result['logical_frames']]
            self.assertEqual([x['actual_RX_status'] for x in rows],['HEADER_FAILURE','CRC_FAILURE'])
            self.assertEqual([x['actual_header_ok'] for x in rows],[False,True])

    def test_closed_pair_requires_actual_source_and_image_bytes(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);target=np.zeros((3,256,256),dtype=np.float32);image=np.full_like(target,.5)
            ip=root/'old.npz';np.savez(ip,image=image);archive=c.core.descriptor(ip);ish=c.metric.array_sha(image)
            metadata={'provider':'unchanged'};identity=c.core.digest(metadata);key=c.metric.pair_cache_key(target,image,identity)
            values=dict(psnr_db=1.,lpips_alex=.7,dinov2_vitl14_cosine=.2,convnext_top1_source_prediction=0)
            row=dict(source_index=0,source_id='original0',metric_pair_key=key,metric_pair_result={'path':'pair'},
                reconstruction_result={'path':'recon'},**values)
            record=dict(source_index=0,source_id='original0',archive={'path':'source'})
            data={'identity':dict(identity=identity,metadata=metadata),'rows':[row]*43200,
                'pair':dict(metrics=values,metric_pair_key=key,metric_identity=identity,reference_archive=record['archive'],reconstruction_archive=archive),
                'recon':dict(reconstruction=dict(image_archive=archive,image_sha256=ish))}
            r=dict(records=[record],reuse_pilot_pairs=dict(metric_identity={'path':'identity'},metric_rows={'path':'rows'}))
            with mock.patch.object(c.core,'reference_pixels',return_value=target):
                cache=c.pilot_pairs(r,types.SimpleNamespace(identity=identity),Local,lambda pin:data[pin['path']],lambda:None)
                self.assertEqual(cache[key][0],values)
                data['pair']['reference_archive']={'path':'wrong-source'}
                with self.assertRaises(RuntimeError):c.pilot_pairs(r,types.SimpleNamespace(identity=identity),Local,lambda pin:data[pin['path']],lambda:None)


class OwnerTests(unittest.TestCase):
    def test_first_phy_request_binds_downstream_caps(self):
        caps=phy.scientific_caps();self.assertEqual(caps['full_visual'],c.VISUAL_CAPS)
        self.assertEqual(caps['full_metrics'],c.METRIC_CAPS)
        self.assertEqual(caps['stage_order'],['source900','fullPHY','fullvisual','fullmetrics'])

    def test_closed_attempt_exact_paths_required(self):
        for owner,fun in [(v,v.cpu_closure),(m,m.visual_closure)]:
            with mock.patch.object(owner.gate,'readpin',side_effect=AssertionError('must reject before reading')):
                with self.assertRaises(RuntimeError):fun({'completion':{'path':'somewhere'},'owner_actual_wait':{'path':'elsewhere'}})

    def test_live_window_is_bounded_and_engine_pins_hold(self):
        import time
        with self.assertRaises(RuntimeError):v.execution(time.time()+30,21601)
        with self.assertRaises(RuntimeError):m.execution(time.time()-1,1)
        self.assertEqual(Local.sha(phy.__file__),v.PHY_SCRIPT_SHA)
        self.assertEqual(Local.sha(c.__file__),v.EVALUATION_SHA)
        self.assertEqual(Local.sha(v.__file__),m.VISUAL_SHA)
        self.assertEqual(Local.sha(m.pilot_owner.__file__),m.PILOT_OWNER_SHA)

    def test_no_replay_and_no_automatic_successor_in_owners(self):
        import inspect
        for owner in (v,m):
            code=inspect.getsource(owner.run)
            self.assertIn('not out.exists()',code);self.assertIn('automatic_successor=False',code)
            self.assertIn('wait_owned',code);self.assertIn('child.wait()',code)
        self.assertEqual(c.PROVIDER_SCOPE,'original_calibration_pilot')
        self.assertEqual(c.POPULATION,'original_calibration_full1000')


if __name__=='__main__':unittest.main()

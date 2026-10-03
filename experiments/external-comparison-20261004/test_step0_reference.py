import copy
import json
import math
from pathlib import Path
import unittest
import numpy as np
from step0_cache_export import identity,pixel_hash
from step0_reference_prepare import CONDITIONS,mild_images,choose_donors
from step0_reference_metrics import clean_metrics,validate_checkpoint,analyze


class ReferencePreparationTests(unittest.TestCase):
    def test_mild_transforms_are_fixed_deterministic_and_independent(self):
        source=np.random.default_rng(17).random((3,256,256),dtype=np.float32)
        original=source.copy()
        images,proof=mild_images(source,'source/a')
        again,proof2=mild_images(source,'source/a')
        self.assertTrue(np.array_equal(source,original))
        self.assertEqual(proof,proof2)
        self.assertTrue(all(np.array_equal(a,b) for a,b in zip(images,again)))
        self.assertTrue(all(x.dtype==np.float32 and x.shape==source.shape and x.min()>=0 and x.max()<=1 for x in images))
        other,_=mild_images(source,'source/b')
        self.assertTrue(np.array_equal(images[0],other[0]))
        self.assertFalse(np.array_equal(images[1],other[1]))
        self.assertTrue(np.array_equal(images[2],other[2]))

    def test_blur_preserves_constant_and_reduces_impulse_without_channel_mix(self):
        source=np.zeros((3,256,256),dtype=np.float32);source[0,128,128]=1
        images,_=mild_images(source,'impulse')
        self.assertAlmostEqual(float(images[0][0].sum()),1.,places=6)
        self.assertEqual(float(images[0][1:].sum()),0.)
        self.assertGreater(float(images[0][0,128,128]),.1)
        self.assertLess(float(images[0][0,128,128]),.2)

    def test_complete_population_selection_metadata_only(self):
        registration={'source_image_bindings':{},'train_ids':[],'calibration_ids':[]}
        manifest={'populations':[]}
        for role,count in (('train',20000),('calibration',1000)):
            members=[dict(image_id=f'{role}/{i}',class_index=i%1000,shard=role+'.pt',index_in_shard=i) for i in range(count)]
            registration['source_image_bindings'][role]=members
            registration['train_ids' if role=='train' else 'calibration_ids']=[x['image_id'] for x in members]
            manifest['populations'].append(dict(name=role,count=count,shards=[dict(path=role+'.pt',sha256='a'*64)]))
        sources=[dict(source_index=i,image_id=f'dev/{i}',class_index=i) for i in range(100)]
        chosen=choose_donors(sources,registration,manifest)
        self.assertEqual(chosen,choose_donors(sources,registration,manifest))
        self.assertEqual(len(chosen),100)
        self.assertTrue(all(r['donor']['class_index']==r['source_index'] for r in chosen))
        registration['source_image_bindings']['train'][0]['shard']='holdout.pt'
        with self.assertRaises(RuntimeError):choose_donors(sources,registration,manifest)


class ReferenceMetricTests(unittest.TestCase):
    def test_confident_error_boundary_and_infinity_serialization(self):
        old=dict(psnr_db=math.inf,lpips_alex=0.,dino_cosine=1.)
        new=dict(resnet50_prediction=1,resnet50_source_prediction=0,resnet50_top1_source_prediction=0)
        score=clean_metrics(old,new,.5)
        self.assertEqual(score['confidently_wrong'],1)
        self.assertIsNone(score['psnr_db']);self.assertTrue(score['psnr_infinite'])
        self.assertIn('null',json.dumps(score,allow_nan=False))
        self.assertEqual(clean_metrics(old,new,.499999)['confidently_wrong'],0)
        new.update(resnet50_prediction=0,resnet50_top1_source_prediction=1)
        self.assertEqual(clean_metrics(old,new,1.)['confidently_wrong'],0)

    def test_negative_infinity_and_nan_probability_rejected(self):
        native=dict(psnr_db=-math.inf,lpips_alex=0.,dino_cosine=1.)
        newer=dict(resnet50_prediction=1,resnet50_source_prediction=0,resnet50_top1_source_prediction=0)
        with self.assertRaises(RuntimeError):clean_metrics(native,newer,.7)
        native['psnr_db']=20
        with self.assertRaises(RuntimeError):clean_metrics(native,newer,math.nan)

    def test_checkpoint_exact_condition_scope_and_true_no_channel(self):
        reg={'x':1};entry=dict(source_index=0,source_id='source',class_index=1,reference_sha256='ref',
            cache=dict(sha256='cache',image_sha256=['image'+str(i) for i in range(6)]))
        rows=[dict(source_id='source',class_index=1,condition=condition,reference_sha256='ref',
            image_sha256='image'+str(i),N=None,snr_db=None,noise_seed=None,resnet50_prediction=1,
            resnet50_source_prediction=1,semantic_error=0,confidently_wrong=0,resnet50_top1_probability=.8,
            psnr_db=None if i==0 else 20.,psnr_infinite=i==0) for i,condition in enumerate(CONDITIONS)]
        cp=dict(registration_sha256=identity(reg),source_index=0,input_cache_sha256='cache',rows=rows)
        cp['payload_sha256']=identity(cp)
        self.assertEqual(validate_checkpoint(cp,reg,entry),rows)
        cp['rows'][0]['N']=0;cp['payload_sha256']=identity({k:v for k,v in cp.items() if k!='payload_sha256'})
        with self.assertRaisesRegex(RuntimeError,'channel result'):validate_checkpoint(cp,reg,entry)

    def test_summary_600_rows_no_triplication_and_100_source_bootstrap(self):
        rows=[]
        for index in range(100):
            for condition in CONDITIONS:
                row=dict(source_index=index,condition=condition,psnr_db=None if condition=='identity' else 20.,
                    lpips_alex=.2,dino_cosine=.5,clip_image_cosine=.6,dists=.3,dreamsim=.4,
                    dinov2_vitl14_cosine=.5,ms_ssim=.7,resnet50_top1_label=1,
                    resnet50_top1_source_prediction=1,semantic_error=0,confidently_wrong=0)
                rows.append(row)
        summary=analyze(rows)
        self.assertEqual(len(summary),72)
        self.assertEqual(sum(x['positive_infinity'] for x in summary),1)
        self.assertTrue(all(x['sources']==100 for x in summary))
        with self.assertRaises(RuntimeError):analyze(rows*3)


if __name__=='__main__':unittest.main()

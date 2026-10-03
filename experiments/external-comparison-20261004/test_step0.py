import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from step0_cache_export import (identity, pixel_hash, selected_row, verified_arrays, sha, seal)
from step0_reference_protocol import unrelated_pairs, select_same_class
from step0_render import render, GROUPS


def fixture(folder):
    source = np.zeros((3,256,256),dtype=np.float32)
    image = np.full_like(source,.25)
    expected = dict(source_index=0,image_id='class0/source',class_index=0,
                    preprocessing_id=hashlib.sha256(np.zeros_like(source,dtype=np.uint8).tobytes()).hexdigest())
    registration = dict(source_identity=[{k:v for k,v in expected.items() if k!='source_index'}],
        modelmanifest_sha256='metric-model',evaluator_identity='evaluator',numerical_runtime={'dtype':'float32'},
        metric_batch_size=1,batch_qualification_sha256='batch',metric_qualified_batch_sizes=[1])
    original = dict(method='P2048',N='2048',snr_db='7',noise_seed='2001',psnr_db='20',lpips_alex='.2')
    row = dict(**original,history_row_id='row1',history_study='FINAL_P2048_P3060',history_source_index=0,
        history_source_id=expected['image_id'],history_preprocessing_id=expected['preprocessing_id'],history_true_class_index=0,
        history_original_row_sha256=identity(original),history_image_sha256=pixel_hash(image),
        history_reference_sha256=pixel_hash(source),history_replay_parity_passed=True,
        history_metadata_json=json.dumps(dict(method_id='P2048',N=2048,decoder='Dc',label_conditioned=False,training_seed=2026092304)),
        new_dinov2_vitl14_cosine=.6,new_resnet50_prediction=1,new_resnet50_source_prediction=0,
        new_resnet50_top1_source_prediction=0)
    path=folder/'0000.npz'
    np.savez_compressed(path,images=image[None],source_rgb=source,row_ids=np.asarray(['row1']),image_slots=np.asarray([0]))
    proof=dict(path=str(path),sha256=sha(path),dtype='float32',layout='CHW',rows=1,row_ids=['row1'],image_slots=[0],
        unique_images=1,image_sha256=[pixel_hash(image,False)],reference_sha256=pixel_hash(source,False))
    cp=dict(binding=identity(registration),source_index=0,rows=[row],float_reconstructions=proof,
        evaluation_identity={k:registration[k] for k in ('modelmanifest_sha256','evaluator_identity','numerical_runtime',
            'metric_batch_size','batch_qualification_sha256','metric_qualified_batch_sizes')},
        metric_batch_sizes_used=[1],unique_images=1,
        parity=[dict(history_row_id='row1',replay_parity_passed=True,synthetic=False)])
    cp['payload_sha256']=identity(cp)
    return cp,registration,expected,path


def reseal(cp):
    cp['payload_sha256']=identity({k:v for k,v in cp.items() if k!='payload_sha256'})


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.cp,self.reg,self.source,self.path=fixture(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def verify(self):
        return verified_arrays(self.cp,self.reg,self.source,'FINAL_P2048_P3060',self.path)

    def test_real_256_dual_hash_and_slots(self):
        target,images,slots=self.verify()
        self.assertEqual(slots,{'row1':0})
        self.assertEqual(target.shape,(3,256,256))
        self.assertNotEqual(pixel_hash(images[0]),pixel_hash(images[0],False))
        self.assertEqual(selected_row(self.cp['rows'][0],'FINAL_P2048_P3060')['semantic_error'],1)

    def test_resealed_metric_or_original_field_changes_rejected(self):
        self.cp['rows'][0]['psnr_db']='99';reseal(self.cp)
        with self.assertRaisesRegex(RuntimeError,'Original scientific row'):
            self.verify()

    def test_broken_row_pixel_hash_rejected(self):
        self.cp['rows'][0]['history_image_sha256']='0'*64;reseal(self.cp)
        with self.assertRaisesRegex(RuntimeError,'Scientific row RGB'):
            self.verify()

    def test_container_tampering_rejected(self):
        with self.path.open('ab') as stream:stream.write(b'changed')
        with self.assertRaisesRegex(RuntimeError,'container'):
            self.verify()

    def test_invalid_slot_rejected(self):
        self.cp['float_reconstructions']['image_slots']=[-1];reseal(self.cp)
        with self.assertRaises(RuntimeError):self.verify()

    def test_source_manifest_change_rejected(self):
        self.source['image_id']='other'
        with self.assertRaisesRegex(RuntimeError,'Frozen source'):
            self.verify()

    def test_only_registered_legacy_preprocessing_alias_with_exact_raw_pixels(self):
        self.reg['source_identity'][0]['preprocessing_id']='original_rounded_uint8_div255_RGB'
        self.cp['binding']=identity(self.reg)
        row=self.cp['rows'][0];row['history_study']='LEGACY_N3060_FINAL'
        row['history_preprocessing_id']='original_rounded_uint8_div255_RGB';reseal(self.cp)
        verified_arrays(self.cp,self.reg,self.source,'LEGACY_N3060_FINAL',self.path)
        self.source['preprocessing_id']='f'*64
        with self.assertRaisesRegex(RuntimeError,'source pixels'):
            verified_arrays(self.cp,self.reg,self.source,'LEGACY_N3060_FINAL',self.path)
        self.source['preprocessing_id']='original_rounded_uint8_div255_RGB'
        with self.assertRaises(RuntimeError):self.verify()

    def test_study_source_and_parity_required(self):
        self.cp['rows'][0]['history_source_id']='other';reseal(self.cp)
        with self.assertRaisesRegex(RuntimeError,'source metadata'):
            self.verify()

    def test_old_protocol_class_label_and_seed_explicit(self):
        row=copy.deepcopy(self.cp['rows'][0])
        row.update(method='raw',history_metadata_json=json.dumps(dict(method_id='final_raw_QPSK_N2048_Dc',N=2048,
            decoder_id='Dc',label_conditioned=True,condition='C')))
        result=selected_row(row,'FINAL_DIGITAL_QPSK')
        self.assertTrue(result['true_class_paid'])
        self.assertFalse(result['classification_main_eligible'])
        self.assertEqual(result['group'],'Digital raw / Dc')
        row['history_metadata_json']=json.dumps(dict(method_id='D_U_QPSK',N=2048,decoder='Dc',label_conditioned=False))
        self.assertIsNone(selected_row(row,'FINAL_DIGITAL_QPSK'))
        row=self.cp['rows'][0].copy();row['noise_seed']='2002'
        self.assertIsNone(selected_row(row,'FINAL_P2048_P3060'))

    def test_changed_class_condition_rejected(self):
        row=self.cp['rows'][0].copy()
        meta=json.loads(row['history_metadata_json']);meta['label_conditioned']=True
        row['history_metadata_json']=json.dumps(meta)
        with self.assertRaisesRegex(RuntimeError,'conditioning'):
            selected_row(row,'FINAL_P2048_P3060')

    def test_seal_never_overwrites_different_protocol(self):
        path=self.root/'sealed.json'
        seal(path,{'a':1});seal(path,{'a':1})
        with self.assertRaises(RuntimeError):seal(path,{'a':2})


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.records=[dict(image_id=f'dev/{i}',class_index=i,source_pixels_sha256=f'{i:064x}') for i in range(100)]
        self.donors=dict(status='VERIFIED_TRAIN_CALIBRATION_MEMBERSHIP',holdout_access=False,verified_before_metrics=True,
            membership_bindings={'old_train_manifest.json':'f'*64},records=[dict(image_id=f'train/{i}',population='train',
                membership_verified=True,class_index=i,preprocessing_id=f'{1000+i:064x}',path=f'train/{i}.npy',file_sha256='a'*64)
                for i in range(100)])

    def test_unrelated_reproducible_derangement_not_quality_selection(self):
        first,attempts=unrelated_pairs(self.records)
        self.assertEqual((first,attempts),unrelated_pairs(self.records))
        self.assertEqual({r['donor_source_index'] for r in first},set(range(100)))
        self.assertTrue(all(r['source_index']!=r['donor_source_index'] for r in first))

    def test_same_class_independent_of_pool_order(self):
        expected=select_same_class(self.records,self.donors)
        self.donors['records'].reverse()
        self.assertEqual(expected,select_same_class(self.records,self.donors))
        self.assertEqual(len(expected[0]),100)
        self.assertEqual(expected[1],[])

    def test_holdout_or_development_donor_rejected(self):
        self.donors['records'][0]['population']='holdout'
        with self.assertRaises(RuntimeError):select_same_class(self.records,self.donors)
        self.donors['records'][0]['population']='train'
        self.donors['records'][0]['image_id']='dev/0'
        with self.assertRaises(RuntimeError):select_same_class(self.records,self.donors)

    def test_missing_class_preserved_as_missing(self):
        self.donors['records']=self.donors['records'][1:]
        selected,missing=select_same_class(self.records,self.donors)
        self.assertEqual(len(selected),99)
        self.assertEqual(missing,[dict(source_index=0,source_id='dev/0',class_index=0)])

    def test_pixel_duplicate_rejected_despite_distinct_id(self):
        self.donors['records'][0]['preprocessing_id']=self.records[0]['source_pixels_sha256']
        with self.assertRaises(RuntimeError):select_same_class(self.records,self.donors)


class RenderTests(unittest.TestCase):
    def test_cpu_resource_grid_preserves_missing_points_and_png_hash(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); name='fixture.png'
            rgb=np.zeros((256,256,3),dtype=np.uint8)
            rgb[:,:,0]=np.arange(256,dtype=np.uint8)[None,:]
            Image.fromarray(rgb).save(root/name)
            source=dict(source_index=0,image_id='synthetic/unit-test',class_index=0,file=name,png_sha256=sha(root/name))
            rows=[]
            for group in GROUPS:
                for n in ((3060,) if group.startswith('Legacy') else (2048,3060,4084)):
                    rows.append(dict(source_index=0,snr_db=7,group=group,N=n,file=name,png_sha256=sha(root/name),
                        resnet50_source_prediction=0,resnet50_prediction=1,semantic_error=1,
                        metrics=dict(psnr_db=20.,lpips_alex=.2,dinov2_vitl14_cosine=.6)))
            manifest=dict(status='EXACT_COMPLETED_FLOAT_RGB_CPU_EXPORT_PASS',sources=[source],frames=rows,
                          source_indices=[0],snrs_db=[7],budgets=[2048,3060,4084])
            seal(root/'step0_manifest.json',manifest)
            result=render(root,root/'figures')
            self.assertEqual(len(result['figures']),2)
            self.assertEqual(render(root,root/'figures'),result)
            with Image.open(root/'figures/step0_source000_snr7.png') as image:
                self.assertGreater(image.width,1500);self.assertGreater(image.height,2000)
            Image.fromarray(np.zeros_like(rgb)).save(root/name)
            with self.assertRaisesRegex(RuntimeError,'PNG identity'):
                render(root,root/'bad_figures')


if __name__=='__main__':unittest.main()

"""Engineering fixtures only; no historical images or GPU evaluation."""
import copy
import csv
import importlib.util
import json
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

spec = importlib.util.spec_from_file_location('engineering_historical_cached', Path(__file__).with_name('historical_cached.py'))
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)

class CacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.manifest = self.root / 'cached.json'
        self.rows = []
        self.records = []
        self.files = []
        for i in range(2):
            target = np.full((3, 4, 4), .7 + i * .1, np.float32)
            images = np.stack([np.full_like(target, .2), np.full_like(target, .4)])
            path = self.root / f'cache{i}.npz'
            np.savez(path, images=images, source_rgb=target)
            self.files.append(path)
            self.records.append(dict(source_index=i, image_id=f'ENGINEERING_{i}', class_index=i,
                preprocessing_id='ENGINEERING_CHW', target_rgb_sha256=h.pixel_sha(target),
                target=dict(path=str(path), key='source_rgb')))
            for slot, arm in [(0,'base'),(1,'control'),(1,'alias')]:
                metric = self.quality(target, [images[slot]])[0]
                self.rows.append(dict(image_index=str(i), image_id=f'ENGINEERING_{i}', arm=arm,
                    snr_db='7.0', seed='2001', image_archive=str(path), image_ref=str(slot),
                    image_sha256=h.pixel_sha(images[slot]), **metric))
        table = self.root / 'per_frame.csv'
        with table.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(self.rows[0]));w.writeheader();w.writerows(self.rows)
        self.files.append(table)
        receipt = self.root / 'complete.json'
        receipt.write_text(json.dumps(dict(status='ENGINEERING_COMPLETE', output_hashes={str(p):h.sha(p) for p in self.files})))
        self.receipt = receipt
        proof = [dict(path=str(p),sha256=h.sha(p),receipt=str(receipt),receipt_sha256=h.sha(receipt),
            pointer='/output_hashes/'+str(p).replace('~','~0').replace('/','~1')) for p in self.files]
        self.document = dict(status='REGISTERED_ORIGINAL_FLOAT_CACHES', studies={'FIXTURE':dict(
            synthetic=True,format='indexed_npz',training_updates=0,policy_selection_updates=0,
            proofs=proof,completion=dict(path=str(receipt),sha256=h.sha(receipt),allowed_statuses=['ENGINEERING_COMPLETE']),
            records=self.records,source_count=2,tables=[dict(path=str(table))],expected_rows=6,rows_per_source=[3,3],
            source_id_column='image_id',source_index_column='image_index',method_key_columns=['arm'],
            archive_column='image_archive',slot_column='image_ref',image_hash_column='image_sha256',
            metric_columns={k:k for k in h.METRICS},methods={m:dict(method=m,model_id='ENGINEERING_MODEL',decoder='ENGINEERING_DECODER',
                label_conditioned=False,reference_only=False,oracle=False) for m in ['base','control','alias']})})
        self.save()

    def save(self):
        self.manifest.write_text(json.dumps(self.document))

    @staticmethod
    def quality(target, images):
        return [dict(psnr_db=float(-10*np.log10(np.mean((image-target)**2))),
            lpips_alex=float(np.mean(abs(image-target))), dino_cosine=float(np.mean(image*target))) for image in images]

    def adapter(self):
        return h.CachedAdapter(self.root,'FIXTURE',manifest=self.manifest,quality_checker=self.quality,engineering=True).setup()

    def test_complete_native_float_alias_rows_and_per_source_target(self):
        a = self.adapter()
        for i in range(2):
            output=list(a.iterate_source(i))
            self.assertEqual(len(output),3)
            self.assertEqual(a.expected_rows(i), [dict(r) for r in a._by_source[i]])
            for row, image, target, parity in output:
                self.assertTrue(parity['replay_parity_passed'])
                self.assertIs(parity['synthetic'],False)
                self.assertEqual(image.dtype,np.float32)
                self.assertEqual(h.pixel_sha(target),self.records[i]['target_rgb_sha256'])
                self.assertNotIn('method',row)
            np.testing.assert_array_equal(output[1][1],output[2][1])
        self.assertFalse(a.metadata(a.rows[0])['label_conditioned'])
        self.assertEqual(h.sha(self.files[-1]),a.bindings[str(self.files[-1])])

    def test_factory_has_no_engineering_bypass(self):
        path=self.root/'outputs/HISTORICAL-METRICS-20261003/cached_studies.json'
        path.parent.mkdir(parents=True);path.write_bytes(self.manifest.read_bytes())
        with self.assertRaisesRegex(RuntimeError,'synthetic'):
            h.create_adapter(self.root,'FIXTURE')

    def test_selected_methods_filter_before_loading_or_scoring(self):
        s = self.document['studies']['FIXTURE']
        s.update(selected_method_keys=['base'], methods={'base':s['methods']['base']},
                 expected_rows=2, rows_per_source=[1,1])
        self.save()
        a = self.adapter()
        self.assertEqual(len(a.rows), 2)
        seen = []
        checker = a._checker
        def tracked(target, images):
            seen.extend(h.pixel_sha(x) for x in images)
            return checker(target, images)
        a._checker = tracked
        result = list(a.iterate_source(0))
        self.assertEqual([r[0]['arm'] for r in result], ['base'])
        self.assertEqual(len(seen), 1)
        self.assertEqual(result[0][3]['original_locations'][0]['row_index'], 0)

    def test_archive_mutation_fails_before_read(self):
        a=self.adapter()
        with self.files[0].open('ab') as f:f.write(b'changed')
        with self.assertRaisesRegex(RuntimeError,'Archive changed'):
            list(a.iterate_source(0))

    def test_not_original_receipt_member_rejected(self):
        self.document['studies']['FIXTURE']['proofs'][0]['pointer']='/status';self.save()
        with self.assertRaisesRegex(RuntimeError,'original receipt member'):self.adapter()

    def test_missing_cache_receipt_proof_rejected(self):
        self.document['studies']['FIXTURE']['proofs']=self.document['studies']['FIXTURE']['proofs'][1:];self.save()
        with self.assertRaisesRegex(RuntimeError,'lacks original receipt'):self.adapter()

    def test_incomplete_completion_rejected(self):
        self.document['studies']['FIXTURE']['completion']['allowed_statuses']=['OTHER_COMPLETE'];self.save()
        with self.assertRaisesRegex(RuntimeError,'incomplete'):self.adapter()

    def test_changed_source_target_rejected(self):
        self.document['studies']['FIXTURE']['records'][0]['target_rgb_sha256']='0'*64;self.save()
        with self.assertRaisesRegex(RuntimeError,'target/preprocessing'):list(self.adapter().iterate_source(0))

    def test_missing_original_row_or_wrong_scope_rejected(self):
        self.document['studies']['FIXTURE']['expected_rows']=7;self.save()
        with self.assertRaisesRegex(RuntimeError,'row count'):self.adapter()

    def test_quality_value_and_missing_metric_rejected(self):
        old=dict(psnr_db=12,lpips_alex=.1,dino_cosine=.9)
        columns={k:k for k in h.METRICS}
        with self.assertRaisesRegex(RuntimeError,'differs'):h.parity(old,{**old,'dino_cosine':.8},columns)
        with self.assertRaisesRegex(RuntimeError,'missing'):h.parity(old,{'psnr_db':12},columns)
        with self.assertRaisesRegex(RuntimeError,'Nonfinite'):h.parity(old,{**old,'dino_cosine':float('nan')},columns)

    def test_tolerance_cannot_be_relaxed(self):
        old=dict(psnr_db=12,lpips_alex=.1,dino_cosine=.9)
        with self.assertRaisesRegex(RuntimeError,'exceeds'):
            h.parity(old,old,{k:k for k in h.METRICS},{k:1 for k in h.METRICS})

    def test_png_uint8_nonfinite_wrongshape_rejected(self):
        for value in [np.zeros((3,4,4),np.uint8),np.zeros((4,4,3),np.float32),np.full((3,4,4),np.nan,np.float32),np.full((3,4,4),1.01,np.float32)]:
            with self.assertRaises(RuntimeError):h.rgb(value)

    def test_conditioning_metadata_mandatory(self):
        del self.document['studies']['FIXTURE']['methods']['base']['label_conditioned'];self.save()
        with self.assertRaisesRegex(RuntimeError,'facets'):self.adapter()

    def test_no_training_or_selection(self):
        self.document['studies']['FIXTURE']['training_updates']=1;self.save()
        with self.assertRaisesRegex(RuntimeError,'cannot train'):self.adapter()

    def test_latent_matrix_exact_grid(self):
        # Separate engineering fixture has one method x one condition per source.
        for i,path in enumerate(self.files[:2]):
            with np.load(path) as a:target=a['source_rgb'].copy();image=a['images'][0].copy()
            np.savez(path,images=image[None,None],source_rgb=target,method_order=np.asarray(['base']))
            self.records[i]['matrix_archive']=str(path)
        table=self.files[-1]
        only=[r for r in self.rows if r['arm']=='base']
        with table.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(only[0]));w.writeheader();w.writerows(only)
        receipt=dict(status='ENGINEERING_COMPLETE',output_hashes={str(p):h.sha(p) for p in self.files})
        self.receipt.write_text(json.dumps(receipt))
        s=self.document['studies']['FIXTURE'];s.update(format='latent_matrix_npz',method_order=['base'],
            frame_order=[[7,2001]],snr_column='snr_db',seed_column='seed',expected_rows=2,rows_per_source=[1,1])
        s['completion']['sha256']=h.sha(self.receipt)
        for p in s['proofs']:p.update(sha256=h.sha(p['path']),receipt_sha256=h.sha(self.receipt))
        self.save();a=self.adapter()
        self.assertEqual(len(list(a.iterate_source(0))),1)
        s['frame_order']=[[4,2001]];self.save()
        with self.assertRaisesRegex(RuntimeError,'grid differs'):self.adapter()

    def test_reference_prefix_dispatch_to_proven_archive(self):
        s=self.document['studies']['FIXTURE'];s['reference_roots']={'input':str(self.root/'cache{source_index}.npz')}
        self.save()
        a=self.adapter();row=dict(a.rows[0],image_ref='input:0')
        self.assertEqual(a.locator(0,row)[0],self.files[0])
        with self.assertRaisesRegex(RuntimeError,'Unknown historical'):
            a.locator(0,dict(row,image_ref='unknown:0'))

    def pixel_fixture(self):
        table=self.files[-1]
        for row in self.rows:
            row['source_rgb_sha256']=self.records[int(row['image_index'])]['target_rgb_sha256']
        with table.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(self.rows[0]));w.writeheader();w.writerows(self.rows)
        self.receipt.write_text(json.dumps(dict(status='ENGINEERING_COMPLETE',output_hashes={str(table):h.sha(table)})))
        s=self.document['studies']['FIXTURE']
        s['completion']['sha256']=h.sha(self.receipt)
        s['proofs']=[dict(path=str(table),sha256=h.sha(table),receipt=str(self.receipt),receipt_sha256=h.sha(self.receipt),
            pointer='/output_hashes/'+str(table).replace('~','~0').replace('/','~1'))]
        s['pixel_proofs']=[]
        for i,path in enumerate(self.files[:2]):
            for key,indices,column,position in [('source_rgb',[],'source_rgb_sha256',i*3),('images',[0],'image_sha256',i*3),('images',[1],'image_sha256',i*3+1)]:
                s['pixel_proofs'].append(dict(path=str(path),sha256=h.sha(path),receipt=str(table),receipt_sha256=h.sha(table),
                    row_index=position,column=column,key=key,indices=indices,pixel_sha256=self.rows[position][column],hash_kind='raw'))
        self.save()

    def test_original_pixel_hash_proof_without_container_hash_receipt(self):
        self.pixel_fixture();a=self.adapter()
        self.assertEqual(len(list(a.iterate_source(0))),3)
        self.assertTrue(a.load_image(0,a.rows[0])[2]['cached_pixels_verified'])

    def test_original_pixel_proof_required_for_every_accessed_slot(self):
        self.pixel_fixture()
        s=self.document['studies']['FIXTURE'];s['pixel_proofs']=[p for p in s['pixel_proofs'] if p['indices']!=[1]]
        self.save()
        with self.assertRaisesRegex(RuntimeError,'Unbound pixels'):list(self.adapter().iterate_source(0))

    def test_fresh_container_hash_cannot_replace_original_pixel_identity(self):
        self.pixel_fixture()
        p=self.files[0]
        with np.load(p) as a:images=a['images'].copy();target=a['source_rgb'].copy()
        images[0]+=np.float32(.01);np.savez(p,images=images,source_rgb=target)
        for proof in self.document['studies']['FIXTURE']['pixel_proofs']:
            if proof['path']==str(p):proof['sha256']=h.sha(p)
        self.save()
        with self.assertRaisesRegex(RuntimeError,'Original pixel proof'):list(self.adapter().iterate_source(0))

    def test_nested_historical_reference_is_exact(self):
        s=self.document['studies']['FIXTURE'];s['reference_roots']={'frozen:previous:new':str(self.root/'cache{source_index}.npz')}
        self.save();a=self.adapter()
        path,key,indices=a.locator(0,dict(a.rows[0],image_ref='frozen:previous:new:1'))
        self.assertEqual(path,self.files[0]);self.assertEqual(indices,[1])

    def test_jsonl_original_rows_preserve_native_types(self):
        table=self.root/'per_image.jsonl'
        typed=[]
        for original in self.rows:
            row=dict(original,image_index=int(original['image_index']),source_only=True,details={'original':'kept'})
            typed.append(row)
        table.write_text(''.join(json.dumps(r)+'\n' for r in typed))
        receipt=dict(status='ENGINEERING_COMPLETE',output_hashes={str(p):h.sha(p) for p in [*self.files[:2],table]})
        self.receipt.write_text(json.dumps(receipt));s=self.document['studies']['FIXTURE']
        s['completion']['sha256']=h.sha(self.receipt)
        s['proofs']=[dict(path=str(p),sha256=h.sha(p),receipt=str(self.receipt),receipt_sha256=h.sha(self.receipt),
            pointer='/output_hashes/'+str(p).replace('~','~0').replace('/','~1')) for p in [*self.files[:2],table]]
        s['tables']=[dict(path=str(table),table_format='jsonl')]
        self.save();a=self.adapter();output=list(a.iterate_source(0))
        self.assertIs(output[0][0]['source_only'],True)
        self.assertIsInstance(output[0][0]['image_index'],int)
        self.assertEqual(output[0][0]['details'],{'original':'kept'})

    def test_typed_tensor_proof_uses_original_dtype_shape_and_bytes(self):
        self.pixel_fixture()
        s=self.document['studies']['FIXTURE'];p=s['pixel_proofs'][0]
        with np.load(p['path']) as a:value=a[p['key']]
        digest=hashlib.sha256();digest.update(b'torch.float32');digest.update(json.dumps([1,*value.shape]).encode());digest.update(value.tobytes())
        self.rows[p['row_index']][p['column']]=digest.hexdigest()
        table=self.files[-1]
        with table.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(self.rows[0]));w.writeheader();w.writerows(self.rows)
        self.receipt.write_text(json.dumps(dict(status='ENGINEERING_COMPLETE',output_hashes={str(table):h.sha(table)})))
        s['completion']['sha256']=h.sha(self.receipt)
        for proof in s['proofs']:proof.update(sha256=h.sha(table),receipt_sha256=h.sha(self.receipt))
        for proof in s['pixel_proofs']:proof['receipt_sha256']=h.sha(table)
        p.update(pixel_sha256=digest.hexdigest(),hash_kind='typed_tensor',dtype='torch.float32',shape=[1,*value.shape])
        self.save();self.assertEqual(len(list(self.adapter().iterate_source(0))),3)
        p['dtype']='torch.float64';self.save()
        with self.assertRaisesRegex(RuntimeError,'Original pixel proof'):list(self.adapter().iterate_source(0))

    def test_negative_scientific_result_is_a_completed_experiment(self):
        sys.path.insert(0,str(Path(__file__).parent))
        try:
            import build_cached_studies as builder
            b=builder.Builder.__new__(builder.Builder);b.hashes={};b.receipts={};b.proof_index={}
            path=self.root/'completion.json'
            for state in ['MATCHED_MECHANISM_ONLY_STRONG_CONTROLS_NOT_BEATEN','STOP_THIS_FINITE_PREFIX_VAR_RECEIVER']:
                b.hashes={};b.receipts={};path.write_text(json.dumps({'status':state}))
                self.assertEqual(b.completion(self.root)['allowed_statuses'],[state])
            for state in ['INCOMPLETE','RUNNING','BUDGET_RUNNING_NOT_COMPLETE']:
                b.hashes={};b.receipts={};path.write_text(json.dumps({'status':state}))
                with self.assertRaisesRegex(RuntimeError,'Unregistered historical completion'):b.completion(self.root)
        finally:sys.path.pop(0)

if __name__=='__main__':unittest.main()

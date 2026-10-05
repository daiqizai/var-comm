"""Pure mapping/cache tests, with fake CPU shards and tiny codec arrays."""
import copy
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np

spec=importlib.util.spec_from_file_location('full_assets_tested',Path(__file__).with_name('full1000_assets.py'))
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)


class AssetsTests(unittest.TestCase):
    def test_owner_precreated_empty_output_accepts_once_and_prior_artifact_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/'owner_precreated';out.mkdir()
            a.claim_empty_output(out,'a'*64)
            attempt=(out/'attempt.json').read_bytes()
            self.assertEqual(a.read(out/'attempt.json')['registration_sha256'],'a'*64)
            with self.assertRaisesRegex(RuntimeError,'independent recovery'):a.claim_empty_output(out,'a'*64)
            self.assertEqual((out/'attempt.json').read_bytes(),attempt)
            other=Path(directory)/'prior';other.mkdir();(other/'source.npz').write_bytes(b'evidence')
            with self.assertRaisesRegex(RuntimeError,'independent recovery'):a.claim_empty_output(other,'b'*64)
            self.assertFalse((other/'attempt.json').exists())
            fresh=Path(directory)/'fresh';a.claim_empty_output(fresh,'c'*64)
            self.assertTrue((fresh/'attempt.json').is_file())

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.pixels=np.zeros((3,256,256),np.uint8);self.pre=hashlib.sha256(self.pixels.tobytes()).hexdigest()
        self.ids=[f'source{i:04d}' for i in range(1000)];self.old_ids=list(reversed(self.ids[50:250]))
        self.model={'models':{'vae':'v','var':'a','decoder':'d'}}
        self.cal=dict(stage='m1_calibration',calibration_or_development='m1_calibration',source_ids=self.ids.copy(),
            preprocessing_ids=[self.pre]*1000,identity=self.model,
            data_bindings=dict(latents={},pixels=[]))
        self.manifest_path=self.root/'pixels/manifest.json';self.manifest={'populations':[{'name':'calibration','count':1000,'shards':[]},
            {'name':'development','count':100,'shards':[{'path':'development/DO_NOT_READ.pt'}]}]}
        self.shards={};self.calls=[]
        for s in range(10):
            rel=f'calibration/shard_{s:04d}.pt';lp=self.root/'latents'/rel;pp=self.root/'pixels'/rel
            lp.parent.mkdir(parents=True,exist_ok=True);pp.parent.mkdir(parents=True,exist_ok=True)
            lp.write_bytes(f'latent{s}'.encode());pp.write_bytes(f'pixels{s}'.encode())
            a.save(lp.with_suffix('.json'),{'sha256':a.sha(lp)})
            self.cal['data_bindings']['latents'][str(lp)]=a.sha(lp)
            self.manifest['populations'][0]['shards'].append({'path':rel,'count':100,'sha256':a.sha(pp)})
            shard_ids=self.ids[s*100:(s+1)*100]
            self.shards[str(lp)]={'image_ids':shard_ids,'full_tokens':np.arange(s*100,(s+1)*100,dtype=np.int64)[:,None]+np.zeros((100,680),np.int64)}
            self.shards[str(pp)]={'image_ids':shard_ids,'labels':np.arange(s*100,(s+1)*100),
                                 'targets_u8':np.broadcast_to(self.pixels,(100,3,256,256))}
            for j in range(100):self.cal['data_bindings']['pixels'].append({'shard':rel,'index_in_shard':j,'image_id':shard_ids[j],'class_index':s*100+j})
        self.plan=a.build_population_plan(self.cal,self.manifest,self.manifest_path,self.old_ids,self.model)

    def loader(self,path):
        self.calls.append(str(path));return self.shards[str(path)]

    def test_reuse_is_by_id_not_old_index_and_original_order_preserved(self):
        self.assertEqual(self.plan['source_ids'],self.ids)
        self.assertEqual(self.plan['records'][249]['legacy_source200_index'],0)
        self.assertIsNone(self.plan['records'][0]['legacy_source200_index'])
        self.assertEqual(sum(r['legacy_source200_index'] is not None for r in self.plan['records']),200)
        self.assertEqual(self.plan['pending_source_encoding_count'],800)

    def test_duplicate_ids_locations_and_wrong_role_model_rejected(self):
        cases=[]
        x=copy.deepcopy(self.cal);x['source_ids'][1]=x['source_ids'][0];cases.append(x)
        x=copy.deepcopy(self.cal);x['data_bindings']['pixels'][1]['index_in_shard']=0;cases.append(x)
        x=copy.deepcopy(self.cal);x['stage']='m1_development';cases.append(x)
        for cal in cases:
            with self.assertRaises(RuntimeError):a.build_population_plan(cal,self.manifest,self.manifest_path,self.old_ids,self.model)
        with self.assertRaisesRegex(RuntimeError,'visual/model'):a.build_population_plan(self.cal,self.manifest,self.manifest_path,self.old_ids,{'models':{}})
        with self.assertRaises(RuntimeError):a.build_population_plan(self.cal,self.manifest,self.manifest_path,[self.old_ids[0]]*200,self.model)

    def test_reader_only_calibration_one_pair_cache_and_no_metric_or_model(self):
        reader=a.CalibrationAssetReader(self.plan,self.loader)
        s=reader.source(249);self.assertEqual(s['source_id'],self.old_ids[0]);self.assertEqual(len(self.calls),2)
        reader.source(248);self.assertEqual(len(self.calls),2)
        reader.source(300);self.assertEqual(len(self.calls),4)
        self.assertTrue(all('/calibration/' in p.replace('\\','/') for p in self.calls))
        self.assertEqual(s['tokens'][0],249);self.assertEqual(s['pixels'].dtype,np.uint8)

    def test_changed_pixel_preprocessing_shard_or_source_identity_rejected(self):
        p=copy.deepcopy(self.plan);p['records'][0]['preprocessing_id']='f'*64
        with self.assertRaisesRegex(RuntimeError,'preprocessing'):a.CalibrationAssetReader(p,self.loader).source(0)
        p=copy.deepcopy(self.plan);p['records'][0]['latent_sha256']='f'*64
        with self.assertRaisesRegex(RuntimeError,'shard SHA'):a.CalibrationAssetReader(p,self.loader).source(0)
        p=copy.deepcopy(self.plan);p['records'][0]['source_id']='other'
        with self.assertRaisesRegex(RuntimeError,'source ID'):a.CalibrationAssetReader(p,self.loader).source(0)

    def test_registered_reader_requires_shard_and_sidecar_pins_before_load(self):
        with self.assertRaisesRegex(RuntimeError,'not bound before execution'):
            a.CalibrationAssetReader(self.plan,self.loader,expected_bindings={}).source(0)
        self.assertEqual(self.calls,[])
        r=self.plan['records'][0];lp=Path(r['latent_shard']);pp=Path(r['pixel_shard'])
        provider=a.CalibrationAssetReader(self.plan,self.loader,expected_bindings=a.bind((lp,pp,lp.with_suffix('.json'))))
        self.assertEqual(provider.source(0)['source_id'],self.ids[0])

    def reuse_fixture(self):
        full=a.CalibrationAssetReader(self.plan,self.loader).source(249)
        s=dict(source_index=0,source_id=full['source_id'],original_calibration_index=249,preprocessing_id=self.pre)
        lengths=[dict(m=m,arithmetic_bits=m+2,raw_bits=12*sum(x*x for x in a.SIZES[:m])) for m in (6,7,8,9)]
        h=dict(source_index=0,source_id=full['source_id'],independent_roundtrip=True,lengths=lengths)
        bits={f'm{m}_bits':np.zeros(m+2,np.uint8) for m in (6,7,8,9)}
        return full,s,h,bits

    def test_exact_reuse_bits_preserved_but_wrong_old_index_or_tokens_refused(self):
        full,s,h,bits=self.reuse_fixture()
        result=a.validate_reuse(full,s,full['tokens'],full['pixels'],h,bits)
        for key in bits:self.assertTrue(np.array_equal(result[key],bits[key]))
        wrong=full['tokens'].copy();wrong[0]+=1
        with self.assertRaisesRegex(RuntimeError,'not exactly'):a.validate_reuse(full,s,wrong,full['pixels'],h,bits)
        h['source_index']=249
        with self.assertRaisesRegex(RuntimeError,'mapping'):a.validate_reuse(full,s,full['tokens'],full['pixels'],h,bits)

    def test_invalid_or_missing_source_codec_never_marked_reusable(self):
        full,s,h,bits=self.reuse_fixture();h['independent_roundtrip']=False
        with self.assertRaisesRegex(RuntimeError,'roundtrip'):a.validate_reuse(full,s,full['tokens'],full['pixels'],h,bits)
        h['independent_roundtrip']=True;bits['m8_bits'][0]=3
        with self.assertRaisesRegex(RuntimeError,'bit vector'):a.validate_reuse(full,s,full['tokens'],full['pixels'],h,bits)
        full['legacy_source200_index']=None
        with self.assertRaisesRegex(RuntimeError,'no registered old'):a.validate_reuse(full,s,full['tokens'],full['pixels'],h,bits)

    def test_ambiguous_shard_basename_never_chooses_first(self):
        cal=copy.deepcopy(self.cal);cal['data_bindings']['latents'][str(self.root/'duplicate/calibration/shard_0000.pt')]='0'*64
        with self.assertRaisesRegex(RuntimeError,'duplicate shard basename'):
            a.build_population_plan(cal,self.manifest,self.manifest_path,self.old_ids,self.model)


if __name__=='__main__':unittest.main()

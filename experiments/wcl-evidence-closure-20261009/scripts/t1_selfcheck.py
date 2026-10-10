"""Synthetic CPU engineering checks, not real-source experimental results."""
from __future__ import annotations
import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np
import t1_entropy_core as c
import t1_static_counts as stats

ROOT=None

class CoreTests(unittest.TestCase):
    def test_integer_codec_exact_static_roundtrip(self):
        entropy,Encoder,Decoder=c.load_integer_codec(ROOT)
        counts=np.ones((10,4096),dtype=np.int64)
        counts[:,0]=100000
        cdf=entropy.probability_cdf(np.log(counts+.5))
        tokens=np.tile(np.array([0,4095,27,3000]),170)
        streams,records=c.static_encode_decode(tokens,cdf,Encoder,Decoder)
        self.assertEqual(set(streams),{5,6,7,8,9})
        self.assertTrue(all(r['zero_extension_reads']==30 for r in records))
        self.assertTrue(np.all(np.diff(cdf,axis=1)>0))

    def test_pure_arithmetic_no_raw_substitution(self):
        candidate=c.candidates()[0]
        lengths={7:2000,6:1000,5:800}
        chosen=c.choose_arithmetic(lengths,candidate)
        self.assertEqual((candidate['source_capacity'],chosen['actual_m']),(927,5))
        self.assertEqual(chosen['arithmetic_bits'],800)

    def test_missing_fallback_is_not_unencodable(self):
        candidate=c.candidates()[0]
        self.assertEqual(c.choose_arithmetic({7:2000,6:1000},candidate)['status'],'BLOCKED_MISSING_PREFIX')
        self.assertEqual(c.choose_arithmetic({7:2000,6:1000,5:950},candidate)['status'],'TX_UNENCODABLE')

    def test_finite_candidate_count_and_real_spacing_budget(self):
        rows=c.candidates(); self.assertEqual(len(rows),36)
        self.assertEqual(len({r['candidate_id'] for r in rows}),36)
        for r in rows:
            self.assertEqual(r['header_symbols']+r['body_symbols'],1024)
            self.assertEqual(r['source_capacity']+29,r['k'])
            self.assertEqual(r['body_symbols']*r['q'],r['n'])

class FakeTensor:
    def __init__(self,values): self.values=np.asarray(values)
    def cpu(self): return self
    def numpy(self): return self.values

class StaticPopulationTests(unittest.TestCase):
    def make_fixture(self,directory):
        cache=directory/'cache'; (cache/'train').mkdir(parents=True)
        ids=[f'train-source-{i:05d}' for i in range(20000)]
        image=directory/'image.json'; training=directory/'training.json'
        descriptors=[dict(path=f'train/shard_{i:04d}.pt',count=100,sha256=f'{i:064x}') for i in range(200)]
        c.write(image,dict(vae_checkpoint_sha256='7c3ec27ae28a3f87055e83211ea8cc8558bd1985d7b51742d074fb4c2fcf186c',
            populations=[dict(name='train',count=20000,shards=descriptors)]))
        c.write(training,dict(train_ids=ids,calibration_ids=['cal-source']))
        c.write(cache/'registration.json',dict(image_manifest_sha256=c.sha(image),precision='float32',old_fp16_fhat_used=False))
        hashes={}; records={}
        for i in range(200):
            path=cache/'train'/f'shard_{i:04d}.pt'; path.write_bytes(f'SYNTHETIC:{i}'.encode())
            hashes[f'train/{path.name}']=c.sha(path)
            c.write(path.with_suffix('.json'),dict(sha256=c.sha(path),count=100,
                source_shard_sha256=descriptors[i]['sha256'],retokenized_prefix_difference_count=0))
            tokens=np.full((100,680),i%4096,dtype=np.int32)
            records[str(path)]=dict(image_ids=ids[i*100:(i+1)*100],source_descriptor=descriptors[i],
                source_indices=FakeTensor(np.arange(i*100,(i+1)*100)),full_tokens=FakeTensor(tokens),
                base_tokens=FakeTensor(tokens[:,:255]))
        c.write(cache/'training_statistics.json',dict(training_shard_sha256=hashes))
        c.write(cache/'completion.json',dict(status='EXACT_FP32_LATENT_CACHE_COMPLETE',source_counts=dict(train=20000),
            registration_sha256=c.sha(cache/'registration.json'),statistics_sha256=c.sha(cache/'training_statistics.json')))
        args=types.SimpleNamespace(root=str(ROOT),cache=str(cache),out=str(directory/'out'),
            image_manifest=str(image),training_registration=str(training))
        fake=types.SimpleNamespace(set_num_threads=lambda n:None,
            load=lambda path,**kwargs:records[str(path)])
        return args,fake,records

    def test_counts_cover_original20k_all_ten_scales(self):
        with tempfile.TemporaryDirectory() as d:
            args,fake,_=self.make_fixture(Path(d))
            with patch.dict(sys.modules,{'torch':fake}),contextlib.redirect_stdout(io.StringIO()):
                result=stats.run(args)
            self.assertEqual(result['status'],'T1_STATIC_TRAIN20K_CDF_COMPLETE')
            with np.load(Path(args.out)/'static_scale_counts_cdf.npz') as z:
                self.assertTrue(np.array_equal(z['counts'].sum(1),20000*np.square(c.SIZES)))
                self.assertTrue(np.all(np.diff(z['cdf'],axis=1)>0))

    def test_shuffled_id_refused(self):
        with tempfile.TemporaryDirectory() as d:
            args,fake,records=self.make_fixture(Path(d)); first=next(iter(records.values()))
            first['image_ids'][0]='cal-source'
            with patch.dict(sys.modules,{'torch':fake}),self.assertRaisesRegex(ValueError,'IDs/order'):
                stats.run(args)

    def test_missing_m9_tokens_refused(self):
        with tempfile.TemporaryDirectory() as d:
            args,fake,records=self.make_fixture(Path(d)); first=next(iter(records.values()))
            first['full_tokens']=FakeTensor(np.zeros((100,255),dtype=np.int32))
            with patch.dict(sys.modules,{'torch':fake}),self.assertRaisesRegex(ValueError,'ten scales'):
                stats.run(args)

    def test_changed_shard_bytes_refused(self):
        with tempfile.TemporaryDirectory() as d:
            args,fake,_=self.make_fixture(Path(d)); (Path(args.cache)/'train/shard_0000.pt').write_bytes(b'changed')
            with patch.dict(sys.modules,{'torch':fake}),self.assertRaisesRegex(ValueError,'SHA256'):
                stats.run(args)

def main():
    global ROOT
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();ROOT=Path(a.root)
    suite=unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    c.write(a.out,dict(status='PASS' if result.wasSuccessful() else 'FAIL',synthetic_only=True,
        tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),
        new_real_model_calls=0,new_real_channel_decodes=0))
    return 0 if result.wasSuccessful() else 1

if __name__=='__main__': raise SystemExit(main())

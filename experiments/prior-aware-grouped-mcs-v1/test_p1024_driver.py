"""CPU interface/provenance tests; no learned model or development data loaded."""
import copy
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
import p1024_driver as p
import source_quality as q
import validation_metrics as v


def source_fixture(directory):
    root=Path(directory);record=dict(image_id='fixture/source',preprocessing_id='fixture/preprocess')
    source=np.zeros((3,256,256),np.float32);images=np.stack([source+np.float32(.2+i/100) for i in range(12)])
    rows=[]
    for slot,(snr,seed) in enumerate((s,n) for s in p.SNRS for n in p.SEEDS):
        row=dict(N=1024,source_index=0,source_id=record['image_id'],preprocessing_id=record['preprocessing_id'],
            method='P1024',snr_db=snr,noise_seed=seed,selected_step=40000,image_slot=slot,
            row_id=f'fixture-{snr}-{seed}',image_sha256=q.rgb_sha(images[slot]),reference_sha256=q.rgb_sha(source))
        row.update({m:1. for m in p.METRICS});rows.append(row)
    npz=root/'0000.npz';np.savez(npz,images=images,source_rgb=source,
        row_ids=np.asarray([r['row_id'] for r in rows]),image_slots=np.arange(12,dtype=np.int64))
    saved=dict(binding='test-binding',source_index=0,source_id=record['image_id'],rows=rows,
        float_reconstructions=dict(path=str(npz),sha256=q.sha(npz)),input_bindings={})
    saved['payload_sha256']=q.identity(saved)
    return saved,record,source,images


class PDriverTests(unittest.TestCase):
    def test_exact_old_grid_excludes_other_methods_snrs_and_duplicates(self):
        rows=[dict(method='P1024',snr_db=str(s),noise_seed=str(n)) for s in (1,4,7,13,19) for n in p.SEEDS]
        rows.append(dict(method='V_policy',snr_db='7',noise_seed='2001'))
        engine=SimpleNamespace(by_source={'N1024':{0:rows}})
        expected=p.expected_source_rows(engine,0)
        self.assertEqual(set(expected),{(s,n) for s in (4,7,13) for n in p.SEEDS})
        rows.append(dict(method='P1024',snr_db='4',noise_seed='2001'))
        with self.assertRaises(RuntimeError):p.expected_source_rows(engine,0)

    def test_hash_bound_archive_and_row_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            saved,record,_,_=source_fixture(directory)
            p.validate_source(saved,'test-binding',0,record)
            bad=copy.deepcopy(saved);bad['rows'][0]['image_slot']=1
            bad['payload_sha256']=q.identity({k:v for k,v in bad.items() if k!='payload_sha256'})
            with self.assertRaises(RuntimeError):p.validate_source(bad,'test-binding',0,record)
            bad=copy.deepcopy(saved);bad['rows'][0]['selected_step']=37500
            bad['payload_sha256']=q.identity({k:v for k,v in bad.items() if k!='payload_sha256'})
            with self.assertRaises(RuntimeError):p.validate_source(bad,'test-binding',0,record)

    def test_missing_metric_fails_before_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            saved,record,_,_=source_fixture(directory)
            saved['rows'][0]['dinov2_vitl14_cosine']=None
            saved['payload_sha256']=q.identity({k:v for k,v in saved.items() if k!='payload_sha256'})
            with self.assertRaises(RuntimeError):p.validate_source(saved,'test-binding',0,record)

    def test_report_renderer_loads_identical_archive(self):
        import report_builder
        with tempfile.TemporaryDirectory() as directory:
            saved,record,source,images=source_fixture(directory)
            # This checks the actual report reader, rather than a copied schema.
            loaded=report_builder.load_images(saved)
            self.assertIsInstance(loaded,tuple)
            self.assertTrue(any(np.array_equal(x,source) for x in loaded if isinstance(x,np.ndarray)))

    def test_complete_summary_pairs_noise_within_source(self):
        rows=[]
        for i in range(100):
            for snr in p.SNRS:
                for seed in p.SEEDS:
                    row=dict(source_index=i,snr_db=snr,noise_seed=seed)
                    row.update({m:float(seed-2000) for m in p.METRICS});rows.append(row)
        result=p.summarize(rows)
        self.assertEqual(len(result),4*len(p.METRICS))
        self.assertTrue(all(r['mean']==r['ci_low']==r['ci_high']==2 for r in result))
        with self.assertRaises(RuntimeError):p.summarize(rows[:-1])

    def test_old_rgb_disagreement_is_not_silently_rescored(self):
        source=np.zeros((3,256,256),np.float32);image=source+.25
        engine=SimpleNamespace(records=[dict(image_id='a',preprocessing_id='p',class_index=0)],target=lambda _:source)
        original=dict(snr_db='4',noise_seed='2001',E='2048',waveform_sha256='w',observation_sha256='o',latent_sq_err_final='8')
        measured=dict(image_sha256=q.rgb_sha(image),reference_sha256=q.rgb_sha(source),source_id='a',
            N=1024,snr_db=4,noise_seed=2001,replay_row_id='old')
        row=p.old_metadata(engine,0,original,measured,image,'cache',{})
        self.assertEqual(row['latent_sq_err_final'],8);self.assertTrue(row['exact_completed_rgb_match'])
        with self.assertRaises(RuntimeError):p.old_metadata(engine,0,original,measured,image+.001,'cache',{})

    def test_cache_inventory_requires_real_complete_receipts(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(p.cache_receipts(directory),({},{}))
            path=Path(directory)/'outputs/EXTERNAL-COMPARISON-20261004/own_controls/export-n1024_completion.json'
            q.write(path,dict(status='RUNNING',synthetic=False,outputs={}))
            with self.assertRaises(RuntimeError):p.cache_receipts(directory)

    def test_cli_import_has_no_torch_or_automatic_gpu_access(self):
        script='import sys,p1024_driver; assert "torch" not in sys.modules; assert "torchvision" not in sys.modules'
        result=subprocess.run([sys.executable,'-c',script],cwd=Path(__file__).parent,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)


if __name__=='__main__':unittest.main()

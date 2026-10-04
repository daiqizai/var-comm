"""CPU-only checks for independent validation and admitted existing float RGB."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
import numpy as np
import validation_metrics as v
import p1024_10db as p


def frame(i,seed,method,truth,source,pred):
    return dict(N=1024,snr_db=10,source_index=i,noise_seed=seed,method=method,
        convnext_used_for_selection=False,**v.diagnostics(truth,source,pred))


def population():
    rows=[]
    for method in ('B0','B3'):
        for i in range(4):
            for seed in (2001,2002,2003):
                # Sources 0,1 were originally correct; 2,3 originally wrong.
                source=0 if i<2 else 1
                pred=source if method=='B0' else (1 if i==0 else 0)
                rows.append(frame(i,seed,method,0,source,pred))
    return rows


class ValidationTests(unittest.TestCase):
    def test_transition_truth_table(self):
        self.assertIn('convnext_source_prediction_agreement',v.METRICS)
        self.assertNotIn('convnext_top1_source_prediction',v.METRICS)
        for source,pred,expected in ((0,0,(False,False)),(0,1,(True,False)),
                                     (1,0,(False,True)),(1,1,(False,False))):
            row=v.diagnostics(0,source,pred)
            self.assertEqual((row['convnext_source_correct_to_wrong'],row['convnext_source_wrong_to_correct']),expected)
            self.assertIs(type(row['convnext_top1_label']),bool)
            self.assertIs(type(row['convnext_source_prediction_agreement']),int)
            self.assertEqual(row['convnext_source_prediction_agreement'],row['convnext_top1_source_prediction'])
        with self.assertRaises(ValueError):v.diagnostics(1000,0,0)
        with self.assertRaises(ValueError):v.diagnostics(True,0,0)

    def test_transitions_use_eligible_original_strata(self):
        summary,transition=v.summarize(population(),source_indices=tuple(range(4)))
        self.assertEqual(len(summary),4)
        rows={r['transition']:r for r in transition if r['method']=='B3'}
        cw=rows['convnext_source_correct_to_wrong'];wc=rows['convnext_source_wrong_to_correct']
        self.assertEqual((cw['numerator_frames'],cw['denominator_frames'],cw['denominator_sources'],cw['mean']),(3,6,2,.5))
        self.assertEqual((wc['numerator_frames'],wc['denominator_frames'],wc['denominator_sources'],wc['mean']),(6,6,2,1.))
        self.assertEqual(cw['bootstrap_population'],'eligible original-correctness source stratum')

    def test_empty_transition_denominator_is_missing_not_zero(self):
        rows=[frame(0,s,'B0',0,0,0) for s in (2001,2002,2003)]
        _,transition=v.summarize(rows,source_indices=(0,))
        empty=next(r for r in transition if r['transition']=='convnext_source_wrong_to_correct')
        self.assertEqual(empty['denominator_frames'],0);self.assertIsNone(empty['mean'])
        self.assertIsNone(empty['ci_low']);self.assertIsNone(empty['ci_high'])

    def test_pairing_is_by_source_after_noise_average(self):
        rows=[]
        for i in (3,17):
            for s in (2001,2002,2003):
                rows += [frame(i,s,'B0',0,0,1),frame(i,s,'B3',0,0,0)]
        result=v.paired(rows,'B3','B0',source_indices=(3,17))
        self.assertEqual(len(result),2)
        for row in result:
            self.assertEqual((row['mean'],row['ci_low'],row['ci_high']),(1.,1.,1.))
            self.assertEqual(row['n_sources'],2);self.assertEqual(row['n_frames'],6)

    def test_population_omission_duplicate_and_source_drift_rejected(self):
        base=population()
        for bad in (base[:-1],base+[base[0]]):
            with self.assertRaises(ValueError):v.summarize(bad,source_indices=tuple(range(4)))
        bad=copy.deepcopy(base);bad[0].update(v.diagnostics(0,2,0))
        with self.assertRaises(ValueError):v.summarize(bad,source_indices=tuple(range(4)))
        bad=copy.deepcopy(base);bad[0]['convnext_source_correct_to_wrong']=True
        with self.assertRaises(ValueError):v.summarize(bad,source_indices=tuple(range(4)))

    def test_no_selection_and_no_model_import_without_admission(self):
        with tempfile.TemporaryDirectory() as d:
            receipt=Path(d)/'policy.json';receipt.write_text('{}')
            v.admission('development',receipt,v.sha(receipt))
            with self.assertRaises(ValueError):v.admission('calibration',receipt,v.sha(receipt))
            with self.assertRaises(ValueError):v.admission('development',receipt,'0'*64)
        script='import sys; import validation_metrics,p1024_10db; assert "torch" not in sys.modules; assert "torchvision" not in sys.modules'
        result=subprocess.run([sys.executable,'-c',script],cwd=Path(__file__).parent,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        rows=population();rows[0]['convnext_used_for_selection']=True
        with self.assertRaises(ValueError):v.summarize(rows,source_indices=tuple(range(4)))

    def test_selected_exact_40k_only(self):
        selected=dict(N=1024,step=40000,checkpoint_sha256='a'*64)
        adapter=SimpleNamespace(p=SimpleNamespace(uses=1024),pmeta=dict(selected=selected,
            selected_sha256='b'*64,decoder_sha256='c'*64),
            pcfg=dict(identity=dict(selected_sha256='b'*64,models={'P1024':'d'*64})))
        self.assertEqual(p.selected_identity(adapter)['selected_step'],40000)
        selected['step']=37500
        with self.assertRaises(ValueError):p.selected_identity(adapter)

    def test_missing_cache_no_reconstruction_fallback(self):
        with tempfile.TemporaryDirectory() as d:
            result=p.read_existing_cache(Path(d)/'missing.json',checkpoint_sha256='a'*64,
                source_index=0,reference_sha256='b'*64)
            self.assertEqual(result['status'],'CACHE_MISSING');self.assertEqual(result['rows'],[])

    def test_original_float_cache_association_and_admitted_hash(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);target=np.zeros((3,256,256),dtype=np.float32);images=np.stack([target+.25,target+.5])
            npz=d/'rgb.npz';np.savez(npz,images=images,native_reference=target,comparison_reference=target)
            proof=dict(path=str(npz),sha256=v.sha(npz),lossless=True,
                image_sha256=[p.rgb_sha(x) for x in images],native_reference_sha256=p.rgb_sha(target),
                comparison_reference_sha256=p.rgb_sha(target))
            rows=[dict(method='P1024',N=1024,snr_db=4,noise_seed=2001,image_slot=0,
                image_sha256=p.rgb_sha(images[0]),reference_sha256=p.rgb_sha(target),original_scientific_row={'method':'P1024'}),
                dict(method='D_U_QPSK',N=1024,snr_db=4,noise_seed=2001,image_slot=1,
                image_sha256=p.rgb_sha(images[1]),reference_sha256=p.rgb_sha(target))]
            cp=d/'0000.json';value=dict(source_index=0,synthetic=False,float_reconstructions=proof,rows=rows)
            cp.write_text(json.dumps(value));admitted=v.sha(cp)
            result=p.read_existing_cache(cp,checkpoint_sha256=admitted,source_index=0,reference_sha256=p.rgb_sha(target))
            self.assertEqual(len(result['rows']),1);self.assertEqual(result['covered_frames'],[(4,2001)])
            np.testing.assert_array_equal(result['images'][0],images[0])
            with self.assertRaises(ValueError):p.read_existing_cache(cp,checkpoint_sha256=admitted,
                source_index=1,reference_sha256=p.rgb_sha(target))
            cp.write_text(json.dumps(dict(value,synthetic=True)))
            with self.assertRaises(ValueError):p.read_existing_cache(cp,checkpoint_sha256=admitted,
                source_index=0,reference_sha256=p.rgb_sha(target))


if __name__=='__main__':unittest.main()

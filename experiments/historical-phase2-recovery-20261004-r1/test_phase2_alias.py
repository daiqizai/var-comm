"""Engineering fixtures for the explicit alias; no scientific output writes."""
import copy
import hashlib
import unittest
import numpy as np
import historical_phase2_alias as alias
from phase2_inheritance import validate_inherited_pure


def fixture(index=81, phase2_psnr=19.31717300415039, grid_psnr=19.317157745361328):
    source='registered_source_'+str(index)
    p=dict(method='pure_continuous',source_index=index,image_id=source,snr_db='1.0',seed='2003',N='4084',E='8168',
        model_context_sha256=alias.CHECKPOINT,decoder_sha256=alias.DECODER,
        psnr_db=phase2_psnr,lpips_alex=.22,dino_cosine=.75)
    g=dict(method='P4084',source_index=index,source_id=source,snr_db='1',noise_seed='2003',N='4084',E='8168',
        checkpoint_sha256=alias.CHECKPOINT,preprocessing_id='a'*64,noise_namespace='VAR-CONTINUOUS-4084',
        waveform_sha256='b'*64,observation_sha256='c'*64,psnr_db=grid_psnr,lpips_alex=.22,dino_cosine=.75,mse=.01)
    image=np.full((3,256,256),.4,np.float32);target=np.full_like(image,.5)
    raw=lambda x:hashlib.sha256(x.tobytes()).hexdigest()
    proof=dict(passed=True,replay_parity_passed=True,synthetic=False,
        original_row_sha256=alias.budget.identity(g),rgb_sha256=raw(image),target_sha256=raw(target),
        exact_fields_checked=['waveform_sha256','observation_sha256'],metric_deltas=dict(psnr_db=0.,lpips_alex=0.,dino_cosine=0.,mse=0.))
    return p,g,image,target,proof


class Phase2AliasTests(unittest.TestCase):
    def test_known_historical_gap_does_not_shift_new_metric(self):
        p,g,image,target,proof=fixture()
        actual=alias.values(g);actual['psnr_db']+=3e-6
        result=alias.observed_alias_proof(p,g,actual,proof,image,target,{'old':'bound'})
        self.assertEqual(result['measured_native_grid_metrics'],actual)
        self.assertFalse(result['new_metric_offset_applied'])
        self.assertFalse(result['original_phase2_scalar_parity_claimed'])
        self.assertEqual(result['historical_alias']['historical_grid_minus_phase2']['psnr_db'],-1.52587890625e-05)
        self.assertGreater(abs(result['measured_grid_minus_original_phase2']['psnr_db']),1e-5)

    def test_known_gap_cannot_hide_new_grid_drift(self):
        p,g,image,target,proof=fixture()
        actual=alias.values(g);actual['psnr_db']+=1.1e-5
        with self.assertRaises(ValueError):alias.observed_alias_proof(p,g,actual,proof,image,target,{})
        for metric in ('lpips_alex','dino_cosine'):
            actual=alias.values(g);actual[metric]+=2.1e-6
            with self.assertRaises(ValueError):alias.observed_alias_proof(p,g,actual,proof,image,target,{})

    def test_unrelated_source_weight_or_noise_cannot_use_alias(self):
        p,g,*_=fixture()
        for key,value in (('source_id','other'),('checkpoint_sha256','x'*64),('noise_seed','2002'),
                          ('noise_namespace','other'),('N','2048')):
            changed=dict(g);changed[key]=value
            with self.assertRaises(ValueError):alias.pair_record(p,changed)
        changed=dict(p,decoder_sha256='x'*64)
        with self.assertRaises(ValueError):alias.pair_record(changed,g)

    def test_actual_float_image_and_physical_proof_required(self):
        p,g,image,target,proof=fixture()
        changed=image.copy();changed[0,0,0]+=.01
        with self.assertRaises(ValueError):alias.observed_alias_proof(p,g,alias.values(g),proof,changed,target,{})
        changed=copy.deepcopy(proof);changed['exact_fields_checked'].remove('observation_sha256')
        with self.assertRaises(ValueError):alias.observed_alias_proof(p,g,alias.values(g),changed,image,target,{})

    def test_inherited_parity_and_float_pixels_remain_untouched(self):
        p,g,image,target,native=fixture(80,19.,19.)
        raw=lambda x:hashlib.sha256(x.tobytes()).hexdigest()
        proof=dict(replay_parity_passed=True,synthetic=False,original_row_sha256=alias.budget.identity(p),
            native_replay_proof=native,native_rgb_sha256=raw(image),tolerances=alias.LIMITS,
            metric_differences={name:0. for name in alias.METRICS})
        row=dict(history_row_id='original_id',history_image_sha256='old_scored_hash')
        cache=dict(image_sha256=[raw(image)],reference_sha256=raw(target))
        before=copy.deepcopy(proof)
        result=validate_inherited_pure(row,proof,p,g,cache,0)
        self.assertEqual(proof,before)
        self.assertTrue(result['original_phase2_parity_preserved'])
        self.assertFalse(result['observed_metric_values_and_original_scores_rewritten'])
        proof['native_replay_proof']['metric_deltas']['psnr_db']=2e-5
        with self.assertRaises(RuntimeError):validate_inherited_pure(row,proof,p,g,cache,0)

    def test_a_different_old_row_cannot_reuse_a_valid_new_image_proof(self):
        p,g,image,target,proof=fixture()
        changed=dict(g,psnr_db=float(g['psnr_db'])+1e-7)
        with self.assertRaises(ValueError):alias.observed_alias_proof(p,changed,alias.values(changed),proof,image,target,{})


if __name__=='__main__':unittest.main()

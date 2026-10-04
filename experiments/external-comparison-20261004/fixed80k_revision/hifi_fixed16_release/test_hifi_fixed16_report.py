"""Small CPU checks for exact paired scope, statistics and honest disclosure."""
from pathlib import Path
import copy
import unittest
import hifi_fixed16_report as r

def population():
    return [dict(source_index=i,N=n,snr_db=s,noise_seed=2001,method=m,synthetic=False,label_conditioned=False,
        audit_used_to_control_receiver=False,selected_step=80000,selected_checkpoint_sha256=r.CHECKPOINT,
        E=2*n,actual_energy=2*n,NFE=0 if m==r.METHODS[0] else 257,t_start=257,complete_posterior_schedule=True,
        header_accepted=True,observed_sha256=f'{i}_{n}_{s}',**{metric:1.0+(m==r.METHODS[1])*.1 for metric in r.METRICS})
        for i in r.FIXED for n in r.BUDGETS for s in r.SNRS for m in r.METHODS]

class Contracts(unittest.TestCase):
    def setUp(self):self.rows=population()
    def test_exact_128_paired_rows(self):self.assertEqual(len(r.validate_rows(self.rows)),128)
    def test_no_extra_SNR_or_noise(self):
        for name,value in [('snr_db',1),('noise_seed',2002),('source_index',99)]:
            rows=copy.deepcopy(self.rows);rows[0][name]=value
            with self.assertRaises(RuntimeError):r.validate_rows(rows)
    def test_duplicate_or_missing_rejected(self):
        for rows in (self.rows[:-1],self.rows[:-1]+[self.rows[0]]):
            with self.assertRaises(RuntimeError):r.validate_rows(rows)
    def test_full_schedule_and_same_waveform(self):
        for name,value in [('NFE',2),('complete_posterior_schedule',False),('observed_sha256','other'),('selected_step',81551)]:
            rows=copy.deepcopy(self.rows);rows[1][name]=value
            with self.assertRaises(RuntimeError):r.validate_rows(rows)
    def test_summary_104(self):
        indexed=r.validate_rows(self.rows)
        summary=[dict(N=n,snr_db=s,method=m,metric=k,n_sources=16,n_frames=16,mean=1+(m==r.METHODS[1])*.1,ci_low=.9,ci_high=1.2)
            for n in r.BUDGETS for s in r.SNRS for m in r.METHODS for k in r.METRICS]
        self.assertEqual(len(r.validate_summary(summary,indexed)),104)
        summary[0]['n_sources']=100
        with self.assertRaises(RuntimeError):r.validate_summary(summary,indexed)
    def pairs(self):
        return [dict(N_A=n,N_B=n,snr_A=s,snr_B=s,method_A=r.METHODS[1],method_B=r.METHODS[0],metric=k,
            comparison_scope='same_source_noise_seed_and_identical_measured_full_waveform',n_sources=16,mean=.1,ci_low=.05,ci_high=.15)
            for n in r.BUDGETS for s in r.SNRS for k in r.METRICS]
    def test_all_52_actual_paired_deltas(self):
        pairs=self.pairs();indexed=r.validate_rows(self.rows)
        self.assertEqual(len(r.validate_pairs(pairs,indexed)),52)
        pairs[0]['mean']=-.1
        with self.assertRaises(RuntimeError):r.validate_pairs(pairs,indexed)
    def test_pair_orientation_and_matching(self):
        for name,value in [('N_B',4084),('snr_B',1),('method_A',r.METHODS[0]),('comparison_scope','independent')]:
            pairs=self.pairs();pairs[0][name]=value
            with self.assertRaises(RuntimeError):r.validate_pairs(pairs,r.validate_rows(self.rows))
    def test_original_fixed16_indices_retained(self):
        self.assertEqual(r.FIXED,(0,25,50,75,4,21,24,29,33,41,52,60,64,87,92,95))
        self.assertEqual(len(r.FIXED)*len(r.BUDGETS)*len(r.SNRS)*len(r.METHODS),128)
    def test_no_full_resume_or_overclaim(self):
        source=Path(r.__file__).read_text(encoding='utf-8')
        for phrase in ('不能代表完整','不是人工语义判决','full_evaluation_resume_authorized=False','ResNet50_Weights.IMAGENET1K_V2'):
            self.assertIn(phrase,source)
        import hifi_fixed16_publish as p
        publication=Path(p.__file__).read_text(encoding='utf-8')
        self.assertIn("phase='hifi_fixed16_release'",publication)
        self.assertIn("swin_early_release/release_publication.json",publication)
        self.assertNotIn("['git','push','--force'",publication)

if __name__=='__main__':unittest.main(verbosity=2)

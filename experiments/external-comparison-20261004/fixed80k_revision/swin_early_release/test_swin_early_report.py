"""CPU contracts for the genuine one-method early report and publication."""
import copy
from pathlib import Path
import unittest
import swin_early_report as r

def population():
    return [dict(source_index=i,N=n,snr_db=s,noise_seed=seed,method=r.METHOD,synthetic=False,label_conditioned=False,
        audit_used_to_control_receiver=False,selected_step=80000,
        selected_checkpoint_sha256='8857b8c5ed91a5c316084e8c252a8e2168e424a767378389436d96268e65ba21',
        E=2*n,actual_energy=2*n,NFE=0,header_accepted=True,**{m:1.0 for m in r.METRICS})
        for i in range(100) for n in r.BUDGETS for s in r.SNRS for seed in r.SEEDS]

class Contracts(unittest.TestCase):
    def setUp(self):self.rows=population()
    def test_exact_one_method_population(self):self.assertEqual(len(r.validate_rows(self.rows)),1800)
    def test_missing_or_duplicate_frames_rejected(self):
        for rows in (self.rows[:-1],self.rows[:-1]+[self.rows[0]],self.rows+self.rows):
            with self.assertRaises(RuntimeError):r.validate_rows(rows)
    def test_hifi_cannot_enter(self):
        self.rows[0]['method']='HiFiDiffCom_SwinJSCC'
        with self.assertRaises(RuntimeError):r.validate_rows(self.rows)
    def test_wrong_checkpoint_rejected(self):
        for field,value in [('selected_step',81551),('selected_checkpoint_sha256','bad'),('synthetic',True),
            ('label_conditioned',True),('audit_used_to_control_receiver',True),('actual_energy',0),('NFE',257)]:
            rows=copy.deepcopy(self.rows);rows[0][field]=value
            with self.assertRaises(RuntimeError):r.validate_rows(rows)
    def test_nonfinite_metric_rejected(self):
        for metric in r.METRICS:
            rows=copy.deepcopy(self.rows);rows[0][metric]=float('nan')
            with self.assertRaises(RuntimeError):r.validate_rows(rows)
    def test_gray_failure_required(self):
        self.rows[0].update(header_accepted=False,fallback='other')
        with self.assertRaises(RuntimeError):r.validate_rows(self.rows)
        self.rows[0]['fallback']='fixed_gray_0.5';r.validate_rows(self.rows)
    def summary(self):
        return [dict(N=n,snr_db=s,method=r.METHOD,metric=m,n_sources=100,n_frames=300,mean=1,ci_low=.9,ci_high=1.1)
            for n in r.BUDGETS for s in r.SNRS for m in r.METRICS]
    def test_summary_all_78_and_exact_frame_means(self):
        indexed=r.validate_rows(self.rows);summary=self.summary()
        self.assertEqual(len(r.validate_summary(summary,indexed)),78)
        summary[0]['mean']=1.001
        with self.assertRaises(RuntimeError):r.validate_summary(summary,indexed)
    def test_summary_cannot_omit_metric(self):
        with self.assertRaises(RuntimeError):r.validate_summary(self.summary()[:-1],r.validate_rows(self.rows))
    def test_original_fixed_selection(self):
        self.assertEqual(r.FIXED,(0,25,50,75,4,21,24,29,33,41,52,60,64,87,92,95))
        self.assertEqual(len(r.FIXED)*len(r.BUDGETS)*len(r.SNRS),96)
    def test_new_publication_destinations(self):
        import swin_early_publish as p
        source=Path(p.__file__).read_text(encoding='utf-8')
        self.assertIn("phase='swin_early_release'",source)
        self.assertIn('full_comparison_complete=False',source)
        self.assertIn('swin_fixed80k_early_20261004.md',source)
        self.assertNotIn("['git','push','--force'",source)
        self.assertNotIn("['git','reset'",source)

if __name__=='__main__':unittest.main(verbosity=2)

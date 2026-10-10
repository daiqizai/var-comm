"""Synthetic checks of the post-hoc freeze/statistics interface; no science calls."""
import argparse,csv
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from t1_entropy_core import candidates,read,sha,write
import t1_holdout_metadata as metadata
from t1_holdout_statistics import metric_value

class Tests(unittest.TestCase):
    def fixture(self,base,holdout=False,bad_rank=False):
        cs=candidates()[:3]
        req=base/'request.json';write(req,dict(phase='full',snrs=[4,10,19],seeds=[4101,4102,4103],holdout_used=holdout,
            records=[dict(source_id='synthetic-cal-'+str(i)) for i in range(1000)],
            static_completion={'synthetic':True},phy_qualification={'synthetic':True},source_model_id={'synthetic':True},
            registered_profile_catalogue={'synthetic':True},noise_rule='synthetic',counter_rule='synthetic'))
        rows=[]
        for family in metadata.FAMILIES:
            for snr in metadata.SNRS:
                for j,c in enumerate(cs):
                    rows.append(dict(family=family,snr_db=snr,candidate_id=c['candidate_id'],target_m=c['target_m'],q=c['q'],
                        nominal_rate=c['nominal_rate'],source_count=1000,noise_count=3,dinov2_vitl14_cosine=.8-j*.1,rank=j+1))
        if bad_rank:rows[0]['dinov2_vitl14_cosine']=.2
        rp=base/'rankings.csv'
        with rp.open('w',newline='') as f:
            w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
        done=base/'complete.json';write(done,dict(status='T1_FULL_CALIBRATION_COMPLETE',source_count=1000,noise_count=3,
            holdout_used=holdout,policy_selection_metric='dinov2_vitl14_cosine_only',ledger={'unresolved':0},families=metadata.FAMILIES,
            request_path=str(req),request_sha256=sha(req),rankings_path=str(rp),outputs={str(rp):sha(rp)}))
        return done

    def test_complete_calibration_freezes_exact_six_policies(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);done=self.fixture(base)
            out=base/'freeze.json';v=metadata.freeze(SimpleNamespace(calibration_completion=[str(done)],out=str(out)))
            self.assertEqual(sum(len(x) for x in v['policies'].values()),6)
            self.assertFalse(v['holdout_used_for_selection']);self.assertEqual(len(v['new_paired_prefixes']),11)
            self.assertEqual(len({tuple(x) for x in v['new_paired_prefixes']}),11)

    def test_holdout_used_cannot_freeze(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);done=self.fixture(base,holdout=True)
            with self.assertRaisesRegex(ValueError,'full calibration closure'):
                metadata.freeze(SimpleNamespace(calibration_completion=[str(done)],out=str(base/'freeze.json')))

    def test_altered_rank_order_cannot_freeze(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);done=self.fixture(base,bad_rank=True)
            with self.assertRaisesRegex(ValueError,'DINOv2-L'):
                metadata.freeze(SimpleNamespace(calibration_completion=[str(done)],out=str(base/'freeze.json')))

    def test_csv_agreement_event_formats(self):
        metric='convnext_top1_source_prediction'
        for text,expected in [('True',1.),('False',0.),('1',1.),('0',0.)]:
            self.assertEqual(metric_value({metric:text},metric),expected)
        self.assertEqual(metric_value({'lpips_alex':'-0.01'},'lpips_alex'),-.01)

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);args=p.parse_args()
    r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    write(args.out,dict(status='PASS' if r.wasSuccessful() else 'FAIL',synthetic_only=True,tests_run=r.testsRun,
        failures=len(r.failures),errors=len(r.errors),new_real_model_calls=0,new_real_bootstrap_calls=0,new_real_channel_decodes=0,
        source_sha256={n:sha(Path(__file__).with_name(n)) for n in ('t1_holdout_metadata.py','t1_holdout_statistics.py','t1_holdout_selfcheck.py')}))
    return 0 if r.wasSuccessful() else 1

if __name__=='__main__':raise SystemExit(main())

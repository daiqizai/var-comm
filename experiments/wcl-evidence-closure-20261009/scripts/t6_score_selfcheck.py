"""Engineering-only T6 admission/statistic tests: no model, PHY or bootstrap."""
import importlib.util, json, tempfile, unittest
from pathlib import Path
import numpy as np
import t6_score as s
import t6_statistics as stats

def fixture():
    family='EC_VAR_WHOLE';ids=['engineering-fixture-source-'+str(i) for i in range(100)];rows=[]
    for method in s.methods(family):
        for snr in s.SNRS:
            for i,sid in enumerate(ids):
                for seed in s.SEEDS:
                    partial=method==s.RAW_PARTIAL
                    # Exact zero PARTIAL-WHOLE at4dB, adverse LPIPS at10/19.
                    increment=0. if snr==4 else .03 if partial else 0.
                    rows.append(dict(N=2048,method_id=method,point_id=s.point(method,snr),snr_db=snr,source_index=i,
                        source_id=sid,noise_seed=seed,status='HEADER_REJECT_GRAY' if seed==9201 else 'ENGINEERING_FIXTURE',
                        psnr_db=20+i/100+increment,lpips_alex=.2+increment,dinov2_vitl14_cosine=.6+increment,
                        convnext_top1_source_prediction=bool((i+(seed-9201))%2)))
    return family,ids,rows

class NoBootstrapFixture:
    def draws(self):return np.tile(np.arange(100,dtype=np.int64),(10000,1))
    def interval(self,vector,samples):
        # Not a scientific confidence interval: checks plumbing and unit/sign only.
        mean=float(np.mean(vector));return dict(mean=mean,ci_low=mean-.001,ci_high=mean+.001,
            source_count=100,noise_count=3,frame_count=1500,bootstrap_seed=2026100701,bootstrap_replicates=10000)

class Tests(unittest.TestCase):
    def test_complete_grid_includes_failures(self):
        family,ids,rows=fixture();keyed=s.frame_grid(rows,ids,family,scored=True)
        self.assertEqual(len(keyed),2700);self.assertEqual(sum(r['status']=='HEADER_REJECT_GRAY' for r in keyed.values()),900)
    def test_ambiguous_missing_and_wrong_budget_rejected(self):
        family,ids,rows=fixture()
        with self.assertRaisesRegex(RuntimeError,'Duplicate'):s.frame_grid(rows+[rows[0]],ids,family,scored=True)
        with self.assertRaisesRegex(RuntimeError,'2700'):s.frame_grid(rows[:-1],ids,family,scored=True)
        rows[0]['N']=1024
        with self.assertRaisesRegex(RuntimeError,'N1024'):s.frame_grid(rows,ids,family,scored=True)
    def test_nonfinite_failure_not_dropped(self):
        family,ids,rows=fixture();rows[0]['lpips_alex']=float('nan')
        with self.assertRaisesRegex(RuntimeError,'finite'):s.frame_grid(rows,ids,family,scored=True)
    def test_three_noise_mean_before_sources(self):
        family,ids,rows=fixture();values,table=stats.source_means(rows,ids,family)
        key=(s.point(s.RAW_WHOLE,4),'convnext_top1_source_prediction')
        self.assertEqual(values[key][0],1/3);self.assertEqual(values[key][1],2/3);self.assertEqual(len(table),3600)
    def test_sign_zero_and_percentage_points(self):
        family,ids,rows=fixture();summary,pairs,means,calls=stats.summarize(rows,ids,family,NoBootstrapFixture())
        self.assertEqual(len(summary),36);self.assertEqual(len(pairs),36)
        zero=[r for r in pairs if r['method_id']==s.RAW_PARTIAL and r['reference_method_id']==s.RAW_WHOLE and r['snr_db']==4]
        self.assertEqual(len(zero),4)
        self.assertTrue(all(r['mean']==r['ci_low']==r['ci_high']==0 for r in zero))
        adverse=next(r for r in pairs if r['method_id']==s.RAW_PARTIAL and r['reference_method_id']==s.RAW_WHOLE and r['snr_db']==10 and r['metric']=='lpips_alex')
        self.assertAlmostEqual(adverse['mean'],.03);self.assertEqual(adverse['interval_direction'],'reference_better')
        self.assertEqual(adverse['improvement_direction'],'negative')
        agreement=stats.decorate(s.METRICS[-1],dict(mean=.02,ci_low=-.01,ci_high=.05),True)
        self.assertEqual(agreement['display_mean'],2.);self.assertEqual(agreement['display_ci_low'],-1.)
        self.assertEqual(agreement['display_ci_high'],5.);self.assertEqual(agreement['display_unit'],'percentage_points')
    def test_pairing_does_not_subtract_marginal_intervals(self):
        family,ids,rows=fixture();summary,pairs,_,_=stats.summarize(rows,ids,family,NoBootstrapFixture())
        zeros=[r for r in pairs if r['method_id']==s.RAW_PARTIAL and r['reference_method_id']==s.RAW_WHOLE and r['snr_db']==4]
        # Marginal intervals each have nonzero width, but paired difference is exactly0.
        self.assertTrue(all(r['ci_low']==r['ci_high']==0 for r in zeros))
        self.assertTrue(all(r['ci_high']>r['ci_low'] for r in summary if r['snr_db']==4))
    def test_actual_wait_exact_worker(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td);request=base/'request.json';request.write_text('{}')
            launch=dict(owner_script_sha256=s.OBSERVER_SHA,child_pid=42,
                argv=['python',str(Path(s.__file__).resolve()),'run','--request',str(request.resolve())])
            (base/'launch.json').write_text(json.dumps(launch));(base/'exit.json').write_text(json.dumps(dict(actual_child_waited=True,exit_code=0)))
            s.actual_wait(base,request,dict(pid=42))
            with self.assertRaisesRegex(RuntimeError,'precise'):s.actual_wait(base,request,dict(pid=43))
            (base/'exit.json').write_text(json.dumps(dict(actual_child_waited=False,exit_code=0)))
            with self.assertRaisesRegex(RuntimeError,'precise'):s.actual_wait(base,request,dict(pid=42))

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    print(json.dumps(dict(engineering_tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
        actual_model_calls=0,actual_PHY_calls=0,actual_bootstrap_calls=0,fixtures_are_scientific_results=False)))
    raise SystemExit(not result.wasSuccessful())

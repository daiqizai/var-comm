"""Synthetic metadata/fake-estimator tests; never execute a scientific bootstrap."""
from pathlib import Path
import copy
import sys
import tempfile
import unittest
from unittest import mock
import numpy as np
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import leo_mismatch_statistics_v2 as s


def fixture():
    ids=[f'fixture-{i:03}' for i in range(100)];rows=[]
    for i in range(100):
        for snr in s.SNRS:
            for lookup in s.LOOKUPS[snr]:
                for family in s.FAMILIES:
                    for n in s.SEEDS:
                        difference=lookup-snr
                        rows.append(dict(source_index=i,source_id=ids[i],actual_snr_db=snr,config_snr_db=lookup,
                            family=family,noise_seed=n,psnr_db=20+i/100+difference,lpips_alex=.4-difference/100,
                            dinov2_vitl14_cosine=.6+difference/100,convnext_top1_source_prediction=int(difference<=0),
                            actual_header_ok=True,actual_crc_accepted=True,body_attempted=True,gray=False))
    return rows,ids


class NoBootstrapFixture:
    """A deterministic test double, not a statistical estimate or resample."""
    def __init__(self):self.draw_calls=0;self.intervals=[]
    def draws(self):
        self.draw_calls+=1
        return np.broadcast_to(np.arange(100,dtype=np.int64),(10000,100))
    def interval(self,vector,samples):
        self.intervals.append(vector.copy());mean=float(vector.mean())
        return dict(mean=mean,ci_low=mean-.1,ci_high=mean+.1,source_count=100,noise_count=3,frame_count=1500,
            bootstrap_seed=s.SEED,bootstrap_replicates=s.REPLICATES,bootstrap_unit='source after original3-noise mean')


class StatisticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows,cls.ids=fixture();cls.fake=NoBootstrapFixture();cls.events=[]
        cls.result=s.summarize(cls.rows,cls.ids,cls.fake,lambda *x:cls.events.append(x))

    def test_exact_contrast_and_estimator_budget(self):
        defs=s.definitions();self.assertEqual(len(defs),72)
        self.assertEqual(sum(d['contrast_kind']=='matched_zero' for d in defs),24)
        self.assertEqual(sum(d['contrast_kind']=='underestimate' for d in defs),24)
        self.assertEqual(sum(d['contrast_kind']=='overestimate' for d in defs),24)
        self.assertEqual(self.fake.draw_calls,1);self.assertEqual(len(self.fake.intervals),48)
        self.assertEqual(len(self.events),98)
        self.assertEqual(self.result[-1],48)

    def test_matched_zero_rows_preserved_without_interval(self):
        rows=[r for r in self.result[0] if r['contrast_kind']=='matched_zero']
        self.assertEqual(len(rows),24)
        self.assertTrue(all(r['mean']==r['ci_low']==r['ci_high']==0 for r in rows))
        self.assertTrue(all(r['interval_provenance']=='EXACT_MATCHED_ZERO_NO_INTERVAL_CALL' for r in rows))

    def test_lpips_original_direction_and_agreement_percentage_points(self):
        rows=[r for r in self.result[0] if r['actual_snr_db']==4 and r['config_snr_db']==7 and r['family']=='PARTIAL']
        lpips=next(r for r in rows if r['metric']=='lpips_alex')
        self.assertAlmostEqual(lpips['mean'],-.03);self.assertEqual(lpips['improvement_direction'],'negative')
        agreement=next(r for r in rows if r['metric']==s.METRICS[-1])
        self.assertEqual(agreement['mean'],-1);self.assertEqual(agreement['display_mean'],-100)
        self.assertEqual(agreement['display_unit'],'percentage_points')
        self.assertEqual(agreement['display_ci_low'],100*agreement['ci_low'])
        self.assertEqual(agreement['display_ci_high'],100*agreement['ci_high'])

    def test_source_mean_before_subtraction(self):
        rows=copy.deepcopy(self.rows)
        target=[r for r in rows if r['source_index']==0 and r['actual_snr_db']==4 and r['family']=='WHOLE']
        for row in target:
            if row['config_snr_db'] in (4,7):
                row['psnr_db']=(1e16,1.,-1e16)[s.SEEDS.index(row['noise_seed'])]+(0 if row['config_snr_db']==4 else 2)
        values,_,_=s.source_vectors(rows,self.ids)
        expected=np.mean([r['psnr_db'] for r in target if r['config_snr_db']==7],dtype=np.float64)-np.mean(
            [r['psnr_db'] for r in target if r['config_snr_db']==4],dtype=np.float64)
        self.assertEqual((values[4,7,'WHOLE','psnr_db']-values[4,4,'WHOLE','psnr_db'])[0],expected)

    def test_incomplete_and_duplicate_grid_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'Complete5400'):s.validate_grid(self.rows[:-1],self.ids)
        with self.assertRaisesRegex(RuntimeError,'Duplicate'):s.validate_grid(self.rows+[self.rows[0]],self.ids)

    def test_changed_source_order_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'outside frozen'):s.validate_grid(self.rows,list(reversed(self.ids)))

    def test_nonfinite_and_nonbinary_values_rejected(self):
        rows=copy.deepcopy(self.rows);rows[0]['lpips_alex']=float('nan')
        with self.assertRaisesRegex(RuntimeError,'Finite'):s.validate_grid(rows,self.ids)
        rows=copy.deepcopy(self.rows);rows[0][s.METRICS[-1]]=.5
        with self.assertRaisesRegex(RuntimeError,'binary'):s.validate_grid(rows,self.ids)

    def test_CRC_KEEP_is_not_gray_and_failure_counts_use_events(self):
        rows=copy.deepcopy(self.rows);r=next(r for r in rows if r['config_snr_db']==1)
        r['actual_crc_accepted']=False
        _,_,fail=s.source_vectors(rows,self.ids);cell=next(x for x in fail if x['actual_snr_db']==4 and x['config_snr_db']==1 and x['family']=='WHOLE')
        self.assertEqual(cell['body_CRC_rejections'],1);self.assertEqual(cell['fixed_gray'],0)
        self.assertEqual(cell['body_CRC_rejections_minus_matched'],1)
        r['gray']=True
        with self.assertRaisesRegex(RuntimeError,'Only actual header'):s.validate_grid(rows,self.ids)

    def test_header_failure_remains_in_quality_rows(self):
        rows=copy.deepcopy(self.rows);rows[0].update(actual_header_ok=False,actual_crc_accepted=False,body_attempted=False,gray=True)
        values,means,fail=s.source_vectors(rows,self.ids)
        self.assertEqual(len(means),7200);self.assertEqual(fail[0]['header_failures'],1)
        self.assertEqual(fail[0]['fixed_gray'],1);self.assertEqual(fail[0]['body_CRC_rejections'],0)
        self.assertEqual(values[4,1,'WHOLE','psnr_db'][0],self.rows[0]['psnr_db'])

    def test_nonclosed_metric_blocks_before_draw_or_interval(self):
        pins=dict(completion=dict(path=str(s.METRIC_ROOT/'registered/run/completion.json'),sha256='a'*64),
            owner_actual_wait=dict(path=str(s.METRIC_ROOT/'run_owner_actual_wait.json'),sha256='b'*64))
        done=dict(status=s.METRIC_PASS,schema=s.METRIC_SCHEMA,actual_wait=dict(success=True),actual_children_waited=True,
            worker_exit_codes=[0],bootstrap_calls=0,policy_selection=False)
        with mock.patch.object(s,'readpin',side_effect=[done,dict(actual_wait=True,returncode=1,timeout=False)]),\
            self.assertRaisesRegex(RuntimeError,'Actually waited complete'):
            s.metric_closure(pins)

    def test_other_attempt_cannot_be_substituted(self):
        pins=dict(completion=dict(path='H800/complete',sha256='a'*64),owner_actual_wait=dict(path='H800/wait',sha256='b'*64))
        with mock.patch.object(s,'readpin') as read,self.assertRaisesRegex(RuntimeError,'Exact leo'):
            s.metric_closure(pins)
        read.assert_not_called()

    def test_existing_receipt_never_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'claim.json';s.write(p,{'phase':'reserved'})
            with self.assertRaises(FileExistsError):s.write(p,{'phase':'completed'})
            self.assertEqual(s.readpin(s.pin(p)),{'phase':'reserved'})

    def test_frozen_statistical_dependencies_and_constants(self):
        bindings=s.source_bindings()
        self.assertEqual(s.REPLICATES,10000);self.assertEqual(s.SEED,2026100701)
        self.assertEqual(len(bindings),6)
        self.assertEqual(s.CAPS,dict(draw_matrix=1,nonmatched_intervals=48,matched_exact_zero_rows=24))


if __name__=='__main__':unittest.main()

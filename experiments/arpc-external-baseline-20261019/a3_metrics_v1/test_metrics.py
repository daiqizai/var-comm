import csv,os,pathlib,tempfile,unittest
import numpy as np
from data import METRICS,BASELINES,means,baseline_rows
from final_stats import summarize
from metric_adapter import adapt
class FixedEvaluation(unittest.TestCase):
    def logical(self):
        return [{'source_id':str(i),'source_index':i,'snr_db':s,'noise_seed':n,**{m:float(n-6200) for m in METRICS}} for i in range(100) for s in [4,10,19] for n in [6201,6202,6203]]
    def paired(self,delta):
        return [{'method':m,'snr_db':s,'metric':k,'source_id':str(i),'source_mean':10*i+(delta if m=='EC_VAR_PARTIAL' else 0)} for m in BASELINES+('ARPC_ADAPTED',) for s in [4,10,19] for k in METRICS for i in range(2)]
    def test_original_math_unchanged(self):
        root=pathlib.Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
        out=adapt((root/'experiments/entropy-partial-mismatch-20261010/scripts/h800_ep_pilot_metrics_v1.py').read_bytes())
        self.assertIn("POPULATION='ARPC_external_common100_v1_fixed_evaluation'",out)
    def test_changed_source_fails(self):
        with self.assertRaises(AssertionError):adapt(b'different metric')
    def test_source_first_three_noise(self):
        rows=means(self.logical(),{str(i) for i in range(100)})
        self.assertEqual(len(rows),2100);self.assertTrue(all(r['source_mean']==2 for r in rows))
    def test_missing_failed_frame_not_dropped(self):
        with self.assertRaises((KeyError,AssertionError)):means(self.logical()[:-1],{str(i) for i in range(100)})
    def test_duplicate_frame_rejected(self):
        rows=self.logical();rows[-1]=rows[0]
        with self.assertRaises(AssertionError):means(rows,{str(i) for i in range(100)})
    def test_pair_before_interval_not_marginal_difference(self):
        summaries,paired=summarize(self.paired(2),['0','1'],np.array([[0,0],[1,1],[0,1]]))
        self.assertTrue(all(r['ci_low']==r['ci_high']==r['mean']==2 for r in paired))
        self.assertTrue(any(r['ci_high']>r['ci_low'] for r in summaries))
    def test_negative_interval_preserved(self):
        _,paired=summarize(self.paired(-2),['0','1'],np.array([[0,0],[1,1],[0,1]]))
        self.assertTrue(all(r['mean']==-2 and r['ci_high']==-2 for r in paired))
    def test_t1c_replaces_only_entropy19_all_seven(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get('ARPC_SYNTHETIC_ROOT')) as directory:
            root=pathlib.Path(directory);paths={k:str(root/(k+'.csv')) for k in ['main','supplement','t1c']}
            sources=[{'source_index':i,'source_id':str(i)} for i in range(100)]
            def write(name,rows):
                with open(paths[name],'w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
            write('main',[{'method':m,'snr_db':s,'metric':k,'source_index':i,'source_id':str(i),'mean':1,'noise_count':3} for m in BASELINES for s in [4,10,19] for k in METRICS[:4] for i in range(100)])
            write('supplement',[{'method':m,'snr_db':s,'metric':k,'source_index':i,'source_mean':1,'noise_count':3} for m in BASELINES for s in [4,10,19] for k in METRICS[4:] for i in range(100)])
            write('t1c',[{'population':'common500','source_index':i,'family':'PARTIAL','metric':k,'new_mean':2} for i in range(100) for k in METRICS])
            rows=baseline_rows(paths,sources);self.assertEqual(len(rows),8400)
            self.assertTrue(all(r['source_mean']==(2 if r['method']=='EC_VAR_PARTIAL' and r['snr_db']==19 else 1) for r in rows))
if __name__=='__main__':unittest.main()

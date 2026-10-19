"""Synthetic contracts for expanded population, unchanged math and paired statistics."""
import ast,unittest
import numpy as np
from common import PE,BASELINES,METRICS
import link_worker,rx_cache,rx_audit,metrics_score,metric_data
from final_statistics import summary_function,conclusion
class Contracts(unittest.TestCase):
    def test_physical_operations_unchanged(self):
        raw=(PE/'a2b_link_v1/worker.py').read_bytes();text=link_worker.adapt(raw)
        self.assertIn('len(sources)==400',text);self.assertIn("RUN=L/'link_v1'",text)
        self.assertNotIn('18000',text)
        with self.assertRaises(AssertionError):link_worker.adapt(raw+b' ')
    def test_receive_audit_population(self):
        raw=(PE/'a3_rx_v1/audit.py').read_bytes();text=rx_audit.adapted(raw);ast.parse(text)
        self.assertIn('len(logical)==3600',text);self.assertNotIn('==900',text)
        with self.assertRaises(AssertionError):rx_audit.adapted(raw+b' ')
    def test_metric_calls_and_caps(self):
        raw=(PE/'a3_metrics_v1/score.py').read_bytes();text=metrics_score.adapted(raw);ast.parse(text)
        self.assertIn('reference_preparations=400',text);self.assertIn('image_scores=3600',text)
        self.assertIn('lpips_alexnet_backbone_forward=7200',text);self.assertIn('len(logical)==3600',text)
        self.assertNotIn('from metric_adapter import load',text)
        with self.assertRaises(AssertionError):metrics_score.adapted(raw+b' ')
    def fixture(self):
        sources=[{'source_id':f'id{i}','source_index':i,'rank':i,'archive':{'sha256':str(i)}}for i in range(500)if i%5]
        rows=[]
        for s in sources:
            for snr in (4,10,19):
                for noise in (6201,6202,6203):
                    rows.append({'key':f'{s["source_id"]}_{snr}_{noise}',**{k:s[k]for k in ('source_id','source_index')},'snr_db':snr,'noise_seed':noise,'candidate_id':'fixed','target_endpoint':7,'endpoint':0,'actual_status':'HEADER_FAIL','receiver_status':'FIXED_GRAY','image':None,'image_float_sha256':None})
        return sources,rows
    def test_gray_keeps_all_denominators_and_original_indices(self):
        s,r=self.fixture();g=metric_data.group(r,s,'gray')
        self.assertEqual(len(g),400);self.assertEqual(sum(len(x['pairs'])for x in g),400)
        self.assertEqual(sum(len(p['frames'])for x in g for p in x['pairs']),3600)
        self.assertEqual([x['source_index']for x in g],[i for i in range(500)if i%5])
    def test_group_duplicate_rejected(self):
        s,r=self.fixture();r[-1]=dict(r[0])
        with self.assertRaises(AssertionError):metric_data.group(r,s,'gray')
    def test_group_no_renumber(self):
        s,r=self.fixture();r[0]['source_index']=0
        with self.assertRaises(AssertionError):metric_data.group(r,s,'gray')
    def test_means_three_noise_equal_source(self):
        s,r=self.fixture()
        for row in r:
            for m in METRICS:row[m]=row['noise_seed']-6200
        out=metric_data.means(r,{x['source_id']for x in s});self.assertEqual(len(out),8400)
        self.assertTrue(all(x['source_mean']==2 and x['noise_count']==3 for x in out))
    def test_incomplete_noise_rejected(self):
        s,r=self.fixture()
        for row in r:
            for m in METRICS:row[m]=0
        with self.assertRaises(AssertionError):metric_data.means(r[:-1],{x['source_id']for x in s})
    def test_exact_cache_includes_endpoint_and_seed(self):
        match,_=rx_cache.load();wire={'sha256':'same','bytes':1};image={'sha256':'pixels','bytes':2}
        states=[{'generation_seed':0,'wire':wire,'endpoint':1,'image':image}]
        jobs=[{'key':'a','input':wire,'paid_profile':1,'generation_seed':0},{'key':'b','input':wire,'paid_profile':2,'generation_seed':0}]
        reused,new=match(jobs,states);self.assertEqual(len(reused),1);self.assertEqual(new[0]['key'],'b')
        jobs[0]['generation_seed']=1
        with self.assertRaises(AssertionError):match(jobs,states)
    def test_conflicting_cache_rejected(self):
        match,_=rx_cache.load();states=[{'generation_seed':0,'wire':{'sha256':'a','bytes':1},'endpoint':1,'image':{'sha256':v,'bytes':1}}for v in ('x','y')]
        with self.assertRaises(AssertionError):match([],states)
    def test_direct_paired_interval_uses_differences(self):
        ids=['s0','s1','s2'];draws=np.array([[0,0,0],[1,1,1],[2,2,2],[0,1,2]],dtype=np.int64)
        rows=[]
        for method in BASELINES+('ARPC_ADAPTED',):
            for snr in (4,10,19):
                for m in METRICS:
                    for i,sid in enumerate(ids):rows.append({'method':method,'snr_db':snr,'metric':m,'source_id':sid,'source_mean':float(i*100+(2 if method=='EC_VAR_PARTIAL'else 0))})
        summary,paired=summary_function()(rows,ids,draws)
        self.assertEqual(len(summary),105);self.assertEqual(len(paired),21)
        self.assertTrue(all(p['mean']==p['ci_low']==p['ci_high']==2 for p in paired))
        # Both marginal intervals are broad; their subtraction cannot yield this.
        self.assertGreater(summary[0]['ci_high']-summary[0]['ci_low'],100)
    def test_zero_negative_and_crossing_conclusions_preserved(self):
        self.assertEqual(conclusion({'ci_low':0,'ci_high':1,'direction':'higher'}),'INTERVAL_INCLUDES_ZERO')
        self.assertEqual(conclusion({'ci_low':-2,'ci_high':-1,'direction':'lower'}),'NeST_FAVORED')
        self.assertEqual(conclusion({'ci_low':-2,'ci_high':-1,'direction':'higher'}),'ARPC_FAVORED')
if __name__=='__main__':unittest.main()

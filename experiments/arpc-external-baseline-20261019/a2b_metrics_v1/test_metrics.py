import ast,copy,pathlib,unittest
from identity import equivalent
from freeze import aggregate,METRICS
import worker
def fixture():
    d={str(s):{'proposed_candidates':['a','b']} for s in [4,10,19]};rows=[]
    for sid in ['s0','s1']:
        for snr in [4,10,19]:
            for cid in ['a','b']:
                for noise in [4101,4102,4103]:rows.append(dict(source_id=sid,snr_db=snr,candidate_id=cid,noise_seed=noise,psnr_db=100 if cid=='b' else 1,lpips_alex=.1,dinov2_vitl14_cosine=.6 if cid=='a' else .4))
    return rows,d,{'s0','s1'}
class Metrics(unittest.TestCase):
    def test_usage_label_only(self):
        x={'DINO_L_metadata':{'metric_selection_usage':'cal100','weights':'same'},'flags':1};y=copy.deepcopy(x);y['DINO_L_metadata']['metric_selection_usage']='cal1000';equivalent(x,y)
    def test_changed_math_rejected(self):
        x={'DINO_L_metadata':{'metric_selection_usage':'cal100','weights':'same'},'flags':1};y=copy.deepcopy(x);y['flags']=2
        with self.assertRaises(AssertionError):equivalent(x,y)
    def test_dino_only_freezing(self):
        m,s,p=aggregate(*fixture());self.assertEqual(set(p.values()),{'a'});self.assertEqual(len(m),12)
    def test_missing_noise_rejected(self):
        rows,d,p=fixture();rows.pop()
        with self.assertRaises(AssertionError):aggregate(rows,d,p)
    def test_duplicate_noise_rejected(self):
        rows,d,p=fixture();rows.append(rows[0])
        with self.assertRaises(AssertionError):aggregate(rows,d,p)
    def test_original_setup_and_compute_ast(self):
        original=(pathlib.Path(__file__).parents[1]/'a2_metrics_v1/score.py').read_bytes();adapted=worker.adapt(original);compile(adapted,'adapted','exec')
        def compute(text):
            nodes=[n for n in ast.walk(ast.parse(text)) if isinstance(n,ast.FunctionDef) and n.name=='compute'];self.assertEqual(len(nodes),1);return ast.dump(nodes[0],include_attributes=False)
        self.assertEqual(compute(original),compute((pathlib.Path(__file__).parent/'pairs.py').read_text()))
if __name__=='__main__':unittest.main(verbosity=2)

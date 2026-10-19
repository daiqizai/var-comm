"""Population, protocol adaptation and exact pairing tests; no model imports."""
import ast,copy,hashlib,importlib.util,unittest
from pathlib import Path
import common,plan,source_native,rx_native,baseline_inputs
class Tests(unittest.TestCase):
    def fixture(self):
        full=[{'source_id':f's{i}','source_index':i,'archive':{'sha256':str(i)},'preprocessing_id':str(i)}for i in range(500)]
        return full,full[::5]
    def test_partition_preserves_indices(self):
        full,old=self.fixture();ordered,rest=common.partition(list(reversed(full)),old)
        self.assertEqual([s['source_index']for s in ordered],list(range(500)))
        self.assertEqual([s['source_index']for s in rest],[i for i in range(500)if i%5])
    def test_duplicate_rejected(self):
        full,old=self.fixture();full[-1]=full[0]
        with self.assertRaises(AssertionError):common.partition(full,old)
    def test_old_pixel_identity_rejected(self):
        full,old=self.fixture();old=copy.deepcopy(old);old[0]['archive']['sha256']='bad'
        with self.assertRaises(AssertionError):common.partition(full,old)
    def test_renumbered_subset_rejected(self):
        full,old=self.fixture();old=copy.deepcopy(old);old[1]['source_index']=1
        with self.assertRaises(AssertionError):common.partition(full,old)
    def test_native_math_unchanged(self):
        original=(common.PE/'a1_native.py').read_bytes()
        for adapter in(source_native,rx_native):
            a=ast.parse(original);b=ast.parse(adapter.adapt(original))
            self.assertEqual([ast.dump(n)for n in a.body if not(isinstance(n,ast.FunctionDef)and n.name=='main')],[ast.dump(n)for n in b.body if not(isinstance(n,ast.FunctionDef)and n.name=='main')])
    def test_native_hash_tamper_rejected(self):
        for adapter in(source_native,rx_native):
            with self.assertRaises(AssertionError):adapter.adapt(b'print(123)')
    def test_tx_no_reference_render(self):
        text=source_native.adapt((common.PE/'a1_native.py').read_bytes())
        self.assertIn("'caption':400,'tokenizer':400,'tx_probability':400,'reference_render':0",text)
        self.assertIn('if False:',text)
    def test_plan_only_cap_changed(self):
        old=(common.PE/'a3_link_v1/plan.py').read_text();new=Path(plan.__file__).read_text()
        self.assertEqual(ast.dump(ast.parse(old)),ast.dump(ast.parse(new.replace('len(out)<=3600','len(out)<=900'))))
    def test_event_counter_and_noise(self):
        streams=[{'endpoint':i,'payload_bits':i*8,'wire':{'sha256':str(i)}}for i in range(1,8)]
        profiles=[{'endpoint':e,'q':q,'nominal_rate':rate,'profile_id':n*8+e,'source_capacity_bits':40}for n,(q,rate)in enumerate([(2,'2/3'),(4,'2/3'),(6,'5/6')])for e in range(8)]
        rows=plan.events([{'source_id':'s499','source_index':499,'streams':streams}],{s:{'proposed_candidates':[cid]}for s,cid in common.POLICIES.items()},profiles)
        self.assertEqual(len(rows),9);self.assertEqual({r['noise_seed']for r in rows},{6201,6202,6203})
        self.assertEqual({r['endpoint']for r in rows},{5})
        for r in rows:self.assertEqual(r['counter'],3000000+96*499+r['profile_id'])
    def test_null_preserved(self):
        streams=[{'endpoint':i,'payload_bits':99999,'wire':{'sha256':str(i)}}for i in range(1,8)]
        profiles=[{'endpoint':e,'q':q,'nominal_rate':rate,'profile_id':n*8+e,'source_capacity_bits':40}for n,(q,rate)in enumerate([(2,'2/3'),(4,'2/3'),(6,'5/6')])for e in range(8)]
        rows=plan.events([{'source_id':'s0','source_index':0,'streams':streams}],{s:{'proposed_candidates':[cid]}for s,cid in common.POLICIES.items()},profiles)
        self.assertEqual(len(rows),9);self.assertTrue(all(r['source_no_fit']and r['wire']is None for r in rows))
    def test_bpg_alias(self):
        self.assertEqual(baseline_inputs.ALIASES,{'BPG_ADAPTIVE':'BPG_ADAPTIVE_DOWNSAMPLING_LDPC_N1024'})
    def test_missing_baseline_rejected(self):
        full,_=self.fixture()
        with self.assertRaises(AssertionError):baseline_inputs.join_rows(full,[],[],[])
if __name__=='__main__':unittest.main()

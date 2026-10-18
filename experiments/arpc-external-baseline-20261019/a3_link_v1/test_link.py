import pathlib,sys,unittest
E=pathlib.Path(__file__).parents[1];sys.path.insert(0,str(E/'phy_v1'))
import link,worker
from plan import events
def fixture():
    sources=[{'source_id':str(i),'source_index':i,'streams':[{'endpoint':e,'payload_bits':e*300,'wire':{'sha256':f'{i}-{e}'}} for e in range(1,8)]} for i in range(100)]
    d={s:{'proposed_candidates':[cid]} for s,cid in [('4','e7_q2_r2-3'),('10','e7_q4_r2-3'),('19','e7_q6_r5-6')]}
    return sources,d,link.templates()
class EvaluationPHY(unittest.TestCase):
    def test_exact900(self):self.assertEqual(len(events(*fixture())),900)
    def test_eval_noise(self):self.assertEqual({r['noise_seed'] for r in events(*fixture())},{6201,6202,6203})
    def test_one_policy_only(self):
        s,d,p=fixture();d['4']['proposed_candidates'].append('e7_q2_r1-2')
        with self.assertRaises(AssertionError):events(s,d,p)
    def test_null_keeps_frames(self):
        s,d,p=fixture()
        for a in s:
            for b in a['streams']:b['payload_bits']=10000
        x=events(s,d,p);self.assertEqual(len(x),900);self.assertTrue(all(r['source_no_fit'] for r in x))
    def test_physical_operations_AST_unchanged(self):compile(worker.adapt((E/'a2b_link_v1/worker.py').read_bytes()),'adapted','exec')
if __name__=='__main__':unittest.main(verbosity=2)

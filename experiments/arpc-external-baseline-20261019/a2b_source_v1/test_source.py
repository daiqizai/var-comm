import ast,pathlib,unittest
import native
from population import missing
SOURCE=pathlib.Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/arpc-external-baseline-20261019/a1_native.py')
def fixture():
    full=[{'source_id':str(i),'source_index':i,'archive':{'sha256':str(i)},'preprocessing_id':'fixed'} for i in range(1000)]
    return full,full[::10]
class Scope(unittest.TestCase):
    def test_no_duplicate100_passes(self):
        a,b=fixture();n=missing(a,b);self.assertEqual(len(n),900);self.assertFalse({s['source_id'] for s in n}&{s['source_id'] for s in b})
    def test_keeps_original_index(self):
        a,b=fixture();self.assertEqual(missing(a,b)[0]['rank'],1)
    def test_source_mismatch_stops(self):
        a,b=fixture();b=[dict(x) for x in b];b[0]['archive']={'sha256':'changed'}
        with self.assertRaises(AssertionError):missing(a,b)
    def test_duplicates_stop(self):
        a,b=fixture();b[0]=b[1]
        with self.assertRaises(AssertionError):missing(a,b)
    def test_reversible_native_edits(self):
        data=SOURCE.read_bytes();s=native.adapt(data)
        for a,b in reversed(native.EDITS):s=s.replace(b,a)
        self.assertEqual(s.encode(),data)
    def test_zero_reference_cap(self):
        s=native.adapt(SOURCE.read_bytes());self.assertIn("'reference_render':0",s)
        tree=ast.parse(s);loops=[n for n in ast.walk(tree) if isinstance(n,ast.If) and isinstance(n.test,ast.Constant) and n.test.value is False]
        self.assertEqual(len(loops),1);self.assertIn('reference_render',ast.unparse(loops[0]))
if __name__=='__main__':unittest.main(verbosity=2)

import unittest,ast,pathlib
import numpy as np
from mixed_arithmetic_v1 import classes,validate_cdf,validate_bits,FULL,HALF,QUARTER,TOTAL
from nested_entropy import encode_candidates,decode_residual
SOURCE=pathlib.Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/src/var_comm/whole_entropy.py')
class MixedTests(unittest.TestCase):
    def test_coarse_bitwise_regression(self):
        E,D=classes(SOURCE);tree=ast.parse(SOURCE.read_text());nodes=[n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='ArithmeticEncoder'];ns=dict(np=np,FULL=FULL,HALF=HALF,QUARTER=QUARTER,TOTAL=TOTAL,validate_cdf=validate_cdf)
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(SOURCE),'exec'),ns)
        rng=np.random.default_rng(2026102801);counts=rng.multinomial(TOTAL-4096,np.full(4096,1/4096))+1;cdf=np.r_[0,counts.cumsum()][None].repeat(30,0);v=rng.integers(0,4096,30)
        a=E();b=ns['ArithmeticEncoder']();a.encode(v,cdf);b.encode(v,cdf);np.testing.assert_array_equal(a.finish(),b.finish());np.testing.assert_array_equal(D(a.finish()).decode(cdf),v)
    def test_all_mixed_endpoints(self):
        E,D=classes(SOURCE);rng=np.random.default_rng(7);v=rng.integers(0,16,64);prob=rng.random((64,16))+.01;prob/=prob.sum(1,keepdims=True);cdf=np.arange(4097,dtype=np.int64)[None]*4096;coarse=np.array([2047])
        for order in('scalar','progressive'):
            streams,_=encode_candidates(E,[coarse],[cdf],v,prob,order)
            for j,bits in streams.items():
                d=D(bits);e=E();rx=d.decode(cdf);e.encode(rx,cdf);q,depth=decode_residual(d,e,j,prob,order);np.testing.assert_array_equal(rx,coarse);np.testing.assert_array_equal(e.finish(),bits);self.assertEqual(d.position-len(bits),30)
                prefix=v>>(4-depth);want=np.where(depth>0,-1+(2*prefix+1)/(2.**depth),0).astype(np.float32);np.testing.assert_array_equal(q,want)
    def test_invalid_tables(self):
        for a in [np.array([[0,0,TOTAL]]),np.array([[0,2,TOTAL-1]]),np.array([[0,.5,TOTAL]]),np.array([[0,2,3,TOTAL]])]:
            with self.assertRaises(ValueError):validate_cdf(a)
if __name__=='__main__':unittest.main()

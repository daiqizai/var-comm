"""Synthetic algorithm parity tests only, not a real-model qualification."""
import ast,contextlib,unittest
from pathlib import Path
import numpy as np
from timed_codec import encode,endpoints,select,OFFSETS

class FakeEncoder:
    def __init__(self):self.bits=[]
    def encode(self,values,cdf):
        for x,c in zip(values,cdf):self.bits += [int(x%2),int(c[0]%2)]
    def finish(self):return np.asarray(self.bits+[1,0],np.uint8)

class Codec:
    Encoder=FakeEncoder
    def _inference(self,f):return contextlib.nullcontext()
    def _context(self,f):return contextlib.nullcontext(self)
    def _cdf(self,f,p,s):return np.full((OFFSETS[s+1]-OFFSETS[s],2),s,np.int64)
    def advance(self,v):pass

class Tests(unittest.TestCase):
    def test_same_traversal_and_nonmutating_finish(self):
        root=Path(__file__).resolve().parents[2]
        text=(root/'experiments/submission-closure-20261016/m10_codec.py').read_text()
        node=next(n for n in ast.parse(text).body if isinstance(n,ast.ClassDef) and n.name=='AuditCodec')
        tokens=np.arange(680)%4096
        for grid in (4,8):
            ns=dict(PartialSourceCodec=Codec,np=np,source_tokens=lambda x:x,OFFSETS=OFFSETS,ENDS=endpoints(grid),asha=lambda x:x.tolist(),WHOLE_FAMILY='EC_VAR_WHOLE')
            exec(compile(ast.Module(body=[node],type_ignores=[]),'original_audit_codec','exec'),ns)
            expected,_=ns['AuditCodec']().encode_all(tokens)
            actual,_,_,counts=encode(Codec(),tokens,grid,lambda:None)
            self.assertEqual(counts['encoded_token_count'],680)
            self.assertEqual(counts['prior_scale_calls'],10)
            self.assertEqual(counts['finish_calls'],len(endpoints(grid)))
            for e,v in expected.items():np.testing.assert_array_equal(v['bits'],actual[e])
        encoder=FakeEncoder();encoder.encode([1],[[2,3]]);before=encoder.bits.copy();encoder.finish();self.assertEqual(before,encoder.bits)
    def test_independent_grid_sets(self):
        self.assertEqual(len(endpoints(4)),25);self.assertEqual(len(endpoints(8)),49)
        self.assertTrue(set(endpoints(4))<set(endpoints(8)))
    def test_longest_feasible_without_monotone_lengths(self):
        es=endpoints(4);streams={e:np.zeros(200,np.uint8) for e in es};streams[es[-2]]=np.zeros(100,np.uint8);streams[es[-3]]=np.zeros(50,np.uint8)
        profiles=[dict(q=4,nominal_rate='2/3',source_capacity_bits=100,m=m,K=k,family='EC_VAR_PARTIAL' if k else 'EC_VAR_WHOLE') for m,k in es]
        p,b,c=select(streams,4,profiles);self.assertEqual((p['m'],p['K']),es[-2]);self.assertEqual(c,2)

if __name__=='__main__':unittest.main()

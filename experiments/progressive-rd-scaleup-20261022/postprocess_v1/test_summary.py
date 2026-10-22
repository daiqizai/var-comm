"""Fixed synthetic arrays only, no production data or scientific bootstrap bank."""
import ast,pathlib,unittest,numpy as np
source=pathlib.Path(__file__).with_name('summarize_scaleup.py')
node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='interval')
ns=dict(np=np);exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),ns);interval=ns['interval']
class StatsTests(unittest.TestCase):
 def test_constant_offset_direct_paired_interval(self):
  bank=np.array([[0,0,1],[2,2,2],[1,0,2],[1,1,0]],dtype=np.int64);b=np.array([1.,4.,9.]);a=b+.25
  result=interval(a-b,bank);self.assertEqual(result,dict(mean=.25,ci_low=.25,ci_high=.25))
  # Subtracting marginal interval endpoints would be incorrect here.
  self.assertNotEqual(interval(a,bank)['ci_low']-interval(b,bank)['ci_high'],.25)
 def test_seed_mean_keeps_source_pair(self):
  seed1=np.array([1.,2.,3.]);seed2=np.array([3.,6.,9.]);base=np.array([2.,4.,6.]);bank=np.array([[0,0,1],[0,1,2]],dtype=np.int64)
  self.assertEqual(interval((seed1+seed2)/2-base,bank),dict(mean=0.,ci_low=0.,ci_high=0.))
 def test_zero_and_negative_retained(self):
  bank=np.array([[0,1,2],[2,1,0]],dtype=np.int64);self.assertEqual(interval([-3.,0.,0.],bank),dict(mean=-1.,ci_low=-1.,ci_high=-1.))
 def test_nonfinite_and_wrong_population_rejected(self):
  bank=np.zeros((2,3),dtype=np.int64)
  with self.assertRaises(AssertionError):interval([1.,float('nan'),2.],bank)
  with self.assertRaises(AssertionError):interval([1.,2.],bank)
if __name__=='__main__':unittest.main()

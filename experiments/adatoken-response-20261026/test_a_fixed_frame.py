import unittest,sys
from common import *
from fixed_frame import catalogue,choose,NULL
import numpy as np
class Grid(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.cat=read(sys.argv[1]);cls.new=catalogue(cls.cat,'A')
 def test_old_profile_bytes_and_payment(self):
  self.assertEqual(self.new['profiles'][:360],self.cat['profiles']);self.assertLess(len(self.new['profiles']),4096)
  self.assertEqual(len({p['profile_id']for p in self.new['profiles']}),len(self.new['profiles']))
 def test_all_nested_grids(self):
  g=grids();self.assertEqual([len(g[str(i)])for i in(1,2,4,8)],[7,13,25,49]);self.assertTrue(set(g['1'])<set(g['2'])<set(g['4'])<set(g['8']))
 def test_exact_fit_and_all_fail(self):
  for q,rate in ((2,'2/3'),(4,'2/3'),(6,'5/6')):
   cap=next(p['source_capacity_bits']for p in self.new['profiles']if p['q']==q and p['nominal_rate']==rate)
   streams={'m%d_K%d'%e:np.zeros(cap+1,np.uint8)for e in grids()['8']};streams['m8_K0']=np.zeros(cap,np.uint8)
   for grid in(1,2,4,8):
    p,b,checks=choose(streams,self.new['profiles'],'A',q,rate,grid);self.assertEqual((p['m'],p['K']),(8,0));self.assertEqual(len(b),cap);self.assertTrue(checks[-1]['fits'])
   streams['m8_K0']=np.zeros(cap+1,np.uint8);p,b,_=choose(streams,self.new['profiles'],'A',q,rate,8);self.assertEqual(p['family'],NULL);self.assertEqual(len(b),2)
 def test_nonmonotonic_finished_lengths(self):
  streams={'m%d_K%d'%e:np.zeros(9000,np.uint8)for e in grids()['8']};streams['m9_K32']=np.zeros(2,np.uint8);streams['m9_K0']=np.zeros(2,np.uint8)
  p,_,_=choose(streams,self.new['profiles'],'A',2,'2/3',8);self.assertEqual((p['m'],p['K']),(9,32))
  p,_,_=choose(streams,self.new['profiles'],'A',2,'2/3',4);self.assertEqual((p['m'],p['K']),(9,0))
 def test_no_model_imports(self):self.assertNotIn('torch',sys.modules);self.assertNotIn('transformers',sys.modules)
if __name__=='__main__':unittest.main(argv=[sys.argv[0]],verbosity=2)

"""Synthetic aggregation checks; no data, models, channel or random draws."""
import unittest
import numpy as np
from formal_statistics import source_aggregate,ci,METRICS,METHODS,SNRS

def grid():
 identity={i:f'source_{i}'for i in range(500)};rows=[]
 for mi,m in enumerate(METHODS):
  for s in SNRS:
   for i in range(500):
    for n in (6201,6202,6203):
     rows.append(dict(method=m,snr_db=s,source_index=i,source_id=identity[i],noise_seed=n,**{k:float(i+mi*10+n-6201)for k in METRICS}))
 return rows,identity

class AggregateTests(unittest.TestCase):
 def test_source_average_and_order(self):
  rows,ids=grid();out,v=source_aggregate(list(reversed(rows)),ids)
  self.assertEqual(len(out),18000);np.testing.assert_array_equal(v['NEST',4,'psnr_db'],np.arange(500)+1)
 def test_failure_keeps_one_third_weight(self):
  rows,ids=grid();rows[0]['psnr_db']=-8;_,v=source_aggregate(rows,ids);self.assertEqual(v['NEST',4,'psnr_db'][0],-5/3)
 def test_missing_noise_rejected(self):
  rows,ids=grid()
  with self.assertRaises(AssertionError):source_aggregate(rows[1:],ids)
 def test_duplicate_rejected(self):
  rows,ids=grid()
  with self.assertRaises(AssertionError):source_aggregate(rows+[rows[0]],ids)
 def test_wrong_identity_rejected(self):
  rows,ids=grid();rows[0]['source_id']='wrong'
  with self.assertRaises(AssertionError):source_aggregate(rows,ids)
 def test_direct_paired_direction(self):
  rows,ids=grid();_,v=source_aggregate(rows,ids)
  bank=np.tile(np.arange(500,dtype=np.int64),(2,1))
  result=ci(v['NEST',19,'dreamsim']-v['ADA',19,'dreamsim'],bank)
  self.assertEqual(result,dict(mean=-20.,ci_low=-20.,ci_high=-20.))

if __name__=='__main__':unittest.main()

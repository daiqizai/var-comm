"""Synthetic edge tests: no real files, model, packet or quality calls."""
import sys,unittest,copy
import tc_common
import numpy as np
import tc_protocol as p
import tb_plan as plan
def grid():
 rows=[]
 for pop,(n,seeds) in p.POPS.items():
  for i in range(n):
   for fam in ('WHOLE','PARTIAL'):
    for seed in seeds:
     rows.append(dict(population=pop,source_index=i,source_global_index=i+(500 if pop=='new100' else 0),source_id=pop+str(i),snr_db=19,noise_seed=seed,strategy_family=fam,public_frame_counter=((5 if pop=='common500' else 2)*n+i)*3+seeds.index(seed),candidate_id=f'm10_{fam.lower()}_q6_r5-6',m=10,K=0,arithmetic_bits=4751,profile_id=371,profile_key='paid',payload_sha256='bits',event_key=f'{pop}/{i}/{seed}',reuse=False,metrics={'v':i+seeds.index(seed)}))
 return rows
class Test(unittest.TestCase):
 def test_shared_event_dedup(self):self.assertEqual(p.validate_grid(grid())['new_physical'],1800)
 def test_duplicate(self):
  a=grid();a[-1]=a[0]
  with self.assertRaises(AssertionError):p.validate_grid(a)
 def test_missing(self):
  with self.assertRaises(AssertionError):p.validate_grid(grid()[:-1])
 def test_counter(self):
  a=grid();a[0]['public_frame_counter']+=1
  with self.assertRaises(AssertionError):p.validate_grid(a)
 def test_mixed_reuse_same_event(self):
  a=grid();a[0]['reuse']=True
  with self.assertRaises(AssertionError):p.validate_grid(a)
 def test_paid_cost(self):
  a=grid();a[0]['arithmetic_bits']=4780
  with self.assertRaises(AssertionError):p.validate_grid(a)
 def test_dataset_overlap(self):
  a=grid()
  for r in a:
   if r['population']=='new100' and r['source_index']==0:r['source_id']='common5000'
  with self.assertRaises(AssertionError):p.validate_grid(a)
 def test_illegal_old_id_not_reusable(self):self.assertFalse(plan.directory_unchanged_on_old_event(dict(header_crc_ok=True,header_fields_legal=False)))
 def test_crc_failure_domain_invariant(self):self.assertTrue(plan.directory_unchanged_on_old_event(dict(header_crc_ok=False,header_fields_legal=False)))
 def test_noise_means(self):np.testing.assert_array_equal(p.source_means(grid(),'new100','WHOLE','v'),np.arange(100)+1)
 def test_missing_noise_rejected(self):
  with self.assertRaises(AssertionError):p.source_means(grid()[:-1],'new100','PARTIAL','v')
 def test_paired_constant_difference(self):
  idx=np.arange(10)[None,:].repeat(10,axis=0);a=np.arange(10.)*100;b=a-2
  self.assertEqual(p.interval(a-b,idx),dict(mean=2.,ci_low=2.,ci_high=2.))
 def test_bool_exit_rejected(self):
  w=dict(actual_wait=True,returncode=False,terminated=False,killed=False,timed_out=False,metadata_error=None,process={},parent_process={})
  with self.assertRaises(AssertionError):p.closed_wait(w,lambda v:True)
 def test_live_parent_rejected(self):
  w=dict(actual_wait=True,returncode=0,terminated=False,killed=False,timed_out=False,metadata_error=None,process={'gone':True},parent_process={'gone':False})
  with self.assertRaises(AssertionError):p.closed_wait(w,lambda v:v['gone'])
 def test_finish_length_boundary_and_fallback(self):
  ls={e:5000 for e in plan.ENDS};ls[10,0]=4752;ls[9,0]=4000;ls[9,192]=4700
  self.assertEqual(plan.choose(ls,'WHOLE',4751),(9,0));self.assertEqual(plan.choose(ls,'PARTIAL',4751),(9,192));ls[10,0]=4751
  self.assertEqual(plan.choose(ls,'PARTIAL',4751),(10,0))
if __name__=='__main__':unittest.main(argv=[sys.argv[0]],verbosity=2)

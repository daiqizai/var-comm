"""Synthetic scope/receiver/statistical invariants; no production assets are read."""
import unittest,json,copy,ast,hashlib,sys
from pathlib import Path
import numpy as np
from nc_protocol import validate_repair_grid,validate_full_grid,closed_wait,receiver_reuse,source_means,interval
def synthetic():
 out=[]
 for snr,count in ((4,497),(10,310)):
  for i in range(count):
   for seed in (6201,6202,6203):out.append(dict(population='common500',source_index=i,source_global_index=i,source_id='synthetic/'+str(i),snr_db=snr,noise_seed=seed,strategy_family='WHOLE',m=6 if snr==4 else 8,K=0,arithmetic_bits=900 if snr==4 else 2400,candidate_id='m7_q2_r2-3' if snr==4 else 'm9_q4_r2-3',reuse=False,public_frame_counter=(([1,4,7,10,13,19].index(snr)*500+i)*3+seed-6201),event_key=f'{i}/{snr}/{seed}'))
 return out
class Tests(unittest.TestCase):
 def test_exact_scope(self):self.assertEqual(validate_repair_grid(synthetic())['packet_cap'],4842)
 def reject(self,key,val):
  x=synthetic();x[0][key]=val
  with self.assertRaises(AssertionError):validate_repair_grid(x)
 def test_reject_19(self):self.reject('snr_db',19)
 def test_reject_partial(self):self.reject('strategy_family','PARTIAL')
 def test_reject_reused_physical(self):self.reject('reuse',True)
 def test_reject_new100(self):self.reject('population','new100')
 def test_reject_new_source(self):self.reject('source_index',500)
 def test_reject_new_noise(self):self.reject('noise_seed',4101)
 def test_reject_unpaid_endpoint(self):self.reject('K',1)
 def test_reject_over_capacity(self):self.reject('arithmetic_bits',1246)
 def test_reject_changed_counter(self):self.reject('public_frame_counter',0)
 def test_reject_reselection(self):self.reject('candidate_id','new_policy')
 def test_reject_duplicate(self):
  x=synthetic();x[-1]=copy.deepcopy(x[0])
  with self.assertRaises(AssertionError):validate_repair_grid(x)
 def test_reject_count(self):
  with self.assertRaises(AssertionError):validate_repair_grid(synthetic()[:-1])
 def test_full_grid(self):
  x=[dict(population='common500',source_index=i,snr_db=s,noise_seed=n,family=f) for i in range(500) for s in (4,10,19) for n in (6201,6202,6203) for f in ('WHOLE','PARTIAL')];validate_full_grid(x)
  x[-1]=x[0]
  with self.assertRaises(AssertionError):validate_full_grid(x)
 def test_closed_exit_strict(self):
  v=dict(actual_wait=True,returncode=0,terminated=False,killed=False,timed_out=False,metadata_error=None,process={'pid':1},parent_process={'pid':2});closed_wait(v,lambda _:True)
  for field,value in [('returncode',False),('returncode',1),('actual_wait',False),('terminated',True),('metadata_error','lost')]:
   with self.subTest(field=field,value=value):
    a=dict(v);a[field]=value
    with self.assertRaises(AssertionError):closed_wait(a,lambda _:True)
  with self.assertRaises(AssertionError):closed_wait(v,lambda _:False)
 def test_received_cache_exact(self):
  cache={'k':dict(source_input_key='k',validated_unique_fixed_pixel=True,source_truth_used=False,reconstruction={'image_sha256':'pixel'})}
  self.assertEqual(receiver_reuse(cache,'k',{},lambda _:'k')['image_sha256'],'pixel')
  self.assertIsNone(receiver_reuse(cache,'other',{},lambda _:'other'))
  with self.assertRaises(AssertionError):receiver_reuse(cache,'k',{},lambda _:'different')
  cache['k']['source_truth_used']=True
  with self.assertRaises(AssertionError):receiver_reuse(cache,'k',{},lambda _:'k')
 def test_source_noise_average_and_negative_difference(self):
  rows=[dict(population='common500',source_index=i,snr_db=4,noise_seed=n,family=f,metrics={'m':i+(n-6201)+(0 if f=='WHOLE' else -2)}) for i in range(500) for n in (6201,6202,6203) for f in ('WHOLE','PARTIAL')]
  a=source_means(rows,4,'PARTIAL','m')-source_means(rows,4,'WHOLE','m');self.assertTrue(np.all(a==-2))
  idx=np.tile(np.arange(500,dtype=np.int64),(10000,1));v=interval(a,idx);self.assertEqual(v,dict(mean=-2.0,ci_low=-2.0,ci_high=-2.0))
 def test_failures_remain_denominator(self):
  rows=[dict(source_index=i,snr_db=4,noise_seed=n,family='WHOLE',metrics={'m':0. if n==6201 else 1.}) for i in range(500) for n in (6201,6202,6203)]
  self.assertTrue(np.allclose(source_means(rows,4,'WHOLE','m'),2/3))
 def test_no_bootstrap_random_draw(self):
  s=(Path(__file__).parent/'nc_delivery.py').read_text();self.assertNotIn('random',s);self.assertIn("r['bootstrap']['indices']",s)
 def test_exact_original_source_transformations(self):
  h=Path(__file__).parent;proof=json.loads((h/'source_transformations.json').read_text())
  for p in proof:
   output=(h/p['output']).read_bytes();self.assertEqual(hashlib.sha256(output).hexdigest(),p['output_sha256'])
   self.assertIsNotNone(ast.parse(output))
 def test_receiver_has_no_TX_input_keys(self):
  s=(Path(__file__).parent/'nc_receiver.py').read_text()
  for bad in ("r['sources']","r['frames']","r['receiver_cache']","r['reused_frames']","r['source_archive']"):self.assertNotIn(bad,s)
  self.assertIn('firewall(out,allow',s);self.assertIn("registration('receiver',expected)",s)
 def test_repair_PHY_exact_SNR(self):
  s=(Path(__file__).parent/'nc_phy.py').read_text();self.assertIn("rt.receive(observed,row['snr_db']",s);self.assertIn("noise=standard*10**(-row['snr_db']/20)",s)
 def test_fresh_stage_is_exclusive(self):
  for name in ('nc_phy.py','nc_receiver.py','nc_metrics.py','nc_delivery.py'):
   s=(Path(__file__).parent/name).read_text();self.assertIn('out.mkdir()',s);self.assertNotIn('exist_ok=True',s)
if __name__=='__main__':
 class Guard:
  def find_spec(self,fullname,path=None,target=None):
   if fullname.split('.')[0] in {'torch','tensorflow','timm','lpips','transformers','diffusers','sionna'}:raise AssertionError('Forbidden model import '+fullname)
 sys.meta_path.insert(0,Guard())
 result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
 if result.wasSuccessful():
  out=Path(sys.argv[1]);out.mkdir(exist_ok=False)
  (out/'completion.json').write_text(json.dumps(dict(status='PASS_SYNTHETIC_ONLY',tests=result.testsRun,production_scientific_calls=0,blocked=[],model_imports=[]),indent=2)+'\n',encoding='utf8')
 sys.exit(not result.wasSuccessful())

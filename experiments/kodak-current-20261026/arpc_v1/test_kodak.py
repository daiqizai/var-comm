"""CPU-only fixtures; imports neither torch nor physical/runtime models."""
import ast,hashlib,importlib.util,pathlib,sys,unittest
E=pathlib.Path(__file__).resolve().parent
R=pathlib.Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
W=E.parents[2]
def load(path,name):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
native=load(E/'native.py','kodak_test_native');phy=load(E/'phy.py','kodak_test_phy');plan=load(E/'kodak_plan.py','kodak_test_plan')
if R.exists():
 old=R/'experiments/arpc-common500-extension-20261019';raw=R/'experiments/arpc-external-baseline-20261019/a1_native.py';worker=R/'experiments/arpc-external-baseline-20261019/a2b_link_v1/worker.py'
else:
 old=W/'experiments/arpc-common500-extension-20261019';snap=W/'.research/kodak_current_20261026_v1/arpc_work/snapshot';raw=snap/'experiments/arpc-external-baseline-20261019/a1_native.py';worker=snap/'experiments/arpc-external-baseline-20261019/a2b_link_v1/worker.py'
source_adapter=load(old/'source_native.py','test_old_source_adapter');rx_adapter=load(old/'rx_native.py','test_old_rx_adapter')
class TestKodak(unittest.TestCase):
 def test_native_source_scope(self):
  s=native.adapt_source(raw.read_bytes(),source_adapter);compile(s,'source','exec')
  self.assertIn("'caption':24,'tokenizer':24,'tx_probability':24,'reference_render':0",s);self.assertIn('len(sources)==24',s);self.assertIn('if False:',s)
 def test_native_rx_scope(self):
  s=native.adapt_rx(raw.read_bytes(),rx_adapter);compile(s,'rx','exec');self.assertIn('len(jobs)<=216',s);self.assertIn("source_error.append(str(error));return None",s)
 def test_math_functions_unchanged(self):
  orig=ast.parse(raw.read_bytes());a=ast.parse(native.adapt_source(raw.read_bytes(),source_adapter));b=ast.parse(native.adapt_rx(raw.read_bytes(),rx_adapter))
  funcs=lambda t:{n.name:ast.dump(n,include_attributes=False)for n in t.body if isinstance(n,ast.FunctionDef)and n.name!='main'}
  self.assertEqual(funcs(orig),funcs(a));self.assertEqual(funcs(orig),funcs(b))
 def test_wrong_native_sha_rejected(self):
  with self.assertRaises(AssertionError):native.adapt_source(raw.read_bytes()+b' ',source_adapter)
 def test_phy_adapter(self):
  s=phy.adapt(worker.read_bytes());compile(s,'phy','exec');self.assertIn('len(sources)==24',s);self.assertNotIn('18000',s);self.assertIn('from kodak_plan import events,digest',s)
 def test_wrong_phy_sha_rejected(self):
  with self.assertRaises(AssertionError):phy.adapt(worker.read_bytes()+b' ')
 def test_population_events(self):
  events=plan.load((old/'plan.py').read_bytes())['events']
  profiles=[]
  for e in range(8):
   for q,rate,capacity in [(2,'2/3',1245),(4,'2/3',2520),(6,'5/6',4751)]:profiles.append({'profile_id':len(profiles),'endpoint':e,'q':q,'nominal_rate':rate,'source_capacity_bits':capacity})
  src=[{'source_index':i,'source_id':f'kodak/kodim{i+1:02d}','streams':[{'endpoint':e,'payload_bits':e*500,'wire':{'sha256':str(i)+'e'+str(e)}}for e in range(1,8)]}for i in range(24)]
  cfg={str(s):{'proposed_candidates':[c]}for s,c in [(4,'e7_q2_r2-3'),(10,'e7_q4_r2-3'),(19,'e7_q6_r5-6')]}
  rows=events(src,cfg,profiles);self.assertEqual(len(rows),216);self.assertEqual(set(r['noise_seed']for r in rows),{2001,2002,2003});self.assertEqual(set(r['source_index']for r in rows),set(range(24)))
  for r in rows:self.assertEqual(r['counter'],3000000+96*r['source_index']+r['profile_id']);self.assertEqual(r['endpoint'],{4:2,10:5,19:7}[r['snr_db']])
 def test_null_keeps_denominator(self):
  events=plan.load((old/'plan.py').read_bytes())['events'];profiles=[]
  for e in range(8):
   for q,rate in [(2,'2/3'),(4,'2/3'),(6,'5/6')]:profiles.append({'profile_id':len(profiles),'endpoint':e,'q':q,'nominal_rate':rate,'source_capacity_bits':1})
  src=[{'source_index':0,'source_id':'fixture','streams':[{'endpoint':e,'payload_bits':100,'wire':{'sha256':str(e)}}for e in range(1,8)]}]
  cfg={str(s):{'proposed_candidates':[c]}for s,c in [(4,'e7_q2_r2-3'),(10,'e7_q4_r2-3'),(19,'e7_q6_r5-6')]};rows=events(src,cfg,profiles);self.assertEqual(len(rows),9);self.assertTrue(all(r['endpoint']==0 and r['wire']is None and r['source_no_fit']for r in rows))
 def test_source_frontier_not_in_rx(self):
  s=native.adapt_rx(raw.read_bytes(),rx_adapter);tree=ast.parse(s);body=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main');branch=next(n for n in ast.walk(body)if isinstance(n,ast.If)and ast.unparse(n.test)=="role == 'tx'")
  rx=ast.Module(body=branch.orelse,type_ignores=[]);reads=[n.id for n in ast.walk(rx)if isinstance(n,ast.Name)and isinstance(n.ctx,ast.Load)];self.assertNotIn('sources',reads);self.assertNotIn('pixels',reads)
 def test_new_control_compiles(self):compile((E/'control.py').read_bytes(),'control','exec')
if __name__=='__main__':unittest.main()

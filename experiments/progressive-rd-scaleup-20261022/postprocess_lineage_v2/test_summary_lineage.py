import unittest,tempfile,pathlib,json,ast
from summary_lineage import verify_routes,pin,NAMES
class LineageTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.L=pathlib.Path(self.tmp.name)/'L';self.O=pathlib.Path(self.tmp.name)/'O';self.O.mkdir();self.jobs=['first',NAMES[0],'last',NAMES[1]]
  def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v))
  self.save=save;fails={}
  for n in NAMES:
   p=self.L/'dispatch_v1'/n/'failure.json';save(p,dict(error="AssertionError(['gpu', '71465', '97'])",traceback='admitted(int(fields[1]),int(fields[2]))'));fails[n]=pin(p)
  save(self.O/'unlaunched_training_recovery_preparation_v1.json',dict(jobs=list(NAMES),prior_scientific_attempts=0,extra_scientific_budget=0,failures=fails,sources=[]))
  for x in ['initial_control_v1','outer_dispatch_v1','recovery_initial_control_v1','recovery_outer_dispatch_v1']:save(self.L/x/'launch.json',dict(process=dict(pid=1,start_ticks=2)))
  save(self.L/'outer_dispatch_v1/actual_wait.json',dict(returncode=1,process=dict(pid=1,start_ticks=2)))
  save(self.L/'recovery_outer_dispatch_v1/actual_wait.json',dict(returncode=0,process=dict(pid=1,start_ticks=2)))
  results=[dict(error="AssertionError(['gpu', '71465', '97'])")if n in NAMES else dict(returncode=0)for n in self.jobs]
  save(self.L/'dispatch_v1/completion.json',dict(all_exit0=False,results=results))
  save(self.L/'dispatch_v1/failure.json',dict(error="AssertionError('A bounded training job failed; preserve and diagnose')"))
  save(self.L/'outer_dispatch_v1/failure.json',dict(error="RuntimeError('Owner exit 1')"))
  save(self.L/'unlaunched_training_recovery_v1/completion.json',dict(all_exit0=True))
 def tearDown(self):self.tmp.cleanup()
 def test_known_prelaunch_failure_route(self):
  paths,record=verify_routes(self.L,self.O,self.jobs,lambda _:True);self.assertEqual(paths[NAMES[0]],self.L/'unlaunched_training_recovery_v1'/NAMES[0]);self.assertFalse(record['original_all_exit0'])
 def test_unexpected_original_exit_rejected(self):
  self.save(self.L/'outer_dispatch_v1/actual_wait.json',dict(returncode=0,process=dict(pid=1,start_ticks=2)))
  with self.assertRaises(AssertionError):verify_routes(self.L,self.O,self.jobs,lambda _:True)
 def test_original_child_launch_rejected(self):
  self.save(self.L/'dispatch_v1'/NAMES[0]/'launch.json',{})
  with self.assertRaises(AssertionError):verify_routes(self.L,self.O,self.jobs,lambda _:True)
 def test_new_unknown_failure_rejected(self):
  self.save(self.L/'unlaunched_training_recovery_v1/failure.json',dict(error='unknown'))
  with self.assertRaises(AssertionError):verify_routes(self.L,self.O,self.jobs,lambda _:True)
 def test_statistics_functions_unchanged(self):
  e=pathlib.Path(__file__).parent
  old=ast.parse((e.parent/'postprocess_v1/summarize_scaleup.py').read_text());new=ast.parse((e/'summarize_scaleup_lineage.py').read_text())
  for n in ['csvread','csvsave','gone','closed_wait','audit_ledger','interval']:
   get=lambda t:ast.dump(next(x for x in t.body if isinstance(x,ast.FunctionDef)and x.name==n))
   self.assertEqual(get(old),get(new))
if __name__=='__main__':unittest.main()

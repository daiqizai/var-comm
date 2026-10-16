"""Synthetic-only interface/asset-boundary checks; never load neural weights."""
from pathlib import Path
import ast,unittest,hashlib,sys
import tb_plan as p
class ExecutionTest(unittest.TestCase):
 def test_native_projection_changes_only_metadata_binding(self):
  root=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
  old=root/'experiments/submission-closure-20261016/native.py'
  if not old.exists():old=Path('experiments/submission-closure-20261016/native.py')
  expected=old.read_text(encoding='utf8').replace('from common import *','from tb_common import *').replace('def build(routes,ledger,boundary,out):','def build(routes,ledger,boundary,out,registered_pci):').replace(";registered_pci=read(O/'source_registration.json')['gpu_pci']",'')
  self.assertEqual(Path(__file__).with_name('tb_native.py').read_text(encoding='utf8'),expected)
 def test_receiver_has_no_tx_encoder_calls(self):
  tree=ast.parse(Path(__file__).with_name('tb_receiver.py').read_text())
  calls=[n.func.attr for n in ast.walk(tree)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)]
  self.assertNotIn('encode_all',calls);self.assertNotIn('encode',calls);self.assertNotIn('source_tokens',calls)
 def test_scoring_reads_truth_after_receiver_completion(self):
  text=Path(__file__).with_name('tb_metrics.py').read_text()
  self.assertLess(text.index("read(T/'receiver/completion.json')"),text.index('reference_pixels(sources[i])'))
 def test_no_real_qualification_hidden(self):
  text=Path(__file__).with_name('tb_control.py').read_text()
  self.assertIn("('phy','receiver','metrics','selection')",text)
  self.assertNotIn('test_',text);self.assertNotIn('qualify_',text)
 def test_receiver_input_signature(self):
  tree=ast.parse(Path(__file__).with_name('tb_receiver.py').read_text())
  fn=next(n for n in ast.walk(tree)if isinstance(n,ast.FunctionDef)and n.name=='decode'and len(n.args.args)>0)
  self.assertEqual([n.arg for n in fn.args.args],['self','family','payload','m','K'])
 def test_bounded_stages(self):
  text=Path(__file__).with_name('tb_control.py').read_text()
  self.assertIn('child.wait(timeout=',text);self.assertIn('automatic_retry=False',text);self.assertIn('actual_wait=True',text)
 def test_control_stage_filenames_exist(self):
  tree=ast.parse(Path(__file__).with_name('tb_control.py').read_text())
  expression=next(n.value for n in ast.walk(tree)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='filename'for t in n.targets))
  code=compile(ast.Expression(expression),'<stage-file>','eval')
  for stage in ('phy','receiver','metrics','selection'):
   self.assertTrue(Path(__file__).with_name(eval(code,{'stage':stage})).is_file())
 def test_cuda_and_nvml_identity_namespaces_separate(self):
  self.assertIn("str(props.uuid)==r['gpu_cuda_uuid']",Path(__file__).with_name('tb_metrics.py').read_text())
  self.assertIn("val[0].strip()==r['gpu_uuid']",Path(__file__).with_name('tb_control.py').read_text())
 def test_no_test_set_paths(self):
  for name in ('tb_phy.py','tb_receiver.py','tb_metrics.py','tb_select.py'):
   text=Path(__file__).with_name(name).read_text()
   self.assertNotIn('common500',text);self.assertNotIn('new100',text)
 def test_all_python_parses(self):
  for q in Path(__file__).parent.glob('*.py'):ast.parse(q.read_text(encoding='utf8'),filename=str(q))
if __name__=='__main__':unittest.main()

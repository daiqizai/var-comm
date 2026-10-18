"""Synthetic population, isolation, closed reuse and process-boundary checks."""
import ast,hashlib,json,pathlib,tempfile,unittest
from unittest import mock
import native,owner,execute

class NativeContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw=(pathlib.Path(__file__).parent.parent/'a1_native.py').read_bytes();cls.text=native.adapt(cls.raw)
    def test_exact_original(self):self.assertEqual(hashlib.sha256(self.raw).hexdigest(),native.ORIGINAL_SHA)
    def test_mutated_original_rejected(self):
        with self.assertRaises(AssertionError):native.adapt(self.raw+b' ')
    def test_new_caps(self):
        tree=ast.parse(self.text);caps=[n.value for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='caps' for t in n.targets)][0]
        self.assertEqual(ast.literal_eval(caps.body),{'model_load':2,'caption':92,'tokenizer':92,'tx_probability':92,'reference_render':0})
        self.assertEqual(ast.literal_eval(caps.orelse),{'model_load':1,'rx_prefix':676,'rx_render':676})
    def test_no_reference_call(self):
        calls=[n for n in ast.walk(ast.parse(self.text)) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='render']
        self.assertEqual(len(calls),1);self.assertEqual(ast.literal_eval(calls[0].args[-1]),'rx_render')
    def test_rx_firewall_unchanged(self):
        a,b=[ast.parse(x) for x in [self.raw,self.text]]
        fn=lambda t:next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='install_rx_boundary')
        self.assertEqual(ast.dump(fn(a)),ast.dump(fn(b)))
    def test_render_math_unchanged(self):
        fn=lambda s:next(n for n in ast.walk(ast.parse(s)) if isinstance(n,ast.FunctionDef) and n.name=='render')
        self.assertEqual(ast.dump(fn(self.raw)),ast.dump(fn(self.text)))
    def test_actual_population(self):
        self.assertIn("['cal100'][8:];assert len(sources)==92",self.text)
        self.assertIn('assert len(jobs)==676',self.text)
        self.assertEqual(92*7+8*4+24,700)
    def test_rx_has_no_source_argument(self):self.assertIn("if role=='rx':assert a.sources is None",self.text)

class Reuse(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=pathlib.Path(self.tmp.name)
        self.run=self.root/'run';self.a1=self.root/'a1';self.base=self.root/'base'
        (self.run/'wire_inputs').mkdir(parents=True);(self.a1/'tx').mkdir(parents=True);self.base.mkdir()
        self.patches=[mock.patch.object(owner,n,v) for n,v in [('RUN',self.run),('A1',self.a1),('BASE',self.base)]]
        for p in self.patches:p.start()
        jobs=[{'key':f'{i:02d}_e{e}','path':str(self.run/'wire_inputs'/f'{i:02d}_e{e}.arc'),'paid_profile':e,'generation_seed':0} for i in range(8,100) for e in range(1,8)]
        (self.run/'wire_inputs/new_jobs.json').write_text(json.dumps(jobs));pins=[]
        for i in range(8):
            for e in [2,3,5,6]:
                p=self.a1/'tx'/f'{i:02d}_e{e}.arc';p.write_bytes(bytes([i,e]));pins.append(owner.desc(p))
        (self.base/'a1_closed_files.json').write_text(json.dumps({'files':pins}))
    def tearDown(self):
        for p in reversed(self.patches):p.stop()
        self.tmp.cleanup()
    def test_reuse_only_missing_and_no_truth(self):
        owner.assemble_jobs();jobs=json.loads((self.run/'wire_inputs/jobs.json').read_text())
        self.assertEqual(len(jobs),676)
        self.assertTrue(all(set(j)=={'key','path','paid_profile','generation_seed'} for j in jobs))
        self.assertFalse(any(j['key']=='00_e1' for j in jobs))
        self.assertEqual((self.run/'wire_inputs/07_e6.arc').read_bytes(),b'\x07\x06')
    def test_altered_old_stream_rejected(self):
        (self.a1/'tx/00_e2.arc').write_bytes(b'bad')
        with self.assertRaises(AssertionError):owner.assemble_jobs()
        self.assertFalse((self.run/'wire_inputs/jobs.json').exists())
    def test_duplicate_new_key_rejected(self):
        p=self.run/'wire_inputs/new_jobs.json';a=json.loads(p.read_text());a[1]=a[0];p.write_text(json.dumps(a))
        with self.assertRaises(AssertionError):owner.assemble_jobs()
    def test_repeat_does_not_overwrite(self):
        owner.assemble_jobs();pin=owner.desc(self.run/'wire_inputs/jobs.json')
        with self.assertRaises(FileExistsError):owner.assemble_jobs()
        self.assertEqual(owner.desc(self.run/'wire_inputs/jobs.json'),pin)

class ExitContract(unittest.TestCase):
    def wait(self,rc=0,timed=False,gone=True):
        with tempfile.TemporaryDirectory() as d:
            p=pathlib.Path(d)/'wait.json';p.write_text(json.dumps({'returncode':rc,'timed_out':timed,'process':{'pid':123}}))
            with mock.patch.object(owner,'gone',return_value=gone):return owner.success(p)
    def test_success(self):self.assertEqual(self.wait()['returncode'],0)
    def test_bool_is_not_integer_exit(self):
        with self.assertRaises(AssertionError):self.wait(False)
    def test_failure_rejected(self):
        with self.assertRaises(AssertionError):self.wait(1)
    def test_timeout_rejected(self):
        with self.assertRaises(AssertionError):self.wait(timed=True)
    def test_live_child_rejected(self):
        with self.assertRaises(AssertionError):self.wait(gone=False)
    def test_no_process_no_signal(self):
        with tempfile.TemporaryDirectory() as d,mock.patch.object(execute,'RUN',pathlib.Path(d)),mock.patch.object(execute.os,'killpg') as kill:
            self.assertEqual(execute.stop_children(),[]);kill.assert_not_called()
if __name__=='__main__':unittest.main(verbosity=2)

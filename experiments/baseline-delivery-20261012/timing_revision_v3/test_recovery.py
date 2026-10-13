"""No-model CPU tests for the exact constructor-only alias recovery."""
import ast
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from module_aliases import ExactByteAliases,sha

ROOT=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')


def actual_loader(relative,modules):
    path=ROOT/relative
    source=path.read_text()
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='load_exact')
    def require(ok,message):
        if not ok: raise RuntimeError(message)
    namespace=dict(Path=Path,sys=SimpleNamespace(modules=modules),importlib=importlib,require=require)
    exec(compile(ast.get_source_segment(source,node),str(path),'exec'),namespace)
    return namespace['load_exact']


class AliasTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.base=Path(self.temp.name);self.pairs=[];self.modules={}
        for name in ['main_raw64_plan_only','main_raw64_packet_adapter']:
            a=self.base/'old'/(name+'.py');b=self.base/'canonical'/(name+'.py')
            a.parent.mkdir(exist_ok=True);b.parent.mkdir(exist_ok=True)
            a.write_text('sentinel = 7\n');b.write_bytes(a.read_bytes())
            self.pairs.append(dict(module_name=name,loaded_path=str(a),requested_path=str(b),sha256=sha(a)))
            self.modules[name]=SimpleNamespace(__file__=str(a),sentinel=7)
        self.modules['main_raw64_packet_adapter'].p=self.modules['main_raw64_plan_only']
        self.alias=ExactByteAliases(self.pairs,self.modules)
        self.t1=SimpleNamespace(load_exact=actual_loader('experiments/wcl-evidence-closure-20261009/scripts/t1_phy.py',self.modules))
        self.ep=SimpleNamespace(load_exact=actual_loader('experiments/entropy-partial-mismatch-20261010/scripts/ep_phy.py',self.modules))

    def tearDown(self): self.temp.cleanup()

    def test_original_path_failure_reproduced_without_models(self):
        for loader in [self.t1.load_exact,self.ep.load_exact]:
            for row in self.pairs:
                with self.assertRaisesRegex(RuntimeError,'Unexpected imported module'):
                    loader(row['requested_path'],row['module_name'])

    def test_four_private_aliases_reuse_exact_original_objects(self):
        original=(self.t1.load_exact,self.ep.load_exact)
        ids={k:id(v) for k,v in self.modules.items()}
        def construct():
            for module in (self.t1,self.ep):
                for row in self.pairs:
                    self.assertIs(module.load_exact(row['requested_path'],row['module_name']),self.modules[row['module_name']])
            return 'constructed_no_science'
        self.assertEqual(self.alias.call(self.t1,self.ep,construct),'constructed_no_science')
        self.assertEqual((self.t1.load_exact,self.ep.load_exact),original)
        self.assertEqual({k:id(v) for k,v in self.modules.items()},ids)
        self.assertEqual(len(self.alias.events),4)
        for row in self.pairs:self.assertEqual(self.modules[row['module_name']].__file__,row['loaded_path'])

    def test_one_byte_drift_rejected(self):
        Path(self.pairs[0]['requested_path']).write_text('sentinel = 8\n')
        with self.assertRaises(AssertionError):self.alias.verify()

    def test_unregistered_path_still_rejected_by_original(self):
        row=self.pairs[0];other=self.base/'other.py';other.write_text('sentinel = 7\n')
        fn=self.alias.wrap(self.t1.load_exact,'test')
        with self.assertRaisesRegex(RuntimeError,'Unexpected imported module'):
            fn(other,row['module_name'])
        self.assertEqual(self.alias.events,[])

    def test_unexpected_preloaded_module_path_rejected(self):
        row=self.pairs[0];self.modules[row['module_name']].__file__=str(self.base/'unexpected.py')
        with self.assertRaisesRegex(AssertionError,'Unexpected prior module path'):
            self.alias.wrap(self.t1.load_exact,'test')(row['requested_path'],row['module_name'])

    def test_packet_planner_split_rejected(self):
        row=self.pairs[1];self.modules[row['module_name']].p=object()
        with self.assertRaisesRegex(AssertionError,'Packet planner object split'):
            self.alias.wrap(self.t1.load_exact,'test')(row['requested_path'],row['module_name'])

    def test_exception_restores_both_original_loaders(self):
        original=(self.t1.load_exact,self.ep.load_exact)
        def fail():raise ValueError('constructor test')
        with self.assertRaisesRegex(ValueError,'constructor test'):self.alias.call(self.t1,self.ep,fail)
        self.assertEqual((self.t1.load_exact,self.ep.load_exact),original)

    def test_canonical_module_uses_original_strict_loader(self):
        row=self.pairs[0];self.modules[row['module_name']].__file__=row['requested_path']
        result=self.alias.wrap(self.t1.load_exact,'test')(row['requested_path'],row['module_name'])
        self.assertIs(result,self.modules[row['module_name']]);self.assertEqual(self.alias.events,[])

    def test_unchanged_name_delegated(self):
        calls=[]
        def original(path,name):calls.append((path,name));return 29
        self.assertEqual(self.alias.wrap(original,'test')('/unused','another_module'),29)
        self.assertEqual(calls,[('/unused','another_module')]);self.assertEqual(self.alias.events,[])


if __name__=='__main__': unittest.main(verbosity=2)

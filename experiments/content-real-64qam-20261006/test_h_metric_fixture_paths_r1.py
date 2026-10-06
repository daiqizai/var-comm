"""Additional path-binding regression tests; separate from the formal28 fixtures."""
import ast
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import test_h_development_metric_adapter_r1 as r


class FixturePathTests(unittest.TestCase):
    def fixture(self,root):
        root=Path(root);assets=root/'separate_assets';assets.mkdir()
        paths={name:str(assets/(name+'.json')) for name in r.ASSET_KEYS}
        bindings={};pins={}
        for name in r.MODULE_SHA:
            p=assets/(name+'.py');p.write_text('# synthetic fixture source\n',encoding='utf-8')
            paths[name]=str(p);pins[name]=r.file_sha(p);bindings[str(p)]=pins[name]
        doc=dict(status='H_ORIGINAL_METRIC_ASSETS_BOUND',paths=paths,source_bindings=bindings,input_bindings={})
        manifest=root/'metric_assets.json';manifest.write_text(json.dumps(doc),encoding='utf-8')
        env=dict(H_METRIC_TEST_ASSETS=str(manifest),H_METRIC_TEST_ASSETS_SHA256=r.file_sha(manifest))
        return doc,manifest,env,pins

    def test_all13_test_semantics_preserved_except_two_path_assignments(self):
        original=ast.parse(Path(r.original.__file__).read_text(encoding='utf-8'))
        replacement=ast.parse(Path(r.__file__).read_text(encoding='utf-8'))
        a=next(n for n in original.body if isinstance(n,ast.ClassDef) and n.name=='AdapterTests')
        b=next(n for n in replacement.body if isinstance(n,ast.ClassDef) and n.name=='AdapterTests')
        old={n.name:n for n in a.body if isinstance(n,ast.FunctionDef)}
        new={n.name:n for n in b.body if isinstance(n,ast.FunctionDef)}
        expected={
            'test_original_derangement_code_exact_source_and100population':('path','replay_module'),
            'test_batch_one_actual_suite_and_independent_API_bridge_with_fakes':('validation','validation_module')}
        self.assertEqual(set(new),set(expected))
        for name,(target,key) in expected.items():
            tree=copy.deepcopy(old[name]);matches=[]
            for n in ast.walk(tree):
                if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id==target:
                    matches.append(n)
            self.assertEqual(len(matches),1)
            matches[0].value=ast.Call(func=ast.Name(id='bound_fixture_module',ctx=ast.Load()),args=[ast.Constant(key)],keywords=[])
            self.assertEqual(ast.dump(tree,include_attributes=False),ast.dump(new[name],include_attributes=False))
        names=unittest.defaultTestLoader.getTestCaseNames(r.AdapterTests)
        self.assertEqual(len(names),13)
        for name in set(old)-set(new):self.assertIs(getattr(r.AdapterTests,name),getattr(r.original.AdapterTests,name))

    def test_explicit_manifest_resolves_both_modules_outside_fixture_directory(self):
        with tempfile.TemporaryDirectory() as td:
            doc,manifest,env,pins=self.fixture(td)
            with patch.dict(os.environ,env,clear=True),patch.object(r,'MODULE_SHA',pins):
                for name in pins:self.assertEqual(r.bound_fixture_module(name),Path(doc['paths'][name]))

    def test_missing_environment_or_wrong_manifest_sha_has_no_fallback(self):
        with patch.dict(os.environ,{},clear=True):
            with self.assertRaisesRegex(RuntimeError,'no fallback'):r.bound_fixture_module('replay_module')
        with tempfile.TemporaryDirectory() as td:
            doc,manifest,env,pins=self.fixture(td);env['H_METRIC_TEST_ASSETS_SHA256']='0'*64
            with patch.dict(os.environ,env,clear=True),patch.object(r,'MODULE_SHA',pins):
                with self.assertRaisesRegex(RuntimeError,'manifest SHA'):r.bound_fixture_module('replay_module')

    def test_unbound_or_changed_module_and_conflicting_bindings_are_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            doc,manifest,env,pins=self.fixture(td);path=doc['paths']['validation_module']
            for failure in ('missing','conflict'):
                changed=copy.deepcopy(doc)
                if failure=='missing':changed['source_bindings'].pop(path)
                else:changed['input_bindings'][path]='0'*64
                manifest.write_text(json.dumps(changed),encoding='utf-8');env['H_METRIC_TEST_ASSETS_SHA256']=r.file_sha(manifest)
                with patch.dict(os.environ,env,clear=True),patch.object(r,'MODULE_SHA',pins):
                    with self.assertRaises(RuntimeError):r.bound_fixture_module('validation_module')
            manifest.write_text(json.dumps(doc),encoding='utf-8');env['H_METRIC_TEST_ASSETS_SHA256']=r.file_sha(manifest)
            Path(path).write_text('changed',encoding='utf-8')
            with patch.dict(os.environ,env,clear=True),patch.object(r,'MODULE_SHA',pins):
                with self.assertRaisesRegex(RuntimeError,'source SHA'):r.bound_fixture_module('validation_module')

    def test_old_test_file_must_stay_byte_identical_and_unknown_module_refused(self):
        self.assertEqual(r.file_sha(r.original.__file__),r.ORIGINAL_TEST_SHA)
        with self.assertRaisesRegex(RuntimeError,'Only two'):r.bound_fixture_module('suite_module')
        with patch.object(r,'ORIGINAL_TEST_SHA','0'*64):
            with self.assertRaisesRegex(RuntimeError,'test source changed'):r.bound_fixture_module('replay_module')


if __name__=='__main__':unittest.main()

"""Check exact historical path expansion with no model imports or real Git."""
import ast
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import visual_source_closure as v

RESEARCH = Path(__file__).resolve().parents[2]
COMMON = RESEARCH/'m1_n2048_m9_20261004/reference/common.py'
QUALITY = RESEARCH/'prior_aware_uep_20261004/runtime/quality_driver.py'
if not COMMON.is_file():
    repo = Path(os.environ.get('VAR_COMM_ROOT', Path.cwd())).resolve()
    COMMON = repo/v.COMMON_DIRECTORY/'common.py'
    QUALITY = repo/v.QUALITY_DIRECTORY/'quality_driver.py'


class ClosureTests(unittest.TestCase):
    def fixture(self, directory):
        base=Path(directory);root=base/'repo';native=root/'experiments/m1-native';quality=root/v.QUALITY_DIRECTORY
        var=base/'VAR';dino=base/'DINO';here=root/v.COMMON_DIRECTORY;old=root/v.OLD_DIRECTORY
        files=[native/'m1_common.py',native/'m1_native.py',native/'m1_phy.py',native/'m1_performance.py',native/'other.py',
            quality/'quality_driver.py',quality/'source_quality.py',here/'common.py',here/'protocol.json',here/'README.md',here/'ignore.txt',
            old/'assets.py',old/'not_bound.json',var/'models/var.py',var/'models/basic_var.py',var/'models/nested/not_bound.py',
            dino/'dinov2/__init__.py',dino/'dinov2/models/vit.py',dino/'dinov2/models/ignore.txt',
            root/'configs/progressive_channel.yaml',root/'configs/example.json',root/'tracked/new_helper.py',root/'tracked/new_test.py',
            root/'tracked/kernel.cpp']+[root/p for p in v.EXPLICIT_RELATIVE]
        for index,p in enumerate(files):p.parent.mkdir(parents=True,exist_ok=True);p.write_text('fixture '+str(index)+'\n')
        tracked=['tracked/new_helper.py','tracked/new_test.py','tracked/kernel.cpp','configs/progressive_channel.yaml',
                 'configs/example.json',v.EXPLICIT_RELATIVE[0]]
        return root,native,var,dino,quality,here,old,tracked

    def original_function_bindings(self, parts):
        root,native,var,dino,quality,here,old,tracked=parts
        tree=ast.parse(COMMON.read_text(encoding='utf-8-sig'))
        function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='source_bindings')
        calls=[]
        def original_git(args,**kw):
            calls.append((args,kw));return '\n'.join(tracked)+'\n'
        env=dict(Path=Path,HERE=here,ROOT=root,OLD=old,sha=v.sha,
            subprocess=SimpleNamespace(check_output=original_git),
            assets=SimpleNamespace(old=SimpleNamespace(b=SimpleNamespace(model_paths=lambda:{'var_source':str(var)}))),
            yaml=SimpleNamespace(safe_load=lambda text:{'quality':{'dino_source':str(dino)}}))
        exec(compile(ast.Module(body=[function],type_ignores=[]),str(COMMON),'exec'),env)
        common_bindings=env['source_bindings']()
        self.assertEqual(calls,[(['git','ls-files',*v.GIT_PATTERNS],{'cwd':root,'text':True})])
        # Execute only the original post-model source-binding suffix. There is
        # no Native constructor, torch import, model loader or GPU function here.
        build=next(n for n in ast.parse(QUALITY.read_text(encoding='utf-8-sig')).body
                   if isinstance(n,ast.FunctionDef) and n.name=='build_native')
        start=next(i for i,n in enumerate(build.body) if isinstance(n,ast.Assign)
            and any(isinstance(t,ast.Attribute) and t.attr=='driver_bindings' for t in n.targets))
        suffix=[]
        for n in build.body[start:]:
            if isinstance(n,ast.Return):break
            suffix.append(n)
        def verify(bindings):
            for p,s in bindings.items():self.assertEqual(v.sha(p),s)
        native_obj=SimpleNamespace(common=SimpleNamespace(source_bindings=lambda:common_bindings,verify_bindings=verify))
        second=dict(native=native_obj,runtime=native,Path=Path,__file__=str(quality/'quality_driver.py'),
                    q=SimpleNamespace(sha=v.sha,__file__=str(quality/'source_quality.py')))
        exec(compile(ast.Module(body=suffix,type_ignores=[]),str(QUALITY),'exec'),second)
        return native_obj.driver_bindings

    def test_matches_actual_old_common_and_quality_driver_ast_path_sets_and_hashes(self):
        with tempfile.TemporaryDirectory() as td:
            parts=self.fixture(td);root,native,var,dino,quality,here,old,tracked=parts
            expected=self.original_function_bindings(parts)
            with patch.object(v,'tracked_paths',return_value=tracked):
                actual=v.collect_bindings(root,native,var,dino,quality)
            self.assertEqual(actual,expected)
            self.assertNotIn(str(native/'other.py'),actual)
            self.assertNotIn(str(var/'models/nested/not_bound.py'),actual)
            self.assertNotIn(str(old/'not_bound.json'),actual)
            self.assertIn(str(dino/'dinov2/models/vit.py'),actual)
            self.assertIn(str(here/'protocol.json'),actual)

    def test_new_tracked_helpers_are_missing_not_changed_and_existing_change_stays_separate(self):
        with tempfile.TemporaryDirectory() as td:
            parts=self.fixture(td);root,native,var,dino,quality,_,_,tracked=parts
            with patch.object(v,'tracked_paths',return_value=tracked):actual=v.collect_bindings(root,native,var,dino,quality)
            registered=dict(actual);missing=[str(root/'tracked/new_helper.py'),str(root/'tracked/new_test.py')]
            for p in missing:registered.pop(p)
            changed=str(native/'m1_native.py');registered[changed]='0'*64;registered['/some/data/receipt.json']='r'*64
            result=v.compare_bindings(actual,registered)
            self.assertEqual(set(result['missing']),set(missing));self.assertEqual(set(result['changed']),{changed})
            self.assertEqual(result['changed'][changed]['actual_sha256'],actual[changed])
            self.assertEqual(result['changed'][changed]['registered_sha256'],'0'*64)
            self.assertEqual(result['extra_registered_bindings'],{'/some/data/receipt.json':'r'*64})
            self.assertFalse(result['registration_modified']);self.assertTrue(result['renderer_guard_unchanged'])
            self.assertEqual(registered[changed],'0'*64)

    def test_matching_expected_sources_does_not_require_removing_data_bindings(self):
        actual={'/code/a.py':'a'*64};reg=dict(actual,**{'/data/input.json':'d'*64})
        result=v.compare_bindings(actual,reg)
        self.assertEqual(result['status'],'EXACT_SOURCE_CLOSURE_MATCH');self.assertEqual(result['matched_files'],1)

    def test_actual_file_missing_or_bad_tracked_path_is_fatal(self):
        with tempfile.TemporaryDirectory() as td:
            parts=self.fixture(td);root,native,var,dino,quality,_,_,tracked=parts
            (root/v.EXPLICIT_RELATIVE[-1]).unlink()
            with patch.object(v,'tracked_paths',return_value=tracked):
                with self.assertRaisesRegex(ValueError,'source file is missing'):v.collect_bindings(root,native,var,dino,quality)
            with self.assertRaisesRegex(ValueError,'Invalid relative'):
                v._collect_paths(root,native,var,dino,quality,['../outside.py'])

    def test_registration_source_input_conflict_is_not_silently_overridden(self):
        self.assertEqual(v.merge_registration(dict(input_bindings={'a':'1'},source_bindings={'a':'1','b':'2'})),{'a':'1','b':'2'})
        with self.assertRaisesRegex(ValueError,'Conflicting'):v.merge_registration(dict(input_bindings={'a':'1'},source_bindings={'a':'2'}))

    def test_git_query_matches_original_patterns_and_has_no_gpu_or_shell(self):
        with patch.object(v.subprocess,'check_output',return_value='x.py\ny.cpp\n') as call:
            self.assertEqual(v.tracked_paths(Path('/repo')),['x.py','y.cpp'])
        args,kw=call.call_args;self.assertEqual(args[0],['git','ls-files','*.py','*.cpp','configs/*.yaml','configs/*.json'])
        self.assertNotIn('shell',kw);self.assertEqual(kw['env']['CUDA_VISIBLE_DEVICES'],'');self.assertEqual(kw['env']['OMP_NUM_THREADS'],'2')

    def test_audit_preserves_registration_bytes_and_has_no_model_execution_claim(self):
        with tempfile.TemporaryDirectory() as td:
            parts=self.fixture(td);root,native,var,dino,quality,_,_,tracked=parts
            reg=root/'registration.json';reg.write_text(json.dumps(dict(source_bindings={},input_bindings={})))
            before=reg.read_bytes()
            with patch.object(v,'tracked_paths',return_value=tracked):result=v.audit(root,native,var,dino,reg,quality)
            self.assertEqual(reg.read_bytes(),before);self.assertEqual(result['registration_sha256'],v.sha(reg))
            self.assertFalse(result['GPU_used']);self.assertFalse(result['model_imports']);self.assertEqual(result['actual_packet_decodes'],0)
            self.assertGreater(len(result['missing']),0)


if __name__=='__main__':unittest.main()

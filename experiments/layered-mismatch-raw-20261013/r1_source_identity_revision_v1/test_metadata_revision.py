"""Synthetic CPU tests; no source arrays, models, packets or GPU calls."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import pin_adapter as a
import runner
import recover_registration as recovery


class PinTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)/'checkpoint.json'; self.path.write_bytes(b'original')
        self.old = dict(path=str(self.path), sha256=a.sha(self.path))
        self.new = dict(self.old, bytes=8)

    def test_legacy_and_optional(self):
        for left, right in ((self.old,self.old),(self.old,self.new),(self.new,self.old),(self.new,self.new)):
            self.assertTrue(a.strict_checkpoint(left,right))

    def test_reject_noninteger_bytes(self):
        for value in (True,8.0,'8',None,-1):
            with self.subTest(value=value), self.assertRaises(RuntimeError):
                a.strict_checkpoint(dict(self.old,bytes=value),self.old)

    def test_reject_wrong_size(self):
        with self.assertRaises(RuntimeError): a.strict_checkpoint(dict(self.old,bytes=9),self.old)

    def test_reject_unknown_checkpoint_key(self):
        with self.assertRaises(RuntimeError): a.strict_checkpoint(dict(self.new,note='ignore'),self.old)

    def test_reject_content_change(self):
        self.path.write_bytes(b'changed!')
        with self.assertRaises(RuntimeError): a.strict_checkpoint(self.new,self.old)

    def test_same_content_other_path_is_not_identity(self):
        other=self.path.with_name('other'); other.write_bytes(b'original')
        with self.assertRaises(RuntimeError): a.strict_checkpoint(dict(self.old,path=str(other)),self.old)

    def test_optional_on_selected_side_checked(self):
        with self.assertRaises(RuntimeError): a.strict_checkpoint(self.old,dict(self.old,bytes=True))

    def test_source_fields_unchanged(self):
        selected=dict(checkpoint=self.old,source_id='x',source_index=1,tokens_sha256='token')
        row=dict(selected,checkpoint=self.new,selection_slot=0)
        self.assertTrue(a.strict_selected_identity(row,selected))
        for key in ('source_id','source_index','tokens_sha256'):
            bad=dict(row);bad[key]='changed'
            with self.subTest(key=key),self.assertRaises(RuntimeError): a.strict_selected_identity(bad,selected)

    def test_missing_identity_field(self):
        with self.assertRaises(RuntimeError): a.strict_selected_identity(dict(checkpoint=self.new),dict(checkpoint=self.old,source_id='x'))


class CodeTests(unittest.TestCase):
    def test_source_proof(self):
        self.assertTrue(a.proof()['inverse_AST_equal'])
        self.assertTrue(recovery.proof()['inverse_AST_equal'])

    def test_fail_closed_original_shape(self):
        text=a.PHY.read_text(encoding='utf-8-sig')
        with self.assertRaises(RuntimeError): a.adapted_source(text.replace(a.OLD,'True'))
        with self.assertRaises(RuntimeError): a.adapted_source(text+'\n'+a.OLD)

    def test_one_private_namespace_for_all_phy_functions(self):
        module=a.install_phy()
        self.assertIs(module,sys.modules['r1_phy'])
        for name in ('validate_registration','run','source_scales','execute_frame'):
            self.assertIs(getattr(module,name).__globals__,vars(module))
        self.assertEqual(module.__file__,str(a.PHY))
        self.assertEqual(a.sha(a.PHY),a.PHY_SHA)

    def test_receivers_and_score_share_adapter(self):
        score=runner.load_entry(runner.RESOURCE/'score_selection.py')
        self.assertIs(score.r1_phy,sys.modules['r1_phy'])
        import receiver_visual
        # audit_phy imports the same explicit private module when called.
        self.assertIn('import r1_phy as phy',Path(receiver_visual.__file__).read_text())
        self.assertEqual(sys.modules['r1_phy'].validate_registration.__globals__['_strict_selected_identity'],a.strict_selected_identity)

    def binding(self):
        return dict(code_bindings={str(Path(runner.__file__).resolve()):a.sha(runner.__file__),
                                   str(Path(a.__file__).resolve()):a.sha(a.__file__)})

    def test_route_retains_entire_original_tail(self):
        for path in runner.ORIGINAL:
            command=['python','-B','-u',path,'run','--registration','bound','--registration-sha256','sha']
            new=runner.route(command,self.binding())
            self.assertEqual(new[:3],command[:3]); self.assertEqual(new[-len(command[3:]):],command[3:])
            self.assertEqual(new[new.index(path)+1],'run')

    def test_cli_envelope_preserves_flag_first_owner_and_dispatch(self):
        for tail in (['--registration','master','--registration-sha256','m'],
                     ['--expected-registration','m'],['run','--registration','phy','--registration-sha256','p']):
            args=runner.parse_envelope(['--runner-sha256','r','--adapter-sha256','a','--original','script',*tail])
            self.assertEqual(args.original_arguments,tail)
            self.assertEqual(args.original,'script')

    def test_route_rejects_unbound_code(self):
        reg=self.binding(); reg['code_bindings'][str(Path(a.__file__).resolve())]='0'*64
        with self.assertRaises(RuntimeError): runner.route(['python','-B','-u',str(a.PHY),'run'],reg)

    def test_owner_routes_all_original_stages(self):
        owner=runner.load_entry(runner.RESOURCE/'owner.py')
        reg=self.binding(); reg.update(runtime=dict(python_cpu='cpu',python_visual='visual'),
            phy_registration=dict(path='phy',sha256='p'),visual_registration=dict(path='visual',sha256='v'))
        master=dict(path='master',sha256='m')
        for stage in ('phy','projection','visual','score','selection'):
            command=owner.stage_command(reg,master,stage,123,456)
            self.assertIn(str(Path(runner.__file__).resolve()),command)
            if stage=='visual':
                self.assertNotIn('master',command);self.assertIn('visual',command)

    def test_dispatch_private_only_routes_argv(self):
        original=(runner.RESOURCE/'dispatch_once.py').read_bytes()
        module=runner.load_entry(runner.RESOURCE/'dispatch_once.py')
        self.assertIs(module.run.__globals__['_metadata_route'],runner.route)
        self.assertEqual((runner.RESOURCE/'dispatch_once.py').read_bytes(),original)

    def test_registration_inverse_rejects_new_shape(self):
        text=recovery.BUILDER.read_text(encoding='utf-8-sig')
        with self.assertRaises(RuntimeError): recovery.adapted_builder_source(text.replace('directory.mkdir(parents=True)','pass'))

    def test_writer_never_rewrites_partial(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td);value={'x':1};canonical=lambda x:json.dumps(x,sort_keys=True)
            (path/'nominations.json').write_bytes((canonical(value)+'\n').encode('utf8'))
            calls=[];stub=types.SimpleNamespace(canonical=canonical,pin=lambda p:dict(path=str(p),sha256=a.sha(p)))
            with mock.patch.dict(recovery.PARTIAL,{'nominations.json':a.sha(path/'nominations.json')},clear=True):
                write=recovery.immutable_or_new_write(stub,path,lambda *args:calls.append(args),{})
                pin=write(path/'nominations.json',value)
                self.assertEqual(calls,[]);self.assertEqual(pin['sha256'],a.sha(path/'nominations.json'))
                with self.assertRaises(RuntimeError):write(path/'nominations.json',{'x':2})
                with self.assertRaises(RuntimeError):write(path/'unexpected.json',{})

    def test_master_discloses_adapter_review_separately(self):
        calls=[];meta=dict(review={'path':'review','sha256':'r'},original_failure={'path':'failure','sha256':'f'})
        write=recovery.immutable_or_new_write(None,Path('/fixed'),lambda p,v:calls.append(v),meta)
        write(Path('/fixed/registration.json'),dict(pins={'root_review':'unchanged'}))
        self.assertEqual(calls[0]['pins']['root_review'],'unchanged')
        self.assertEqual(calls[0]['metadata_identity_revision'],meta)
        self.assertEqual(calls[0]['pins']['metadata_adapter_review'],meta['review'])

    def test_partial_gate_requires_exact_files_runtime_and_no_attempt(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td); out=base/'out'; directory=out/'registered'; directory.mkdir(parents=True)
            temporary=base/'tmp'/'r1_execution_v1';temporary.mkdir(parents=True)
            (directory/'nominations.json').write_bytes(b'{}')
            phy=dict(runtime=dict(deadline_unix=recovery.DEADLINE,max_seconds=43200,cpu_affinity=[2,3]))
            (directory/'phy_registration.json').write_text(json.dumps(phy))
            pins={p.name:a.sha(p) for p in directory.iterdir()}
            shared=types.SimpleNamespace(OUT=out,TMP=temporary,read=lambda p:json.loads(Path(p).read_text()))
            with mock.patch.dict(recovery.PARTIAL,pins,clear=True),mock.patch.object(recovery.time,'time',return_value=recovery.DEADLINE-1):
                self.assertEqual(recovery.verify_partial(directory,shared),phy)
                unexpected=directory/'registration.json';unexpected.write_bytes(b'{}')
                with self.assertRaises(RuntimeError):recovery.verify_partial(directory,shared)
                unexpected.unlink()
                for marker in ('owner_launch.json','owner_failure.json','completion.json','phy','projection','visual','score','selection'):
                    path=temporary/marker;path.write_bytes(b'{}')
                    with self.subTest(marker=marker),self.assertRaises(RuntimeError):recovery.verify_partial(directory,shared)
                    path.unlink()
                with mock.patch.object(recovery.time,'time',return_value=recovery.DEADLINE):
                    with self.assertRaises(RuntimeError):recovery.verify_partial(directory,shared)
                (directory/'nominations.json').write_bytes(b'{"changed":true}')
                with self.assertRaises(RuntimeError):recovery.verify_partial(directory,shared)

    def test_writer_keeps_visual_source_free(self):
        calls=[];visual=dict(pins={'common_ledger':'bound'},runtime={'deadline_unix':recovery.DEADLINE})
        write=recovery.immutable_or_new_write(None,Path('/fixed'),lambda p,v:calls.append(v),
                                            {'review':{'path':'source-capable-review'}})
        write(Path('/fixed/visual_registration.json'),visual)
        self.assertEqual(calls,[visual]);self.assertNotIn('metadata_identity_revision',calls[0])

    def test_recovery_never_restarts_deadline(self):
        changed=recovery.adapted_builder_source(recovery.BUILDER.read_text(encoding='utf-8-sig'))
        self.assertIn('deadline_unix=_original_deadline',changed)
        self.assertNotIn('deadline_unix=time.time() + max_seconds',changed)
        self.assertEqual(recovery.MAX_SECONDS,43200)
        self.assertEqual(recovery.DEADLINE,1791954813.2221007)

    def test_no_scientific_import(self):
        self.assertNotIn('torch',sys.modules)
        self.assertNotIn('sionna',sys.modules)


if __name__=='__main__': unittest.main()

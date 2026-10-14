"""Bounded synthetic metadata tests; no actual scientific/preflight execution.

R1_PROBE_LINEAGE_SOURCE_FIXTURES may name a directory containing delivery.py,
dispatcher.py, figures.py, backup.py with their exact frozen bytes. Otherwise
use the existing local reviewed source copies. Every loaded source is SHA checked.
"""
from __future__ import annotations
import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('tested_probe_lineage',HERE/'adapter.py')
A=importlib.util.module_from_spec(spec);spec.loader.exec_module(A)


def frozen(role):
    fixtures=os.environ.get('R1_PROBE_LINEAGE_SOURCE_FIXTURES')
    path=Path(fixtures)/(role+'.py') if fixtures else HERE.parent/A.SOURCES[role][0]
    data=path.read_bytes()
    if A.digest(data)!=A.SOURCES[role][1]:raise RuntimeError('Synthetic source fixture changed: '+role)
    spec=importlib.util.spec_from_file_location('_synthetic_'+role,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def write(folder,name,value):
    path=folder/name;path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf8');return A.pin(path)


def fixture(folder):
    specs=[];prior=[]
    for n in range(1,7):
        failure=n in (1,2,5)
        value=dict(schema='R1_RX_ZERO_MODEL_IMPORT_PROBE_V1',
            status='FAILED_METADATA_ONLY' if failure else 'PASS_ZERO_MODEL_IMPORTS',
            CUDA_VISIBLE_DEVICES='',CUDA_initialized=False,actual_R1_registration_created=False,blocked=[])
        value.update({k:0 for k in A.ZERO})
        evidence=write(folder,str(n)+'.json',value);specs.append((n,failure,evidence))
        if n<6:prior.append(dict(attempt=n,record=evidence,historical_only=True))
    report=dict(schema='R1_RX_IMPORT_COMPATIBILITY_REVIEW_V1',status='PASS_ZERO_MODEL_IMPORTS',
        CUDA_visible_devices='',CUDA_initialized=False,actual_R1_registration_created=False,
        prior_attempts_preserved=prior,final_probe=specs[-1][2],channel_calls=0,qualification_frames=0,
        real_model_constructors=0,scientific_calls=0,source_archive_reads=0,source_token_reads=0,tensor_operations_requested=0)
    rp=write(folder,'report.json',report);bp=write(folder,'bindings.json',dict(review_receipts=[rp]))
    cp=write(folder,'review.json',dict(approved=True,independent_review_bundle=bp))
    return rp,bp,cp,specs


class LineageTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.folder=Path(self.temp.name).resolve()
        self.args=fixture(self.folder)
    def tearDown(self):self.temp.cleanup()
    def test_authenticates_exact_three_failed_metadata_records(self):
        proof=A.authenticate(*self.args)
        self.assertEqual([Path(x['path']).name for x in proof['exact_historical_failure_allowances']],['1.json','2.json','5.json'])
        self.assertEqual(proof['validated_probes'],6)
    def test_altered_failure_bytes_rejected(self):
        Path(self.args[-1][0][2]['path']).write_text('{}')
        with self.assertRaisesRegex(RuntimeError,'pin changed'):A.authenticate(*self.args)
    def test_changed_report_pin_rejected(self):
        args=list(self.args);args[0]=dict(args[0],sha256='0'*64)
        with self.assertRaises(RuntimeError):A.authenticate(*args)
    def test_bad_review_chain_rejected(self):
        args=list(self.args);args[2]=write(self.folder,'other.json',dict(approved=True,independent_review_bundle={}))
        with self.assertRaisesRegex(RuntimeError,'Root review'):A.authenticate(*args)
    def test_nonzero_scientific_counter_rejected(self):
        for key in A.ZERO:
            value=A.readpin(self.args[-1][0][2]);value[key]=1
            with self.subTest(key=key),self.assertRaises(RuntimeError):A.validate_probe(value,1,True)
    def test_boolean_zero_counter_rejected(self):
        value=A.readpin(self.args[-1][0][2]);value['scientific_calls']=False
        with self.assertRaises(RuntimeError):A.validate_probe(value,1,True)
    def test_cuda_exposure_rejected(self):
        value=A.readpin(self.args[-1][0][2]);value['CUDA_VISIBLE_DEVICES']='0'
        with self.assertRaises(RuntimeError):A.validate_probe(value,1,True)
    def test_nonempty_final_blocked_list_rejected(self):
        value=A.readpin(self.args[-1][-1][2]);value['blocked']=['x']
        with self.assertRaises(RuntimeError):A.validate_probe(value,6,False)
    def test_registration_created_rejected(self):
        value=A.readpin(self.args[-1][0][2]);value['actual_R1_registration_created']=True
        with self.assertRaises(RuntimeError):A.validate_probe(value,1,True)
    def test_unknown_attempt_and_extra_allowance_rejected(self):
        args=list(self.args);args[-1]=args[-1]+[(7,True,args[-1][0][2])]
        with self.assertRaises(RuntimeError):A.authenticate(*args)
    def test_symlink_rejected(self):
        target=self.folder/'link.json'
        try:target.symlink_to(self.folder/'1.json')
        except OSError:
            with patch.object(Path,'lstat',return_value=types.SimpleNamespace(st_mode=0o120777)):
                with self.assertRaisesRegex(RuntimeError,'Symlink'):A.regular(self.folder/'1.json')
        else:
            with self.assertRaisesRegex(RuntimeError,'Symlink'):A.pin(target)
    def test_parent_traversal_rejected(self):
        with self.assertRaises(RuntimeError):A.regular(self.folder/'x'/'..'/'1.json')
    def test_byte_bool_rejected(self):
        with self.assertRaises(RuntimeError):A.checked(dict(self.args[0],bytes=True))


class RoutingTests(unittest.TestCase):
    def context(self):
        c=object.__new__(A.Context);c.proofs=[]
        c.adapter=dict(path=str(A.E/A.NAME/'adapter.py'),sha256='a'*64,bytes=1)
        c.runner=dict(path=str(A.E/A.NAME/'runner.py'),sha256='b'*64,bytes=2)
        c.inventory_pin=dict(path=str(A.E/A.NAME/'runtime_inventory.json'),sha256=A.INVENTORY_SHA,bytes=33978)
        c.lineage={};c.authorization_pin=dict(path=str(A.O/'r1_delivery_probe_lineage_preparation_v1/authorization.json'),sha256='c'*64,bytes=3)
        c.authorization=dict(files=[]);c.inventory=dict(empty_directories=[],published_receipts={})
        return c
    def test_actual_route_contains_runner_role_and_auth(self):
        c=self.context();argv=c.command('delivery')
        self.assertEqual(argv[:4],[A.CPY,'-B',c.runner['path'],'delivery'])
        self.assertIn(c.authorization_pin['sha256'],argv)
        self.assertNotIn(str(A.E/A.SOURCES['delivery'][0]),argv)
    def test_unknown_role_rejected(self):
        with self.assertRaises(RuntimeError):self.context().command('science')
    def test_dispatcher_inverse_ast_only_metadata(self):
        module=frozen('dispatcher');before=module.identity.__code__;c=self.context();c.install_dispatcher(module)
        self.assertIs(module.identity.__code__,before)
        self.assertEqual(c.proofs[0]['substitutions'],2)
        self.assertTrue(c.proofs[0]['inverse_AST'])
    def test_figures_inverse_ast_leaves_math_and_render_identical(self):
        module=frozen('figures');before={x:getattr(module,x).__code__ for x in ('load_tables','render','caption','axis_limits','run')}
        c=self.context();c.install_figures(module)
        self.assertTrue(all(getattr(module,x).__code__ is value for x,value in before.items()))
        self.assertTrue(c.proofs[0]['inverse_AST'])
    def test_backup_inverse_ast_leaves_archive_and_science_functions(self):
        module=frozen('backup');before={x:getattr(module,x).__code__ for x in ('build','deduplicate','verify_archive','check_archived_references')}
        c=self.context();c.inventory_files=lambda:[];c.install_backup(module)
        self.assertTrue(all(getattr(module,x).__code__ is value for x,value in before.items()))
        self.assertTrue(c.proofs[0]['inverse_AST'])
    def test_revision_receipt_exact_match_required(self):
        c=self.context();c.check_revision(dict(metadata_revision=c.revision()))
        wrong=copy.deepcopy(c.revision());wrong['runner']['sha256']='x'
        with self.assertRaises(RuntimeError):c.check_revision(dict(metadata_revision=wrong))
        with self.assertRaises(RuntimeError):c.check_revision({})
    def test_original_guard_rejects_new_failure(self):
        module=frozen('delivery');c=self.context()
        with tempfile.TemporaryDirectory() as td:
            runtime=Path(td).resolve();old=runtime/'r1_execution_v1';old.mkdir()
            known=old/'old_failure.json';known.write_text('{}')
            c.lineage=dict(exact_historical_failure_allowances=[dict(path=str(known))])
            with patch.object(A,'L',runtime),patch.object(A,'authenticate',return_value=c.lineage):
                original_preflight=module.preflight;c.install_delivery(module)
                module.no_failure_files(old)
                (old/'new_failure.json').write_text('{}')
                with self.assertRaisesRegex(RuntimeError,'Unreviewed failure'):module.no_failure_files(old)
                self.assertIs(module.failure_scope.__globals__['preflight'],module.preflight)
                self.assertNotEqual(module.preflight,original_preflight)
    def test_allowance_does_not_apply_other_root(self):
        module=frozen('delivery');c=self.context();c.lineage=dict(exact_historical_failure_allowances=[])
        with tempfile.TemporaryDirectory() as td:
            p=Path(td).resolve();(p/'failure.json').write_text('{}');c.install_delivery(module)
            with self.assertRaises(RuntimeError):module.no_failure_files(p)
    def test_installer_only_two_delivery_functions_changed(self):
        m=frozen('delivery');before={k:v for k,v in vars(m).items() if isinstance(v,types.FunctionType)}
        self.context().install_delivery(m)
        self.assertEqual({k for k,v in before.items() if getattr(m,k) is not v},{'preflight','no_failure_files'})


class FetchScopeTests(unittest.TestCase):
    def setUp(self):
        self.inventory=json.loads((HERE/'runtime_inventory.json').read_text(encoding='utf-8-sig'))
        self.assertEqual(A.digest((HERE/'runtime_inventory.json').read_bytes()),A.INVENTORY_SHA)
        def reject(name):raise RuntimeError('Original rejects '+name)
        self.fetch=types.SimpleNamespace(payload_name=reject,validate_manifest=lambda *x:{})
        self.build=types.SimpleNamespace(safe_parts=lambda n:self.assertNotIn('..',n.split('/')))
        A.install_fetch_scope(self.fetch,self.build,self.inventory)
    def manifest(self):
        prefix=(A.L/'r1_execution_v1').as_posix()+'/'
        rows=[dict(path='runtime/r1_execution_v1/'+p['path'][len(prefix):],source_path=p['path'],sha256=p['sha256'],bytes=p['bytes']) for p in self.inventory['files']]
        return dict(files=rows,aliases=[],empty_directories=['runtime/r1_execution_v1/'+p[len(prefix):] for p in self.inventory['empty_directories']],
            verification=dict(metadata_revision=dict(schema='R1_CPU_PREPARATION_PROBE_LINEAGE_REVISION_V1',historical_inventory=dict(sha256=A.INVENTORY_SHA))))
    def test_exact_sixteen_files_and_six_empty_names(self):
        m=self.manifest()
        for row in m['files']:self.fetch.payload_name(row['path'])
        for name in m['empty_directories']:self.fetch.payload_name(name+'/.empty_directory_metadata.json')
        self.fetch.validate_manifest(m,{},[])
    def test_unknown_preparation_name_not_allowed(self):
        with self.assertRaises(RuntimeError):self.fetch.payload_name('runtime/r1_execution_v1/preparation_zero_model_import_probe_v99/failure.json')
    def test_changed_archive_historical_sha_rejected(self):
        m=self.manifest();m['files'][0]['sha256']='0'*64
        with self.assertRaises(RuntimeError):self.fetch.validate_manifest(m,{},[])
    def test_missing_historical_file_rejected(self):
        m=self.manifest();m['files'].pop()
        with self.assertRaises(RuntimeError):self.fetch.validate_manifest(m,{},[])
    def test_missing_empty_directory_rejected(self):
        m=self.manifest();m['empty_directories'].pop()
        with self.assertRaises(RuntimeError):self.fetch.validate_manifest(m,{},[])
    def test_runtime_projection_has_exact_two_key_renames(self):
        projection=copy.deepcopy(self.inventory)
        self.assertEqual(projection.pop('original_inventory_sha256'),
            '8d3850bfa6e56ada3f9efd93f95b43bec09c196d4fea6548305660f6b12efdfd')
        proof=projection.pop('runtime_projection');self.assertEqual(len(proof['renamed_fields']),2)
        for key in ('manifest_pin','local_verification_pin'):
            row=projection['prior_verified_backup_evidence'][key];row['path']=row.pop('local_path')
        original=HERE/'prior_preparation_inventory.json'
        if original.exists():
            self.assertEqual(A.digest(original.read_bytes()),'8d3850bfa6e56ada3f9efd93f95b43bec09c196d4fea6548305660f6b12efdfd')
            self.assertEqual(projection,json.loads(original.read_text(encoding='utf-8-sig')))
        else:
            # Linux qualification snapshots include the immutable original as
            # test evidence; require it instead of silently weakening the proof.
            self.fail('Original local inventory fixture required for projection differential proof')
    def test_runtime_projection_has_no_windows_file_pins(self):
        build=frozen('backup')
        pins=list(build.inner_pins(self.inventory));self.assertTrue(pins)
        self.assertTrue(all(x['path'].startswith('/') for x in pins))
    def test_local_fetch_requires_same_adapter_and_runner(self):
        fetch=types.SimpleNamespace(payload_name=lambda n:n,validate_manifest=lambda *x:{})
        A.install_fetch_scope(fetch,self.build,self.inventory,dict(adapter='a',runner='b'))
        m=self.manifest();m['verification']['metadata_revision'].update(adapter=dict(sha256='a'),runner=dict(sha256='b'))
        fetch.validate_manifest(m,{},[])
        m['verification']['metadata_revision']['runner']['sha256']='changed'
        with self.assertRaises(RuntimeError):fetch.validate_manifest(m,{},[])


class InventoryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name).resolve()
        self.context=object.__new__(A.Context)
        self.context.inventory=dict(schema='R1_DELIVERY_PRIOR_PREPARATION_INVENTORY_V1',files=[],empty_directories=[])
        for n in range(16):
            self.context.inventory['files'].append(write(self.root,'r1_execution_v1/prep'+str(n%9)+'/'+str(n)+'.json',{}))
        for n in range(6):
            p=self.root/'r1_execution_v1'/('prep'+str(n))/'cache';p.mkdir()
            self.context.inventory['empty_directories'].append(str(p))
        self.patch=patch.object(A,'L',self.root);self.patch.start()
    def tearDown(self):self.patch.stop();self.temp.cleanup()
    def test_exact_metadata_files_verified(self):
        self.assertEqual(len(self.context.inventory_files()),16)
    def test_extra_file_in_preparation_root_rejected(self):
        write(self.root,'r1_execution_v1/prep0/unknown.json',{})
        with self.assertRaisesRegex(RuntimeError,'Unexpected or omitted'):self.context.inventory_files()
    def test_changed_known_metadata_rejected(self):
        Path(self.context.inventory['files'][0]['path']).write_text('changed')
        with self.assertRaises(RuntimeError):self.context.inventory_files()
    def test_missing_or_nonempty_historical_directory_rejected(self):
        path=Path(self.context.inventory['empty_directories'][0]);path.rmdir()
        with self.assertRaisesRegex(RuntimeError,'missing or nonempty'):self.context.inventory_files()
        path.mkdir();(path/'unexpected.txt').write_text('x')
        with self.assertRaisesRegex(RuntimeError,'missing or nonempty'):self.context.inventory_files()


if __name__=='__main__':unittest.main()

"""CPU startup/launch-scope fixtures; no original sources or GPU are changed."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

HERE=Path(__file__).parent
spec=importlib.util.spec_from_file_location('numeric_startup_tested',HERE/'launch_recovery.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class Fixture:
    def __init__(self,root):
        self.paths=m.layout(root);self.root=self.paths['root'];self.runtime=self.paths['runtime'];self.runtime.mkdir(parents=True)
        for name in ('sitecustomize.py','launch_recovery.py','numeric_snr.py','test_startup.py','README.md'):
            (self.runtime/name).write_text('bound '+name)
        self.failure=self.paths['out']/'archive/failed.log';self.failure.parent.mkdir();self.failure.write_text('original failure preserved')
        frozen={}
        for folder,names in ((self.paths['original'],('replay.py','runner.py')),
                             (self.paths['cache']/'runtime',('concurrent_runner.py','replay_compat.py')),
                             (self.paths['controller'].parent,('controller.py','final_runner.py'))):
            folder.mkdir(parents=True)
            for name in names:(folder/name).write_text('original '+name)
            frozen.update(m.sources(folder))
        assets=self.root/'outputs/UNIFIED-METRICS-20261002/assets_complete.json';assets.parent.mkdir(parents=True)
        assets.write_text(json.dumps({'environment':'/usr/bin/python3'}));frozen[str(assets)]=m.sha(assets)
        table=self.root/'actual.csv';table.write_text('snr_db\n1.0\n')
        policy=self.root/'policy.json';policy.write_text('{"snr_db":1.0}')
        self.qualification=self.paths['out']/'numeric_snr_qualification.json'
        q=dict(status='REAL_NUMERIC_SNR_COMPATIBILITY_PASS',training_updates=0,policy_selection_updates=0,
            raw_strings_unchanged=True,source_row_hashes_unchanged=True,row_ids_unchanged=True,
            actual_grid_equal=True,integer_grid_valid=True,other_studies_unchanged=True,
            source_bindings={str(self.runtime/'numeric_snr.py'):m.sha(self.runtime/'numeric_snr.py'),
                             str(self.paths['original']/'replay.py'):m.sha(self.paths['original']/'replay.py')},
            table_bindings={str(table):m.sha(table),str(policy):m.sha(policy)})
        m.write(self.qualification,q)
        self.manifest=self.paths['out']/'manifest.json'
        self.value=dict(status='REGISTERED_METRIC_NUMERIC_RECOVERY',root=str(self.root),
            controller_entry=str(self.paths['controller']),final_runner_entry=str(self.paths['final_runner']),
            cache_dir=str(self.paths['cache']),runtime_source_bindings=m.sources(self.runtime),frozen_source_bindings=frozen,
            previous_failure_bindings={str(self.failure):m.sha(self.failure)},qualification_path=str(self.qualification),
            qualification_sha256=m.sha(self.qualification),python_executable='/usr/bin/python3',training_updates=0,
            policy_selection_updates=0,inference_functions_changed=False,metric_values_changed=False)
        self.save()
    def save(self):m.write(self.manifest,self.value)
    def argv(self,child=False):
        return [str(self.paths['final_runner'] if child else self.paths['controller']),'--root',str(self.root)]+(
            ['--cache-dir',str(self.paths['cache'])] if child else [])
    def validate(self,child=False):return m.validate(self.manifest,self.argv(child),self.runtime,root=self.root)


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.f=Fixture(self.tmp.name)
    def test_full_manifest_parent_and_child_pass(self):
        for child in (False,True):
            value,paths,scope,bound=self.f.validate(child)
            self.assertEqual(scope,'final_runner' if child else 'controller')
            self.assertEqual(value,self.f.value);self.assertIn(str(self.f.qualification),bound)
            self.assertIn(str(self.f.runtime/'numeric_snr.py'),bound)
    def test_manifest_source_policy_and_argument_mutations_refused(self):
        for field,bad in (('metric_values_changed',True),('training_updates',1),('root','other'),('python_executable','wrong')):
            old=self.f.value[field];self.f.value[field]=bad;self.f.save()
            with self.assertRaises(RuntimeError):self.f.validate()
            self.f.value[field]=old;self.f.save()
        with self.assertRaises(RuntimeError):m.validate(self.f.manifest,self.f.argv()+['extra'],self.f.runtime,root=self.f.root)
        (self.f.paths['original']/'replay.py').write_text('mutated')
        with self.assertRaises(RuntimeError):self.f.validate()
    def test_partial_source_inventory_and_stale_actual_qualification_rejected(self):
        original=dict(self.f.value['frozen_source_bindings'])
        self.f.value['frozen_source_bindings'].pop(str(self.f.paths['controller']));self.f.save()
        with self.assertRaises(RuntimeError):self.f.validate()
        self.f.value['frozen_source_bindings']=original;self.f.save()
        q=m.read(self.f.qualification);q['row_ids_unchanged']=False;m.write(self.f.qualification,q)
        self.f.value['qualification_sha256']=m.sha(self.f.qualification);self.f.save()
        with self.assertRaises(RuntimeError):self.f.validate()
    def test_archived_failure_hash_is_required(self):
        self.f.failure.write_text('discarded historical failure')
        with self.assertRaises(RuntimeError):self.f.validate()


class HookTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.f=Fixture(self.tmp.name)
        self.value,self.paths,self.scope,self.bound=self.f.validate();self.events=[]
    def test_parent_only_injects_exact_final_child_env_and_preserves_other_settings(self):
        calls=[]
        def original(*a,**k):calls.append((a,k));return 'child'
        with mock.patch.object(subprocess,'Popen',original):
            m.install_parent(self.value,self.paths,self.bound,lambda *a,**k:self.events.append((a,k)))
            command=[self.value['python_executable'],'-u',*self.f.argv(True)]
            supplied=dict(PYTHONPATH='original-path',OMP_NUM_THREADS='6',CUDA_VISIBLE_DEVICES='0')
            self.assertEqual(subprocess.Popen(command,env=supplied,cwd=self.f.root,stdout=123),'child')
            self.assertEqual(supplied['PYTHONPATH'],'original-path')
            actual=calls[-1][1];self.assertEqual(actual['stdout'],123);self.assertEqual(actual['cwd'],self.f.root)
            self.assertEqual(actual['env']['OMP_NUM_THREADS'],'6');self.assertEqual(actual['env']['CUDA_VISIBLE_DEVICES'],'0')
            self.assertEqual(actual['env'][m.ENV],str(self.f.manifest))
            self.assertEqual(actual['env']['PYTHONPATH'],str(self.f.runtime)+os.pathsep+'original-path')
            other=['python','original_analysis.py'];subprocess.Popen(other,env=supplied)
            self.assertIs(calls[-1][1]['env'],supplied)
            with self.assertRaises(RuntimeError):subprocess.Popen(command+['--other'],env=supplied)
    def test_child_origin_failure_and_preimport_rejected(self):
        module=types.SimpleNamespace(__file__='wrong',imports=lambda root:())
        with self.assertRaises(RuntimeError):m.adapt_imports(module,self.value,self.paths,self.bound,lambda *a,**k:None)
        with mock.patch.object(m,'validate',return_value=(self.value,self.paths,'final_runner',self.bound)),\
                mock.patch.object(m,'recorder',return_value=lambda *a,**k:None),\
                mock.patch.dict(sys.modules,{'concurrent_runner':module}):
            with self.assertRaises(RuntimeError):m.install(self.f.manifest,self.f.argv(True),self.f.runtime,root=self.f.root)
    def test_child_hook_preserves_imports_and_binds_extra_inputs_without_source_map_edit(self):
        adapter=self.f.runtime/'numeric_snr.py';adapter.write_text("def install(replay,root):\n replay._numeric_snr_receipt={'name':'STRICT_STRING_INT'}\n return replay._numeric_snr_receipt\n")
        self.bound[str(adapter)]=m.sha(adapter)
        class Engine:
            rows={'N512':[], 'M2_ACTUAL':[]}
            def __init__(self):self.artifacts={};self.source_bindings={'frozen':'unchanged'}
            def bind(self,path):self.artifacts[path]=m.sha(path);return self.artifacts[path]
            def manifest(self):return dict(model_identity='frozen',source_ids=['original'],parity_tolerances='unchanged')
        replay=types.SimpleNamespace(create_engine=lambda root:Engine())
        answer=('runner',replay,'models','batch');namespace={'answer':answer}
        exec(compile('def imports(root):\n return answer\n',str(self.paths['cache']/'runtime/concurrent_runner.py'),'exec'),namespace)
        module=types.SimpleNamespace(__file__=str(self.paths['cache']/'runtime/concurrent_runner.py'),imports=namespace['imports'])
        m.adapt_imports(module,self.value,self.paths,self.bound,lambda *a,**k:self.events.append((a,k)))
        self.assertIs(module.imports(self.f.root),answer)
        engine=replay.create_engine(self.f.root)
        self.assertEqual(engine.source_bindings,{'frozen':'unchanged'})
        self.assertEqual(engine.artifacts,self.bound)
        self.assertEqual(engine.manifest()['model_identity'],'frozen')
        proof=engine.manifest()['numeric_snr_compatibility']
        self.assertEqual(proof['manifest_sha256'],m.sha(self.f.manifest));self.assertTrue(proof['raw_csv_unchanged'])
    def test_meta_import_adapter_delegates_other_names_and_refuses_wrong_origin(self):
        hook=m.ImportAdapter(self.value,self.paths,self.bound,lambda *a,**k:None)
        self.assertIsNone(hook.find_spec('unrelated'))
        with mock.patch.object(m.importlib.machinery.PathFinder,'find_spec',return_value=types.SimpleNamespace(origin='wrong')):
            with self.assertRaises(RuntimeError):hook.find_spec('concurrent_runner')


class StartupTests(unittest.TestCase):
    def test_non_target_inert(self):
        with mock.patch.object(sys,'argv',['unrelated.py']),mock.patch.dict(os.environ,{},clear=True),mock.patch.object(os,'_exit') as ex:
            exec(compile((HERE/'sitecustomize.py').read_text(),str(HERE/'sitecustomize.py'),'exec'),{'__file__':str(HERE/'sitecustomize.py')})
            ex.assert_not_called()
    def test_target_failure_exits78_instead_of_ignored_site_error(self):
        entry='/home/liulu/projects/VAR_COMM/outputs/METRIC-FINAL-CACHE-20261002/runtime/controller.py'
        with mock.patch.object(sys,'argv',[entry]),mock.patch.dict(os.environ,{},clear=True),mock.patch.object(os,'_exit',side_effect=SystemExit(78)):
            with self.assertRaises(SystemExit) as failure:
                exec(compile((HERE/'sitecustomize.py').read_text(),str(HERE/'sitecustomize.py'),'exec'),{'__file__':str(HERE/'sitecustomize.py')})
            self.assertEqual(failure.exception.code,78)
    def test_no_torch_import_in_startup_adapter(self):
        self.assertNotIn('import torch',(HERE/'launch_recovery.py').read_text())


class LaunchTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.f=Fixture(self.tmp.name)
        self.old=dict(pid=100,start_ticks='11');self.f.value['previous_controller']=self.old;self.f.save()
        patch=mock.patch.object(m,'__file__',str(self.f.runtime/'launch_recovery.py'));patch.start();self.addCleanup(patch.stop)
        self.final=self.f.paths['controller'].parent.parent
        m.write(self.final/'controller_launch.json',self.old)
        m.write(self.final/'controller_status.json',dict(status='FAILED',pid=100,worker_pid=None))
    def test_active_previous_owner_or_unchecked_ticks_rejected(self):
        with mock.patch.object(m,'process',return_value=dict(self.old,state='S')):
            with self.assertRaises(RuntimeError):m.launch(self.f.manifest,root=self.f.root)
        m.write(self.final/'controller_launch.json',dict(pid=100,start_ticks='999'))
        with mock.patch.object(m,'process',return_value=None):
            with self.assertRaises(RuntimeError):m.launch(self.f.manifest,root=self.f.root)
    def test_original_frozen_entry_launch_and_checked_retry_keep_manifest(self):
        calls=[];pid=[200]
        def spawn(*a,**k):calls.append((a,k));return types.SimpleNamespace(pid=pid[0])
        def reader(value):return dict(pid=value,start_ticks=str(value+1),state='S') if value==pid[0] else None
        fcntl=types.SimpleNamespace(LOCK_EX=1,LOCK_NB=2,flock=lambda *a:None)
        with mock.patch.dict(sys.modules,{'fcntl':fcntl}),mock.patch.object(m.subprocess,'Popen',spawn),mock.patch.object(m,'process',side_effect=reader):
            first=m.launch(self.f.manifest,root=self.f.root)
            self.assertEqual(calls[0][0][0],[self.f.value['python_executable'],'-u',*self.f.argv()])
            self.assertTrue(calls[0][1]['start_new_session']);self.assertEqual(first['previous_controller'],self.old)
            self.assertEqual(calls[0][1]['env'][m.ENV],str(self.f.manifest))
            m.write(self.final/'controller_launch.json',dict(pid=first['pid'],start_ticks=first['start_ticks']))
            m.write(self.final/'controller_status.json',dict(status='FAILED',pid=first['pid'],worker_pid=None))
            pid[0]=300
            second=m.launch(self.f.manifest,root=self.f.root)
            self.assertEqual(second['manifest_sha256'],first['manifest_sha256'])
            self.assertEqual(second['inputs'],first['inputs']);self.assertEqual(second['previous_controller'],self.old)
            self.assertTrue((self.f.paths['out']/'launch_history/200_201.json').exists())
            self.assertTrue((self.f.paths['out']/'launch_history/300_301.json').exists())


if __name__=='__main__':unittest.main()

"""Synthetic CPU tests only: no model, channel, actual runtime or SSH access."""
from pathlib import Path
import ast, collections, copy, importlib.util, json, os, sys, tempfile, types, unittest
from unittest import mock
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import common as c
import adapter as a
import registration as b
import runner
import dispatch_once as dispatch

W=HERE.parents[2]
OLD=W/'experiments/layered-mismatch-raw-20261013/r1_execution_resource_v2'

def original_runtime():
    r=dict(root=str(c.ANCESTOR),python_cpu='/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/var_comm_runtime_20261010/envs/frozen_ldpc_py311/bin/python',
       python_visual='/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/var_comm_runtime_20261010/envs/frozen_visual_py310/bin/python',
       target_gpu=3,cpu_affinity=[0,1],phy_cpu_affinity=[2,3],max_seconds=43200,deadline_unix=c.DEADLINE,stop_file=str(c.OLD_OUT/'STOP'))
    r.update({s+'_directory':str(c.ANCESTOR/s) for s in ('phy','projection',*c.STAGES)})
    return r

class RecoveryTests(unittest.TestCase):
    def test_probe_real_zero_qualification_and_rejected_variants(self):
        done=dict(schema='R1_RX_CACHE_ZERO_MODEL_QUALIFICATION_V2',status='PASS_ZERO_MODEL_CACHE_IMPORT_FLAGS',probe=c.PROBE_SOURCE,
            source=dict(path=str(c.OLD/'receiver_visual.py'),sha256=c.OLD_SHA['receiver_visual.py']),
            bundle=dict(path=str(c.OLD_OUT/'registered/model_bundle.json'),sha256='c1d25aaf76fdb36098e225c8dba1157b0d4ea92c9afccb39f348e2121c87a0a0'),
            scientific_calls=0,model_or_channel_calls=0,qualification_frames=0,source_truth_reads=0,model_weight_bytes_read=0,
            actual_scientific_registration_created=False,CUDA_VISIBLE_DEVICES='',cuda_initialized_before_flags=False,cuda_initialized_after_flags=False,
            blocked=[],storage_blocked=[],rejected_requests=[],request_counts={k:0 for k in c.PROBE_COUNTERS})
        wait=dict(actual_wait=True,returncode=0,pid=10,CUDA_VISIBLE_DEVICES='',automatic_retry=False,probe_sha256=c.PROBE_SOURCE['sha256'])
        auth=dict(probe_source=c.PROBE_SOURCE,zero_model_probe=dict(path=str(c.PROBE_OUT/'completion.json'),sha256='a'*64),
            zero_model_probe_actual_wait=dict(path=str(c.PROBE_WAIT),sha256='b'*64))
        def validate(d,w,identity=None,failed=False):
            with mock.patch.object(c,'checked'),mock.patch.object(c,'readpin',side_effect=[d,w]),mock.patch.object(c,'identity',return_value=identity),mock.patch.object(Path,'exists',return_value=failed):
                return c.validate_probe(auth)
        self.assertEqual(validate(done,wait),done)
        for updates in ({'schema':'old'},{'status':'STARTED'}, {'scientific_calls':1},{'scientific_calls':False},
            {'cuda_initialized_after_flags':True},{'blocked':['denied']},{'storage_blocked':['write']},{'rejected_requests':['tensor']},
            {'request_counts':dict(done['request_counts'],tensor_requests=1)}):
            with self.assertRaises(RuntimeError):validate(dict(done,**updates),wait)
        for updates in ({'returncode':False},{'actual_wait':False},{'pid':True},{'timed_out':True},{'probe_sha256':'0'*64}):
            with self.assertRaises(RuntimeError):validate(done,dict(wait,**updates))
        with self.assertRaises(RuntimeError):validate(done,wait,identity={'pid':10,'start_ticks':1})
        with self.assertRaises(RuntimeError):validate(done,wait,failed=True)

    def test_receiver_exact_inverse_ast_and_math_unchanged(self):
        text=(OLD/'receiver_visual.py').read_text(encoding='utf8');changed=a.receiver_source(text)
        self.assertEqual(ast.dump(ast.parse(changed.replace(a.CACHE_NEW,a.CACHE_OLD))),ast.dump(ast.parse(text)))
        before={n.name:ast.dump(n) for n in ast.parse(text).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        after={n.name:ast.dump(n) for n in ast.parse(changed).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        self.assertEqual({k:v for k,v in before.items() if k!='visual'},{k:v for k,v in after.items() if k!='visual'})
        self.assertIn('out.mkdir();_prepare_receiver_cache(out)',changed)
        self.assertLess(changed.index(a.CACHE_NEW),changed.index('firewall=SourceReadFirewall'))

    def test_no_ambiguous_or_repeated_cache_adaptation(self):
        text=(OLD/'receiver_visual.py').read_text(encoding='utf8')
        for bad in ('',text.replace(a.CACHE_OLD,a.CACHE_OLD+';'+a.CACHE_OLD),a.receiver_source(text)):
            with self.assertRaises(RuntimeError):a.receiver_source(bad)

    def test_frozen_original_paths_deadline_resources(self):
        old=original_runtime();new=c.frozen_runtime(old)
        self.assertEqual(old,original_runtime())
        for k in old:
            if k not in ('root','visual_directory','score_directory','selection_directory'):self.assertEqual(old[k],new[k])
        a.validate_runtime(new,old);a.validate_visual_runtime(new)
        for key,value in [('deadline_unix',c.DEADLINE+1),('target_gpu',0),('max_seconds',999999),('cpu_affinity',[2,3])]:
            bad=copy.deepcopy(new);bad[key]=value
            with self.assertRaises(RuntimeError):a.validate_visual_runtime(bad)

    def test_private_cache_created_only_after_new_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);out=root/'visual'
            with mock.patch.object(c,'RUN',root),mock.patch.dict(os.environ,{},clear=False),mock.patch.object(tempfile,'tempdir',None):
                with self.assertRaises(RuntimeError):a.prepare_receiver_cache(out)
                out.mkdir();r=a.prepare_receiver_cache(out)
                self.assertEqual(tempfile.tempdir,str(out/'runtime_cache/tmp'))
                for k in ('TMPDIR','TEMP','TMP'):self.assertEqual(os.environ[k],str(out/'runtime_cache/tmp'))
                self.assertEqual(os.environ['TORCHINDUCTOR_CACHE_DIR'],str(out/'runtime_cache/inductor'))
                self.assertEqual(os.environ['TRITON_CACHE_DIR'],str(out/'runtime_cache/triton'))
                self.assertFalse(r['source_truth_access_added'])
                with self.assertRaises(RuntimeError):a.prepare_receiver_cache(out)

    def test_visual_has_no_cpu_recovery_source_descriptor(self):
        old_visual=dict(runtime=original_runtime(),code_bindings={'a.py':'b'},model_bundle={'path':'/old/pinned/model_bundle.json','sha256':'1'*64},
                        pins={'common_ledger':{'path':'/old/ledger.py','sha256':'2'*64}},schema='visual')
        master=dict(runtime=c.frozen_runtime(original_runtime()),code_bindings={'new.py':'3'*64},
                    recovery={'original_registration':c.ORIGINAL,'failure_audit':c.FAILURE_AUDIT})
        control=dict(schema='R1_RX_ONLY_INHERITED_PROJECTION_V1',original_master_sha256=c.ORIGINAL['sha256'],actual_wait={'path':str(c.ANCESTOR/'projection/actual_wait.json'),'sha256':'4'*64},
            projection_process={'pid':2,'start_ticks':3},owner_process={'pid':4,'start_ticks':5},source_truth_access_added=False)
        child=b.visual_projection(master,old_visual,control)
        self.assertNotIn('recovery',child);self.assertNotIn(c.ORIGINAL['path'],json.dumps(child));self.assertNotIn(c.FAILURE_AUDIT['path'],json.dumps(child))
        self.assertEqual(child['model_bundle'],old_visual['model_bundle']);self.assertEqual(child['pins'],old_visual['pins'])
        self.assertEqual(child['inherited_projection'],control)

    def test_frozen_firewall_accepts_private_cache_not_sources(self):
        text=(OLD/'receiver_visual.py').read_text(encoding='utf8');node=next(n for n in ast.parse(text).body if isinstance(n,ast.ClassDef) and n.name=='SourceReadFirewall')
        ns=dict(Path=Path,os=os,collections=collections);exec(compile(ast.Module(body=[node],type_ignores=[]),'fixture','exec'),ns)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve();out=root/'new/visual';out.mkdir(parents=True)
            f=ns['SourceReadFirewall'].__new__(ns['SourceReadFirewall']);f.files=set();f.output=out;f.codes=set();f.root=root
            f.env=root/'env';f.metadata_dirs=set();f.absent_stdlib_zip=None
            self.assertTrue(f.permitted(out/'runtime_cache/test',True))
            for p in (root/'old/cache/test',root/'sources/image.npz',root/'sources/tokens.npz',root/'old/phy/frame.json'):
                self.assertFalse(f.permitted(p,True));self.assertFalse(f.permitted(p,False))

    def test_actual_wait_integer_exit_and_original_gone(self):
        child={'pid':123,'start_ticks':456};valid=dict(actual_wait=True,pid=123,returncode=0,terminated=False,killed=False,timed_out=False)
        c.actual(valid,child,0)
        for update in ({'returncode':False},{'returncode':1},{'actual_wait':False},{'timed_out':True},{'pid':124}):
            with self.assertRaises(RuntimeError):c.actual(dict(valid,**update),child,0)
        for state in ('R','S','D','Z'):
            with self.assertRaises(RuntimeError):c.gone(child,lambda p:dict(child,state=state))
        c.gone(child,lambda p:None);c.gone(child,lambda p:dict(pid=123,start_ticks=789))
        for bad in ({'pid':True,'start_ticks':456},{'pid':123,'start_ticks':True},{'pid':0,'start_ticks':456}):
            with self.assertRaises(RuntimeError):c.actual(valid,bad,0)

    def test_private_install_routes_same_shared_and_only_missing_workers(self):
        runtime=original_runtime();old=dict(runtime=runtime,code_bindings={})
        master=dict(runtime=c.frozen_runtime(runtime),code_bindings={})
        binding=dict(path=str(c.OUT/'registered/registration.json'),sha256='a'*64)
        window_calls=[];source_calls=[]
        shared=types.ModuleType('shared_fixture');shared.MASTER_SCHEMA='master';shared.validate_registration=lambda v:master
        receiver=types.ModuleType('receiver_fixture');receiver.require_successful_wait=lambda d:None
        score=types.ModuleType('score_fixture')
        owner=types.ModuleType('owner_fixture');owner.__file__=str(c.OLD/'owner.py');owner.s=shared
        tree=ast.parse((OLD/'owner.py').read_text(encoding='utf8'))
        command=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='stage_command')
        exec(compile(ast.Module(body=[command],type_ignores=[]),'fixture','exec'),dict(Path=Path,__file__=owner.__file__,s=shared),owner.__dict__)
        shared.require=c.require
        owner.acquire_window=lambda reg,stage,flock:(window_calls.append(stage) or (None,{}))
        def load(name,path,text=None):
            source_calls.append((name,path,text));m={'shared':shared,'receiver_visual':receiver,'score_selection':score,'_cache_recovery_original_owner':owner}[name]
            if name!='shared':m.s=shared
            return m
        pin_adapter=types.SimpleNamespace(install_phy=mock.Mock())
        spec=types.SimpleNamespace(loader=types.SimpleNamespace(exec_module=lambda m:None))
        install_score=mock.Mock()
        with mock.patch.object(c,'readpin',side_effect=[master,old]),mock.patch.object(c,'sha',side_effect=lambda p:c.OLD_SHA[Path(p).name]), \
             mock.patch.object(a,'load_source',side_effect=load),mock.patch.object(a.importlib.util,'spec_from_file_location',return_value=spec), \
             mock.patch.object(a.importlib.util,'module_from_spec',return_value=pin_adapter),mock.patch.dict(sys.modules,{'score_provenance':types.SimpleNamespace(install=install_score)}), \
             mock.patch.object(Path,'read_text',return_value=(OLD/'receiver_visual.py').read_text(encoding='utf8')):
            s,rx,sc,own=a.install(binding,'owner')
        self.assertIs(s,shared);self.assertIs(rx.s,s);self.assertIs(sc.s,s);self.assertIs(own.s,s)
        self.assertEqual(own.STAGES,('visual','score','selection'));install_score.assert_called_once_with(score)
        reg=dict(runtime=master['runtime'],visual_registration=dict(path='visual.json',sha256='b'*64))
        for stage,script,action in [('visual','receiver_visual.py','visual'),('score','score_selection.py','score'),('selection','score_selection.py','select')]:
            argv=own.stage_command(reg,binding,stage,100,200)
            self.assertEqual(argv[3:7],[str(c.HERE/'runner.py'),'--original',str(c.OLD/script),action])
        with self.assertRaises(RuntimeError):own.stage_command(reg,binding,'phy',100,200)
        own.acquire_window(reg,'phy',None);self.assertEqual(window_calls,['visual'])

    def test_owned_termination_actual_wait_and_group_boundary(self):
        child=types.SimpleNamespace(pid=8,poll=mock.Mock(return_value=None),wait=mock.Mock(side_effect=[dispatch.subprocess.TimeoutExpired('fake',15),-9]))
        with mock.patch.object(dispatch.os,'getpgid',return_value=8,create=True),mock.patch.object(dispatch.os,'killpg',create=True) as kill,mock.patch.object(dispatch.signal,'SIGKILL',9,create=True):
            value=dispatch.terminate_owned(child)
        self.assertEqual(value,dict(actual_wait=True,pid=8,returncode=-9,terminated=True,killed=True))
        self.assertEqual(child.wait.call_count,2);self.assertEqual(kill.call_count,2)
        with mock.patch.object(dispatch.os,'getpgid',return_value=9,create=True),mock.patch.object(dispatch.os,'killpg',side_effect=AssertionError('No foreign group'),create=True):
            with self.assertRaises(RuntimeError):dispatch.terminate_owned(child)
        tree=ast.parse((HERE/'dispatch_once.py').read_text(encoding='utf8'))
        popen=next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='Popen')
        fds=next(k.value for k in popen.keywords if k.arg=='pass_fds')
        self.assertEqual([n.func.value.id for n in fds.elts],['lock','old_lock','old_dispatch'])

    def test_source_free_projection_wait_reads_only_explicit_pin(self):
        child={'pid':123,'start_ticks':456};owner={'pid':100,'start_ticks':400}
        wait=dict(actual_wait=True,pid=123,returncode=0)
        control=dict(schema='R1_RX_ONLY_INHERITED_PROJECTION_V1',original_master_sha256=c.ORIGINAL['sha256'],source_truth_access_added=False,
             actual_wait={'path':str(c.ANCESTOR/'projection/actual_wait.json'),'sha256':'a'*64},projection_process=child,owner_process=owner)
        calls=[]
        with mock.patch.object(c,'readpin',side_effect=lambda p:(calls.append(p) or wait)),mock.patch.object(c,'gone'),mock.patch.object(c,'pin',return_value=control['actual_wait']):
            self.assertEqual(a.receiver_inherited_wait(c.ANCESTOR/'projection',control,None),control['actual_wait'])
        self.assertEqual(calls,[control['actual_wait']])

    def test_runner_has_no_completed_scientific_routes(self):
        for path,action in ((c.E/'r1_phy.py','run'),(c.OLD/'receiver_visual.py','project')):
            with self.assertRaises(RuntimeError):runner.run(['--original',str(path),action])

    def test_source_no_register_model_bundle_or_train_calls(self):
        for name in ('common.py','registration.py','owner.py','dispatch_once.py','runner.py','adapter.py'):
            tree=ast.parse((HERE/name).read_text(encoding='utf8'))
            for node in ast.walk(tree):
                if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
                    self.assertNotIn(node.func.attr,('build_model_bundle','physical_registration','run_phy','backward','optimizer_step'))
        self.assertEqual(c.STAGES,('visual','score','selection'))

    def test_no_registration_if_no_authorization(self):
        with mock.patch.object(c,'readpin',side_effect=RuntimeError('unapproved')),mock.patch.object(Path,'mkdir',side_effect=AssertionError('No output')):
            with self.assertRaisesRegex(RuntimeError,'unapproved'):b.build({'path':'/none','sha256':'a'*64})

    def test_no_import_exec_science(self):
        for name in ('common.py','adapter.py','registration.py','runner.py','owner.py','dispatch_once.py'):
            tree=ast.parse((HERE/name).read_text(encoding='utf8'))
            for node in tree.body:
                if isinstance(node,ast.Import):self.assertFalse(any(a.name.split('.')[0] in ('torch','numpy','sionna','tensorflow') for a in node.names))

if __name__=='__main__':unittest.main()

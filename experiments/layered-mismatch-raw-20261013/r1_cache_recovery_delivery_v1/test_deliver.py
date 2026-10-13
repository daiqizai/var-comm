"""Synthetic metadata only. No real run, quality file, model or channel is read."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest import mock

import deliver as recovery

WORKSPACE = Path(__file__).resolve().parents[3]
original = recovery.load_original(WORKSPACE)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding='utf8')
    return original.pin(path)


def short(value):
    return {k:value[k] for k in ('path', 'sha256')}


def fixture(base):
    root, temporary = base/'repo', base/'personal'
    p = recovery.paths(root, temporary)
    code = {}
    for name in recovery.CODE:
        path = p['recovery']/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# synthetic source '+name+'\n', encoding='utf8')
        code[name] = original.sha(path)
    experiment = p['resource'].parent
    old_code = {}
    for name in ('runner.py', 'pin_adapter.py'):
        path = experiment/'r1_source_identity_revision_v1'/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# synthetic old route '+name, encoding='utf8')
        old_code[str(path)] = original.sha(path)
    common = root/'experiments/entropy-main-common500-20261013/common.py'
    common.parent.mkdir(parents=True, exist_ok=True);common.write_text('# synthetic ledger', encoding='utf8')
    runtime = dict(root=str(p['old_run']), python_cpu='cpu', python_visual='visual',
        cpu_affinity=[0,1], phy_cpu_affinity=[2,3], deadline_unix=123456.75)
    runtime.update({stage+'_directory':str(p['old_run']/stage) for stage in original.STAGES})
    old = dict(runtime=runtime, code_bindings=old_code, pins=dict(common_ledger=original.pin(common)),
        expected_physical_frames=32400, expected_logical_var_images=43200,
        phy_registration=short(save(p['old_result']/'registered/phy_registration.json', {'fixture':'phy'})),
        visual_registration=short(save(p['old_result']/'registered/visual_registration.json', {'fixture':'old visual'})))
    oldpin = short(save(p['old_result']/'registered/registration.json', old))
    oldowner = dict(pid=200, start_ticks=20, state='S')
    oldsupervisor = dict(pid=100, start_ticks=10, state='S')
    save(p['old_run']/'owner_launch.json', dict(process=oldowner, registration=oldpin))
    save(p['old_dispatch']/'launch.json', dict(process=oldsupervisor, registration=oldpin,
        argv=['cpu','-B','-u',*original.routed(old,p['resource']/'owner.py',
            ['--registration',oldpin['path'],'--registration-sha256',oldpin['sha256']])]))
    save(p['old_initial']/'launch.json', dict(process=oldsupervisor, registration=oldpin,
        argv=['cpu','-B','-u',*original.routed(old,p['resource']/'dispatch_once.py',
            ['--expected-registration',oldpin['sha256']])]))
    save(p['old_dispatch']/'owner_identity.json', dict(process=oldowner, registration=oldpin))
    save(p['old_dispatch']/'actual_wait.json', dict(actual_wait=True, pid=200, returncode=1,
        identity_at_launch=oldowner, registration=oldpin))

    def stage(reg, regpin, name, inherited, code, index):
        run=p['old_run'] if inherited else p['run'];owner=oldowner if inherited else newowner
        process=dict(pid=(300 if inherited else 600)+index, start_ticks=(30 if inherited else 60)+index, state='S')
        launch=save(run/(name+'_launch.json'), dict(stage=name, process=process, registration=regpin,
            deadline_unix=reg['runtime']['deadline_unix'],
            cpu_affinity=reg['runtime']['cpu_affinity' if name in ('visual','score') else 'phy_cpu_affinity'],
            argv=recovery.stage_argv(p,reg,regpin,name,owner,inherited,original)))
        wait=dict(actual_wait=True, pid=process['pid'], returncode=code, stage=name)
        wp=save(run/name/'actual_wait.json',wait);save(run/(name+'_actual_wait.json'),wait)
        result=dict(stage=name,launch=launch,actual_wait=wp)
        if code==0:
            child=reg['phy_registration'] if name=='phy' else reg['visual_registration'] if name=='visual' else regpin
            result['completion']=save(run/name/'completion.json',dict(status=original.STAGE_STATUS[name],
                registration=child, channel_crc_accept=False))
        return result,process

    oldstages=[];oldprocesses=[]
    for index,name in enumerate(('phy','projection','visual')):
        item,identity=stage(old,oldpin,name,True,1 if name=='visual' else 0,index)
        oldstages.append(item);oldprocesses.append(identity)
    save(p['old_run']/'owner_failure.json',dict(error='preserved synthetic initialization failure'))
    save(p['old_run']/'visual/failure.json',dict(error='denied synthetic temporary write'))
    save(p['old_run']/'visual/partial_logical_frames.json',[])
    save(p['old_run']/'visual/resource_history_at_failure.json',[dict(synthetic=True)])
    (p['old_run']/'visual/images').mkdir()
    ledger=p['old_run']/'visual/calls.sqlite'
    db=sqlite3.connect(ledger)
    db.execute('CREATE TABLE calls(id INTEGER,kind TEXT,metadata TEXT,state TEXT)')
    db.execute("INSERT INTO calls VALUES(1,'model_load','{}','FAILED')");db.commit();db.close()
    evidence={str(path):original.pin(path) for directory in (p['old_run'],p['old_dispatch'])
        for path in directory.rglob('*') if path.is_file() and path.name!='resource_history_at_failure.json'}
    failure=dict(schema='R1_CLOSED_VISUAL_CACHE_FAILURE_AUDIT_V1',status='DIAGNOSED_NO_AUTOMATIC_RETRY',
        original_registration=oldpin,original_outer_returncode=1,original_deadline_unix=runtime['deadline_unix'],
        completed_stages=['phy','projection'],remaining_unlaunched_stages=['score','selection'],
        failed_calls=[dict(id=1,kind='model_load',metadata='{}',state='FAILED')],
        extra_failed_initialization_attempts=1,evidence=evidence,
        identities={str(i):dict(expected=item,original_identity_gone=True,actual=None)
                    for i,item in enumerate((oldowner,oldsupervisor,*oldprocesses))})
    failure.update({k:0 for k in ('completed_model_loads','completed_VAR_renders','completed_prior_scales',
        'completed_decoder_forwards','output_images','partial_logical_frames','scientific_calls_in_this_audit')})
    failurepin=save(p['failure'],failure)
    reg=copy.deepcopy(old);reg['runtime']['root']=str(p['run'])
    for name in recovery.NEW_STAGES:reg['runtime'][name+'_directory']=str(p['run']/name)
    reg['code_bindings'].update({str(p['recovery']/name):digest for name,digest in code.items()})
    reg['visual_registration']=short(save(p['result']/'registered/visual_registration.json',dict(fixture='new visual')))
    reg['recovery']=dict(schema='R1_CACHE_RECOVERY_V1',original_registration=oldpin,
        original_closure=dict(stages=[{k:v for k,v in item.items() if k!='completion'} for item in oldstages]))
    registration=short(save(p['result']/'registered/registration.json',reg))
    newowner=dict(pid=500,start_ticks=50,state='S');newsupervisor=dict(pid=400,start_ticks=40,state='S')
    save(p['run']/'owner_launch.json',dict(process=newowner,registration=registration,deadline_unix=runtime['deadline_unix']))
    save(p['dispatch']/'owner_identity.json',dict(process=newowner,registration=registration))
    save(p['dispatch']/'launch.json',dict(process=newsupervisor,registration=registration,deadline_unix=runtime['deadline_unix'],
        argv=['cpu','-B','-u',str(p['recovery']/'owner.py'),'--registration',registration['path'],'--registration-sha256',registration['sha256']]))
    save(p['initial']/'dispatch_snapshot.json',dict(process=newsupervisor,registration=registration,
        original_deadline_unix=runtime['deadline_unix'],argv=['cpu','-B','-u',str(p['recovery']/'dispatch_once.py'),
        '--expected-registration',registration['sha256']]))
    save(p['dispatch']/'actual_wait.json',dict(actual_wait=True,pid=500,returncode=0,identity_at_launch=newowner,
        registration=registration,scientific_owner_launch_exists=True))
    newstages=[]
    for index,name in enumerate(recovery.NEW_STAGES):
        item,_=stage(reg,registration,name,False,0,index)
        newstages.append({k:v for k,v in item.items() if k!='launch'})
    complete=save(p['result']/'completion.json',dict(status='COMPLETE_R1_CHAIN',registration=registration,
        actual_exit_zero_all=True,automatic_R2=False,R2='NOT_REGISTERED_NOT_RUN',stages=newstages,
        frozen_policy_completion=newstages[-1]['completion']))
    save(p['result']/'recovery_completion.json',dict(status='COMPLETE_R1_CACHE_RECOVERY_CHAIN',registration=registration,
        original_registration=oldpin,new_segment_completion=complete,new_stages=newstages,old_owner_exit=1,
        new_segment_stage_actual_exits_zero=True,new_owner_actual_wait_required=True,failed_initialization_attempts=1,
        additional_scientific_budget=0,R2='NOT_REGISTERED_NOT_RUN',automatic_R2=False,
        inherited_stages=reg['recovery']['original_closure']['stages'][:2]))
    aux=original.pin(p['old_run']/'visual/resource_history_at_failure.json')
    constants=dict(MASTER_SHA=registration['sha256'],ORIGINAL_SHA=oldpin['sha256'],
        VISUAL_SHA=reg['visual_registration']['sha256'],FAILURE_SHA=failurepin['sha256'],CODE=code)
    constants.update(AUXILIARY_FAILURE_SHA=aux['sha256'],AUXILIARY_FAILURE_BYTES=aux['bytes'])
    return root,temporary,registration,p,constants


class ClosureTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root,self.tmp,self.registration,self.p,constants=fixture(Path(self.temp.name))
        self.addCleanup(mock.patch.stopall)
        for key,value in constants.items():mock.patch.object(recovery,key,value).start()
        mock.patch.object(original,'unheld_locks',lambda p:None).start()

    def call(self,probe=lambda pid:None):
        return recovery.preflight(original,self.root,self.tmp,self.registration,probe)

    def update(self,path,change):
        value=original.read(path);change(value);save(path,value)

    def test_complete_mixed_provenance_retains_old_failed_segment_and_makes_no_output(self):
        before={str(x) for x in Path(self.temp.name).rglob('*')}
        _,proof=self.call()
        self.assertEqual(proof['original_owner_actual_exit'],1);self.assertEqual(proof['new_owner_actual_exit'],0)
        self.assertEqual(proof['original_failed_visual']['actual_exit'],1)
        self.assertEqual([r['stage'] for r in proof['inherited_successful_stages']],['phy','projection'])
        self.assertEqual(proof['original_failure']['completed_original_forward_calls'],0)
        self.assertEqual(before,{str(x) for x in Path(self.temp.name).rglob('*')})

    def test_new_owner_failure_or_missing_wait_stops_before_old_evidence_and_quality(self):
        waited=self.p['dispatch']/'actual_wait.json'
        self.update(waited,lambda v:v.update(returncode=1))
        read=original.read
        def spy(path):
            self.assertNotEqual(Path(path),self.p['failure'])
            self.assertFalse(any(k in str(path) for k in ('scored_rows','selection.json','source_means')))
            return read(path)
        with mock.patch.object(original,'read',spy),self.assertRaises(RuntimeError):self.call()
        waited.unlink()
        with mock.patch.object(original,'read',spy),self.assertRaises(FileNotFoundError):self.call()

    def test_same_identity_different_state_is_valid_but_wrong_generation_is_not(self):
        self.update(self.p['dispatch']/'owner_identity.json',lambda v:v['process'].update(state='R'))
        self.call()
        self.update(self.p['dispatch']/'owner_identity.json',lambda v:v['process'].update(start_ticks=51))
        with self.assertRaisesRegex(RuntimeError,'process chain'):self.call()

    def test_live_or_zombie_process_refused_but_pid_reuse_allowed(self):
        for state in ('R','Z'):
            with self.subTest(state=state),self.assertRaises(RuntimeError):
                self.call(lambda pid:dict(pid=pid,start_ticks={500:50,400:40}.get(pid,1),state=state))
        self.call(lambda pid:dict(pid=pid,start_ticks=99999,state='R'))

    def test_new_failure_marker_refused_before_quality(self):
        save(self.p['run']/'visual/failure.json',dict(error='new failure'))
        with self.assertRaisesRegex(RuntimeError,'Unreviewed'):self.call()

    def test_unreviewed_original_failure_is_not_ignored(self):
        save(self.p['old_run']/'phy/failure_extra.json',{})
        with self.assertRaisesRegex(RuntimeError,'Unreviewed'):self.call()

    def test_original_ledger_and_byte_evidence_cannot_change(self):
        path=self.p['old_run']/'visual/calls.sqlite'
        db=sqlite3.connect(path);db.execute("UPDATE calls SET state='COMPLETE'");db.commit();db.close()
        with self.assertRaises(RuntimeError):self.call()

    def test_auxiliary_history_not_in_old_audit_still_requires_exact_independent_pin(self):
        self.call()
        path=self.p['old_run']/'visual/resource_history_at_failure.json'
        self.assertNotIn(str(path),original.read(self.p['failure'])['evidence'])
        save(path,[dict(changed=True)])
        with self.assertRaises(RuntimeError):self.call()

    def test_original_owner_failure_cannot_be_relabelled_success(self):
        self.update(self.p['old_dispatch']/'actual_wait.json',lambda v:v.update(returncode=0))
        with self.assertRaisesRegex(RuntimeError,'Wrong actual exit'):self.call()

    def test_known_failure_scope_does_not_allow_hidden_forward_work(self):
        path=self.p['old_run']/'visual/images'/'unexpected.png';path.write_bytes(b'fake')
        with self.assertRaisesRegex(RuntimeError,'not empty'):self.call()

    def test_wrong_argv_or_deadline_refused(self):
        path=self.p['run']/'score_launch.json'
        self.update(path,lambda v:v['argv'].append('--unregistered'))
        with self.assertRaisesRegex(RuntimeError,'argv'):self.call()

    def test_runtime_deadline_extension_refused(self):
        self.update(self.p['dispatch']/'launch.json',lambda v:v.update(deadline_unix=999999))
        with self.assertRaisesRegex(RuntimeError,'deadline'):self.call()

    def test_child_registration_and_status_checked_not_just_completion_hash(self):
        path=self.p['run']/'visual/completion.json';self.update(path,lambda v:v.update(registration=dict(path='other',sha256='wrong')))
        # Semantic check happens before comparing the owner's saved completion pin.
        with self.assertRaisesRegex(RuntimeError,'child registration'):self.call()

    def test_unknown_recovery_code_change_refused(self):
        (self.p['recovery']/'adapter.py').write_text('# changed',encoding='utf8')
        with self.assertRaisesRegex(RuntimeError,'source changed'):self.call()

    def test_locks_required_without_writing_them(self):
        with mock.patch.object(original,'unheld_locks',side_effect=RuntimeError('held')) as lock:
            with self.assertRaisesRegex(RuntimeError,'held'):self.call()
        self.assertEqual(len(lock.call_args.args[0]),4)


class PureAdapterTests(unittest.TestCase):
    def test_strict_exit_booleans_and_interruptions_refused(self):
        identity=dict(pid=1,start_ticks=2);base=dict(actual_wait=True,pid=1,returncode=0)
        for changed in ({'pid':True},{'returncode':False},{'start_ticks':True},
                        *({flag:True} for flag in ('terminated','killed','timed_out','timeout','stopped'))):
            with self.subTest(changed=changed),self.assertRaises(RuntimeError):
                recovery.actual_exit(original,dict(base,**changed),identity,0)
        with self.assertRaises(RuntimeError):recovery.actual_exit(original,base,dict(pid=True,start_ticks=2),0)

    def test_only_metadata_routing_and_output_identity_change_original_functions(self):
        module=recovery.load_original(WORKSPACE)
        before={name:value for name,value in vars(module).items() if callable(value)}
        proof=recovery.adapt_original(module,WORKSPACE)
        self.assertTrue(proof['deep_audit_inverse_AST']);self.assertEqual(len(proof['deep_audit_metadata_substitutions']),3)
        for name,value in before.items():
            if name not in ('deep_audit','publish','report_text'):
                self.assertIs(vars(module)[name],value,name)
        self.assertIn(recovery.NAME,module.publish.__code__.co_consts)
        self.assertNotIn('r1_delivery_v1',module.publish.__code__.co_consts)

    def test_no_science_or_selection_execution_in_adapter(self):
        tree=ast.parse(Path(recovery.__file__).read_text(encoding='utf8'))
        for node in ast.walk(tree):
            if isinstance(node,ast.Call):
                name=node.func.attr if isinstance(node.func,ast.Attribute) else node.func.id if isinstance(node.func,ast.Name) else ''
                self.assertNotIn(name,('select_from_complete_rows','select','score','visual','bootstrap','run','Popen'))

    def test_report_keeps_saved_negative_gate_and_discloses_initialization_failure(self):
        module=recovery.load_original(WORKSPACE);recovery.adapt_original(module,WORKSPACE)
        comparisons={key:dict(mean=-.012,per_hat={'7':-.02,'10':-.004}) for key in ('S_robust','S_back3')}
        data=dict(selected=dict(gate=dict(status='NO_GO_STOP',comparisons=comparisons),
            separate_engineering_targets=dict(per_hat=[dict(snr_hat=7,
                matched_U_minus_S_nom_psnr_db=-.25,same_U_overestimate_psnr_loss_db=1.5)])))
        report=module.report_text(data)
        self.assertIn('NO_GO_STOP',report);self.assertIn('-0.012000',report);self.assertIn('-0.250000',report)
        self.assertIn('one failed model-load reservation',report);self.assertIn('zero completed model loads',report)
        self.assertIn('without confidence intervals',report);self.assertIn('R2 is not registered',report)
        self.assertIn(r'NO\_GO\_STOP',module.tex_fragment(report))

    def test_source_binding_failure_stops_before_original_import(self):
        with tempfile.TemporaryDirectory() as td,self.assertRaisesRegex(RuntimeError,'Frozen original'):
            recovery.load_original(Path(td))


if __name__=='__main__':unittest.main()

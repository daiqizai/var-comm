"""Synthetic owner fixtures; no real jobs, neural models, or PHY are invoked."""
import copy
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch
import stage_owner as o


class Process:
    pid=12345
    def __init__(self,codes=None,code=0):self.codes=list(codes or [0]);self.code=code;self.waited=False
    def poll(self):
        if self.waited:return self.code
        return self.codes.pop(0) if len(self.codes)>1 else self.codes[0]
    def wait(self):self.waited=True;return self.code
    def kill(self):raise AssertionError('No forced kill permitted')
    terminate=kill


class Tests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve();self.out=self.root/'H';self.out.mkdir()
        self.driver=self.root/'implemented_stage.py';self.driver.write_text('# fixture only\n')
        self.regpath=self.out/'execution_registration.json';self.config_path=self.out/'owner_config.json'
        self.budget=self.out/'resource_budget.json'
        self.limits=dict(qualification=2000,coarse=49152,refine=12288,initial_true200=20000,
                         whole_calibration=48000,partial_calibration=36000,development=13200,engineering_reserve=19360)
        o.save(self.budget,dict(schema='H_PACKET_BUDGET_V1',status='FROZEN',branch='H',total_cap=200000,phase_limits=self.limits))
        source=self.root/'S1_result.json';o.save(source,{'complete':True})
        gate=self.root/'S1_completion.json'
        o.save(gate,dict(status='S1_BRIDGE_COMPLETE_AWAIT_NEXT_DECISION',outputs={str(source):o.sha(source)}))
        self.cfg=dict(branch='H',root=str(self.root),out=str(self.out),owner_out=str(self.out/'owner'/'r1'),
            registration=str(self.regpath),budget_registration=str(self.budget),budget_registration_sha256=o.sha(self.budget),
            budget_path=str(self.out/'budget.sqlite'),phase_limits=self.limits,
            S1_gate=dict(path=str(gate),sha256=o.sha(gate),status='S1_BRIDGE_COMPLETE_AWAIT_NEXT_DECISION'),
            qualification_started_unix=o.QUALIFICATION_START,qualification_deadline_unix=o.QUALIFICATION_DEADLINE,
            overall_deadline_unix=o.QUALIFICATION_DEADLINE+86400,cpu_affinities=[[24,25],[26,27]],
            gpu_affinity=list(range(24,30)),gpu_threads=6,gpu_device=0,nvidia_smi=str(self.driver),
            visual_lock_path=str(self.root/'visual.lock'),stages=[self.stage('qualification')])
        self.register()
    def stage(self,name,resource='cpu',jobs=1):
        return dict(id=name,resource=resource,max_seconds=3600,requires=[],jobs=[dict(id=str(i),
            argv=[str(self.driver),'-B',str(self.driver),'--stage',name],cwd=str(self.root),
            out=str(self.out/name/str(i)),completion=str(self.out/name/str(i)/'completion.json'),
            accepted_statuses=['PASS'],receipt_expect={'real_stage':name}) for i in range(jobs)])
    def register(self):
        o.save(self.config_path,self.cfg)
        self.reg=dict(status='H_EXECUTION_REVISION_REGISTERED',branch='H',owner_config_sha256=o.sha(self.config_path),
            allowed_stage_ids=[s['id'] for s in self.cfg['stages']],phase_limits=self.limits,
            source_bindings={str(Path(o.__file__).resolve()):o.sha(o.__file__),str(self.driver):o.sha(self.driver)},
            input_bindings={str(self.budget):o.sha(self.budget),self.cfg['S1_gate']['path']:self.cfg['S1_gate']['sha256']})
        o.save(self.regpath,self.reg)
    def owner(self):
        owner=o.Owner(self.config_path);owner.owner_out.mkdir(parents=True,exist_ok=True)
        owner.available_affinity=set(range(24,30));return owner
    def make_child(self,owner,job,code=0):
        execution=owner.owner_out/'stages'/owner.current['id']/'workers'/job['id'];execution.mkdir(parents=True,exist_ok=True)
        directory=Path(job['out']);directory.mkdir(parents=True,exist_ok=True)
        proc=Process(code=code,codes=[code])
        row=dict(job=job,process=proc,execution=execution,log=(execution/'worker.log').open('xb'),started=time.time(),identity={'pid':proc.pid})
        owner.active.append(row);return row
    def successful_launch(self,owner,job,slot):
        self.make_child(owner,job)
        output=Path(job['out'])/'actual.json';o.save(output,{'fixture':True})
        o.save(job['completion'],dict(status='PASS',registration_sha256=owner.regsha,
                real_stage=owner.current['id'],outputs={str(output):o.sha(output)}))
    def test_stage_revision_must_exactly_match_implemented_order(self):
        self.owner()
        self.reg['allowed_stage_ids']+=['source'];o.save(self.regpath,self.reg)
        with self.assertRaisesRegex(RuntimeError,'implemented stages'):self.owner()
        self.cfg['stages']=[self.stage('holdout')];self.register()
        with self.assertRaisesRegex(RuntimeError,'Unknown'):self.owner()
    def test_max_two_CPU_one_GPU_and_safe_job_ids(self):
        for resource,jobs in [('cpu',3),('gpu',2)]:
            self.cfg['stages']=[self.stage('qualification',resource,jobs)];self.register()
            with self.assertRaisesRegex(RuntimeError,'worker limit'):self.owner()
        self.cfg['stages']=[self.stage('qualification')];self.cfg['stages'][0]['jobs'][0]['id']='../../escape';self.register()
        with self.assertRaisesRegex(RuntimeError,'unsafe'):self.owner()
    def test_S1_complete_and_source_hash_required(self):
        path=Path(self.cfg['S1_gate']['path']);o.save(path,{'status':'RUNNING'})
        with self.assertRaisesRegex(RuntimeError,'SHA changed'):self.owner()
    def test_no_shell_entry_and_venv_spelling_preserved(self):
        with self.assertRaisesRegex(RuntimeError,'shortcuts'):o.command_entry([str(self.driver),'-c','print(1)'])
        with patch.object(o.Path,'resolve',side_effect=AssertionError('do not resolve interpreter')):
            self.assertEqual(o.absolute(self.driver,False),self.driver)
    def test_successful_partial_batch_does_not_claim_full_pipeline(self):
        owner=self.owner()
        with patch.object(owner,'launch',side_effect=lambda j,s:self.successful_launch(owner,j,s)),patch.object(owner,'wall',return_value=o.QUALIFICATION_START+1):owner.run()
        done=o.read(owner.owner_out/'completion.json')
        self.assertEqual(done['status'],'REGISTERED_H_STAGE_BATCH_COMPLETE')
        self.assertTrue(done['qualification_passed'])
        self.assertFalse(done['H_full_delivery_claimed'] or done['future_stages_started'] or done['C_started'])
        self.assertEqual(len(done['completed']),1)
    def test_qualification_deadline_stops_before_any_job_then_prioritizes_main(self):
        owner=self.owner()
        with patch.object(owner,'wall',return_value=o.QUALIFICATION_DEADLINE+1),patch.object(owner,'launch') as launch:
            with self.assertRaisesRegex(RuntimeError,'qualification deadline'):owner.run()
        launch.assert_not_called()
        handoff=o.read(owner.owner_out/'handoff_request.json')
        self.assertEqual(handoff['target'],'MAIN_SYSTEM_FULL_CALIBRATION')
        self.assertFalse(handoff['placeholder_command_launched'] or handoff['C_started'])
    def test_failed_qualification_does_not_run_source(self):
        self.cfg['stages'].append(self.stage('source'));self.cfg['stages'][1]['requires']=['qualification'];self.register()
        owner=self.owner();launched=[]
        def fail(job,slot):launched.append(owner.current['id']);self.make_child(owner,job,1)
        with patch.object(owner,'launch',side_effect=fail),patch.object(owner,'wall',return_value=o.QUALIFICATION_START+1):
            with self.assertRaisesRegex(RuntimeError,'Worker failed'):owner.run()
        self.assertEqual(launched,['qualification'])
        self.assertTrue((owner.owner_out/'failure.json').exists())
        self.assertTrue((owner.owner_out/'handoff_request.json').exists())
    def test_source_failure_after_qualification_does_not_reclassify_as_qualification(self):
        self.cfg['stages'].append(self.stage('source'));self.cfg['stages'][1]['requires']=['qualification'];self.register()
        owner=self.owner()
        def launch(job,slot):
            if owner.current['id']=='qualification':self.successful_launch(owner,job,slot)
            else:self.make_child(owner,job,1)
        with patch.object(owner,'launch',side_effect=launch),patch.object(owner,'wall',return_value=o.QUALIFICATION_START+1):
            with self.assertRaises(RuntimeError):owner.run()
        self.assertFalse((owner.owner_out/'handoff_request.json').exists())
    def test_launch_owns_child_before_identity_failure_and_drains_without_kill(self):
        owner=self.owner();owner.current=self.cfg['stages'][0];owner.stage_began=time.monotonic();proc=Process([None])
        with patch.object(o.subprocess,'Popen',return_value=proc),patch.object(o.os,'sched_setaffinity',create=True), \
             patch.object(o,'identity',side_effect=RuntimeError('proc race')),patch.object(owner,'wall',return_value=o.QUALIFICATION_START+1):
            with self.assertRaisesRegex(RuntimeError,'proc race'):owner.launch(owner.current['jobs'][0],0)
        self.assertIs(owner.active[0]['process'],proc)
        owner.request_stop('failure');proc.codes=[0];owner.drain()
        self.assertTrue(proc.waited)
        self.assertFalse(o.read(owner.owner_out/'drain_receipt.json')['force_kill'])
    def test_exit_race_allowed_but_reused_live_identity_rejected(self):
        expected=dict(pid=12345,start_ticks=1,uid=1002,argv=['python'])
        def gone(pid):raise ProcessLookupError()
        self.assertFalse(o.check_live(Process([None,0]),expected,gone))
        with self.assertRaisesRegex(RuntimeError,'identity changed'):
            o.check_live(Process([None]),expected,lambda pid:dict(expected,start_ticks=2))
    def test_scientific_receipt_cannot_bind_open_log(self):
        directory=self.out/'qualification'/'0';directory.mkdir(parents=True)
        log=directory/'worker.log';log.write_text('live')
        path=directory/'completion.json';o.save(path,dict(status='PASS',registration_sha256='science',outputs={str(log):o.sha(log)}))
        with self.assertRaisesRegex(RuntimeError,'mutable'):o.receipt(path,['PASS'],'science',directory=directory)
    def test_source_requires_qualification_receipt_or_stage(self):
        self.cfg['stages']=[self.stage('source')];self.register()
        with self.assertRaisesRegex(RuntimeError,'actual qualification'):self.owner()
    def test_NVML_stale_reaped_zombie_must_disappear_before_admission(self):
        expected=dict(pid=12345,start_ticks=12,uid=1002,argv=['python','gpu.py'])
        history=[dict(identity=expected,wait_completed=True)]
        answers=[[12345],[12345],[]];now=[0.0]
        def query(timeout):return answers.pop(0)
        def sleep(seconds):now[0]+=seconds
        result=o.wait_gpu_idle(query,history,lambda p:dict(expected,state='Z',argv=['']),clock=lambda:now[0],sleep=sleep)
        self.assertEqual(result['stale_polls'],2)
        self.assertEqual(result['previously_reaped_stale_pids'],[12345])
        self.assertEqual(now[0],2)
    def test_NVML_foreign_absent_and_reused_live_PID_are_never_waived(self):
        expected=dict(pid=12345,start_ticks=12,uid=1002,argv=['python','gpu.py'])
        with self.assertRaisesRegex(RuntimeError,'Foreign/unproven'):
            o.wait_gpu_idle(lambda timeout:[999],[],lambda p:None)
        history=[dict(identity=expected,wait_completed=True)]
        with self.assertRaisesRegex(RuntimeError,'Live or reused'):
            o.wait_gpu_idle(lambda timeout:[12345],history,lambda p:dict(expected,state='R'))
        with self.assertRaisesRegex(RuntimeError,'Live or reused'):
            o.wait_gpu_idle(lambda timeout:[12345],history,lambda p:dict(expected,state='Z',start_ticks=13))
    def test_NVML_owned_absent_is_bounded30seconds_and_never_treated_as_idle(self):
        expected=dict(pid=12345,start_ticks=12,uid=1002,argv=['python','gpu.py'])
        history=[dict(identity=expected,wait_completed=True)];now=[0.0];calls=[]
        def query(timeout):calls.append(timeout);return [12345]
        def sleep(seconds):now[0]+=seconds
        with self.assertRaisesRegex(RuntimeError,'within30seconds'):
            o.wait_gpu_idle(query,history,lambda p:None,clock=lambda:now[0],sleep=sleep)
        self.assertEqual(now[0],30);self.assertEqual(len(calls),30)


if __name__=='__main__':unittest.main()

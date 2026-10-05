"""Synthetic sequencing and receipt tests; no real process or packet is run."""
import copy
from pathlib import Path
import unittest
from unittest.mock import patch
import stage_owner as o
import wait_then_coarse as w
import test_stage_owner as fixture


class Tests(unittest.TestCase):
    def setUp(self):
        f = fixture.Tests(); f.setUp(); self.addCleanup(f.temp.cleanup); self.f = f
        self.root = f.root; self.out = f.out
        f.cfg['stages'] = [f.stage('qualification'), f.stage('source', 'gpu'), f.stage('clean_quality', 'gpu')]
        f.cfg['stages'][1]['requires'] = ['qualification']; f.cfg['stages'][2]['requires'] = ['source']
        f.register(); self.original = copy.deepcopy(f.cfg); self.original_reg = copy.deepcopy(f.reg)
        self.oldid = dict(pid=111, start_ticks=11, uid=1002,
                          argv=[str(f.driver), '-B', str(Path(o.__file__).resolve()), '--config', str(f.config_path)])
        self.launch = self.out / 'initial_launch.json'
        o.save(self.launch, dict(identity=self.oldid, argv=self.oldid['argv'], registration_sha256=o.sha(f.regpath),
                               owner_config_sha256=o.sha(f.config_path)))
        self.childids = []; self.stagefiles = []
        self.complete_original()
        driver = self.root / 'h_coarse_driver.py'; driver.write_text('# synthetic only\n')
        n = copy.deepcopy(self.original); n['owner_out'] = str(self.out / 'coarse_owner')
        n['registration'] = str(self.out / 'coarse_registration.json')
        n['stages'] = [f.stage('coarse', jobs=2), f.stage('report')]; n['stages'][1]['requires'] = ['coarse']
        for stage in n['stages']:
            for i, job in enumerate(stage['jobs']):
                phase = 'worker' if stage['id'] == 'coarse' else 'merge'
                job['argv'] = [str(f.driver), '-B', str(driver), '--config', 'synthetic', '--stage', phase]
                if phase == 'worker': job['argv'] += ['--worker-index', str(i)]
        self.next_config = self.out / 'coarse_owner_config.json'; o.save(self.next_config, n); self.next = n
        self.waitcfg = self.out / 'waiter_config.json'
        self.cfg = dict(schema='H_COARSE_WAIT_V1', root=str(self.root), out=str(self.out),
            waiter_out=str(self.out / 'coarse_wait'), registration=n['registration'],
            next_owner_config=str(self.next_config), python=str(f.driver), stage_owner=str(Path(o.__file__).resolve()),
            initial=dict(registration=str(f.regpath), config=str(f.config_path), launch=str(self.launch),
                         owner_out=self.original['owner_out']), deadline_unix=w.DEADLINE, poll_seconds=10)
        o.save(self.waitcfg, self.cfg)
        self.reg = copy.deepcopy(self.original_reg)
        self.reg.update(owner_config_sha256=o.sha(self.next_config), allowed_stage_ids=['coarse', 'report'])
        self.reg['source_bindings'].update({str(driver):o.sha(driver),str(Path(w.__file__).resolve()):o.sha(w.__file__)})
        for p in (self.waitcfg, self.next_config, f.regpath, f.config_path, self.launch):
            self.reg['input_bindings'][str(p)] = o.sha(p)
        o.save(n['registration'], self.reg)

    def complete_original(self):
        rs = o.sha(self.f.regpath); base = Path(self.original['owner_out']); completed = []
        for index, stage in enumerate(self.original['stages']):
            directory = base / 'stages' / stage['id']; job = stage['jobs'][0]
            evidence = directory / 'workers' / job['id']; evidence.mkdir(parents=True)
            output = Path(job['out']) / 'data.json'; o.save(output, {'fixture':True})
            cp = Path(job['completion']); o.save(cp, dict(status='PASS', registration_sha256=rs,
                real_stage=stage['id'], outputs={str(output):o.sha(output)}, development_used=False, holdout_used=False))
            log = evidence / 'worker.log'; log.write_text('CLOSED\n')
            ident = dict(pid=200+index,start_ticks=20+index,uid=1002,argv=job['argv']); self.childids.append(ident)
            o.save(evidence / 'launch.json', dict(identity=ident,registration_sha256=rs))
            event = dict(job_id=job['id'],identity=ident,exit_code=0,registration_sha256=rs,
                         closed_log_sha256=o.sha(log),completion=str(cp),completion_sha256=o.sha(cp))
            o.save(evidence / 'exit_receipt.json', event)
            sp = directory / 'completion.json'; o.save(sp, dict(status='REGISTERED_STAGE_COMPLETE',stage=stage['id'],
                registration_sha256=rs,budget={'unresolved':0},jobs=[event]))
            completed.append(dict(stage=stage['id'],completion=str(sp),sha256=o.sha(sp))); self.stagefiles.append(sp)
        self.done = base / 'completion.json'
        o.save(self.done, dict(status='REGISTERED_H_STAGE_BATCH_COMPLETE',registration_sha256=rs,
            allowed_stage_ids=['qualification','source','clean_quality'],qualification_passed=True,
            budget={'unresolved':0},completed=completed,future_stages_started=False,H_full_delivery_claimed=False,
            C_started=False,holdout_started=False))

    def waiter(self):
        x = w.Waiter(self.waitcfg); x.waiter_out.mkdir(); return x

    def test_valid_config_and_full_receipt_chain(self):
        x = self.waiter()
        gate = w.verify_batch(o, self.f.config_path, self.f.regpath, 1002, lambda _:None)
        self.assertEqual(len(gate['child_identities']),3)
        self.assertIn(str(self.done),gate['bindings'])
        self.assertEqual(x.next['stages'][1]['jobs'][0]['argv'][-1],'merge')

    def test_receipt_or_output_tampering_blocks_gate(self):
        cp = Path(self.original['stages'][0]['jobs'][0]['out']) / 'data.json'; cp.write_text('changed')
        with self.assertRaisesRegex(RuntimeError,'SHA changed'):
            w.verify_batch(o, self.f.config_path, self.f.regpath, 1002, lambda _:None)

    def test_Zombie_and_live_identity_wait_reused_PID_proves_old_exit(self):
        for state in ('R','S','Z'):
            row = dict(self.oldid,state=state)
            self.assertFalse(w.exited(self.oldid,lambda _,r=row:r))
        self.assertTrue(w.exited(self.oldid,lambda _:None))
        self.assertTrue(w.exited(self.oldid,lambda _:dict(self.oldid,state='R',start_ticks=999)))
        with self.assertRaises(PermissionError):w.exited(self.oldid,lambda _:(_ for _ in ()).throw(PermissionError()))

    def test_gate_waits_even_when_completion_exists_until_owner_absent(self):
        x = self.waiter(); states = [dict(self.oldid,state='Z'),None]
        def reader(pid):
            return states.pop(0) if pid == self.oldid['pid'] and states else None
        with patch.object(x,'guard'),patch.object(x.a,'raw_process_state',side_effect=reader),patch.object(w.time,'sleep') as sleep:
            x.wait_gate()
        sleep.assert_called_once_with(10)
        self.assertTrue((x.waiter_out/'release_gate.json').exists())

    def test_exit_without_completion_and_initial_failure_never_launch(self):
        x = self.waiter(); self.done.unlink()
        with patch.object(x,'guard'),patch.object(x.a,'raw_process_state',return_value=None):
            with self.assertRaisesRegex(RuntimeError,'without a complete'):x.wait_gate()
        o.save(Path(self.original['owner_out'])/'failure.json',{'failed':True})
        with patch.object(x,'wall',return_value=w.DEADLINE-1):
            with self.assertRaisesRegex(RuntimeError,'Initial owner failed'):x.guard()
        self.assertFalse((x.waiter_out/'release_gate.json').exists())

    def test_live_child_invalidates_complete_batch(self):
        ident = self.childids[-1]
        def reader(pid):return dict(ident,state='S') if pid==ident['pid'] else None
        with self.assertRaisesRegex(RuntimeError,'still present'):
            w.verify_batch(o,self.f.config_path,self.f.regpath,1002,reader)

    def test_CPU_only_scope_and_shared_budget_cannot_change(self):
        self.next['stages'][0]['resource']='gpu';o.save(self.next_config,self.next)
        self.reg['owner_config_sha256']=o.sha(self.next_config)
        self.reg['input_bindings'][str(self.next_config)]=o.sha(self.next_config)
        o.save(self.cfg['registration'],self.reg)
        with self.assertRaisesRegex(RuntimeError,'worker limit|CPU table'):w.Waiter(self.waitcfg)

    def test_deadline_STOP_and_failure_do_not_touch_initial_owner(self):
        x = self.waiter()
        with patch.object(x,'wall',return_value=w.DEADLINE):
            with self.assertRaisesRegex(RuntimeError,'96-hour'):x.guard()
        with patch.object(w.subprocess,'Popen') as launch,patch.object(x,'wait_gate',side_effect=RuntimeError('blocked')):
            with self.assertRaisesRegex(RuntimeError,'blocked'):x.run()
        launch.assert_not_called()
        self.assertTrue((x.waiter_out/'failure.json').exists())
        self.assertFalse((self.out/'STOP').exists())

    def test_Popen_identity_failure_keeps_child_owned_and_safe_drains(self):
        x = self.waiter(); child = fixture.Process(codes=[None], code=1)
        with patch.object(x,'wait_gate'),patch.object(x,'guard'), \
             patch.object(w.subprocess,'Popen',return_value=child), \
             patch.object(x.a,'identity',side_effect=RuntimeError('capture failed')):
            with self.assertRaisesRegex(RuntimeError,'capture failed'):x.run()
        self.assertTrue(child.waited)
        self.assertTrue((self.out/'STOP').exists())
        failure = o.read(x.waiter_out/'failure.json'); exit_row = o.read(x.waiter_out/'owner_exit.json')
        self.assertEqual(failure['child_identity']['pid'],child.pid)
        self.assertEqual(exit_row['exit_code'],1)
        self.assertEqual(exit_row['closed_log_sha256'],o.sha(x.waiter_out/'owner.log'))


if __name__ == '__main__':unittest.main()

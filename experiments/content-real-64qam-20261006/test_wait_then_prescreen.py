"""Synthetic closure/race/scope tests. No real process, PHY or model is run."""
import copy
import os
from pathlib import Path
import unittest
from unittest.mock import patch
import stage_owner as o
import wait_then_coarse as w
import wait_then_prescreen as p
import test_wait_then_coarse as prior_fixture
import test_stage_owner as owner_fixture


def complete_batch(config, registration, first_pid):
    cfg=o.read(config);rs=o.sha(registration);base=Path(cfg['owner_out']);completed=[];ids=[]
    for si,stage in enumerate(cfg['stages']):
        directory=base/'stages'/stage['id'];exits=[]
        for ji,job in enumerate(stage['jobs']):
            evidence=directory/'workers'/job['id'];evidence.mkdir(parents=True)
            output=Path(job['out'])/'data.json';o.save(output,{'synthetic':True})
            cp=Path(job['completion']);o.save(cp,dict(status=job['accepted_statuses'][0],registration_sha256=rs,real_stage=stage['id'],
                outputs={str(output):o.sha(output)},new_packet_decodes=0,new_visual_inference=0,
                ready_for_real_calibration=job['accepted_statuses'][0]=='H_PRESCREEN_COMPLETE_FINAL'))
            log=evidence/'worker.log';log.write_text('CLOSED\n')
            ident=dict(pid=first_pid+si*10+ji,start_ticks=first_pid+si*10+ji,uid=1002,argv=job['argv']);ids.append(ident)
            o.save(evidence/'launch.json',dict(identity=ident,registration_sha256=rs))
            ev=dict(job_id=job['id'],identity=ident,exit_code=0,registration_sha256=rs,closed_log_sha256=o.sha(log),
                    completion=str(cp),completion_sha256=o.sha(cp));o.save(evidence/'exit_receipt.json',ev);exits.append(ev)
        path=directory/'completion.json';o.save(path,dict(status='REGISTERED_STAGE_COMPLETE',stage=stage['id'],
            registration_sha256=rs,budget={'unresolved':0},jobs=exits))
        completed.append(dict(stage=stage['id'],completion=str(path),sha256=o.sha(path)))
    o.save(base/'completion.json',dict(status='REGISTERED_H_STAGE_BATCH_COMPLETE',registration_sha256=rs,
        allowed_stage_ids=[s['id'] for s in cfg['stages']],qualification_passed=True,budget={'unresolved':0},completed=completed,
        future_stages_started=False,H_full_delivery_claimed=False,C_started=False,holdout_started=False))
    return ids


class Tests(unittest.TestCase):
    def setUp(self):
        f=prior_fixture.Tests();f.setUp();self.addCleanup(f.doCleanups);self.f=f
        self.prior=f.waiter();old=self.prior;self.a=o
        self.ids=complete_batch(f.next_config,f.cfg['registration'],500)
        self.waitid=dict(pid=444,start_ticks=44,uid=1002,argv=[str(f.f.driver),'-B',str(Path(w.__file__).resolve()),'--config',str(f.waitcfg)])
        self.launch=old.waiter_out/'launch.json';self.identity=old.waiter_out/'identity.json'
        o.save(self.launch,dict(identity=self.waitid,argv=self.waitid['argv'],registration_sha256=old.regsha,config_sha256=o.sha(f.waitcfg)))
        o.save(self.identity,dict(**self.waitid,registration_sha256=old.regsha,config_sha256=o.sha(f.waitcfg)))
        self.ownerid=dict(pid=450,start_ticks=45,uid=1002,argv=[str(f.f.driver),'-B',str(Path(o.__file__).resolve()),'--config',str(f.next_config)])
        o.save(old.waiter_out/'owner_launch.json',dict(identity=self.ownerid,argv=self.ownerid['argv'],registration_sha256=old.regsha))
        (old.waiter_out/'owner.log').write_text('CLOSED\n')
        o.save(old.waiter_out/'owner_exit.json',dict(identity=self.ownerid,exit_code=0,registration_sha256=old.regsha,
             closed_log_sha256=o.sha(old.waiter_out/'owner.log')))
        original=w.verify_batch(o,f.f.config_path,f.f.regpath,1002,lambda _:None)
        o.save(old.waiter_out/'release_gate.json',dict(**original,registration_sha256=old.regsha,initial_identity=old.oldid))
        closed=w.verify_batch(o,f.next_config,f.cfg['registration'],1002,lambda _:None)
        ocp=Path(old.next['owner_out'])/'completion.json'
        o.save(old.waiter_out/'completion.json',dict(status='H_COARSE_WAIT_AND_CPU_BATCH_COMPLETE',registration_sha256=old.regsha,
            owner_completion=str(ocp),owner_completion_sha256=o.sha(ocp),output_bindings=closed['bindings'],
            real_decodes_in_waiter=0,GPU_used=False))
        entry=f.root/'h_prescreen.py';entry.write_text('# synthetic only\n')
        n=copy.deepcopy(f.next);n['registration']=str(f.out/'prescreen_registration.json');n['owner_out']=str(f.out/'prescreen_owner')
        stage=f.f.stage('freeze');job=stage['jobs'][0];self.science=f.out/'prescreen_config.json'
        job['argv']=[str(f.f.driver),'-B',str(entry),'--config',str(self.science)]
        job['accepted_statuses']=['H_PRESCREEN_COMPLETE_REFINEMENT_REQUIRED','H_PRESCREEN_COMPLETE_FINAL']
        n['stages']=[stage];self.n=n;self.nextconfig=f.out/'prescreen_owner_config.json';o.save(self.nextconfig,n)
        o.save(self.science,dict(registration=n['registration'],phase='coarse',out=job['out']))
        self.cfg=dict(schema='H_PRESCREEN_WAIT_V1',root=str(f.root),out=str(f.out),waiter_out=str(f.out/'prescreen_wait'),
            registration=n['registration'],next_owner_config=str(self.nextconfig),python=str(f.f.driver),
            stage_owner=str(Path(o.__file__).resolve()),wait_then_coarse=str(Path(w.__file__).resolve()),
            coarse_wait=dict(config=str(f.waitcfg),launch=str(self.launch),identity=str(self.identity)),
            deadline_unix=p.DEADLINE,poll_seconds=10)
        self.config=f.out/'prescreen_wait_config.json';o.save(self.config,self.cfg)
        self.reg=copy.deepcopy(f.reg);self.reg.update(owner_config_sha256=o.sha(self.nextconfig),allowed_stage_ids=['freeze'])
        self.reg['source_bindings'].update({str(entry):o.sha(entry),str(Path(p.__file__).resolve()):o.sha(p.__file__)})
        for path in (self.config,self.nextconfig,self.science,f.waitcfg,self.launch,self.identity,f.next_config,f.cfg['registration']):
            self.reg['input_bindings'][str(path)]=o.sha(path)
        o.save(n['registration'],self.reg)

    def waiter(self):
        v=p.Waiter(self.config);v.waiter_out.mkdir();return v

    def verify(self,reader=lambda _:None):
        return p.verify_coarse_closed(o,w,self.prior,o.read(self.launch),self.waitid,reader)

    def test_full_chain_and_exact_cpu_only_config(self):
        v=self.waiter();got=self.verify()
        self.assertEqual(len(got['child_identities']),6)
        self.assertEqual(v.science['phase'],'coarse')
        self.assertEqual(v.reg['allowed_stage_ids'],['freeze'])

    def test_waiter_owner_or_science_child_live_or_zombie_block(self):
        for ident in (self.waitid,self.ownerid,self.ids[-1],self.prior.oldid):
            for state in ('S','Z'):
                with self.subTest(pid=ident['pid'],state=state):
                    with self.assertRaises(RuntimeError):
                        self.verify(lambda pid,i=ident,s=state:dict(i,state=s) if pid==i['pid'] else None)

    def test_closed_log_and_output_or_receipt_tampering_rejected(self):
        path=self.prior.waiter_out/'owner.log';original=path.read_bytes();path.write_text('changed')
        with self.assertRaisesRegex(RuntimeError,'log differs'):self.verify()
        path.write_bytes(original)
        cp=self.prior.waiter_out/'completion.json';d=o.read(cp);d['output_bindings'].pop(next(iter(d['output_bindings'])));o.save(cp,d)
        with self.assertRaisesRegex(RuntimeError,'output closure'):self.verify()

    def test_completed_waiter_still_waits_for_exit_and_requires_quiescent(self):
        v=self.waiter();states=[dict(self.waitid,state='Z'),None]
        def reader(pid):return states.pop(0) if pid==self.waitid['pid'] and states else None
        with patch.object(v,'guard'),patch.object(v.a,'raw_process_state',side_effect=reader),patch.object(p.time,'sleep') as sleep:
            v.wait_gate()
        sleep.assert_called_once_with(10);self.assertTrue((v.waiter_out/'release_gate.json').exists())

    def test_orphan_budget_reservation_prevents_release(self):
        v=self.waiter()
        with patch.object(v,'guard'),patch.object(v.a,'raw_process_state',return_value=None), \
             patch.object(v,'snapshot',side_effect=RuntimeError('RESERVED unresolved')):
            with self.assertRaisesRegex(RuntimeError,'RESERVED'):v.wait_gate()
        self.assertFalse((v.waiter_out/'release_gate.json').exists())

    def test_failure_or_exit_without_completion_never_launches(self):
        v=self.waiter();(self.prior.waiter_out/'completion.json').unlink()
        with patch.object(v,'guard'),patch.object(v.a,'raw_process_state',return_value=None),patch.object(p.subprocess,'Popen') as launch:
            with self.assertRaisesRegex(RuntimeError,'without completion'):v.run()
        launch.assert_not_called();self.assertFalse((v.out/'STOP').exists())
        self.assertTrue((v.waiter_out/'failure.json').exists())

    def test_scope_rejects_refine_phase_extra_cli_and_gpu(self):
        sc=o.read(self.science);sc['phase']='refined';o.save(self.science,sc)
        self.reg['input_bindings'][str(self.science)]=o.sha(self.science);o.save(self.n['registration'],self.reg)
        with self.assertRaisesRegex(RuntimeError,'phase/config'):p.Waiter(self.config)

    def test_registered_waiter_identity_mismatch_rejected(self):
        stamp=o.read(self.identity);stamp['start_ticks']+=1;o.save(self.identity,stamp)
        self.reg['input_bindings'][str(self.identity)]=o.sha(self.identity);o.save(self.n['registration'],self.reg)
        with self.assertRaisesRegex(RuntimeError,'identity/launch'):p.Waiter(self.config)

    def test_deadline_or_prerequisite_failure_stops_without_signalling_prior(self):
        v=self.waiter()
        with patch.object(v,'wall',return_value=p.DEADLINE):
            with self.assertRaisesRegex(RuntimeError,'96-hour'):v.guard()
        o.save(self.prior.waiter_out/'failure.json',{'original':'preserved'})
        with patch.object(v,'wall',return_value=p.DEADLINE-1):
            with self.assertRaisesRegex(RuntimeError,'Prerequisite failed'):v.guard()
        self.assertFalse((v.out/'STOP').exists())

    def test_new_Popen_capture_failure_keeps_owned_process_and_safe_drains(self):
        v=self.waiter();child=owner_fixture.Process(codes=[None],code=1)
        with patch.object(v,'wait_gate'),patch.object(v,'guard'),patch.object(p.subprocess,'Popen',return_value=child), \
             patch.object(v.a,'identity',side_effect=RuntimeError('capture failed')):
            with self.assertRaisesRegex(RuntimeError,'capture failed'):v.run()
        self.assertTrue(child.waited);self.assertTrue((v.out/'STOP').exists())
        self.assertEqual(o.read(v.waiter_out/'failure.json')['child_identity']['pid'],child.pid)
        self.assertEqual(o.read(v.waiter_out/'owner_exit.json')['exit_code'],1)

    def test_refinement_required_is_success_and_never_starts_refinement(self):
        v=self.waiter();child=owner_fixture.Process(codes=[0],code=0)
        ident=dict(pid=child.pid,start_ticks=77,uid=1002,
                   argv=[self.cfg['python'],'-B',self.cfg['stage_owner'],'--config',self.cfg['next_owner_config']])
        def launch(*args,**kwargs):
            self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'],'')
            complete_batch(self.nextconfig,self.n['registration'],700)
            return child
        def gate():v.before=v.snapshot()
        with patch.object(v,'wait_gate',side_effect=gate),patch.object(v,'guard'), \
             patch.object(p.subprocess,'Popen',side_effect=launch) as popen,patch.object(v.a,'identity',return_value=ident), \
             patch.object(v.a,'raw_process_state',return_value=None),patch.object(os,'getuid',return_value=1002,create=True):
            v.run()
        self.assertEqual(popen.call_count,1)
        d=o.read(v.waiter_out/'completion.json')
        self.assertTrue(d['refinement_requested']);self.assertFalse(d['refinement_started'])
        self.assertFalse(d['ready_for_real_calibration']);self.assertFalse(d['H_full_delivery_claimed'])
        self.assertFalse((v.out/'STOP').exists())


if __name__=='__main__':unittest.main()

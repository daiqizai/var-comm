"""Pure admission/planning tests; no owner process or physical decoder starts."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
sys.path.insert(0,str(HERE.parent/'phy_codec'));sys.path.insert(0,str(HERE.parent/'runner'))
import register_initial_payload as b
import wait_then_coarse as oldwait


class ClosedGateTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.base=self.root/'wait';self.base.mkdir();self.h=self.root/'H';self.h.mkdir()
        self.regsha='a'*64;self.cfgpath=self.root/'config.json';b.save(self.cfgpath,{'bound':True})
        self.wfile=self.root/'wait_then_prescreen.py';self.wfile.write_text('# frozen verifier\n')
        self.ident=dict(pid=11,start_ticks=111,uid=1002,argv=['python','-B',str(self.wfile),'--config',str(self.cfgpath)])
        self.launch=dict(identity=self.ident,argv=self.ident['argv'],registration_sha256=self.regsha,config_sha256=b.sha(self.cfgpath))
        self.stamp=dict(self.ident,registration_sha256=self.regsha,config_sha256=b.sha(self.cfgpath))
        self.ownerid=dict(pid=12,start_ticks=222,uid=1002,argv=['python','-B','owner.py','--config','owner.json'])
        self.states={};self.state=lambda pid:self.states.get(pid)
        self.final=self.h/'prescreen.json';b.save(self.final,dict(status='H_PRESCREEN_COMPLETE_FINAL',
            ready_for_real_calibration=True,registration_sha256=self.regsha,outputs={},source_bindings={},input_bindings={}))
        self.closure=dict(bindings={str(self.final):b.sha(self.final)},child_identities=[dict(pid=13,start_ticks=333,uid=1002,argv=['worker'])])
        self.prior=dict(status='COARSE_WAITER_OWNER_AND_CHILDREN_CLOSED',bindings={},child_identities=[],
                        coarse_waiter_identity={'old':'wait'},coarse_owner_identity={'old':'owner'})
        self.dump('release_gate.json',dict(self.prior,registration_sha256=self.regsha))
        self.dump('owner_launch.json',dict(identity=self.ownerid,argv=self.ownerid['argv'],registration_sha256=self.regsha))
        (self.base/'owner.log').write_text('closed log\n')
        self.dump('owner_exit.json',dict(identity=self.ownerid,registration_sha256=self.regsha,exit_code=0,
                                       closed_log_sha256=b.sha(self.base/'owner.log')))
        self.dump('completion.json',dict(status='H_COARSE_WAIT_AND_PRESCREEN_COMPLETE',registration_sha256=self.regsha,
            output_bindings=self.closure['bindings'],prescreen_status='H_PRESCREEN_COMPLETE_FINAL',ready_for_real_calibration=True,
            refinement_requested=False,refinement_started=False,GPU_used=False,H_full_delivery_claimed=False,C_started=False,
            holdout_started=False,new_packet_decodes=0,new_visual_inference=0))
        def verify(mapping):
            for p,s in mapping.items():b.require(b.sha(p)==s,'changed bound evidence')
        self.api=SimpleNamespace(read=b.read,sha=b.sha,verify=verify,same_identity=lambda a,c:
            all(a[k]==c[k] for k in ('pid','start_ticks','uid','argv')))
        def verify_batch(*_):
            verify(self.closure['bindings'])
            for ident in self.closure['child_identities']:b.require(oldwait.exited(ident,self.state),'worker not reaped')
            return self.closure
        self.p=SimpleNamespace(exited=oldwait.exited,verify_batch=verify_batch)
        self.w=SimpleNamespace(a=self.api,p=self.p,cfg=dict(python='python',stage_owner='owner.py',next_owner_config='owner.json',registration='reg.json'),
            config_path=self.cfgpath,waiter_out=self.base,out=self.h,regsha=self.regsha,prior_launch={},priorid={},
            prior=SimpleNamespace(old={'stages':[]},next={'stages':[]}),next={'stages':[]},job={'completion':str(self.final)})
        self.wapi=SimpleNamespace(__file__=str(self.wfile),verify_coarse_closed=lambda *_:self.prior)
    def dump(self,name,value):
        p=self.base/name;p.write_text(json.dumps(value));return p
    def run_gate(self):return b.verify_prescreen_closed(self.w,self.wapi,self.launch,self.stamp,self.state)
    def test_closed_chain_is_read_only_and_preserves_all_receipt_bindings(self):
        before={str(p):b.sha(p) for p in self.root.rglob('*') if p.is_file()}
        result=self.run_gate();self.assertEqual(result['status'],'ALL_H_INITIAL_COARSE_PRESCREEN_PROCESSES_CLOSED')
        self.assertEqual(result['bindings'][str(self.final)],b.sha(self.final))
        self.assertEqual(before,{str(p):b.sha(p) for p in self.root.rglob('*') if p.is_file()})
    def test_live_or_zombie_waiter_owner_and_worker_each_block(self):
        for identity in (self.ident,self.ownerid,self.closure['child_identities'][0]):
            for state in ('S','Z'):
                self.states.clear();self.states[identity['pid']]=dict(identity,state=state)
                with self.subTest(pid=identity['pid'],state=state),self.assertRaisesRegex(RuntimeError,'reaped'):
                    self.run_gate()
    def test_refinement_pending_does_not_release_initial_calibration(self):
        data=b.read(self.base/'completion.json');data.update(prescreen_status='H_PRESCREEN_COMPLETE_REFINEMENT_REQUIRED',
            ready_for_real_calibration=False,refinement_requested=True);self.dump('completion.json',data)
        with self.assertRaisesRegex(RuntimeError,'FINAL'):self.run_gate()
    def test_failure_stop_and_closed_log_tampering_block(self):
        for name in ('failure.json','STOP'):
            p=self.base/name;p.write_text('failed')
            with self.assertRaisesRegex(RuntimeError,'failed or stopped'):self.run_gate()
            p.unlink()
        (self.base/'owner.log').write_text('changed after close')
        with self.assertRaisesRegex(RuntimeError,'closed-log'):self.run_gate()
    def test_unbound_final_receipt_or_changed_release_gate_blocks(self):
        self.closure['bindings'][str(self.final)]='0'*64
        with self.assertRaisesRegex(RuntimeError,'changed bound'):self.run_gate()
        self.closure['bindings'][str(self.final)]=b.sha(self.final)
        changed=b.read(self.base/'release_gate.json');changed['coarse_owner_identity']={'unexpected':'PID'};self.dump('release_gate.json',changed)
        with self.assertRaisesRegex(RuntimeError,'release gate'):self.run_gate()


class PlanTests(unittest.TestCase):
    def test_plan_has_only_two_cpu_workers_then_one_read_only_merge(self):
        previous=dict(stages=[{'id':'freeze'}],phase_limits=copy.deepcopy(b.PHASES),gpu_threads=6,gpu_affinity=list(range(4,10)),
                      prior_qualification_gate={'path':'frozen'},overall_deadline_unix=b.DEADLINE)
        request=dict(root='/repo',cpu_out='/H/initial_true200',runtime_dir='/runtime',execution_dir='/H/initial_execution_v1',
                     cpu_affinities=[[0,1],[2,3]],python='/python',max_worker_seconds=14400)
        original=copy.deepcopy(previous);cfg=b.owner_plan(request,previous,'/config.json','/reg.json')
        self.assertEqual(previous,original);self.assertEqual(cfg['phase_limits'],b.PHASES)
        self.assertEqual([s['id'] for s in cfg['stages']],['initial_true200','report'])
        self.assertEqual([len(s['jobs']) for s in cfg['stages']],[2,1])
        self.assertTrue(all(s['resource']=='cpu' for s in cfg['stages']))
        for i,job in enumerate(cfg['stages'][0]['jobs']):self.assertEqual(job['argv'][-2:],['--worker-index',str(i)])
        self.assertEqual(cfg['stages'][1]['requires'],['initial_true200'])
        self.assertNotIn('--worker-index',cfg['stages'][1]['jobs'][0]['argv'])
        self.assertEqual(cfg['overall_deadline_unix'],b.DEADLINE)
    def test_contract_has_no_registration_hash_cycle_and_runtime_injection_is_separate(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'core.py';path.write_text('# tested core')
            digest=lambda obj:hashlib.sha256(json.dumps(obj,sort_keys=True).encode()).hexdigest()
            core=SimpleNamespace(__file__=str(path),digest=digest,ENGINEERING_CHOICES={'fixed':'counter'})
            seal=b.contract(core,{'protocol':1},{'catalogue':2},{'shortlist':3})
            self.assertNotIn('execution_registration_sha256',seal)
            self.assertEqual(seal['core_source_sha256'],b.sha(path))
            actual=dict(seal,execution_registration_sha256='f'*64)
            self.assertNotIn('execution_registration_sha256',seal);self.assertEqual(actual['execution_registration_sha256'],'f'*64)
    def test_conflicting_sources_and_changed_pins_cannot_be_silently_rebound(self):
        with self.assertRaisesRegex(RuntimeError,'Conflicting'):b.merge({'/core':'a'},{'/core':'b'})
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'proof';p.write_text('original');record={'path':str(p),'sha256':b.sha(p)}
            self.assertEqual(b.pinned(record),p);p.write_text('changed')
            with self.assertRaisesRegex(RuntimeError,'Pinned'):b.pinned(record)


if __name__=='__main__':unittest.main()

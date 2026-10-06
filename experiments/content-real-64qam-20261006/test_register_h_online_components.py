"""Pure metadata fixtures; no real GPU, model, decoder, process or registration."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import register_h_online_components as r


class Owner:
    def __init__(self,budget):self.budget=budget
    def budget_snapshot(self,*args,**kwargs):return copy.deepcopy(self.budget)
    def validate_config(self,*args):pass


class Tests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.base=Path(temp.name)
        h=self.base/'H';h.mkdir();self.e=h/'timing_execution';self.out=h/'timing'
        budget=h/'budget.json';budget.write_text('{}');core=h/'core.py';core.write_text('# fake only\n')
        log=h/'tests.log';log.write_text('CPU fixtures only\n')
        self.cfg=dict(schema='H_ONLINE_COMPONENTS_CONFIG_V1',root=str(self.base),H_out=str(h),out=str(self.out),
            registration=str(self.e/'execution_registration.json'),visual_owner_config=str(self.e/'owner_config.json'),
            timing_driver_module=str(Path(r.d.__file__).absolute()),timing_core_module=str(core),
            max_seconds=60,overall_deadline_unix=r.d.DEADLINE,ledger=str(h/'ledger'),budget_registration=str(budget),
            phase_limits={'development':13200},stop_file=str(h/'STOP'),protocol=str(budget),metric_batch={})
        self.before=dict(charged=164760,phase_charged={'development':10800},development_remaining=2400,failed=0,unresolved=0)
        self.owner=Owner(self.before)
        old=dict(out=str(h),owner_out='old',registration='old',gpu_threads=6,gpu_affinity=[0,1],
            stages=[dict(id='development',resource='gpu',jobs=[dict(id='development_metrics',argv=['/original/UM/bin/python','-B','old'])])])
        self.ctx=dict(cfg=self.cfg,before=self.before,owner=self.owner,scope={'fixed':'16x18'},source_ids=['s'+str(i) for i in range(100)],
            prior=dict(closed={'owner':old},bindings={}))
        self.p600=dict(bindings={},closed={'normal_owner_success':True})
        sources=r.d.bind((Path(r.__file__).absolute(),Path(r.d.__file__).absolute(),core))
        q=dict(status=r.QUALIFIED,python='/original/UM/bin/python',GPU_used=False,models_constructed=False,
            new_packet_decodes=0,new_noise_draws=0,results=[dict(exit_code=0,test_count=16)],
            source_bindings=sources,input_bindings={},outputs=r.d.bind((log,)))
        self.qp=h/'qualification.json';self.qp.write_text(json.dumps(q))
        self.req=dict(schema='H_ONLINE_COMPONENTS_REGISTRATION_REQUEST_V1',config=self.cfg,execution_dir=str(self.e),
            source_bindings=sources,input_bindings=r.d.merge(q['outputs'],r.d.bind((self.qp,))),
            prepared_qualification=str(self.qp),p600_batch={})
        self.rp=h/'request.json';self.rp.write_text(json.dumps(self.req))

    def run_register(self):
        with mock.patch.object(r.sys,'platform','linux'),mock.patch.object(r.d,'inspect_inputs',return_value=self.ctx) as i,\
            mock.patch.object(r,'normal_p600_closed',return_value=self.p600) as p,\
            mock.patch.object(r.os,'sched_getaffinity',return_value={0,1},create=True),\
            mock.patch.object(r.shutil,'disk_usage',return_value=SimpleNamespace(free=20*1024**3)):
            result=r.register(self.rp);self.assertEqual(i.call_count,1);self.assertEqual(p.call_count,1);return result

    def test_single_fixed_timing_owner_no_successor_or_self_exit_circularity(self):
        old=copy.deepcopy(self.ctx['prior']['closed']['owner']);result=self.run_register()
        self.assertEqual(result['status'],r.REGISTERED);self.assertFalse(result['workers_started']);self.assertFalse(self.out.exists())
        reg=r.d.read(self.e/'execution_registration.json');owner=r.d.read(self.e/'owner_config.json')
        job=owner['stages'][0]['jobs'][0];expect=job['receipt_expect']
        self.assertEqual(job['id'],'h_online_components');self.assertEqual(job['argv'][0],'/original/UM/bin/python')
        self.assertEqual((expect['source_count'],expect['component_case_count'],expect['noise_seed']),(16,288,6201))
        self.assertEqual((expect['warmup_repetitions'],expect['measured_repetitions']),(1,3))
        self.assertTrue(expect['own_owner_success_not_yet_certified']);self.assertFalse(expect['end_to_end_latency_measured'])
        self.assertFalse(reg['future_stage_automatic']);self.assertFalse(reg['MAIN_started'])
        self.assertEqual(self.ctx['prior']['closed']['owner'],old)

    def test_fresh_independent_directories_and_four_hour_cap(self):
        bad=copy.deepcopy(self.req);bad['config']['max_seconds']=14401
        with self.assertRaisesRegex(RuntimeError,'cap'):r.validate_request(bad)
        bad=copy.deepcopy(self.req);bad['config']['out']=str(self.e/'nested')
        with self.assertRaisesRegex(RuntimeError,'independent'):r.validate_request(bad)
        self.e.mkdir();(self.e/'failure.json').write_text('preserved')
        with self.assertRaisesRegex(RuntimeError,'Prior'):r.validate_request(self.req)
        self.assertEqual((self.e/'failure.json').read_text(),'preserved')

    def test_failed_or_not_normal_P600_blocks_before_writes(self):
        with mock.patch.object(r.sys,'platform','linux'),mock.patch.object(r.d,'inspect_inputs',return_value=self.ctx),\
            mock.patch.object(r,'normal_p600_closed',side_effect=RuntimeError('P600 owner still live')):
            with self.assertRaisesRegex(RuntimeError,'P600 owner'):r.register(self.rp)
        self.assertFalse(self.e.exists());self.assertFalse(self.out.exists())

    def test_unbound_P600_evidence_is_not_silently_accepted(self):
        p=self.base/'P600.json';p.write_text('{}');self.p600['bindings']=r.d.bind((p,))
        with self.assertRaisesRegex(RuntimeError,'P600 lineage'):self.run_register()
        self.assertFalse(self.e.exists())

    def test_qualification_changed_interpreter_or_GPU_test_is_rejected(self):
        q=r.d.read(self.qp)
        for change in ({'python':'/usr/bin/python'},{'GPU_used':True}):
            with self.subTest(change=change):
                value=dict(q,**change);self.qp.write_text(json.dumps(value))
                self.req['input_bindings'][str(self.qp)]=r.d.sha(self.qp);self.rp.write_text(json.dumps(self.req))
                with self.assertRaisesRegex(RuntimeError,'CPU-only'):self.run_register()
        self.assertFalse(self.e.exists())

    def test_budget_stop_and_failure_evidence_are_preserved(self):
        self.owner.budget=dict(self.before,charged=164761)
        with self.assertRaisesRegex(RuntimeError,'budget changed'):r.budget_guard(self.ctx)
        self.owner.budget=self.before;Path(self.cfg['stop_file']).write_text('STOP')
        with self.assertRaisesRegex(RuntimeError,'STOP'):r.budget_guard(self.ctx)
        Path(self.cfg['stop_file']).unlink()
        self.owner.validate_config=mock.Mock(side_effect=RuntimeError('invalid owner'))
        with self.assertRaisesRegex(RuntimeError,'invalid owner'):self.run_register()
        self.assertTrue((self.e/'registration_failure.json').exists())
        self.assertFalse((self.e/'registration_completion.json').exists())

    def test_P600_normal_gate_uses_original_closed_batch_without_reenumerating_graph(self):
        cfg=dict(self.cfg,schema='P600_RECEIVED_REPLAY_CONFIG_V1',metric_batch={'completion':str(self.qp)},
            replay_driver_module='/frozen/p_driver.py',visual_source_closure=str(self.base/'oldgraph.json'),
            owner_module='/frozen/owner.py',wait_module='/frozen/wait.py',cpu_driver_module='/frozen/cpu.py')
        self.cfg.update({k:cfg[k] for k in ('owner_module','wait_module','cpu_driver_module','metric_batch')})
        graph={'status':'EXACT_SOURCE_CLOSURE_MATCH','source_bindings':{}}
        Path(cfg['visual_source_closure']).write_text(json.dumps(graph))
        cp=self.base/'Pconfig.json';cp.write_text(json.dumps(cfg));op=Path(cfg['visual_owner_config']);op.parent.mkdir();op.write_text('{}')
        regpath=Path(cfg['registration']);reg=dict(source_bindings={},input_bindings=r.d.bind((cp,op,cfg['visual_source_closure'])))
        regpath.write_text(json.dumps(reg))
        spec=dict(config=str(cp),owner_config=str(op),registration=str(regpath),launch='launch',completion='done')
        job=dict(id='p600_received_replay',argv=['/original/UM/bin/python','-B',cfg['replay_driver_module'],'--config',str(cp)])
        done=dict(source_ids=self.ctx['source_ids'],visual_source_bindings={},budget_before=self.before,budget_after=self.before,
            metric_predecessor_completion_sha256=r.d.sha(self.qp),outputs={})
        closed=dict(owner={'stages':[dict(id='development',resource='gpu',jobs=[job])]},
            done=done,owner_done={'budget':self.before},bindings={})
        cpu=SimpleNamespace(closed_batch=mock.Mock(return_value=closed));self.owner.raw_process_state=mock.Mock()
        self.ctx['prior']['context']=dict(cpu=cpu,wait=object())
        # This fixture replaces only final file hashing for synthetic absent launch/done.
        original_bind=r.d.bind
        with mock.patch.object(r.d,'bind',side_effect=lambda paths: {} if set(paths)==set(spec.values()) else original_bind(paths)):
            value=r.normal_p600_closed(spec,self.ctx)
        self.assertEqual(cpu.closed_batch.call_count,1);self.assertIs(value['closed'],closed)
        self.assertEqual(cpu.closed_batch.call_args.args[4]['frame_count'],600)
        done['source_ids']=['wrong']
        with mock.patch.object(r.d,'bind',return_value={}):
            with self.assertRaisesRegex(RuntimeError,'population'):r.normal_p600_closed(spec,self.ctx)


if __name__=='__main__':unittest.main()

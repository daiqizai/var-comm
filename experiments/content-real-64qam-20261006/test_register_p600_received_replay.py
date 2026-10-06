import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import register_p600_received_replay as r


class Owner:
    def __init__(self,budget): self.budget=budget;self.validated=[]
    def budget_snapshot(self,*args,**kwargs): return copy.deepcopy(self.budget)
    def validate_config(self,*args): self.validated.append(args)


class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.base=Path(self.tmp.name);h=self.base/'H';h.mkdir();e=h/'p600_execution';out=h/'p600'
        budget_file=h/'budget.json';budget_file.write_text('{}')
        core=h/'core.py';core.write_text('# synthetic fixture only\n')
        log=h/'tests.log';log.write_text('synthetic passed\n')
        self.cfg=dict(schema='P600_RECEIVED_REPLAY_CONFIG_V1',H_out=str(h),root=str(self.base),
            out=str(out),registration=str(e/'execution_registration.json'),visual_owner_config=str(e/'owner_config.json'),
            max_seconds=60,overall_deadline_unix=r.d.DEADLINE,max_archive_bytes=1024**3,
            replay_driver_module=str(Path(r.d.__file__).absolute()),core_module=str(core),
            ledger=str(h/'ledger'),budget_registration=str(budget_file),phase_limits={'development':13200},
            stop_file=str(h/'STOP'),protocol=str(budget_file))
        self.before=dict(charged=164760,phase_charged={'development':10800},development_remaining=2400,failed=0,unresolved=0)
        self.owner=Owner(self.before)
        old=dict(out=str(h),owner_out='old',registration='old',gpu_threads=6,gpu_affinity=[0,1],
            stages=[dict(id='development',resource='gpu',jobs=[dict(id='development_metrics',argv=['/original/UM/bin/python','-B','original'])])])
        self.ctx=dict(cfg=self.cfg,before=copy.deepcopy(self.before),owner=self.owner,
            prior=dict(closed={'owner':old},bindings={}))
        sources=r.d.bind((Path(r.__file__).absolute(),Path(r.d.__file__).absolute(),core))
        q=dict(status=r.QUALIFIED,GPU_used=False,models_constructed=False,new_packet_decodes=0,new_noise_draws=0,
            python='/original/UM/bin/python',results=[dict(exit_code=0,test_count=8)],source_bindings=sources,
            input_bindings={},outputs=r.d.bind((log,)))
        qp=h/'qualification.json';qp.write_text(json.dumps(q));self.qp=qp
        self.request=dict(schema='P600_RECEIVED_REPLAY_REGISTRATION_REQUEST_V1',config=self.cfg,execution_dir=str(e),
            source_bindings=sources,input_bindings=r.d.merge(q['outputs'],r.d.bind((qp,))),prepared_qualification=str(qp))
        self.rp=h/'request.json';self.rp.write_text(json.dumps(self.request));self.execution=e;self.out=out

    def run_register(self):
        with mock.patch.object(r.sys,'platform','linux'),mock.patch.object(r.d,'inspect_inputs',return_value=self.ctx) as inspect,\
             mock.patch.object(r.os,'sched_getaffinity',return_value={0,1},create=True),\
             mock.patch.object(r.shutil,'disk_usage',return_value=type('Disk',(),{'free':20*1024**3})()):
            value=r.register(self.rp)
            self.assertEqual(inspect.call_count,1)
            return value

    def test_registers_only_p600_no_launch_or_extra_pipeline(self):
        before=copy.deepcopy(self.ctx['prior']['closed']['owner'])
        value=self.run_register();self.assertEqual(value['status'],r.REGISTERED)
        self.assertFalse(value['workers_started']);self.assertFalse(value['GPU_used']);self.assertFalse(self.out.exists())
        reg=r.d.read(self.execution/'execution_registration.json');owner=r.d.read(self.execution/'owner_config.json')
        job=owner['stages'][0]['jobs'][0]
        self.assertEqual(job['id'],'p600_received_replay');self.assertEqual(len(owner['stages']),1)
        self.assertEqual(job['argv'][0],'/original/UM/bin/python');self.assertEqual(job['receipt_expect']['noise_seeds'],[2001,2002,2003])
        self.assertTrue(job['receipt_expect']['RGB_exact_parity']);self.assertEqual(job['receipt_expect']['new_noise_draws'],0)
        self.assertFalse(reg['MAIN_started']);self.assertEqual(reg['frame_count'],600)
        self.assertEqual(self.ctx['prior']['closed']['owner'],before)
        self.assertEqual(value['registration_sha256'],r.d.sha(self.execution/'execution_registration.json'))

    def test_prior_attempt_or_path_overlap_not_overwritten(self):
        self.execution.mkdir();(self.execution/'failure.json').write_text('original failure')
        with self.assertRaisesRegex(RuntimeError,'Prior'):r.validate_request(self.request)
        self.assertEqual((self.execution/'failure.json').read_text(),'original failure')
        bad=copy.deepcopy(self.request);bad['config']['out']=str(self.execution/'child')
        with self.assertRaisesRegex(RuntimeError,'independent'):r.validate_request(bad)

    def test_qualification_or_tampered_pin_fails_without_registration(self):
        q=r.d.read(self.qp);q['GPU_used']=True;self.qp.write_text(json.dumps(q))
        # Even a newly repinned non-CPU qualification must be rejected.
        self.request['input_bindings'][str(self.qp)]=r.d.sha(self.qp);self.rp.write_text(json.dumps(self.request))
        with self.assertRaisesRegex(RuntimeError,'CPU interface'):self.run_register()
        self.assertFalse(self.execution.exists());self.assertFalse(self.out.exists())

    def test_budget_change_stop_or_deadline_blocks_without_calls(self):
        for kind in ('budget','STOP','deadline'):
            with self.subTest(kind=kind):
                self.owner.budget=copy.deepcopy(self.before)
                stop=Path(self.cfg['stop_file'])
                if stop.exists():stop.unlink()
                if kind=='budget':self.owner.budget['charged']+=1
                if kind=='STOP':stop.write_text('hold')
                with mock.patch.object(r.time,'time',return_value=r.d.DEADLINE+1 if kind=='deadline' else r.d.DEADLINE-1):
                    with self.assertRaises(RuntimeError):r.budget_guard(self.ctx)

    def test_predecessor_failure_blocks_before_output_creation(self):
        with mock.patch.object(r.sys,'platform','linux'),mock.patch.object(r.d,'inspect_inputs',side_effect=RuntimeError('normal owner missing')):
            with self.assertRaisesRegex(RuntimeError,'normal owner'):r.register(self.rp)
        self.assertFalse(self.execution.exists());self.assertFalse(self.out.exists())

    def test_failed_seal_preserves_evidence_no_completion(self):
        self.owner.validate_config=mock.Mock(side_effect=RuntimeError('receipt mismatch'))
        with self.assertRaisesRegex(RuntimeError,'receipt mismatch'):self.run_register()
        self.assertTrue((self.execution/'registration_failure.json').exists())
        self.assertFalse((self.execution/'registration_completion.json').exists())
        self.assertFalse(self.out.exists())

    def test_actual_venv_and_single_gpu_shape_not_replaced(self):
        bad=copy.deepcopy(self.ctx);bad['prior']['closed']['owner']['gpu_threads']=12
        with self.assertRaisesRegex(RuntimeError,'Original single'):r.make_owner(self.cfg,bad,self.execution)
        bad=copy.deepcopy(self.request);bad['config']['max_seconds']=7201
        with self.assertRaisesRegex(RuntimeError,'Bounded'):r.validate_request(bad)
        bad=copy.deepcopy(self.request);bad['config']['max_archive_bytes']=r.d.RGB_BYTES
        with self.assertRaisesRegex(RuntimeError,'Bounded'):r.validate_request(bad)


if __name__=='__main__': unittest.main()

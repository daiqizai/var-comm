"""Synthetic CPU control-plane tests; never model or scientific measurements."""
from __future__ import annotations
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import swin_early_delivery_common as c
import swin_early_delivery as d
import hifi_resume as h

class Contracts(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve();self.p=c.locations(self.root)
        self.config=dict(root=str(self.root),parent_config=str(self.p['parent_config']),parent_config_sha256='parent',
            science_config=str(self.p['child']/'config.json'),pause_evidence=str(self.p['child']/'pause_verified.json'),
            pause_admission=str(self.p['delivery']/'pause_admission.json'),output=str(self.p['delivery']),poll_seconds=1)
        parent=dict(swin_python='/swin/python',metric_python='/metric/python',report_python='/report/python',
            swin_environment=dict(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='6'),
            metric_environment=dict(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='6'))
        c.write(self.p['parent_config'],parent)

    def paused_snapshot(self):
        owners=[dict(pid=31,start_ticks='100'),dict(pid=32,start_ticks='101')]
        c.write(self.config['pause_evidence'],dict(processes={'31':{'exists':False},'32':{'exists':False}},
            completed_physical_frames=33,**{'delivery/status.json':dict(status='PAUSED',pid=31,config_sha256='parent',training_resumed=False),
            'evaluation/reconstruct_status.json':dict(status='PAUSED',pid=32,safe_pause_handler_installed=True)}))
        c.write(self.config['pause_admission'],dict(owners=owners,completed_physical_frames=33));return owners

    def test_exact_process_identity(self):
        owner=dict(pid=20,start_ticks='123')
        self.assertTrue(c.live(owner,lambda _:dict(state='S',start_ticks='123')))
        self.assertFalse(c.live(owner,lambda _:dict(state='S',start_ticks='124')))
        self.assertFalse(c.live(owner,lambda _:dict(state='Z',start_ticks='123')))

    def test_pause_requires_both_exact_owners_exited(self):
        self.paused_snapshot();self.assertEqual(c.snapshot_gate(self.config,lambda _:None)['completed_physical_frames'],33)
        with self.assertRaisesRegex(RuntimeError,'remains alive'):
            c.snapshot_gate(self.config,lambda pid:dict(state='S',start_ticks='101') if pid==32 else None)

    def test_missing_owner_pause_proof_rejected(self):
        owners=self.paused_snapshot();c.write(self.config['pause_admission'],dict(owners=owners[:1],completed_physical_frames=33))
        with self.assertRaisesRegex(RuntimeError,'coverage'):c.snapshot_gate(self.config,lambda _:None)

    def test_changed_pair_frame_count_rejected(self):
        owners=self.paused_snapshot();c.write(self.config['pause_admission'],dict(owners=owners,completed_physical_frames=32))
        with self.assertRaisesRegex(RuntimeError,'count'):c.snapshot_gate(self.config,lambda _:None)

    def test_stage_order_publish_before_hifi(self):
        self.assertEqual(c.STAGES,('reconstruct','score','report','publish','resume_hifi'))
        runner=object.__new__(d.Controller);runner.stop=False;stages=[];runner.stage=lambda stage:stages.append(stage)
        runner.release_receipt=lambda:stages.append('published_receipt')
        runner.run_stages([]);self.assertEqual(stages,['reconstruct','score','report','publish','published_receipt','resume_hifi'])

    def test_takeover_stop_reaches_only_live_worker(self):
        runner=object.__new__(d.Controller);runner.stop=True;stages=[];runner.stage=lambda stage:stages.append(stage)
        with self.assertRaises(d.Paused):runner.run_stages(['resume_hifi'])
        self.assertEqual(stages,['resume_hifi'])

    def test_python_and_gpu_isolation(self):
        for stage,python,gpu in [('reconstruct','/swin/python',True),('score','/metric/python',True),
            ('report','/report/python',False),('publish','/report/python',False),('resume_hifi','/report/python',False)]:
            command,env,actual=c.stage_spec(self.config,stage,'token');self.assertEqual(command[0],python);self.assertEqual(actual,gpu)
            self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'0' if gpu else '')

    def test_parent_pause_uses_parent_hash_and_exact_wrapper_pid(self):
        launch=dict(pid=41,start_ticks='111',started_ns=0,launch_id='new')
        path=self.p['parent_delivery']/'status.json'
        c.write(path,dict(pid=41,config_sha256='parent',status='RUNNING',safe_pause_handler_installed=True))
        self.assertTrue(d.safe_pause(self.config,'resume_hifi',launch))
        c.write(path,dict(pid=40,config_sha256='parent',status='RUNNING',safe_pause_handler_installed=True))
        self.assertFalse(d.safe_pause(self.config,'resume_hifi',launch))

    def test_never_signal_publisher_transaction(self):
        launch=dict(pid=41,start_ticks='111',started_ns=0,launch_id='new')
        self.assertFalse(d.safe_pause(self.config,'publish',launch))

    def test_early_worker_pause_token_must_match(self):
        launch=dict(pid=41,start_ticks='111',started_ns=0,launch_id='new')
        path=self.p['evaluation']/'score_status.json'
        c.write(path,dict(pid=41,launch_id='old',status='PAUSED',safe_pause_handler_installed=True))
        self.assertFalse(d.was_paused(self.config,'score',launch))
        c.write(path,dict(pid=41,launch_id='new',status='PAUSED',safe_pause_handler_installed=True))
        self.assertTrue(d.was_paused(self.config,'score',launch))

    def test_manual_bypass_retires_only_proven_manual_owner(self):
        owner=dict(pid=32,start_ticks='101');original=lambda config:owner;parent={'frozen':True}
        with patch.object(c,'snapshot_gate',return_value={'owners':[owner]}),patch.object(c,'live',return_value=False):
            wrapper=h.manual_bypass(original,parent,self.config);self.assertIsNone(wrapper(parent))
            with self.assertRaisesRegex(RuntimeError,'another parent'):wrapper({'frozen':False})

    def test_manual_bypass_rejects_unproven_or_live_owner(self):
        owner=dict(pid=32,start_ticks='101')
        for owners,alive in [([],False),([owner],True)]:
            with patch.object(c,'snapshot_gate',return_value={'owners':owners}),patch.object(c,'live',return_value=alive):
                with self.assertRaises(RuntimeError):h.manual_bypass(lambda _:owner,{},self.config)({})

    def test_scientific_receipt_requires_all_1800_rows(self):
        path=c.receipt_path(self.config,'score')
        base=dict(status='SWIN_EARLY_EVALUATION_COMPLETE',synthetic=False,sources=100,rows=1800,metric_groups=6,
            full_comparison_complete=False,bindings={},outputs={})
        c.write(path,base);self.assertTrue(c.stage_proof(self.config,'score'))
        for updates in ({'rows':1799},{'metric_groups':12},{'full_comparison_complete':True},{'synthetic':True}):
            c.write(path,dict(base,**updates))
            with self.assertRaises(RuntimeError):c.stage_proof(self.config,'score')

    def test_report_cannot_claim_hifi_or_missing_examples(self):
        path=c.receipt_path(self.config,'report')
        base=dict(status='SWIN_EARLY_REPORT_COMPLETE',synthetic=False,sources=100,rows=1800,metric_groups=6,
            full_comparison_complete=False,png_figures=24,pdf_figures=6,figure_cells=96,inputs={},outputs={})
        c.write(path,base);self.assertTrue(c.stage_proof(self.config,'report'))
        for updates in ({'figure_cells':95},{'full_comparison_complete':True},{'png_figures':23}):
            c.write(path,dict(base,**updates))
            with self.assertRaises(RuntimeError):c.stage_proof(self.config,'report')

    def test_resume_requires_normal_pushed_release(self):
        path=c.receipt_path(self.config,'publish')
        base=dict(status='PUSHED',checks='PASS',phase='swin_early_release',rows=1800,
            full_comparison_complete=False,commit='abc',remote_commit='abc',inputs={})
        c.write(path,base);self.assertTrue(c.stage_proof(self.config,'publish'))
        for updates in ({'status':'COMMITTED'},{'checks':'FAIL'},{'remote_commit':'other'}):
            c.write(path,dict(base,**updates))
            with self.assertRaises(RuntimeError):c.stage_proof(self.config,'publish')

    def test_manual_swin_adoption_binds_exact_original_command(self):
        c.write(self.config['science_config'],{'registered':True})
        command,environment,gpu=c.stage_spec(self.config,'reconstruct','manual')
        path=self.p['child']/'reconstruct_launch.json'
        raw=dict(pid=60,start_ticks='600',stage='reconstruct',launch_id='manual',command=command,environment=environment,
            started_unix=1,config_sha256=c.sha(self.config['science_config']))
        c.write(path,raw);self.config.update(manual_reconstruct=str(path),bindings={str(path):c.sha(path)})
        result=d.manual_launch(self.config,'delivery');self.assertEqual(result['config_sha256'],'delivery');self.assertEqual(result['started_ns'],10**9)
        c.write(path,dict(raw,command=['unregistered']))
        with self.assertRaises(RuntimeError):d.manual_launch(self.config,'delivery')

    def test_git_blobs_allow_later_live_manifest_changes(self):
        path=self.root/'release_manifest.json';path.write_text('later current manifest');original=b'old manifest'
        record=dict(status='PUSHED',checks='PASS',commit='a'*40,remote_commit='a'*40,
            published_files={str(path):hashlib.sha256(original).hexdigest()})
        with (patch.object(c.subprocess,'check_output',return_value=original) as read,
            patch.object(c.subprocess,'run') as run):
            c.verify_publication_blobs(self.root,record)
            self.assertEqual(read.call_args.args[0],['git','show','a'*40+':release_manifest.json'])
            self.assertEqual(run.call_count,2)

if __name__=='__main__':unittest.main()

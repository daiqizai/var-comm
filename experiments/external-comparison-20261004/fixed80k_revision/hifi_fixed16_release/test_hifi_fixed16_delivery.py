"""Synthetic CPU lifecycle tests; no scientific measurements or GPU calls."""
from __future__ import annotations
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import hifi_fixed16_delivery_common as c
import hifi_fixed16_delivery as d

class Contracts(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name).resolve()
        self.p=c.locations(self.root);self.config=dict(root=str(self.root),parent_config=str(self.p['parent_config']),
            science_config=str(self.p['child']/'config.json'),pause_evidence=str(self.p['child']/'pause_verified.json'),
            pause_admission=str(self.p['delivery']/'pause_admission.json'))
        c.write(self.p['parent_config'],dict(swin_python='/swin/python',metric_python='/metric/python',report_python='/report/python',
            swin_environment={'CUDA_VISIBLE_DEVICES':'0','OMP_NUM_THREADS':'6'},metric_environment={'CUDA_VISIBLE_DEVICES':'0','OMP_NUM_THREADS':'6'}))

    def paused(self):
        owners=[dict(pid=31+i,start_ticks=str(100+i)) for i in range(3)]
        snapshot={'processes':{str(o['pid']):dict(o,exists=False) for o in owners}}
        keys=('swin_early_release/delivery/status.json','delivery/status.json','evaluation/reconstruct_status.json')
        for owner,(launch,status),key in zip(owners,c.old_queue_paths(self.config),keys):
            row=dict(pid=owner['pid'],status='PAUSED',safe_pause_handler_installed=True)
            c.write(launch,owner);c.write(status,row);snapshot[key]=row
        c.write(self.config['pause_evidence'],snapshot)
        c.write(self.config['pause_admission'],dict(status='FULL_QUEUES_SAFELY_PAUSED_FOR_FIXED16',owners=owners,
            pause_evidence_sha256=c.sha(self.config['pause_evidence']),training_resume_allowed=False))
        return owners

    def test_no_resume_stage_exists(self):
        self.assertEqual(c.STAGES,('reconstruct','score','report','publish'))
        for stage in ('resume_hifi','train','full_report'):
            with self.assertRaises(RuntimeError):c.stage_spec(self.config,stage,'x')

    def test_stage_order_ends_with_publication(self):
        runner=object.__new__(d.Controller);runner.stop=False;calls=[];runner.stage=lambda stage:calls.append(stage)
        runner.run_stages([]);self.assertEqual(calls,['reconstruct','score','report','publish'])

    def test_takeover_stop_reaches_existing_worker_only(self):
        runner=object.__new__(d.Controller);runner.stop=True;calls=[];runner.stage=lambda stage:calls.append(stage)
        with self.assertRaises(d.Paused):runner.run_stages(['score'])
        self.assertEqual(calls,['score'])

    def test_all_three_old_queues_remain_paused(self):
        self.paused();self.assertEqual(len(c.paused_queues(self.config,lambda _:None)['owners']),3)

    def test_live_old_worker_is_blocker(self):
        self.paused()
        with self.assertRaisesRegex(RuntimeError,'paused and exited'):
            c.paused_queues(self.config,lambda pid:dict(state='S',start_ticks='102') if pid==33 else None)

    def test_replaced_old_controller_is_blocker(self):
        self.paused();path=c.old_queue_paths(self.config)[0][0];c.write(path,dict(pid=200,start_ticks='999'))
        with self.assertRaisesRegex(RuntimeError,'replaced'):c.paused_queues(self.config,lambda _:None)

    def test_old_full_queue_running_status_is_blocker(self):
        owners=self.paused();path=c.old_queue_paths(self.config)[1][1];c.write(path,dict(pid=owners[1]['pid'],status='RUNNING'))
        with self.assertRaises(RuntimeError):c.paused_queues(self.config,lambda _:None)

    def test_pause_proof_hash_is_permanent(self):
        self.paused();c.write(self.config['pause_evidence'],{})
        with self.assertRaisesRegex(RuntimeError,'evidence changed'):c.paused_queues(self.config,lambda _:None)

    def test_exact_process_identity_and_zombie(self):
        owner=dict(pid=20,start_ticks='10')
        self.assertTrue(c.live(owner,lambda _:dict(state='S',start_ticks='10')))
        self.assertFalse(c.live(owner,lambda _:dict(state='S',start_ticks='11')))
        self.assertFalse(c.live(owner,lambda _:dict(state='Z',start_ticks='10')))

    def test_runtime_split(self):
        for stage,python,gpu in [('reconstruct','/swin/python',True),('score','/metric/python',True),('report','/report/python',False),('publish','/report/python',False)]:
            command,env,actual=c.stage_spec(self.config,stage,'x');self.assertEqual(command[0],python)
            self.assertEqual(actual,gpu);self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'0' if gpu else '')

    def test_manual_launch_exact_adoption(self):
        c.write(self.config['science_config'],{'scope':64});command,env,gpu=c.stage_spec(self.config,'reconstruct','manual')
        path=self.p['child']/'reconstruct_launch.json';row=dict(pid=90,start_ticks='190',stage='reconstruct',launch_id='manual',
            started_ns=100,command=command,environment=env,config_sha256=c.sha(self.config['science_config']))
        c.write(path,row);self.config.update(manual_reconstruct=str(path),bindings={str(path):c.sha(path)})
        launch=d.manual_launch(self.config,'driver');self.assertEqual(launch['pid'],90);self.assertEqual(launch['config_sha256'],'driver')
        c.write(path,dict(row,command=['changed']))
        with self.assertRaises(RuntimeError):d.manual_launch(self.config,'driver')

    def test_pause_handshake_needs_exact_pid_token_ready(self):
        launch=dict(pid=22,launch_id='new',started_ns=0)
        path=d.status_path(self.config,'score');row=dict(pid=22,launch_id='new',status='PAUSED',safe_pause_handler_installed=True)
        c.write(path,row);self.assertTrue(d.safe_pause(self.config,'score',launch));self.assertTrue(d.was_paused(self.config,'score',launch))
        for changes in ({'pid':23},{'launch_id':'old'},{'safe_pause_handler_installed':False}):
            c.write(path,dict(row,**changes));self.assertFalse(d.safe_pause(self.config,'score',launch))

    def test_publication_is_not_interrupted(self):
        self.assertFalse(d.safe_pause(self.config,'publish',dict(pid=22,launch_id='new',started_ns=0)))

    def test_reconstruction_requires_64_real_frames(self):
        path=c.receipt_path(self.config,'reconstruct');base=dict(status='HIFI_FIXED16_RECONSTRUCTIONS_COMPLETE',rows=128,
            physical_frames=64,sources=16,synthetic=False,bindings={},outputs={})
        c.write(path,base);self.assertTrue(c.stage_proof(self.config,'reconstruct'))
        for changes in ({'physical_frames':63},{'sources':100},{'rows':3600},{'synthetic':True}):
            c.write(path,dict(base,**changes))
            with self.assertRaises(RuntimeError):c.stage_proof(self.config,'reconstruct')

    def test_score_requires_paired_eight_groups(self):
        path=c.receipt_path(self.config,'score');base=dict(status='HIFI_FIXED16_EVALUATION_COMPLETE',rows=128,
            sources=16,synthetic=False,metric_groups=8,bindings={},outputs={})
        c.write(path,base);self.assertTrue(c.stage_proof(self.config,'score'))
        c.write(path,dict(base,metric_groups=4))
        with self.assertRaises(RuntimeError):c.stage_proof(self.config,'score')

    def test_report_cannot_claim_full_comparison(self):
        path=c.receipt_path(self.config,'report');base=dict(status='HIFI_FIXED16_REPORT_COMPLETE',rows=128,sources=16,
            synthetic=False,metric_groups=8,png_figures=16,pdf_figures=4,figure_cells=128,full_comparison_complete=False,
            physical_frames=64,fixed_sources=16,paired_metric_groups=52,selected_step=80000,exploratory_fixed_subset=True,
            full_evaluation_resume_authorized=False,inputs={},outputs={})
        c.write(path,base);self.assertTrue(c.stage_proof(self.config,'report'))
        for changes in ({'full_comparison_complete':True},{'figure_cells':127},{'png_figures':15}):
            c.write(path,dict(base,**changes))
            with self.assertRaises(RuntimeError):c.stage_proof(self.config,'report')

    def test_only_confirmed_normal_push_finishes(self):
        path=c.receipt_path(self.config,'publish');base=dict(status='PUSHED',checks='PASS',phase='hifi_fixed16_release',
            rows=128,full_comparison_complete=False,commit='abc',remote_commit='abc',inputs={})
        c.write(path,base);self.assertTrue(c.stage_proof(self.config,'publish'))
        for changes in ({'status':'COMMITTED'},{'remote_commit':'other'},{'full_comparison_complete':True}):
            c.write(path,dict(base,**changes))
            with self.assertRaises(RuntimeError):c.stage_proof(self.config,'publish')

    def test_changed_scientific_output_is_rejected(self):
        artifact=self.root/'rows.csv';artifact.write_text('original');path=c.receipt_path(self.config,'score')
        c.write(path,dict(status='HIFI_FIXED16_EVALUATION_COMPLETE',sources=16,rows=128,synthetic=False,metric_groups=8,
            outputs={str(artifact):c.sha(artifact)}));artifact.write_text('changed')
        with self.assertRaisesRegex(RuntimeError,'Bound file changed'):c.stage_proof(self.config,'score')

if __name__=='__main__':unittest.main()

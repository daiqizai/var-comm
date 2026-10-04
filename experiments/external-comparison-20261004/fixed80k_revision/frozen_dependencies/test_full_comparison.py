"""CPU engineering tests; fixtures are synthetic and never scientific results."""
from __future__ import annotations
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import full_comparison_common as c
import full_comparison_delivery as d
import publish_full_comparison as p


class Contracts(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve();self.paths=c.locations(self.root)
        self.config=dict(root=str(self.root),runtime=str(self.root/'runtime'),output=str(self.paths['delivery']),
            external_controller_config=str(self.paths['out']/'controller_config.json'),
            external_eval_config=str(self.paths['out']/'external_eval_config.json'),fixed_examples=str(self.paths['out']/'fixed_examples.json'),
            native_python='/native/python',metric_python='/metric/python',report_python='/report/python',swin_python='/swin/python',
            native_environment={'CUDA_VISIBLE_DEVICES':'0'},metric_environment={'CUDA_VISIBLE_DEVICES':'0'},
            swin_environment={'CUDA_VISIBLE_DEVICES':'0'},poll_seconds=1)

    def report(self,**extra):
        row=dict(status='MATCHED_FIVE_METHOD_REPORT_COMPLETE',synthetic=False,sources=100,rows=9000,method_groups=30,
            metrics_per_group=13,paired_comparisons=780,source_bootstrap_replicates=10000,full_plan_complete=True,
            remaining_requirements=[],receiver_cost_complete=True,reference_conditions_complete=True,
            fixed_sources=16,png_figures=24,pdf_figures=6,figure_cells=480,resource_figures=2,
            sample_selection_uses_quality=False,holdout_access=False,inputs={},outputs={})
        row.update(extra);c.write(self.paths['final']/'completion.json',row);return row

    def table(self,edit=None):
        path=self.paths['result']/'final_comparison/metrics_per_frame.csv';path.parent.mkdir(parents=True,exist_ok=True)
        columns=['source_index','N','snr_db','noise_seed','comparison_family','synthetic','label_conditioned','E',*c.METRICS]
        with path.open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,columns);writer.writeheader()
            for i in range(100):
                for n in (1024,2048):
                    for s in (1,7,13):
                        for seed in (2001,2002,2003):
                            for m in c.FAMILIES:
                                row=dict(source_index=i,N=n,snr_db=s,noise_seed=seed,comparison_family=m,
                                    synthetic=False,label_conditioned=False,E=2*n,**{metric:.5 for metric in c.METRICS})
                                if edit:row=edit(row)
                                if row is not None:writer.writerow(row)
        return path

    def test_complete_9000_cartesian_grid(self):
        path=self.table();self.assertEqual(c.final_table_gate(self.config),{str(path):c.sha(path)})

    def test_missing_point_not_filled(self):
        self.table(lambda row:None if row['source_index']==99 else row)
        with self.assertRaisesRegex(RuntimeError,'9000'):c.final_table_gate(self.config)

    def test_class_conditioning_rejected(self):
        self.table(lambda row:dict(row,label_conditioned=True))
        with self.assertRaisesRegex(RuntimeError,'class conditioned'):c.final_table_gate(self.config)

    def test_nonfinite_metric_rejected(self):
        self.table(lambda row:dict(row,dists='nan'))
        with self.assertRaisesRegex(RuntimeError,'nonfinite'):c.final_table_gate(self.config)

    def test_complete_full_report(self):
        self.report();self.assertTrue(c.stage_proof(self.config,'full_report'))

    def test_incomplete_report_rejected(self):
        for change in ({'full_plan_complete':False},{'receiver_cost_complete':False},{'figure_cells':479},
            {'metrics_per_group':12},{'source_bootstrap_replicates':9999}):
            with self.subTest(change=change):
                self.report(**change)
                with self.assertRaises(RuntimeError):c.stage_proof(self.config,'full_report')

    def test_actual_bound_report_changed(self):
        file=self.root/'artifact.txt';file.write_text('exact');self.report(outputs={str(file):c.sha(file)})
        file.write_text('changed')
        with self.assertRaisesRegex(RuntimeError,'Bound input changed'):c.stage_proof(self.config,'full_report')

    def test_all_five_receiver_population_required(self):
        receipt=dict(status='OWN_CONTROLS_RECEIVER_COST_COMPLETE',rows=120,same_population_all5=True,synthetic=False,
            source_indices=[0,1,2,3],receiver_boundary='received_full_waveform_to_RGB',cuda_synchronized=True,
            includes_header_decode=True,includes_TX=False,includes_model_loading=False,includes_metric_scoring=False,
            summary=[{}]*30,outputs={},bindings={})
        path=self.paths['cost']/'completion.json';c.write(path,receipt);self.assertTrue(c.stage_proof(self.config,'cost_report'))
        receipt['summary']=[{}]*18;c.write(path,receipt)
        with self.assertRaises(RuntimeError):c.stage_proof(self.config,'cost_report')

    def test_stage_python_and_gpu_split(self):
        a,env,gpu=c.stage_spec(self.config,'cost_external','x');self.assertEqual(a[0],'/swin/python');self.assertTrue(gpu)
        a,env,gpu=c.stage_spec(self.config,'cost_own','x');self.assertEqual(a[0],'/native/python');self.assertTrue(gpu)
        for stage in ('cost_report','full_report','publish'):
            a,env,gpu=c.stage_spec(self.config,stage,'x');self.assertEqual(a[0],'/report/python')
            self.assertFalse(gpu);self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'')

    def test_default_uses_actual_frozen_eval_path_and_environment(self):
        path=self.paths['out']/'runtime/external_eval_config.json'
        env=dict(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='6',MKL_NUM_THREADS='6',OPENBLAS_NUM_THREADS='6',PYTHONPATH=str(self.root/'src'))
        c.write(self.paths['out']/'controller_config.json',dict(evaluation=dict(config=str(path),score_environment=env,reconstruction_environment=env)))
        c.write(self.root/'outputs/SCALE-CAUSAL-PARTIAL-RESIDUAL-20261002/m1_calibration_registration.json',
            dict(identity=dict(decoder_gate='/frozen/decoder.json')))
        config=c.default_config(self.root)
        self.assertEqual(config['external_eval_config'],str(path));self.assertEqual(config['metric_environment'],env)
        self.assertEqual(config['swin_environment'],env);self.assertEqual(config['native_environment']['OMP_NUM_THREADS'],'6')
        self.assertTrue(Path(config['native_environment']['PYTHONPATH']).as_posix().endswith('mechanisms/src'))

    def test_pid_reuse_is_not_old_owner(self):
        old=dict(pid=22,start_ticks='100')
        self.assertTrue(c.live(old,lambda _:dict(state='S',start_ticks='100')))
        self.assertFalse(c.live(old,lambda _:dict(state='S',start_ticks='101')))
        self.assertFalse(c.live(old,lambda _:dict(state='Z',start_ticks='100')))

    def upstream(self):
        folder=self.paths['out']/'controller';config=Path(self.config['external_controller_config']);c.write(config,{'immutable':True})
        self.config['bindings']={str(config):c.sha(config)}
        stages={}
        for stage in ('qualification','hifi_preflight','training','hifi_qualification','reconstruct','score'):
            path=self.root/(stage+'.json');c.write(path,{'status':stage+'_DONE'})
            stages[stage]=dict(path=str(path),sha256=c.sha(path),status=stage+'_DONE')
        c.write(folder/'completion.json',dict(status='EXTERNAL_TRAINING_AND_EVALUATION_COMPLETE',config_sha256=c.sha(config),stages=stages))
        c.write(folder/'controller_launch.json',dict(pid=30,start_ticks='200',config_sha256=c.sha(config)))
        c.write(folder/'launches/worker.json',dict(pid=31,start_ticks='201'))

    def test_completion_waits_for_actual_worker_exit(self):
        self.upstream()
        with self.assertRaises(d.Waiting):d.upstream_gate(self.config,lambda pid:dict(state='S',start_ticks='201') if pid==31 else None)
        result=d.upstream_gate(self.config,lambda _:None)
        self.assertEqual(result['status'],'EXTERNAL_SIX_STAGES_COMPLETE_AND_OWNERS_EXITED')
        self.assertFalse(any('launch' in key for key in result['bindings']))

    def test_upstream_failure_never_retries(self):
        self.upstream();c.write(self.paths['out']/'controller/failure.json',{'failed':True})
        with self.assertRaisesRegex(RuntimeError,'review required'):d.upstream_gate(self.config,lambda _:None)

    def test_native_pause_requires_real_new_status(self):
        launch=dict(pid=20,start_ticks='90',launch_id='token',started_ns=0)
        path=self.paths['own']/'screen_status.json';c.write(path,dict(pid=20,stage='screen',status='RUNNING'))
        self.assertTrue(d.pause_ready(self.config,'own_controls',launch))
        self.assertFalse(d.pause_ready(self.config,'publish',launch))
        launch['pid']=21;self.assertFalse(d.pause_ready(self.config,'own_controls',launch))

    def test_score_pause_requires_token_and_handler(self):
        launch=dict(pid=20,start_ticks='90',launch_id='token',started_ns=0)
        path=self.paths['own']/'score_status.json';state=dict(pid=20,launch_id='wrong',safe_pause_handler_installed=True)
        c.write(path,state);self.assertFalse(d.pause_ready(self.config,'score',launch))
        state['launch_id']='token';c.write(path,state);self.assertTrue(d.pause_ready(self.config,'score',launch))

    def test_paused_receipt_must_belong_to_same_worker(self):
        launch=dict(pid=20,start_ticks='90',launch_id='token',started_ns=0)
        path=self.paths['own']/'score_status.json';c.write(path,dict(status='PAUSED',pid=21,launch_id='token'))
        self.assertFalse(d.was_paused(self.config,'score',launch))
        c.write(path,dict(status='PAUSED',pid=20,launch_id='token'));self.assertTrue(d.was_paused(self.config,'score',launch))

    def test_stop_on_takeover_reaches_only_active_worker(self):
        runner=object.__new__(d.Delivery);runner.stop=True;called=[];runner.run_stage=lambda stage:called.append(stage)
        with self.assertRaises(d.Paused):runner.run_stages(['score'])
        self.assertEqual(called,['score'])

    def test_new_stages_run_in_exact_order(self):
        runner=object.__new__(d.Delivery);runner.stop=False;called=[];runner.run_stage=lambda stage:called.append(stage)
        runner.run_stages([]);self.assertEqual(called,list(c.STAGES))

    def test_active_takeover_does_not_pause_before_admission(self):
        runner=object.__new__(d.Delivery);runner.stop=True;runner.config=self.config;runner.out=self.paths['delivery']
        with patch.object(d,'upstream_gate',return_value={'status':'verified'}):runner.wait_upstream(active=True)
        self.assertTrue((runner.out/'upstream_admission.json').exists())

    def test_byte_archive_preserves_binary_utf8_and_original(self):
        path=self.root/'proof.json';raw=('中文🙂\n'*100).encode()+b'\x00\xff';path.write_bytes(raw)
        record,parts,index=p.archive_bytes(path,limit=64);self.assertEqual(path.read_bytes(),raw)
        self.assertTrue(all(part.stat().st_size<=64 for part in parts if part.name.endswith('.b64.txt')))
        p.verify_byte_archive(self.root,record);self.assertEqual(record['sha256'],hashlib.sha256(raw).hexdigest())

    def test_byte_archive_corruption_rejected(self):
        path=self.root/'proof.json';path.write_text('some data'*100)
        record,parts,index=p.archive_bytes(path,limit=128);part=self.root/record['parts'][0]['path'];part.write_text('AAAA')
        with self.assertRaises(RuntimeError):p.verify_byte_archive(self.root,record)

    def test_copy_never_overwrites_old_scientific_bytes(self):
        source=self.root/'source.py';target=self.root/'repository/source.py';source.write_text('new')
        target.parent.mkdir();target.write_text('old')
        with self.assertRaisesRegex(RuntimeError,'Existing published artifact differs'):p.copy_exact(source,target,self.root)
        self.assertEqual(target.read_text(),'old')

    def test_git_blob_validation_ignores_new_live_manifest(self):
        path=self.root/'release_manifest.json';path.write_text('new unrelated manifest')
        old=b'old manifest';calls=[]
        def command(args,capture=False):
            calls.append(args);return old if args[1]=='show' else b''
        record=dict(status='PUSHED',checks='PASS',commit='a'*40,remote_commit='a'*40,
            published_files={str(path):hashlib.sha256(old).hexdigest()})
        p.verify_prior_publication(record,self.root,command)
        self.assertIn(['git','show','a'*40+':release_manifest.json'],calls)
        self.assertIn(['git','merge-base','--is-ancestor','a'*40,'origin/main'],calls)

    def test_binary_capture_not_written_to_log(self):
        response=type('Result',(),dict(returncode=0,stdout=b'PRIVATE_BINARY\x00\xff',stderr=b''))()
        log=io.StringIO()
        with patch.object(p.subprocess,'run',return_value=response):
            result=p.run_command(['git','show','revision:figure.pdf'],self.root,log,capture=True)
        self.assertEqual(result,response.stdout);self.assertNotIn('PRIVATE_BINARY',log.getvalue())

    def test_lossless_csv_helper_keeps_quoted_newlines(self):
        candidates=[Path(__file__).resolve().parents[2]/'historical_eval_20261003/source/experiments/unified-metrics-20261002/publish.py',
            Path(__file__).resolve().parents[3]/'experiments/unified-metrics-20261002/publish.py']
        # Deployment runtime is OUT/runtime; the canonical helper is ROOT/experiments.
        helper=next((x for x in candidates if x.is_file()),None)
        if helper is None:self.skipTest('Canonical archive helper not in this checkout fixture')
        spec=importlib.util.spec_from_file_location('_full_csv_fixture',helper);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        path=self.root/'rows.csv';raw=b'a,b\r\n'+b'1,"two\nlines"\r\n'*30;path.write_bytes(raw)
        manifest=module.archive_tables(self.root,limit=80);entry=manifest['tables']['rows.csv']
        module.verify_archive_entry(self.root,entry);self.assertEqual(path.read_bytes(),raw);self.assertEqual(entry['rows'],30)


if __name__=='__main__':unittest.main()

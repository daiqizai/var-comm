"""CPU-only adversarial controller tests; no Torch, GPU, SSH or real training."""
from __future__ import annotations
import copy
import os
from pathlib import Path
import signal
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import external_controller as c


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.runtime = self.root/'runtime'; self.runtime.mkdir()
        self.source = self.runtime/'bound_source.py'; self.source.write_text('real_cpu_fixture = True\n')
        self.old = self.root/'outputs'/c.OLD_NAME; self.old.mkdir(parents=True)
        self.vendor = self.root/'vendor'; self.vendor.mkdir()
        (self.vendor/'model.py').write_text('official_cpu_fixture = True\n')
        self.cache = self.root/'original_cache'; self.cache.mkdir()
        for name in ('swin_train.py', 'hifi_preflight.py', 'hifi_qualification.py', 'external_eval.py'):
            (self.runtime/name).write_text('# CPU fixture; never executed\n')
        self.train_config = self.runtime/'swin_training_protocol.json'
        c.write(self.train_config, dict(seed=2026100401, total_N=[1024, 2048]))
        self.population = self.root/'population.json'; c.write(self.population, dict(train=20000, calibration=1000))
        self.adm = self.root/'adm.pt'; self.adm.write_bytes(b'CPU fixture, not a model')
        self.old_fixture()
        training = dict(python=sys.executable, script=str(self.runtime/'swin_train.py'), vendor=str(self.vendor),
            output=str(self.root/'outputs'/'training'), config=str(self.train_config), image_cache=str(self.cache),
            reference_registration=str(self.population), environment={'CUDA_VISIBLE_DEVICES':'0'})
        hifi = dict(python=sys.executable, preflight_script=str(self.runtime/'hifi_preflight.py'),
            qualification_script=str(self.runtime/'hifi_qualification.py'), diffcom_vendor=str(self.vendor),
            adm_checkpoint=str(self.adm), preflight_output=str(self.root/'outputs'/'early_hifi'),
            qualification_output=str(self.root/'outputs'/'selected_hifi'), environment={'CUDA_VISIBLE_DEVICES':'0'})
        self.config = dict(version=c.VERSION, root=str(self.root), output=str(self.root/'outputs'/'controller'),
            mode='train', poll_seconds=1, old={'output':str(self.old)}, training=training, hifi=hifi, evaluation=None)
        for key in ('queue_registration', 'controller_launch', 'controller_dispatch'):
            self.config['old'][key+'_sha256'] = c.sha(self.old/(key+'.json'))
        self.config['bindings'] = {str(p):c.sha(p) for p in c.required_sources(self.config)}
        self.config_path = self.root/'controller_config.json'; c.write(self.config_path, self.config)

    def tearDown(self):
        self.temporary.cleanup()

    def old_fixture(self):
        sources = {str(self.source):c.sha(self.source)}
        queue = dict(status='REGISTERED', training_updates=0, policy_selection_updates=0,
                     source_bindings=sources, jobs=[{'study':'study_'+str(i)} for i in range(12)])
        c.write(self.old/'queue_registration.json', queue)
        qsha = c.sha(self.old/'queue_registration.json')
        launch = dict(pid=222, start_ticks='111', queue_registration_sha256=qsha, source_bindings=sources)
        c.write(self.old/'controller_launch.json', launch)
        c.write(self.old/'controller_dispatch.json', dict(**launch, status='CONTROLLER_LAUNCHED'))
        inputs = {}
        for item in queue['jobs']:
            path = self.old/item['study']/'completion.json'
            c.write(path, dict(status='HISTORICAL_STUDY_METRICS_COMPLETE', study=item['study'], sources=100,
                parity_passed=True, synthetic=False, training_updates=0, policy_selection_updates=0))
            inputs[str(path)] = c.sha(path)
        publication = dict(status='PUSHED', phase='results', checks='PASS', commit='1'*40, remote_commit='1'*40,
            runtime_source_bindings=sources, source_bindings=sources, published_files=sources,
            inputs=inputs, queue_registration_sha256=qsha)
        c.write(self.old/'results_publication.json', publication)
        c.write(self.old/'completion.json', dict(**launch, status='HISTORICAL_METRICS_COMPLETE', publication=publication,
            training_updates=0, policy_selection_updates=0, stop=True))

    def qualification(self):
        out = Path(self.config['training']['output'])
        value = dict(status='PASS', synthetic=False, real_training_probe=True, formal_training_updates=0,
            populated_adam_resume_exact=True, initial_model_and_rng_restored=True,
            initial_model_sha256='9'*64, config_sha256=c.identity(c.read(self.train_config)),
            bindings=copy.deepcopy(self.config['bindings']))
        c.write(out/'qualification.json', value)
        return value

    def launch(self, stage='qualification'):
        token = stage+'_cpu_fixture'
        command, env, _ = c.stage_spec(self.config, stage, token)
        return dict(pid=333, start_ticks='444', command=command, environment=env, stage=stage,
            launch_id=token, started_ns=time.time_ns()-2_000_000_000,
            config_sha256=c.sha(self.config_path), source_bindings=self.config['bindings'], log='fixture')

    def store_launch(self, controller, launch):
        c.write(controller.out/'launches'/(launch['launch_id']+'.json'), launch)
        c.write(controller.out/(launch['stage']+'_current_launch.json'), launch)

    def add_evaluation(self):
        output = self.root/'outputs'/'evaluation'; path = self.runtime/'external_eval_config.json'
        c.write(path, dict(root=str(self.root), output=str(output), training_output=self.config['training']['output'],
            hifi_qualification_path=str(Path(self.config['hifi']['qualification_output'])/'qualification.json'),
            swin_vendor=str(self.vendor), diffcom_vendor=str(self.vendor), adm_checkpoint=str(self.adm)))
        self.config['evaluation'] = dict(script=str(self.runtime/'external_eval.py'), config=str(path), output=str(output),
            reconstruction_python=sys.executable, score_python=sys.executable,
            reconstruction_environment={'CUDA_VISIBLE_DEVICES':'0'}, score_environment={'CUDA_VISIBLE_DEVICES':'0'})
        self.config['bindings'] = {str(p):c.sha(p) for p in c.required_sources(self.config)}
        c.write(self.config_path, self.config)
        return output

    def test_all_old_studies_published_and_pid_reuse_is_an_exit(self):
        reader = lambda pid: dict(pid=pid, start_ticks='999', state='S')
        gate = c.old_gate(self.config, reader)
        self.assertEqual(gate['status'], 'HISTORICAL_R3_COMPLETE_PUSHED_AND_EXITED')
        self.assertFalse(gate['original_processes_signalled'])

    def test_old_completion_waits_for_exact_owner_exit(self):
        with self.assertRaises(c.Waiting):
            c.old_gate(self.config, lambda pid: dict(pid=pid, start_ticks='111', state='S'))
        self.assertTrue(c.old_gate(self.config, lambda pid: dict(pid=pid, start_ticks='111', state='Z')))

    def test_old_live_without_final_receipt_waits_but_dead_fails(self):
        (self.old/'completion.json').unlink()
        with self.assertRaises(c.Waiting):
            c.old_gate(self.config, lambda pid: dict(start_ticks='111', state='S'))
        with self.assertRaisesRegex(RuntimeError, 'exited without'):
            c.old_gate(self.config, lambda pid: None)

    def test_old_child_liveness_also_blocks(self):
        c.write(self.old/'launches'/'analysis.json', dict(pid=555, start_ticks='666', job='analysis',
            queue_registration_sha256=self.config['old']['queue_registration_sha256']))
        reader = lambda pid: dict(start_ticks='666', state='S') if pid == 555 else None
        with self.assertRaisesRegex(c.Waiting, 'child'):
            c.old_gate(self.config, reader)

    def test_unpushed_and_changed_old_study_are_rejected(self):
        publication = c.read(self.old/'results_publication.json'); publication['remote_commit'] = '2'*40
        c.write(self.old/'results_publication.json', publication)
        with self.assertRaisesRegex(RuntimeError, 'checked push'):
            c.old_gate(self.config, lambda pid: None)
        self.old_fixture()
        c.write(self.old/'study_5'/'completion.json', {'status':'PARTIAL'})
        with self.assertRaisesRegex(RuntimeError, 'Bound file changed'):
            c.old_gate(self.config, lambda pid: None)

    def test_pinned_historical_launch_cannot_be_substituted(self):
        launch = c.read(self.old/'controller_launch.json'); launch['pid'] += 1
        c.write(self.old/'controller_launch.json', launch)
        with self.assertRaisesRegex(RuntimeError, 'Bound file changed'):
            c.old_gate(self.config, lambda pid: None)

    def test_configuration_freezes_real_dependencies_and_requires_preflight(self):
        c.validate_config(self.config)
        changed = copy.deepcopy(self.config); changed['hifi'] = None
        with self.assertRaisesRegex(RuntimeError, 'early HiFi preflight'):
            c.validate_config(changed)
        changed['mode'] = 'qualification_only'; c.validate_config(changed)
        (self.runtime/'swin_train.py').write_text('changed implementation')
        with self.assertRaisesRegex(RuntimeError, 'Bound file changed'):
            c.validate_config(self.config)

    def test_gpu_query_fails_closed(self):
        self.assertEqual(c.parse_gpu_pids(''), [])
        self.assertEqual(c.parse_gpu_pids('42\n 99\n'), [42, 99])
        for text in ('N/A', 'permission denied', '0', '-2'):
            with self.subTest(text=text), self.assertRaises(RuntimeError):
                c.parse_gpu_pids(text)

    def test_source_qualification_not_process_exit_defines_success(self):
        self.assertIsNone(c.completion(self.config, 'qualification'))
        value = self.qualification(); self.assertEqual(c.completion(self.config, 'qualification')['status'], 'PASS')
        value['real_training_probe'] = False
        c.write(Path(self.config['training']['output'])/'qualification.json', value)
        with self.assertRaisesRegex(RuntimeError, 'real Swin qualification'):
            c.completion(self.config, 'qualification')

    def test_any_training_failure_blocks_even_old_success(self):
        self.qualification()
        c.write(Path(self.config['training']['output'])/'failure.json', {'status':'FAILED_REQUIRES_REVIEW'})
        with self.assertRaisesRegex(RuntimeError, 'not automatic retry'):
            c.completion(self.config, 'qualification')

    def test_no_signal_to_reused_pid_or_changed_command(self):
        launch = self.launch('training'); calls = []
        sender = lambda pid, sig: calls.append((pid, sig))
        self.assertFalse(c.signal_owned(launch, lambda pid: dict(start_ticks='different', state='S'), sender))
        self.assertEqual(calls, [])
        with self.assertRaisesRegex(RuntimeError, 'command changed'):
            c.signal_owned(launch, lambda pid: dict(start_ticks='444', state='S', command=['unrelated']), sender)
        self.assertEqual(calls, [])
        self.assertTrue(c.signal_owned(launch,
            lambda pid: dict(start_ticks='444', state='S', command=launch['command']), sender))
        self.assertEqual(calls, [(333, signal.SIGTERM)])

    def test_handlers_must_be_ready_before_safe_stop(self):
        launch = self.launch('training'); launch['started_ns'] = time.time_ns()
        self.assertFalse(c.ready_to_signal(self.config, 'qualification', launch))
        self.assertFalse(c.ready_to_signal(self.config, 'hifi_preflight', launch))
        self.assertFalse(c.ready_to_signal(self.config, 'training', launch))
        latest = Path(self.config['training']['output'])/'latest.json'
        c.write(latest, {'step':0, 'reason':'before_calibration'})
        os.utime(latest, ns=(launch['started_ns']+1_000_000, launch['started_ns']+1_000_000))
        self.assertTrue(c.ready_to_signal(self.config, 'training', launch))
        os.utime(latest, ns=(launch['started_ns']-1, launch['started_ns']-1))
        self.assertFalse(c.ready_to_signal(self.config, 'training', launch))

    def test_adoption_does_not_spawn_duplicate_or_require_gpu_idle(self):
        self.qualification(); controller = c.Controller(self.config_path)
        launch = self.launch(); self.store_launch(controller, launch)
        with patch.object(c, 'live', side_effect=[True, False]), \
             patch.object(c, 'process', return_value=dict(command=launch['command'])), \
             patch.object(c.subprocess, 'Popen') as spawn, \
             patch.object(controller, 'wait_gate') as gate:
            proof = controller.stage('qualification')
        self.assertEqual(proof['status'], 'PASS'); spawn.assert_not_called(); gate.assert_not_called()
        receipt = c.read(controller.out/'exits'/(launch['launch_id']+'.json'))
        self.assertIsNone(receipt['returncode']); self.assertEqual(receipt['status'], 'COMPLETE')

    def test_unregistered_child_and_fake_success_are_rejected(self):
        controller = c.Controller(self.config_path); launch = self.launch()
        self.store_launch(controller, launch)
        launch['environment']['CUDA_VISIBLE_DEVICES'] = '1'
        c.write(controller.out/'qualification_current_launch.json', launch)
        with self.assertRaisesRegex(RuntimeError, 'launch identity differs'):
            controller.existing_child('qualification')
        with self.assertRaisesRegex(RuntimeError, 'without verified completion'):
            controller.child_done('qualification', self.launch(), 0)

    def test_safe_pause_is_distinct_from_failure_and_needs_fresh_receipt(self):
        controller = c.Controller(self.config_path); launch = self.launch('training')
        with self.assertRaisesRegex(RuntimeError, 'fresh scientific pause'):
            controller.child_done('training', launch, 75)
        status = Path(self.config['training']['output'])/'status.json'
        c.write(status, dict(status='PAUSED', reason='safe checkpoint'))
        with self.assertRaises(c.Paused):
            controller.child_done('training', launch, 75)
        self.assertEqual(c.read(controller.out/'exits'/(launch['launch_id']+'.json'))['status'], 'PAUSED')
        self.assertFalse((controller.out/'failure.json').exists())

    def test_preflight_cannot_substitute_selected_checkpoint_qualification(self):
        admitted = self.qualification()
        command, _, out = c.stage_spec(self.config, 'hifi_preflight', '')
        request = {command[i][2:].replace('-', '_'):str(Path(command[i+1]).resolve())
                   for i in range(3, len(command), 2)}
        value = dict(status='ENGINEERING_PREFLIGHT', passed=True, request=request,
            synthetic=False, scientific_result=False, qualification_only=True, full_input_gradient_pass=True,
            native_forward_parity_pass=True, model_parameters_unchanged=True, selected_checkpoint_qualification=False,
            training_qualification_replacement=False, initialization='random', formal_training_updates=0,
            source_role='original_calibration', source_indices=[0,1], development_read=False, holdout_read=False,
            swin_qualification_sha256=c.sha(Path(self.config['training']['output'])/'qualification.json'),
            initial_model_sha256=admitted['initial_model_sha256'],
            source_bindings={command[2]:c.sha(command[2])}, input_bindings={str(self.adm):c.sha(self.adm)},
            real_ADM_probes=[dict(N=N, sampler_receipt=dict(NFE=2, diagnostic_probe=True)) for N in (1024,2048)])
        c.write(out/'preflight.json', value)
        self.assertEqual(c.completion(self.config, 'hifi_preflight')['status'], 'ENGINEERING_PREFLIGHT')
        self.assertIsNone(c.completion(self.config, 'hifi_qualification'))
        value['training_qualification_replacement'] = True; c.write(out/'preflight.json', value)
        with self.assertRaisesRegex(RuntimeError, 'cannot replace'):
            c.completion(self.config, 'hifi_preflight')

    def test_eval_is_not_invoked_when_its_interface_is_absent(self):
        with self.assertRaisesRegex(RuntimeError, 'no frozen real interface'):
            c.stage_spec(self.config, 'score', 'unused')
        command, _, _ = c.stage_spec(self.config, 'qualification', 'unused')
        self.assertEqual(command[-1], '--qualification-only')
        self.assertNotIn('--qualification-only', c.stage_spec(self.config, 'training', 'unused')[0])

    def test_stop_during_adoption_reaches_existing_worker_before_controller_exit(self):
        controller = c.Controller(self.config_path); controller.stop = True
        stages = ['qualification', 'hifi_preflight', 'training', 'hifi_qualification']
        with patch.object(c, 'completion', return_value={'status':'completed prerequisite'}) as done, \
             patch.object(controller, 'stage', side_effect=c.Paused('worker now safe')) as supervise:
            with self.assertRaisesRegex(c.Paused, 'worker now safe'):
                controller.run_stages(stages, ['training'])
        self.assertEqual(done.call_count, 2)
        supervise.assert_called_once_with('training')

    def test_eval_receipt_rejects_diagnostic_sampler_or_incomplete_population(self):
        out = self.add_evaluation(); c.validate_config(self.config)
        bindings = {self.config['evaluation'][key]:c.sha(self.config['evaluation'][key]) for key in ('script','config')}
        receipt = dict(status='EXTERNAL_RECONSTRUCTIONS_COMPLETE', synthetic=False, sources=100, rows=3600,
            physical_frames=1800, sampler_step_limit=None, selection_uses_development=False,
            bindings=bindings, outputs={str(self.source):c.sha(self.source)})
        path = out/'reconstruction_completion.json'; c.write(path, receipt)
        self.assertEqual(c.completion(self.config, 'reconstruct')['status'], 'EXTERNAL_RECONSTRUCTIONS_COMPLETE')
        for key, value in [('sources',2), ('sampler_step_limit',2), ('selection_uses_development',True)]:
            wrong = dict(receipt); wrong[key] = value; c.write(path, wrong)
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, 'full external'):
                c.completion(self.config, 'reconstruct')

    def test_eval_pause_ready_marker_matches_both_pid_and_launch_id(self):
        out = self.add_evaluation(); launch = self.launch('reconstruct')
        status = dict(status='STARTING', pid=launch['pid'], launch_id=launch['launch_id'], safe_pause_handler_installed=True)
        path = out/'reconstruct_status.json'; c.write(path, status)
        self.assertTrue(c.ready_to_signal(self.config, 'reconstruct', launch))
        status['launch_id'] = 'another launch'; c.write(path, status)
        self.assertFalse(c.ready_to_signal(self.config, 'reconstruct', launch))

    def test_eval_uses_explicit_distinct_interpreters_and_same_registered_training(self):
        self.add_evaluation()
        self.config['evaluation']['reconstruction_python'] = 'native-python'
        self.config['evaluation']['score_python'] = 'metric-python'
        native = c.stage_spec(self.config, 'reconstruct', 'launch-token')[0]
        metrics = c.stage_spec(self.config, 'score', 'launch-token')[0]
        self.assertEqual(native[0], 'native-python'); self.assertEqual(metrics[0], 'metric-python')
        self.assertEqual(native[-2:], ['--launch-id', 'launch-token'])
        self.assertEqual(metrics[-2:], ['--launch-id', 'launch-token'])


if __name__ == '__main__':
    unittest.main()

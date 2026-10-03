"""CPU-only gate, scope, archive and controller integration tests."""
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import history_controller as c
import history_publish as p
from history_common import write, read, sha, source_bindings


def file(path, text='proof\n'):
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(text)
    return path


def public():
    return {'status': 'PUSHED', 'checks': 'PASS', 'commit': 'a'*40, 'remote_commit': 'a'*40}


def parent_fixture(root):
    helper = file(root/'experiments/m2-json-recovery-20261002/publish_when_complete.py', '# Frozen fixture\n')
    proof = file(root/'proof.txt'); bound = {str(proof): sha(proof)}
    previous = dict(public(), source_bindings={str(helper): sha(helper)})
    write(root/'outputs/M2-JSON-RECOVERY-20261002/publication.json', previous)
    folder = root/'outputs'/c.NUMERIC
    publication = dict(public(), published_files=bound, inputs=bound, source_bindings=bound)
    write(folder/'publication.json', publication)
    write(folder/'publication_status.json', {'status': 'COMPLETE', 'publication': publication, 'stop': True, 'pid': 200})
    write(folder/'delivery_launch.json', {'pid': 200, 'start_ticks': '20'})
    unified = root/'outputs/UNIFIED-METRICS-20261002/completion.json'
    write(unified, {'status': 'UNIFIED_METRICS_COMPLETE', 'publication': public()})
    return folder, lambda *_: {'inputs': {str(unified): sha(unified)}, 'commits': ['a'*40]}


def queue_fixture(root):
    runtime = root/'outputs'/c.NAME/'runtime'; runtime.mkdir(parents=True)
    for name in ('historical_fixture.py', 'history_controller.py', 'history_publish.py', 'history_score.py', 'history_analysis.py', 'test_fixture.py'):
        file(runtime/name, '# CPU fixture\n')
    shared = {}
    for name in ('runner.py', 'replay.py', 'metric_models.py', 'batch_speed.py', 'publish.py'):
        path = file(root/'experiments/unified-metrics-20261002'/name, '# Frozen shared fixture\n')
        shared[str(path)] = sha(path)
    proof = file(root/'proof.txt'); coverage = root/'coverage.json'; write(coverage, {'complete': []})
    queue = dict(status='REGISTERED', training_updates=0, policy_selection_updates=0,
        source_bindings=source_bindings(runtime), shared_metric_bindings=shared,
        input_proof_bindings={str(proof): sha(proof)},
        coverage_manifest={'path': str(coverage), 'sha256': sha(coverage)},
        jobs=[{'adapter': 'historical_fixture', 'study': 'TEST_STUDY'}], contrasts=[])
    path = root/'outputs'/c.NAME/'queue_registration.json'; write(path, queue)
    return runtime, path, queue


class Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name).resolve()

    def tearDown(self):
        self.temp.cleanup()

    def test_waits_for_actual_publication_and_same_owner_exit(self):
        folder, base = parent_fixture(self.root)
        gate = c.parent_gate(self.root, {}, base_gate=base)
        self.assertEqual(gate['status'], 'ORIGINAL_DELIVERY_PUSHED_AND_ALL_OWNERS_EXITED')
        with self.assertRaises(c.NotReady):
            c.parent_gate(self.root, {200: {'state': 'S', 'start_ticks': '20'}}, base_gate=base)
        self.assertTrue(c.parent_gate(self.root, {200: {'state': 'S', 'start_ticks': '21'}}, base_gate=base))
        (folder/'publication.json').unlink()
        with self.assertRaises(c.NotReady):
            c.parent_gate(self.root, {}, base_gate=base)

    def test_rejects_forged_status_remote_hash_and_changed_published_file(self):
        folder, base = parent_fixture(self.root)
        status = read(folder/'publication_status.json'); status['pid'] = 201; write(folder/'publication_status.json', status)
        with self.assertRaisesRegex(RuntimeError, 'identity differs'):
            c.parent_gate(self.root, {}, base_gate=base)
        parent_fixture(self.root)
        value = read(folder/'publication.json'); value['remote_commit'] = 'b'*40; write(folder/'publication.json', value)
        with self.assertRaisesRegex(RuntimeError, 'remote SHA'):
            c.parent_gate(self.root, {}, base_gate=base)
        parent_fixture(self.root); file(self.root/'proof.txt', 'changed')
        with self.assertRaisesRegex(RuntimeError, 'changed'):
            c.parent_gate(self.root, {}, base_gate=base)

    def test_queue_inventory_and_path_injection_are_rejected(self):
        runtime, path, queue = queue_fixture(self.root)
        self.assertEqual(c.validate_queue(self.root, runtime, path), queue)
        queue['jobs'][0]['adapter'] = '../old_main'; write(path, queue)
        with self.assertRaises(RuntimeError):
            c.validate_queue(self.root, runtime, path)
        queue['jobs'][0]['adapter'] = 'historical_fixture'; write(path, queue)
        file(runtime/'unregistered.py')
        with self.assertRaisesRegex(RuntimeError, 'inventory'):
            c.validate_queue(self.root, runtime, path)

    def test_gpu_query_unknown_lines_fail_closed(self):
        with patch.object(c.subprocess, 'check_output', return_value='123\n456\n'):
            self.assertEqual(c.gpu_pids(), [123, 456])
        with patch.object(c.subprocess, 'check_output', return_value='N/A\n'):
            with self.assertRaises(RuntimeError):
                c.gpu_pids()

    def test_required_jobs_must_precede_supplements(self):
        runtime, path, queue = queue_fixture(self.root)
        queue['jobs'].append(dict(adapter='historical_fixture',study='OPTIONAL_STUDY'))
        queue['execution_groups']=dict(required=['TEST_STUDY'],analysis_supplements=['OPTIONAL_STUDY'])
        write(path,queue)
        self.assertEqual(c.validate_queue(self.root,runtime,path),queue)
        queue['jobs'].reverse();write(path,queue)
        with self.assertRaisesRegex(RuntimeError,'precede'):
            c.validate_queue(self.root,runtime,path)

    def test_only_analysis_sealed_figures_can_publish(self):
        result=self.root/'results';path=result/'figures'/'curve.png';path.parent.mkdir(parents=True)
        path.write_bytes(b'\x89PNG\r\n\x1a\nENGINEERING_FIXTURE')
        archive={'tables':{}}
        with self.assertRaisesRegex(RuntimeError,'Unsealed'):
            p.selected_result_files(result,archive,{})
        self.assertEqual(p.selected_result_files(result,archive,{str(path):sha(path)}),[path])
        path.write_bytes(b'not a PNG')
        with self.assertRaisesRegex(RuntimeError,'format'):
            p.selected_result_files(result,archive,{str(path):sha(path)})

    def test_waiting_controller_does_not_launch_scoring_or_publication(self):
        runtime, path, queue = queue_fixture(self.root)
        controller = c.Controller(self.root, path, runtime=runtime, once=True)
        with patch.object(c, 'parent_gate', side_effect=c.NotReady('not done')), patch.object(c, 'gpu_pids') as gpu:
            self.assertIsNone(controller.wait_gate())
            gpu.assert_not_called()
        state = read(controller.out/'controller_status.json')
        self.assertEqual(state['status'], 'WAITING_FOR_ORIGINAL_DELIVERY_AND_FREE_GPU')

    def test_controller_exact_serial_order_and_final_receipt(self):
        runtime, path, queue = queue_fixture(self.root)
        controller = c.Controller(self.root, path, runtime=runtime)
        calls = []
        def fake_job(name, script, args=(), cpu=True):
            calls.append((name, script, cpu))
            if name == 'result_publication':
                write(controller.out/'results_publication.json', public())
        with patch.object(c, 'lock', return_value=nullcontext()), patch.object(c, 'proc', return_value={'start_ticks': '99'}), \
                patch.object(controller, 'wait_gate', return_value={'inputs': {}}), patch.object(controller, 'job', side_effect=fake_job):
            # nullcontext has no close, so use a tiny explicit lock substitute.
            class Lock:
                def close(self):
                    pass
            with patch.object(c, 'lock', return_value=Lock()):
                result = controller.run()
        self.assertEqual([x[0] for x in calls], ['test_fixture', 'source_publication', 'TEST_STUDY', 'analysis', 'result_publication'])
        self.assertEqual([x[2] for x in calls], [True, True, False, True, True])
        self.assertEqual(result['status'], 'PUSHED')
        self.assertEqual(read(controller.out/'completion.json')['status'], 'HISTORICAL_METRICS_COMPLETE')

    def test_source_copy_rejects_existing_different_bytes(self):
        a, b = file(self.root/'a.py', 'a'), file(self.root/'b.py', 'b')
        with self.assertRaisesRegex(RuntimeError, 'bytes differ'):
            p.copy_exact(a, b)
        self.assertEqual(b.read_text(), 'b')

    def test_archive_helper_checks_pin_and_never_calls_original_main(self):
        runtime, path, queue = queue_fixture(self.root)
        original = self.root/'experiments/unified-metrics-20261002/publish.py'
        file(original, 'raise RuntimeError("not imported if pin is wrong")\n')
        with self.assertRaisesRegex(RuntimeError, 'hash differs'):
            p.archive_helper(self.root, queue)

    def test_source_publication_contains_only_reviewed_source_and_small_proofs(self):
        runtime, path, queue = queue_fixture(self.root)
        out = self.root/'outputs'/c.NAME
        write(out/'cpu_checks.json', {'status': 'PASS', 'source_bindings': queue['source_bindings']})
        gate = {'inputs': {str(self.root/'proof.txt'): sha(self.root/'proof.txt')}, 'commits': []}
        files, inputs, canonical = p.prepare_files(self.root, runtime, path, queue, 'source', gate)
        self.assertTrue(all(x.suffix in ('.py', '.json') for x in files))
        self.assertTrue(all('/outputs/' not in x.as_posix() for x in files))
        self.assertEqual(set(canonical), {str(self.root/'experiments/historical-metrics-20261003'/Path(x).name) for x in queue['source_bindings']})
        self.assertEqual(inputs[str(path)], sha(path))
        copied = self.root/'results/historical_metrics_20261003/provenance/queue_registration.json'
        self.assertEqual(copied.read_bytes(), path.read_bytes())

    def test_source_publication_does_not_overwrite_extra_canonical_code(self):
        runtime, path, queue = queue_fixture(self.root)
        unexpected = file(self.root/'experiments/historical-metrics-20261003/unreviewed.py')
        with self.assertRaisesRegex(RuntimeError, 'unrelated files'):
            p.prepare_files(self.root, runtime, path, queue, 'source', {'inputs': {}, 'commits': []})
        self.assertTrue(unexpected.exists())

    def test_other_publisher_is_not_signalled_or_ignored(self):
        with self.assertRaises(c.NotReady):
            p.no_other_writers({999999: {'state': 'S', 'command': 'python /repo/outputs/runtime/history_publish.py'}})
        p.no_other_writers({999999: {'state': 'Z', 'command': 'python /repo/outputs/runtime/history_publish.py'}})

    def test_result_gate_rejects_no_real_first_source(self):
        runtime, path, queue = queue_fixture(self.root)
        out = self.root/'outputs'/c.NAME
        proof = self.root/'proof.txt'; bound = {str(proof): sha(proof)}
        write(out/'source_publication.json', dict(public(), runtime_source_bindings=queue['source_bindings'],
             source_bindings=queue['source_bindings'], published_files=bound, inputs=bound))
        study = 'TEST_STUDY'; folder = out/study
        write(folder/'completion.json', dict(status='HISTORICAL_STUDY_METRICS_COMPLETE', study=study,
            parity_passed=True, synthetic=False, training_updates=0, policy_selection_updates=0,
            source_bindings=queue['source_bindings'], inputs=bound, outputs=bound, sources=100, frames=300))
        reg = self.root/'results/historical_metrics_20261003'/study/'registration.json'
        write(reg, dict(queue_registration_sha256=sha(path), source_bindings=queue['source_bindings'],
             source_count=100, frame_count=300, evaluator_identity='same'))
        write(folder/'first_source_qualification.json', {'status': 'SYNTHETIC_PASS'})
        with self.assertRaisesRegex(RuntimeError, 'real first-source'):
            p.result_evidence(self.root, path, queue)


if __name__ == '__main__':
    unittest.main()

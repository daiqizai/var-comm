"""Synthetic CPU checks of scheduling identities; no GPU or image claims."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from contextlib import nullcontext
from types import SimpleNamespace

import scheduling_guard as g


class SchedulingGuardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.out = self.root / 'outputs' / g.NAME
        self.runtime = self.out / 'runtime'
        self.runtime.mkdir(parents=True)
        for name in ('controller.py', 'm2_controller.py', 'scheduled_process.py', 'concurrent_runner.py'):
            (self.runtime / name).write_text('# engineering source fixture\n')
        (self.runtime / 'README.md').write_text('Synthetic scheduling fixture only.\n')
        self.guard = self.root / 'original_guard.py'
        self.guard.write_text('# frozen original thermal/resource guard\n')
        self.bound = g.sources(self.runtime)
        self.admission_path = self.out / 'admission.json'
        self.admission = dict(status='REGISTERED_SCHEDULING_ADMISSION', training_updates=0,
            policy_selection_updates=0, allowed_roles=list(g.ROLES), exclusive_stages=['timing'],
            source_bindings=self.bound, original_guard_bindings={str(self.guard): g.sha(self.guard)})
        g.write(self.admission_path, self.admission)
        self.digest = g.sha(self.admission_path)
        self.procs = {}
        self.original_calls = 0

    def tearDown(self):
        self.tmp.cleanup()

    def peer(self, role='m2'):
        owner = dict(pid=200, start_ticks='100', admission_sha256=self.digest, source_bindings=self.bound)
        lease = dict(status='ACTIVE', pid=300, start_ticks='101', role=role,
            owner=g.identity(owner), admission_sha256=self.digest, source_bindings=self.bound, exclusive=False)
        g.write(g.owner_path(self.out, role), owner)
        g.write(self.out / f'active_{role}.json', lease)
        owner_entry = 'controller.py' if role == 'metrics' else 'm2_controller.py'
        worker_entry = 'concurrent_runner.py' if role == 'metrics' else 'scheduled_process.py'
        self.procs = {
            200: dict(pid=200, start_ticks='100', state='S', ppid=1,
                command='python ' + (self.runtime / owner_entry).as_posix()),
            300: dict(pid=300, start_ticks='101', state='S', ppid=200,
                command='python ' + (self.runtime / worker_entry).as_posix()),
        }
        return owner, lease

    def filtered(self, foreign, role='metrics', exclusive=False):
        def original():
            self.original_calls += 1
            return foreign
        return g.filtered_foreign(original, self.out, self.admission, self.admission_path,
            role, exclusive=exclusive, reader=self.procs.get)()

    def mutate_lease(self, **changes):
        p = self.out / 'active_m2.json'; value = g.read(p); value.update(changes); g.write(p, value)

    def test_python_cleanup_retains_live_cuda_peer_until_actual_exit(self):
        owner,lease = self.peer(role='metrics'); callbacks = []
        runtime = SimpleNamespace(__file__=str(self.guard),foreign_gpu_processes=lambda:[])
        with mock.patch.object(g,'__file__',str(self.runtime/'scheduling_guard.py')),mock.patch.object(g.os,'getpid',return_value=300),\
                mock.patch.object(g,'process',side_effect=self.procs.get),mock.patch.object(g,'admission_lock',return_value=nullcontext()),\
                mock.patch.object(g.importlib,'import_module',return_value=runtime),mock.patch.object(g.atexit,'register',side_effect=callbacks.append):
            g.install_admission(self.root,self.admission_path,role='metrics')
            callbacks[0]()
        saved = g.read(self.out/'active_metrics.json')
        self.assertEqual(saved['status'],'ACTIVE'); self.assertIn('cleanup_started_time',saved)
        self.assertEqual(self.filtered([dict(pid=300,description='CUDA teardown still active')],role='m2'),[])
        self.procs[300]['state'] = 'Z'
        self.assertEqual(self.filtered([dict(pid=300,description='stale CUDA listing')],role='m2'),[])
        with mock.patch.object(g.time,'time',return_value=saved['cleanup_started_time']+5):
            self.assertEqual(self.filtered([dict(pid=300,description='stale CUDA listing')],role='m2'),
                             [dict(pid=300,description='stale CUDA listing')])

    def cleanup_peer(self,stamp=100):
        self.peer(); self.mutate_lease(cleanup_started_time=stamp)
        self.procs[300]['state'] = 'Z'
        return [dict(pid=300,description='[No data],6564'),dict(pid=999,description='unknown GPU user')]

    def test_known_zombie_cleanup_grace_keeps_unknown_users(self):
        foreign = self.cleanup_peer()
        with mock.patch.object(g.time,'time',return_value=100.65):
            self.assertEqual(self.filtered(foreign),[foreign[1]])

    def test_genuinely_exited_known_cleanup_pid_gets_bounded_grace(self):
        foreign = self.cleanup_peer(); del self.procs[300]
        with mock.patch.object(g.time,'time',return_value=104.99): self.assertEqual(self.filtered(foreign),[foreign[1]])

    def test_cleanup_grace_expires_at_five_seconds(self):
        foreign = self.cleanup_peer()
        with mock.patch.object(g.time,'time',return_value=105): self.assertEqual(self.filtered(foreign),foreign)

    def test_future_cleanup_stamp_rejected(self):
        foreign = self.cleanup_peer()
        with mock.patch.object(g.time,'time',return_value=99.9): self.assertEqual(self.filtered(foreign),foreign)

    def test_cleanup_grace_requires_live_owner(self):
        foreign = self.cleanup_peer(); self.procs[200]['state'] = 'Z'
        with mock.patch.object(g.time,'time',return_value=101): self.assertEqual(self.filtered(foreign),foreign)

    def test_cleanup_grace_never_approves_live_pid_reuse(self):
        foreign = self.cleanup_peer(); self.procs[300].update(state='S',start_ticks='999')
        with mock.patch.object(g.time,'time',return_value=101): self.assertEqual(self.filtered(foreign),foreign)

    def test_cleanup_grace_requires_same_zombie_identity(self):
        foreign = self.cleanup_peer(); self.procs[300]['start_ticks'] = '999'
        with mock.patch.object(g.time,'time',return_value=101): self.assertEqual(self.filtered(foreign),foreign)

    def test_cleanup_grace_cannot_change_exclusive_timing(self):
        foreign = self.cleanup_peer()
        with mock.patch.object(g.time,'time',return_value=101): self.assertEqual(self.filtered(foreign,exclusive=True),foreign)

    def test_valid_admission_binds_original_guard(self):
        self.assertEqual(g.validate_admission(self.root, self.admission_path, self.runtime), self.admission)

    def test_admission_cannot_change_registered_grid(self):
        value = copy.deepcopy(self.admission); value['exclusive_stages'] = []
        g.write(self.admission_path, value)
        with self.assertRaises(RuntimeError):
            g.validate_admission(self.root, self.admission_path, self.runtime)

    def test_guard_file_must_remain_frozen(self):
        self.guard.write_text('# changed thermal implementation\n')
        with self.assertRaises(RuntimeError):
            g.validate_admission(self.root, self.admission_path, self.runtime)

    def test_runtime_inventory_must_remain_frozen(self):
        (self.runtime / 'new.py').write_text('# unregistered\n')
        with self.assertRaises(RuntimeError):
            g.validate_admission(self.root, self.admission_path, self.runtime)

    def test_unregistered_directory_is_rejected(self):
        with self.assertRaises(RuntimeError):
            g.validate_admission(self.root, self.admission_path, self.root)

    def test_only_registered_live_peer_is_filtered(self):
        self.peer()
        peer = dict(pid=300, description='registered M2')
        other = dict(pid=400, description='other user')
        self.assertEqual(self.filtered([peer, other]), [other])
        self.assertEqual(self.original_calls, 1)

    def test_other_role_peer_is_filtered_symmetrically(self):
        self.peer('metrics')
        self.assertEqual(self.filtered([dict(pid=300)], role='m2'), [])

    def test_unknown_processes_are_preserved_in_order(self):
        self.peer()
        values = [dict(pid=401, description='unknown1'), dict(pid=402, description='unknown2')]
        self.assertEqual(self.filtered(values), values)

    def test_no_active_lease_preserves_every_foreign_process(self):
        values = [dict(pid=300), dict(pid=400)]
        self.assertEqual(self.filtered(values), values)

    def test_exclusive_timing_preserves_even_registered_peer(self):
        self.peer(); values = [dict(pid=300), dict(pid=400)]
        self.assertEqual(self.filtered(values, exclusive=True), values)

    def test_peer_exclusive_lease_is_not_admitted(self):
        self.peer(); self.mutate_lease(exclusive=True)
        self.assertEqual(self.filtered([dict(pid=300)]), [dict(pid=300)])

    def test_reused_worker_pid_is_not_admitted(self):
        self.peer(); self.procs[300]['start_ticks'] = '102'
        self.assertEqual(self.filtered([dict(pid=300)]), [dict(pid=300)])

    def test_reused_owner_pid_is_not_admitted(self):
        self.peer(); self.procs[200]['start_ticks'] = '102'
        self.assertEqual(self.filtered([dict(pid=300)]), [dict(pid=300)])

    def test_zombie_peer_is_not_admitted(self):
        self.peer(); self.procs[300]['state'] = 'Z'
        self.assertEqual(self.filtered([dict(pid=300)]), [dict(pid=300)])

    def test_departed_peer_is_not_admitted(self):
        self.peer(); del self.procs[300]
        self.assertEqual(self.filtered([dict(pid=300)]), [dict(pid=300)])

    def test_changed_admission_sha_in_lease_is_rejected(self):
        self.peer(); self.mutate_lease(admission_sha256='f' * 64)
        with self.assertRaises(RuntimeError): self.filtered([dict(pid=300)])

    def test_changed_peer_source_lineage_is_rejected(self):
        self.peer(); self.mutate_lease(source_bindings={str(self.guard): 'f' * 64})
        with self.assertRaises(RuntimeError): self.filtered([dict(pid=300)])

    def test_changed_owner_lineage_is_rejected(self):
        self.peer(); owner = g.read(g.owner_path(self.out, 'm2')); owner['start_ticks'] = '102'
        g.write(g.owner_path(self.out, 'm2'), owner)
        with self.assertRaises(RuntimeError): self.filtered([dict(pid=300)])

    def test_forged_parent_relationship_is_rejected(self):
        self.peer(); self.procs[300]['ppid'] = 999
        with self.assertRaises(RuntimeError): self.filtered([dict(pid=300)])

    def test_wrong_worker_entrypoint_is_rejected(self):
        self.peer(); self.procs[300]['command'] = 'python /different/scheduled_process.py'
        with self.assertRaises(RuntimeError): self.filtered([dict(pid=300)])

    def test_wrong_owner_entrypoint_is_rejected(self):
        self.peer(); self.procs[200]['command'] = 'python /different/m2_controller.py'
        with self.assertRaises(RuntimeError): self.filtered([dict(pid=300)])

    def test_runtime_admission_mutation_after_install_is_rejected(self):
        self.peer()
        fn = g.filtered_foreign(lambda: [dict(pid=300)], self.out, self.admission,
            self.admission_path, 'metrics', reader=self.procs.get)
        value = g.read(self.admission_path); value['unexpected'] = True; g.write(self.admission_path, value)
        with self.assertRaises(RuntimeError): fn()

    def test_malformed_original_gpu_response_is_rejected(self):
        for bad in (None, {}, [300], [dict(pid=True)], [dict(pid='300')], [dict(pid=0)]):
            with self.subTest(response=bad), self.assertRaises(RuntimeError): self.filtered(bad)


if __name__ == '__main__':
    unittest.main()

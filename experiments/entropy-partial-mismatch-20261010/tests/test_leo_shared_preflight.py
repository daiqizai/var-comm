"""Local synthetic metadata and real owned-child tests; no SSH/GPU/science."""
from pathlib import Path
import copy
import importlib.util
import os
import subprocess
import sys
import tempfile
import time
import unittest

MODULE = Path(__file__).resolve().parents[1]/'scripts/leo_shared_preflight.py'
spec = importlib.util.spec_from_file_location('leo_shared_preflight', MODULE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
ROOT = MODULE.parents[3]


def snapshot():
    now = time.time()
    return dict(started_unix=now-.3, completed_unix=now,
        gpu=dict(uuid=m.GPU_UUID, free_MiB=67024, utilization_percent=19),
        cpu=dict(allowed_cpus=list(range(8)), effective_cpu_cores=8,
            host_busy_percent=24, load1=1000, available_memory_bytes=8*(1 << 30)),
        personal_filesystem_free_bytes=20*(1 << 30))


class SharedPreparationTests(unittest.TestCase):
    def test_gpu_query_parsing_scoped_uuid_and_real_name(self):
        row = m.parse_gpu(m.GPU_UUID+', 5, NVIDIA RPBZZZ6, 97887, 67024, 19\n')
        self.assertEqual(row['uuid'],m.GPU_UUID)
        self.assertEqual(row['name'],'NVIDIA RPBZZZ6')
        self.assertFalse(row['exclusivity_verified'])
        for text in ['GPU-other,5,name,97887,67024,19',
                     m.GPU_UUID+',5,name,97887,100000,19',
                     m.GPU_UUID+',5,name,97887,NaN,19',
                     (m.GPU_UUID+',5,name,97887,67024,19\n')*2]:
            with self.subTest(text=text), self.assertRaises((RuntimeError,ValueError)):
                m.parse_gpu(text)

    def test_cpu_only_admission_never_claims_gpu_exclusive_or_science(self):
        result=m.check_resources(snapshot(),2)
        self.assertEqual(result['workers'],1)
        self.assertEqual(result['cpu_affinity'],[0,1])
        self.assertFalse(result['GPU_allocations_allowed'])
        self.assertFalse(result['science_calls_allowed'])
        # High loadavg alone is not high sampled CPU utilization.
        self.assertTrue(result['sampled_GPU_headroom_estimate'])

    def test_other_users_busy_gpu_does_not_block_cpu_check(self):
        value=snapshot();value['gpu'].update(free_MiB=100,utilization_percent=100)
        result=m.check_resources(value,2)
        self.assertFalse(result['sampled_GPU_headroom_estimate'])
        self.assertTrue(result['GPU_headroom_does_not_block_CPU_only_checks'])

    def test_resource_shortage_stale_and_thread_escalation_refused(self):
        variants=[]
        v=snapshot();v['started_unix']-=20;variants.append(v)
        v=snapshot();v['cpu']['host_busy_percent']=96;variants.append(v)
        v=snapshot();v['cpu']['available_memory_bytes']=1 << 30;variants.append(v)
        v=snapshot();v['personal_filesystem_free_bytes']=1 << 30;variants.append(v)
        v=snapshot();v['cpu']['effective_cpu_cores']=1.5;variants.append(v)
        v=snapshot();v['gpu']['uuid']='other';variants.append(v)
        for value in variants:
            with self.assertRaises(RuntimeError):m.check_resources(value,2)
        for threads in (1,5,True):
            with self.assertRaises(RuntimeError):m.check_resources(snapshot(),threads)

    def test_private_paths_reject_escape_and_traversal(self):
        with tempfile.TemporaryDirectory() as t:
            base=Path(t)/'personal';base.mkdir()
            self.assertEqual(m.inside(base/'logs'/'run1',base),base/'logs'/'run1')
            for p in (Path(t)/'other',base/'logs'/'..'/'other',Path('relative')):
                with self.assertRaises(RuntimeError):m.inside(p,base)

    def test_private_symlink_escape_if_platform_permits(self):
        with tempfile.TemporaryDirectory() as t:
            base=Path(t)/'personal';base.mkdir();other=Path(t)/'other';other.mkdir()
            try:(base/'escape').symlink_to(other,target_is_directory=True)
            except OSError as e:self.skipTest('Local symlink permission unavailable: '+str(e))
            with self.assertRaises(RuntimeError):m.inside(base/'escape'/'write',base)

    def test_venv_spelling_stays_under_personal_base(self):
        with tempfile.TemporaryDirectory() as t:
            base=Path(t);venv=base/'env';(venv/'bin').mkdir(parents=True)
            (venv/'pyvenv.cfg').write_text('synthetic venv')
            p=venv/'bin/python';p.write_text('not executed')
            self.assertEqual(m.interpreter_path(str(p),base),p)
            (venv/'pyvenv.cfg').unlink()
            with self.assertRaises(RuntimeError):m.interpreter_path(str(p),base)

    def test_real_personal_file_lock_rejects_duplicate_owner(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'owner.lock'
            with m.owner_lock(p):
                with self.assertRaises((OSError,RuntimeError)):
                    with m.owner_lock(p):pass
            with m.owner_lock(p):pass

    def test_all_cache_tmp_environment_paths_are_private_and_gpu_hidden(self):
        with tempfile.TemporaryDirectory() as t:
            base=Path(t); original=dict(os.environ)
            env, controlled=m.controlled_environment(base,base/'run',2)
            self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'')
            for name in m.CACHE_NAMES:
                self.assertTrue(Path(controlled[name]).is_relative_to(base))
                self.assertTrue(Path(controlled[name]).is_dir())
            self.assertNotIn('HOME',controlled)
            self.assertNotIn('CODEX_HOME',controlled)
            self.assertEqual(dict(os.environ),original)
            self.assertEqual(env['OMP_NUM_THREADS'],'2')
            self.assertNotIn('PYTHONPATH',env)

    def test_exact_whitelist_refuses_arbitrary_or_changed_engineering_code(self):
        with tempfile.TemporaryDirectory() as t:
            base=Path(t);project=base/'project';project.mkdir()
            for rel in m.CHECK_FILES['plan']:
                target=project/rel;target.parent.mkdir(parents=True,exist_ok=True)
                target.write_bytes((ROOT/rel).read_bytes())
            script,pins=m.bind_check(project,'plan',base)
            self.assertEqual(len(pins),2)
            self.assertEqual(script.name,'test_ep_plan.py')
            with self.assertRaises(RuntimeError):m.bind_check(project,'qualification',base)
            script.write_text('print("changed")')
            with self.assertRaises(RuntimeError):m.bind_check(project,'plan',base)

    def test_real_owned_child_exit_is_waited_not_inferred_from_pid(self):
        p=subprocess.Popen([sys.executable,'-c','raise SystemExit(0)'],stdout=subprocess.DEVNULL)
        result=m.wait_owned(p,5)
        self.assertTrue(result['actual_child_waited'])
        self.assertTrue(result['success'])
        self.assertEqual(p.returncode,0)

    def test_real_owned_child_failure_stops_no_retry(self):
        p=subprocess.Popen([sys.executable,'-c','raise SystemExit(7)'],stdout=subprocess.DEVNULL)
        result=m.wait_owned(p,5)
        self.assertFalse(result['success'])
        self.assertFalse(result['automatic_retry'])
        self.assertEqual(result['child_exit_code'],7)

    def test_real_owned_child_timeout_terminates_and_waits_only_that_child(self):
        p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'],stdout=subprocess.DEVNULL)
        result=m.wait_owned(p,.05)
        self.assertFalse(result['success'])
        self.assertEqual(result['stop_reason'],'ENGINEERING_DEADLINE_EXCEEDED')
        self.assertIsNotNone(p.returncode)

    def test_output_records_are_write_once(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'receipt.json';m.save(p,dict(synthetic=True))
            with self.assertRaises(FileExistsError):m.save(p,dict(synthetic=False))


if __name__=='__main__':
    unittest.main(verbosity=2)

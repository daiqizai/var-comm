"""New inert tests of only the startup-resource revision; no real GPU calls."""
import ast
import copy
import csv
import json
from pathlib import Path
import types
import unittest
from unittest.mock import patch
import resource_adapter as a
import timing_resource_owner as owner

RESOURCE_SOURCE = "def resource_snapshot(repository, allowed_observer=None):\n    reg = read(OUT / 'registration.json')\n    original = old_snapshot(repository, allowed_observer)\n    gpu_rows = subprocess.check_output(['nvidia-smi',\n        '--query-gpu=index,uuid,name,memory.total,memory.free,utilization.gpu,utilization.memory',\n        '--format=csv,noheader,nounits'], text=True)\n    parsed = list(csv.reader(gpu_rows.splitlines()))\n    entry = next(row for row in parsed if int(row[0]) == reg['GPU_index'])\n    idx, uuid, name = int(entry[0]), entry[1].strip(), entry[2].strip()\n    total, free, util, mem_util = map(float, entry[3:])\n    processes = subprocess.check_output(['nvidia-smi',\n        '--query-compute-apps=gpu_uuid,pid,process_name,used_memory',\n        '--format=csv,noheader,nounits'], text=True)\n    selected = [row for row in csv.reader(processes.splitlines()) if row and row[0].strip() == uuid]\n    own = [row for row in selected if int(row[1]) == os.getpid()]\n    # The latest authorization permits E2 CPU statistics in parallel. Preserve\n    # their presence in the record; only scientific E1 jobs / repository GPU\n    # jobs block this explicitly shared timing window.\n    blocked = []\n    science = ROOT / 'experiments/entropy-main-common500-20261013'\n    for pid in original['conflicting_repository_python_pids']:\n        try:\n            argv = Path(f'/proc/{pid}/cmdline').read_bytes().decode().split('\\0')\n        except FileNotFoundError:\n            continue\n        if pid in original['gpu_pids'] or any(\n            str(science/name) in argv for name in ('owner.py','source.py','phy.py','visual.py','score.py')):\n            blocked.append(pid)\n    constructing_or_running = allowed_observer is not None\n    memory_ok = free >= 2048 if constructing_or_running else free >= 18432\n    utilization_ok = constructing_or_running or util <= reg['initial_GPU_utilization_max_percent']\n    coordination = coordinated()\n    snapshot = dict(admitted=(coordination['released'] and memory_ok and utilization_ok and\n                          not blocked),\n                checked_unix=time.time(), selected_GPU=dict(index=idx, uuid=uuid, name=name,\n                    memory_total_MiB=total, memory_free_MiB=free,\n                    utilization_percent=util, memory_utilization_percent=mem_util),\n                resident_compute_processes=selected, other_GPU_pids=original['gpu_pids'],\n                conflicting_repository_python_pids=blocked,\n                observed_repository_python_pids=original['conflicting_repository_python_pids'],\n                coordination=coordination, memory_admitted=memory_ok,\n                initial_low_load_admitted=utilization_ok, shared_environment=True,\n                exclusive_or_uncontended_claimed=False, jobs_stopped=0,\n                GPU_global_settings_changed=False)\n    if RUN.exists():\n        with (RUN / 'resource_snapshots.jsonl').open('a') as handle:\n            handle.write(json.dumps(snapshot, sort_keys=True) + '\\n')\n    return snapshot\n"
POST_SOURCE = "def validate_resources(history, numerical, device, reg, admission, worker_pid, expected_boundaries=1730):\n    require(len(history) == expected_boundaries, 'Missing/extra before/after or per-case resource snapshots')\n    require(admission['admitted'] is True and admission['selected_GPU']['memory_free_MiB'] >= 18432\n            and admission['selected_GPU']['utilization_percent'] <= 25, 'Original low-load admission missing')\n    require(numerical['FP32'] is True and numerical['TF32'] is False and numerical['deterministic'] is True,\n            'Numeric precision changed')\n    require(numerical['shared_GPU'] is True and numerical['timing_measurement'] is True, 'Wrong runtime scope')\n    require(numerical['threads'] == 6 and numerical['interop_threads'] == 2\n            and sorted(numerical['affinity']) == sorted(reg['CPU_slots']), 'Thread/affinity changed')\n    require(numerical['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8' and numerical['allocator_cap_bytes'] == 16 * 1024**3,\n            'Numeric/runtime cap differs')\n    binding = device['configured_device_binding']\n    require(device['same_physical_PCI'] is True and numerical['device_identity']['same_physical_PCI'] is True, 'Device PCI identity not established')\n    require(device['actual_pci_bus_id'] == binding['pci_bus_id'] == numerical['device_identity']['actual_pci_bus_id'], 'Wrong physical PCI device')\n    require(device['actual_cuda_uuid'] == binding['cuda_uuid'] == numerical['device_identity']['actual_cuda_uuid'], 'CUDA identity mismatch')\n    require(device['configured_nvml_uuid'] == binding['nvml_uuid'] == numerical['device_identity']['configured_nvml_uuid'], 'NVML identity mismatch')\n    require(device['configured_nvml_index'] == reg['GPU_index'] == binding['nvml_gpu_index'], 'GPU index changed')\n    times = []; free = []; util = []; foreign = []; releases = set()\n    for row in history:\n        require(row['admitted'] is True and row['shared_environment'] is True and row['exclusive_or_uncontended_claimed'] is False, 'Resource window violated')\n        require(row['jobs_stopped'] == 0 and row['GPU_global_settings_changed'] is False\n                and not row['conflicting_repository_python_pids'], 'Interference/setting guard violated')\n        gpu = row['selected_GPU']\n        require(gpu['index'] == reg['GPU_index'] and gpu['uuid'] == device['configured_nvml_uuid'] and gpu['name'] == numerical['device'], 'Resource device differs')\n        require(gpu['memory_free_MiB'] >= 2048 and row['memory_admitted'] is True, 'Free-memory floor violated')\n        require(row['coordination']['released'] is True, 'Quality/timing coordination was not released')\n        releases.add((row['coordination']['release']['path'], row['coordination']['release']['sha256']))\n        times.append(finite(row['checked_unix'], 'resource timestamp')); free.append(gpu['memory_free_MiB']); util.append(gpu['utilization_percent'])\n        foreign.append(sum(int(p[1]) != worker_pid for p in row['resident_compute_processes']))\n    require(times == sorted(times), 'Resource snapshots reordered')\n    require(len(releases) == 1, 'Coordination release changed')\n    return dict(boundary_samples=len(history), minimum_free_MiB=min(free), maximum_GPU_utilization_percent=max(util),\n                minimum_foreign_resident_processes=min(foreign), maximum_foreign_resident_processes=max(foreign),\n                first_sample_unix=times[0], last_sample_unix=times[-1], continuous_resource_monitoring=False,\n                CUDA_and_NVML_UUID_namespaces_differ=device['actual_cuda_uuid'] != device['configured_nvml_uuid'],\n                same_physical_PCI=True, pci_bus_id=device['actual_pci_bus_id'], numerical_runtime=numerical), releases\n"


class AbsentRun:
    def exists(self): return False


class FakeProc:
    def __init__(self, path): self.path = path
    def read_bytes(self): return b'foreign_worker\0'


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.free=8192; self.util=0; self.released=True; self.conflicts=[]; self.gpu_pids=[]
        self.ns=dict(read=lambda p: dict(GPU_index=0,initial_GPU_utilization_max_percent=25),
            old_snapshot=lambda *args: dict(conflicting_repository_python_pids=self.conflicts,gpu_pids=self.gpu_pids),
            subprocess=types.SimpleNamespace(check_output=self.query), csv=csv,
            os=types.SimpleNamespace(getpid=lambda:42), Path=FakeProc,
            ROOT=Path('/inert'), OUT=Path('/inert'), RUN=AbsentRun(),
            coordinated=lambda:dict(released=self.released),time=types.SimpleNamespace(time=lambda:123),json=json)
        self.f,self.proof=a.compile_function(RESOURCE_SOURCE,self.ns,'embedded_resource_fixture','resource_snapshot',a.STARTUP_OLD,a.STARTUP_NEW)

    def query(self, args, **kwargs):
        if 'index,uuid,name' in args[1]: return f'0, UUID, H800, 81559, {self.free}, {self.util}, 0\n'
        return ''

    def test_startup_exact_boundary(self):
        self.free=8191; self.assertFalse(self.f(None)['admitted'])
        self.free=8192; self.assertTrue(self.f(None)['admitted'])

    def test_current_observed_free(self):
        self.free=10197; self.assertTrue(self.f(None)['admitted'])

    def test_original_high_startup_preserved_as_history(self):
        ns=dict(self.ns); exec(RESOURCE_SOURCE,ns)
        self.assertFalse(ns['resource_snapshot'](None)['admitted'])

    def test_initial_utilization_preserved(self):
        self.util=25; self.assertTrue(self.f(None)['admitted'])
        self.util=26; self.assertFalse(self.f(None)['admitted'])

    def test_runtime_floor_preserved(self):
        self.free=2047; self.assertFalse(self.f(None,{'observer':True})['admitted'])
        self.free=2048; self.assertTrue(self.f(None,{'observer':True})['admitted'])

    def test_runtime_does_not_require_idle_device(self):
        self.util=99; self.free=2048; self.assertTrue(self.f(None,{'observer':True})['admitted'])

    def test_coordination_preserved(self):
        self.released=False; self.assertFalse(self.f(None)['admitted'])

    def test_conflicting_gpu_process_preserved(self):
        self.conflicts=[999]; self.gpu_pids=[999]; self.assertFalse(self.f(None)['admitted'])

    def test_cpu_only_other_job_still_allowed(self):
        self.conflicts=[999]; self.assertTrue(self.f(None)['admitted'])

    def test_inverse_and_one_constant(self):
        changed,proof=a.changed_function(RESOURCE_SOURCE,'resource_snapshot',a.STARTUP_OLD,a.STARTUP_NEW)
        self.assertEqual(changed.replace(a.STARTUP_NEW,a.STARTUP_OLD),RESOURCE_SOURCE)
        self.assertTrue(proof['inverse_AST_equal'])
        before=[n.value for n in ast.walk(ast.parse(RESOURCE_SOURCE)) if isinstance(n,ast.Constant)]
        after=[n.value for n in ast.walk(ast.parse(changed)) if isinstance(n,ast.Constant)]
        self.assertEqual(sum(x!=y for x,y in zip(before,after)),1)

    def test_reject_unexpected_comparator(self):
        with self.assertRaises(AssertionError):
            a.changed_function(RESOURCE_SOURCE.replace('18432','19000'),'resource_snapshot',a.STARTUP_OLD,a.STARTUP_NEW)

    def test_reject_duplicate_function(self):
        with self.assertRaises(AssertionError):
            a.changed_function(RESOURCE_SOURCE*2,'resource_snapshot',a.STARTUP_OLD,a.STARTUP_NEW)

    def test_shared_snapshot_reference(self):
        v2=types.SimpleNamespace(**self.ns); v2.resource_snapshot=lambda:None
        old=types.SimpleNamespace(exclusive_snapshot=lambda:None)
        v6=types.SimpleNamespace(v2=v2,old=old)
        a.install_snapshot(v6,RESOURCE_SOURCE,'embedded')
        self.assertIs(v2.resource_snapshot,old.exclusive_snapshot)
        self.assertTrue(old.exclusive_snapshot(None)['admitted'])


class PostprocessTests(unittest.TestCase):
    def setUp(self):
        def require(v,m):
            if not v: raise AssertionError(m)
        ns=dict(require=require,finite=lambda v,n:float(v))
        oldns=dict(ns);exec(POST_SOURCE,oldns);self.original=oldns['validate_resources']
        self.relaxed,self.proof=a.compile_function(POST_SOURCE,ns,'embedded_post_fixture','validate_resources',a.POST_OLD,a.POST_NEW)
        self.reg=dict(schema='FROZEN_SIX_METHOD_FIXED16_TIMING_V6_PARTIAL_RECOVERY',CPU_slots=[25,29,31,35,39,43],GPU_index=0)
        self.admission=dict(admitted=True,selected_GPU=dict(memory_free_MiB=10197,utilization_percent=0))
        binding=dict(pci_bus_id='PCI',cuda_uuid='CUDA',nvml_uuid='NVML',nvml_gpu_index=0)
        self.device=dict(configured_device_binding=binding,same_physical_PCI=True,actual_pci_bus_id='PCI',actual_cuda_uuid='CUDA',configured_nvml_uuid='NVML',configured_nvml_index=0)
        self.num=dict(FP32=True,TF32=False,deterministic=True,shared_GPU=True,timing_measurement=True,threads=6,interop_threads=2,
            affinity=self.reg['CPU_slots'],CUBLAS_WORKSPACE_CONFIG=':4096:8',allocator_cap_bytes=16*1024**3,
            device_identity=self.device,device='H800')
        self.history=[dict(admitted=True,shared_environment=True,exclusive_or_uncontended_claimed=False,jobs_stopped=0,GPU_global_settings_changed=False,
            conflicting_repository_python_pids=[],selected_GPU=dict(index=0,uuid='NVML',name='H800',memory_free_MiB=4096,utilization_percent=80),memory_admitted=True,
            coordination=dict(released=True,release=dict(path='prior-release',sha256='same')),checked_unix=1,resident_compute_processes=[])]
        self.f=a.resource_validator(self.original,self.relaxed,copy.deepcopy(self.reg),copy.deepcopy(self.admission),42)

    def call(self,reg=None,admission=None,pid=42):
        return self.f(self.history,self.num,self.device,self.reg if reg is None else reg,self.admission if admission is None else admission,pid,1)

    def test_bound_new_segment_uses_8192(self):
        self.assertEqual(self.call()[0]['startup_free_threshold_MiB'],8192)

    def test_historical_segment_retains_18432(self):
        reg=dict(self.reg,schema='FROZEN_SIX_METHOD_FIXED16_TIMING_V5_RECOVERY')
        with self.assertRaises(AssertionError): self.call(reg)
        high=copy.deepcopy(self.admission);high['selected_GPU']['memory_free_MiB']=18432
        self.assertNotIn('startup_resource_revision_applied',self.call(reg,high)[0])

    def test_unknown_segment_no_relaxation(self):
        with self.assertRaises(AssertionError): self.call(dict(self.reg,schema='UNKNOWN'))

    def test_unbound_registration_rejected(self):
        with self.assertRaises(AssertionError): self.call(dict(self.reg,extra='changed'))

    def test_unbound_admission_rejected(self):
        wrong=copy.deepcopy(self.admission);wrong['selected_GPU']['memory_free_MiB']+=1
        with self.assertRaises(AssertionError): self.call(admission=wrong)

    def test_unbound_worker_rejected(self):
        with self.assertRaises(AssertionError): self.call(pid=43)
        with self.assertRaises(AssertionError): self.call(pid=True)

    def test_numeric_guards_preserved(self):
        self.num['TF32']=True
        with self.assertRaises(AssertionError): self.call()

    def test_allocator_cap_preserved(self):
        self.num['allocator_cap_bytes']=12*1024**3
        with self.assertRaises(AssertionError): self.call()

    def test_postprocess_inverse_only_comparator(self):
        changed,_=a.changed_function(POST_SOURCE,'validate_resources',a.POST_OLD,a.POST_NEW)
        self.assertEqual(changed.replace(a.POST_NEW,a.POST_OLD),POST_SOURCE)


class IdentityTests(unittest.TestCase):
    def test_expected_registration_sha_before_read(self):
        with patch.object(owner,'sha',return_value='b'*64), patch.object(owner,'read',side_effect=AssertionError('must not read')):
            with self.assertRaisesRegex(AssertionError,'Supplemental registration'):
                owner.verify_revision('a'*64)

    def test_missing_expected_sha_rejected(self):
        with self.assertRaises(AssertionError): owner.verify_revision(None)

    def test_owner_source_does_not_edit_original_files(self):
        source=Path(owner.__file__).read_text(encoding='utf8')
        self.assertNotIn("OLD_OUT / 'registration.json', reg",source)
        self.assertIn("v6.old.__file__ = str(HERE / 'timing_resource_owner.py')",source)


class ClosedIdentityTests(unittest.TestCase):
    def setUp(self):
        self.expected='a'*64
        def pin(p): return dict(path=str(p),sha256='f'*64,bytes=1)
        self.pin=pin
        self.revision=dict(sources={'frozen':'source'}, original_registration=pin(owner.OLD_OUT/'registration.json'), adapter_proof={'inverse':True})
        ownproc=dict(pid=42,start_ticks='123',argv=['python',str(owner.HERE/'timing_resource_owner.py'),'run','--expected-resource-registration',self.expected])
        workerproc=dict(pid=43,start_ticks='124',argv=['python',str(owner.HERE/'timing_resource_owner.py'),'_worker'])
        def execution(role,process,parent):
            return dict(schema='TIMING_V6_ACTUAL_RESOURCE_EXECUTION_V1',role=role,process=process,parent_process=parent,
                resource_registration=pin(owner.OUT/'registration.json'), original_registration=self.revision['original_registration'],
                sources=self.revision['sources'],original_source_sha256=owner.ORIGINAL_PINS,
                adapter_proof=self.revision['adapter_proof'],actual_startup_free_MiB=8192)
        self.owner=execution('run',ownproc,dict(pid=41,start_ticks='122'))
        self.worker=execution('_worker',workerproc,ownproc)
        def returned(process):
            return dict(status='ORIGINAL_V6_RETURNED_SUCCESS',process=process,
                resource_registration=pin(owner.OUT/'registration.json'), original_completion=pin(owner.OLD_OUT/'completion.json'))
        self.data={
            str(owner.CONTROL/'owner_resource_execution.json'):self.owner,
            str(owner.CONTROL/'worker_resource_execution.json'):self.worker,
            str(owner.CONTROL/'owner_returned.json'):returned(ownproc),
            str(owner.CONTROL/'worker_returned.json'):returned(workerproc),
            str(owner.OLD_OUT/'launch.json'):dict(owner=copy.deepcopy(ownproc)),
            str(owner.OLD_OUT/'worker_launch.json'):dict(pid=43,owner_pid=42),
            str(owner.OLD_OUT/'actual_wait.json'):dict(actual_child_waited=True,exit_code=0,timed_out=False)}
        self.live=None
    def call(self):
        with patch.object(owner,'verify_revision',return_value=self.revision),patch.object(owner,'pin',side_effect=self.pin), \
             patch.object(owner,'read',side_effect=lambda p:self.data[str(p)]),patch.object(owner,'proc',side_effect=lambda pid:self.live), \
             patch.object(Path,'glob',return_value=[]):
            return owner.closed_binding(self.expected)
    def test_complete_closed_identity_passes(self):
        self.assertEqual(self.call()[1],43)
    def test_wrong_actual_owner_argv_rejected(self):
        self.owner['process']['argv'][-1]='b'*64
        with self.assertRaises(AssertionError): self.call()
    def test_wrong_returned_registration_rejected(self):
        self.data[str(owner.CONTROL/'worker_returned.json')]['resource_registration']={'wrong':True}
        with self.assertRaises(AssertionError): self.call()
    def test_still_live_generation_rejected(self):
        self.live=dict(pid=42,start_ticks='123')
        with self.assertRaisesRegex(AssertionError,'still exists'): self.call()
    def test_original_failed_wait_rejected(self):
        self.data[str(owner.OLD_OUT/'actual_wait.json')]['exit_code']=1
        with self.assertRaises(AssertionError): self.call()
    def test_bool_is_not_actual_exit_code(self):
        self.data[str(owner.OLD_OUT/'actual_wait.json')]['exit_code']=False
        with self.assertRaises(AssertionError): self.call()
    def test_wrong_worker_parent_generation_rejected(self):
        self.worker['parent_process']=dict(pid=42,start_ticks='WRONG')
        with self.assertRaises(AssertionError): self.call()


if __name__=='__main__': unittest.main(verbosity=2)

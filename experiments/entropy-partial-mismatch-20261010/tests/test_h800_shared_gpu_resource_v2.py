"""CPU-only fake-device resource ownership and failure-evidence tests."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import h800_shared_gpu_resource_v2 as r


def snapshot(free=4096,util=99):
    return dict(started_unix=r.time.time(),gpu=dict(index=2,free_MiB=free,utilization_percent=util),
        cpu=dict(effective_cpu_cores=2,allowed_cpus=[7,8],host_busy_percent=25,available_memory_bytes=4*r.GiB),
        personal_filesystem_free_bytes=20*r.GiB)


def own(reserved=16*r.GiB,initialized=True):
    return dict(pid=os.getpid(),initialized=initialized,identity_verified=initialized,reserved_bytes=reserved,
        actual_identity={'uuid':'GPU2'},allocated_bytes=reserved,peak_reserved_bytes=reserved)


class FakeCuda:
    initialized=True
    def is_initialized(self):return self.initialized
    def device_count(self):return 1
    def current_device(self):return 0
    def get_device_properties(self,i):return types.SimpleNamespace(uuid='expected')
    def memory_reserved(self,i):return 12*r.GiB
    def memory_allocated(self,i):return 10*r.GiB
    def max_memory_reserved(self,i):return 13*r.GiB


class ResourceTests(unittest.TestCase):
    def test_runtime_can_credit_only_own_reserved(self):
        s,_=r.assess_snapshot(snapshot(),own(),own(),False)
        self.assertEqual(s['resource_guard']['credited_own_reserved_bytes'],16*r.GiB)

    def test_prelaunch_never_credits_allocator(self):
        with self.assertRaises(r.ResourceBusy):r.assess_snapshot(snapshot(4096,10),own(),own(),True)
        s,_=r.assess_snapshot(snapshot(20480,50),own(),own(),True)
        self.assertEqual(s['resource_guard']['credited_own_reserved_bytes'],0)

    def test_prelaunch_utilization_gate_is_unchanged(self):
        with self.assertRaises(r.ResourceBusy):r.assess_snapshot(snapshot(20480,51),own(),own(),True)

    def test_runtime_absolute_floor(self):
        with self.assertRaises(r.ResourceBusy):r.assess_snapshot(snapshot(4095),own(),own(),False)

    def test_runtime_adjusted_margin(self):
        with self.assertRaises(r.ResourceBusy):r.assess_snapshot(snapshot(4096),own(15*r.GiB),own(),False)

    def test_cache_release_uses_lower_observation(self):
        with self.assertRaises(r.ResourceBusy):r.assess_snapshot(snapshot(8192),own(16*r.GiB),own(11*r.GiB),False)

    def test_uninitialized_cuda_has_no_credit(self):
        with self.assertRaises(r.ResourceBusy):r.assess_snapshot(snapshot(),own(0,False),own(),False)
        s,_=r.assess_snapshot(snapshot(20480),own(0,False),own(0,False),False)
        self.assertEqual(s['resource_guard']['credited_own_reserved_bytes'],0)

    def test_pid_identity_must_remain_exact(self):
        x=own();x['pid']+=1
        with self.assertRaises(RuntimeError):r.assess_snapshot(snapshot(),x,own(),False)

    def test_device_identity_must_remain_exact(self):
        x=own();x['actual_identity']={'uuid':'OTHER'}
        with self.assertRaises(RuntimeError):r.assess_snapshot(snapshot(),x,own(),False)

    def test_wrong_gpu_is_not_admitted(self):
        s=snapshot(20480);s['gpu']['index']=0
        with self.assertRaises(RuntimeError):r.assess_snapshot(s,own(),own(),False)

    def test_all_cpu_host_disk_gates_preserved(self):
        for field,value in [('effective_cpu_cores',1),('allowed_cpus',[7]),('host_busy_percent',96),
            ('host_busy_percent',float('nan')),('available_memory_bytes',2*r.GiB-1)]:
            with self.subTest(field=field,value=value):
                s=snapshot(20480);s['cpu'][field]=value
                with self.assertRaises(r.ResourceBusy):r.assess_snapshot(s,own(),own(),False)
        s=snapshot(20480);s['personal_filesystem_free_bytes']=10*r.GiB-1
        with self.assertRaises(r.ResourceBusy):r.assess_snapshot(s,own(),own(),False)

    def fake_device(self):
        return types.SimpleNamespace(index=2,creator_pid=os.getpid(),
            check_cuda_environment=lambda:None,
            validate_cuda_device_identity=lambda uuid,pci:{'uuid':uuid,'pci':pci})

    def test_same_process_observation_reads_live_local_allocator(self):
        with patch.object(r.original_math,'cuda_driver_pci_bus_id',return_value='PCI2'):
            p=r.own_allocator_observation(self.fake_device(),types.SimpleNamespace(cuda=FakeCuda()))
        self.assertTrue(p['identity_verified']);self.assertEqual(p['reserved_bytes'],12*r.GiB)

    def test_absent_torch_never_imports_or_initializes(self):
        with patch.dict(sys.modules,{'torch':None}):
            p=r.own_allocator_observation(self.fake_device())
        self.assertFalse(p['initialized']);self.assertEqual(p['reserved_bytes'],0)

    def test_allocator_peak_cap_cannot_be_hidden(self):
        cuda=FakeCuda();cuda.max_memory_reserved=lambda i:17*r.GiB
        with patch.object(r.original_math,'cuda_driver_pci_bus_id',return_value='PCI2'):
            with self.assertRaises(RuntimeError):r.own_allocator_observation(self.fake_device(),types.SimpleNamespace(cuda=cuda))

    def test_observer_rejects_parent_process_adapter(self):
        device=self.fake_device();device.creator_pid+=1
        with self.assertRaises(RuntimeError):r.own_allocator_observation(device,types.SimpleNamespace(cuda=FakeCuda()))

    def test_observer_rejects_wrong_cuda_mask_and_actual_device(self):
        device=self.fake_device()
        def reject(*unused):raise RuntimeError('wrong physical CUDA device')
        device.check_cuda_environment=reject
        with self.assertRaises(RuntimeError):r.own_allocator_observation(device,types.SimpleNamespace(cuda=FakeCuda()))
        device=self.fake_device();device.validate_cuda_device_identity=reject
        with patch.object(r.original_math,'cuda_driver_pci_bus_id',return_value='WRONG'):
            with self.assertRaises(RuntimeError):r.own_allocator_observation(device,types.SimpleNamespace(cuda=FakeCuda()))

    def test_observer_rejects_multiple_visible_devices_or_wrong_logical_device(self):
        for name,value in [('device_count',2),('current_device',1)]:
            cuda=FakeCuda();setattr(cuda,name,lambda:value)
            with self.subTest(name=name),self.assertRaises(RuntimeError):
                r.own_allocator_observation(self.fake_device(),types.SimpleNamespace(cuda=cuda))

    def test_observer_rejects_invalid_allocator_values(self):
        for name,value in [('memory_reserved',True),('memory_reserved',-1),('memory_allocated',13*r.GiB)]:
            cuda=FakeCuda();setattr(cuda,name,lambda unused:value)
            with self.subTest(name=name,value=value),patch.object(r.original_math,'cuda_driver_pci_bus_id',return_value='PCI2'):
                with self.assertRaises(RuntimeError):r.own_allocator_observation(self.fake_device(),types.SimpleNamespace(cuda=cuda))

    def test_explicit_uninitialized_torch_never_queries_device(self):
        cuda=types.SimpleNamespace(is_initialized=lambda:False)
        observed=r.own_allocator_observation(self.fake_device(),types.SimpleNamespace(cuda=cuda))
        self.assertFalse(observed['initialized']);self.assertEqual(observed['reserved_bytes'],0)

    def test_failure_snapshot_is_persisted_when_old_gpu_gate_fails(self):
        with tempfile.TemporaryDirectory() as td:
            device=object.__new__(r.DeviceAdapter);device.index=2;device.creator_pid=os.getpid()
            device.evidence_directory=Path(td);device.request_sha='a'*64
            old=r.ResourceBusy('old margin');old.resource_snapshot=snapshot(4095)
            with patch.object(r,'own_allocator_observation',return_value=own()),patch.object(
                r.host.original_rx.DeviceAdapter,'resource_snapshot',side_effect=old):
                with self.assertRaises(r.ResourceBusy):device.resource_snapshot(object())
            receipts=list((Path(td)/'resource_guard_failures').glob('*.json'));self.assertEqual(len(receipts),1)
            j=json.loads(receipts[0].read_text());self.assertEqual(j['snapshot']['gpu']['free_MiB'],4095)
            self.assertEqual(j['request_sha256'],'a'*64)

    def test_old_gpu_margin_can_be_corrected_without_bypassing_cpu_gate(self):
        with tempfile.TemporaryDirectory() as td:
            device=object.__new__(r.DeviceAdapter);device.index=2;device.creator_pid=os.getpid()
            device.evidence_directory=Path(td);device.request_sha='a'*64
            old=r.ResourceBusy('old margin');old.resource_snapshot=snapshot()
            with patch.object(r,'own_allocator_observation',return_value=own()),patch.object(
                r.host.original_rx.DeviceAdapter,'resource_snapshot',side_effect=old):
                actual,ad=device.resource_snapshot(object())
            self.assertEqual(ad['cpu_affinity'],[7,8]);self.assertEqual(actual['resource_guard']['credited_own_reserved_bytes'],16*r.GiB)

    def test_collection_failure_still_has_failure_receipt(self):
        with tempfile.TemporaryDirectory() as td:
            device=object.__new__(r.DeviceAdapter);device.index=2;device.creator_pid=os.getpid()
            device.evidence_directory=Path(td);device.request_sha='b'*64
            with patch.object(r,'own_allocator_observation',return_value=own()),patch.object(
                r.host.original_rx.DeviceAdapter,'resource_snapshot',side_effect=RuntimeError('NVML unavailable')):
                with self.assertRaises(RuntimeError):device.resource_snapshot(object())
            self.assertEqual(len(list((Path(td)/'resource_guard_failures').glob('*.json'))),1)

    def test_stale_snapshot_is_not_admitted_and_is_preserved(self):
        with tempfile.TemporaryDirectory() as td:
            device=object.__new__(r.DeviceAdapter);device.index=2;device.creator_pid=os.getpid()
            device.evidence_directory=Path(td);device.request_sha='c'*64
            old=r.ResourceBusy('old margin');old.resource_snapshot=snapshot();old.resource_snapshot['started_unix']-=20
            with patch.object(r,'own_allocator_observation',return_value=own()),patch.object(
                r.host.original_rx.DeviceAdapter,'resource_snapshot',side_effect=old):
                with self.assertRaises(RuntimeError):device.resource_snapshot(object())
            self.assertEqual(len(list((Path(td)/'resource_guard_failures').glob('*.json'))),1)

    def test_both_new_engines_bind_only_new_adapter(self):
        import h800_ep_full_visual_v2 as visual
        import h800_ep_full_metric_owner_v2 as metric
        for engine in [visual.engine(),metric.engine()]:
            self.assertIs(engine.device_from_request.__globals__['DeviceAdapter'],r.DeviceAdapter)

    def test_scientific_providers_caps_and_methods_are_unchanged(self):
        import ast
        import h800_ep_full_visual_v1 as visual1
        import h800_ep_full_visual_v2 as visual2
        import h800_ep_full_metric_owner_v1 as metric1
        import h800_ep_full_metric_owner_v2 as metric2
        self.assertEqual(visual1.CAPS,visual2.CAPS);self.assertEqual(metric1.CAPS,metric2.CAPS)
        for old,new,names in [(visual1,visual2,['cpu_closure','reusable_pilot','images']),
            (metric1,metric2,['pilot_pair_certificate','scores'])]:
            def nodes(module):
                return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(Path(module.__file__).read_text()).body if isinstance(n,ast.FunctionDef)}
            a,b=nodes(old),nodes(new)
            for name in names:self.assertEqual(a[name],b[name])


if __name__=='__main__':unittest.main()

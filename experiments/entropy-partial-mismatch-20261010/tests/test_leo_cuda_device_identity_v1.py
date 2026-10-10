"""CPU-only mocked identity checks; no CUDA driver, tensor, or model call."""
import ctypes
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import leo_whole_gate_v1 as gate
import leo_whole_math_v1 as math_adapter


class Function:
    def __init__(self,operation):self.operation=operation
    def __call__(self,*args):return self.operation(*args)


def fake_driver(device_code=0,pci_code=0,pci=b'0000:A2:00.0'):
    calls=[]
    def device(pointer,ordinal):
        calls.append(('device',ordinal));ctypes.cast(pointer,ctypes.POINTER(ctypes.c_int))[0]=7
        return device_code
    def bus(buffer,length,device):
        calls.append(('pci',length,device));buffer.value=pci
        return pci_code
    return SimpleNamespace(cuDeviceGet=Function(device),cuDeviceGetPCIBusId=Function(bus)),calls


class Tests(unittest.TestCase):
    def test_pci_domain_width_and_case_are_equivalent(self):
        for value in ('00000000:A2:00.0','0000:a2:00.0',' 0000:A2:00.0 '):
            self.assertEqual(gate.normalize_pci_bus_id(value),gate.PCI_BUS_ID)
        for value in ('A2:00.0','000:a2:00.0','0000:a2:00.8','0000:a2:20.0','0000:a2:00.0\nextra'):
            with self.subTest(value=value),self.assertRaises(RuntimeError):gate.normalize_pci_bus_id(value)

    def test_selected_nvml_query_is_one_read_only_device(self):
        runner=mock.Mock(return_value=SimpleNamespace(stdout=f'5, {gate.UUID}, 00000000:A2:00.0\n'))
        receipt=gate.query_nvml_device_identity(runner)
        self.assertEqual(runner.call_count,1)
        argv=runner.call_args.args[0]
        self.assertEqual(argv,['nvidia-smi','--id='+gate.UUID,'--query-gpu=index,uuid,pci.bus_id','--format=csv,noheader,nounits'])
        self.assertEqual(runner.call_args.kwargs['timeout'],10)
        self.assertEqual(receipt['expected_cuda_uuid'],gate.CUDA_UUID)
        self.assertTrue(receipt['cuda_identity_not_yet_observed'])
        self.assertEqual(receipt['device_setting_changes'],0)

    def test_wrong_nvml_index_uuid_pci_and_extra_devices_stop(self):
        good=f'5, {gate.UUID}, 00000000:A2:00.0'
        for output in (good.replace('5,','4,'),good.replace(gate.UUID,gate.CUDA_UUID),
                       good.replace('A2','A3'),good+'\n'+good,''):
            with self.subTest(output=output),self.assertRaises(RuntimeError):
                gate.query_nvml_device_identity(mock.Mock(return_value=SimpleNamespace(stdout=output)))

    def test_cuda_uuid_and_actual_pci_both_required(self):
        actual=gate.validate_cuda_device_identity(gate.CUDA_UUID.upper(),'00000000:A2:00.0')
        self.assertEqual(actual['actual_cuda_uuid'],gate.CUDA_UUID)
        self.assertEqual(actual['configured_nvml_uuid'],gate.UUID)
        self.assertNotEqual(gate.UUID,gate.CUDA_UUID)
        for uuid,pci in ((gate.UUID,gate.PCI_BUS_ID),(gate.CUDA_UUID,'0000:a3:00.0'),
                         (gate.CUDA_UUID,'0001:a2:00.0'),('not-a-uuid',gate.PCI_BUS_ID)):
            with self.subTest(uuid=uuid,pci=pci),self.assertRaises(RuntimeError):gate.validate_cuda_device_identity(uuid,pci)

    def test_environment_requires_admitted_ordinal_and_rejects_uuid_masks(self):
        env=gate.cuda_environment_binding();gate.check_cuda_environment(env)
        self.assertEqual(env,dict(CUDA_VISIBLE_DEVICES='5',CUDA_DEVICE_ORDER='PCI_BUS_ID'))
        for changed in ({'CUDA_VISIBLE_DEVICES':gate.UUID},{'CUDA_VISIBLE_DEVICES':gate.CUDA_UUID},
                        {'CUDA_VISIBLE_DEVICES':'0'},{'CUDA_VISIBLE_DEVICES':'5,0'},{'CUDA_DEVICE_ORDER':'FASTEST_FIRST'},
                        {'CUDA_DEVICE_ORDER':None}):
            with self.subTest(changed=changed),self.assertRaises(RuntimeError):gate.check_cuda_environment(env|changed)

    def test_driver_queries_logical_zero_and_returned_handle_no_gpu_runtime(self):
        driver,calls=fake_driver()
        self.assertEqual(math_adapter.cuda_driver_pci_bus_id(driver),'0000:A2:00.0')
        self.assertEqual(calls,[('device',0),('pci',64,7)])
        self.assertEqual(driver.cuDeviceGet.restype,ctypes.c_int)
        self.assertEqual(driver.cuDeviceGetPCIBusId.restype,ctypes.c_int)
        for device_code,pci_code in ((100,0),(0,101)):
            driver,calls=fake_driver(device_code,pci_code)
            with self.assertRaises(RuntimeError):math_adapter.cuda_driver_pci_bus_id(driver)
            self.assertEqual(len(calls),1 if device_code else 2)

    def test_old_request_cannot_be_silently_replayed(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'request.json'
            value=dict(schema=gate.SCHEMA,status='REGISTERED_NOT_EXECUTED',tool_bindings={})
            p.write_text(json.dumps(value),encoding='utf-8')
            with self.assertRaisesRegex(RuntimeError,'Fresh dual-identity registration'):
                gate.prepared(p,hashlib.sha256(p.read_bytes()).hexdigest())
            value['device_identity_binding']=gate.device_identity_binding()
            p.write_text(json.dumps(value),encoding='utf-8')
            self.assertEqual(gate.prepared(p,hashlib.sha256(p.read_bytes()).hexdigest()),value)
            # The earlier CUDA-UUID request had neither explicit mask field.
            for key in gate.cuda_environment_binding():value['device_identity_binding'].pop(key)
            p.write_text(json.dumps(value),encoding='utf-8')
            with self.assertRaisesRegex(RuntimeError,'Fresh dual-identity registration'):
                gate.prepared(p,hashlib.sha256(p.read_bytes()).hexdigest())


if __name__=='__main__':unittest.main(verbosity=2)

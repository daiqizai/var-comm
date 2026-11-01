"""Verify the recorded CUDA/NVML identity pair through their shared physical PCI address."""
from followup_common import *
import ctypes, subprocess


def normalized_pci(value):
    domain, bus, slot = value.lower().split(':')
    device, function = slot.split('.')
    return int(domain,16),int(bus,16),int(device,16),int(function,16)


def verify_device(torch, request):
    expected = cr(request['device_mapping'])
    lib = ctypes.CDLL('libcuda.so.1')
    assert lib.cuInit(0) == 0
    buf = ctypes.create_string_buffer(64)
    assert lib.cuDeviceGetPCIBusId(buf,64,0) == 0
    pci = buf.value.decode()
    rows = [v.split(', ') for v in subprocess.check_output(['nvidia-smi',
        '--query-gpu=uuid,pci.bus_id','--format=csv,noheader,nounits'],text=True).splitlines()]
    found = [v for v in rows if v[0] == request['gpu_uuid']]
    assert len(found) == 1
    assert normalized_pci(pci) == normalized_pci(found[0][1]) == normalized_pci(request['pci']) == normalized_pci(expected['pci'])
    actual = str(torch.cuda.get_device_properties(0).uuid)
    assert actual == expected['CUDA_UUID'] and request['gpu_uuid'] == expected['NVML_UUID']
    save(Path(request['output'])/'device_mapping_verified.json',dict(status='MATCHED_PHYSICAL_PCI',
         CUDA_UUID=actual,NVML_UUID=found[0][0],driver_PCI=pci,NVML_PCI=found[0][1],
         registered_PCI=request['pci'],mapping=request['device_mapping']))

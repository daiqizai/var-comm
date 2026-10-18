"""Validate the known NVML/CUDA identity mapping, then run frozen native code."""
import ctypes,json,os,pathlib,runpy,sys
E=pathlib.Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/arpc-external-baseline-20261019')

def check_mapping(lib,expected,visible):
    assert visible==str(expected['visible_index']),'Unexpected CUDA mask'
    def ok(rc):assert rc==0,('CUDA driver error',rc)
    ok(lib.cuInit(0));device=ctypes.c_int();ok(lib.cuDeviceGet(ctypes.byref(device),0))
    pci=ctypes.create_string_buffer(64);ok(lib.cuDeviceGetPCIBusId(pci,64,device))
    uuid=(ctypes.c_ubyte*16)();ok(lib.cuDeviceGetUuid(ctypes.byref(uuid),device))
    actual={'pci_bus_id':pci.value.decode().lower(),'cuda_uuid_hex':bytes(uuid).hex(),'visible_index':expected['visible_index']}
    assert actual['pci_bus_id']==expected['pci_bus_id'].lower(),'Physical device changed'
    assert actual['cuda_uuid_hex']==expected['cuda_uuid_hex'],'CUDA device UUID changed'
    return actual

def main():
    assert sys.argv[1] in ['tx','rx']
    i=sys.argv.index('--public');cfg=json.loads(pathlib.Path(sys.argv[i+1]).read_text())
    actual=check_mapping(ctypes.CDLL('libcuda.so.1'),cfg['cuda_admission'],os.environ.get('CUDA_VISIBLE_DEVICES'))
    p=pathlib.Path(cfg['run'])/(sys.argv[1]+'.cuda_mapping.json')
    with p.open('x') as f:json.dump({'actual':actual,'expected':cfg['cuda_admission'],'pid':os.getpid(),'model_calls':0},f,indent=2)
    sys.argv[0]=str(E/'a1_native.py')
    runpy.run_path(sys.argv[0],run_name='__main__')

if __name__=='__main__':main()

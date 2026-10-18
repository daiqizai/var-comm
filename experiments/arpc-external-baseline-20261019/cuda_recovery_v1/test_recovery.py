"""Synthetic checks for the new mapping and bounded dispatch only."""
import ctypes,unittest
import runtime,recovery

class Driver:
    def __init__(self,pci='0000:A1:00.0',uuid='e3133767ae42afba35150873255753fc',rc=0):self.pci=pci;self.uuid=uuid;self.rc=rc
    def cuInit(self,x):return self.rc
    def cuDeviceGet(self,pointer,index):ctypes.cast(pointer,ctypes.POINTER(ctypes.c_int))[0]=0;return 0
    def cuDeviceGetPCIBusId(self,buf,size,device):buf.value=self.pci.encode();return 0
    def cuDeviceGetUuid(self,pointer,device):ctypes.memmove(pointer,bytes.fromhex(self.uuid),16);return 0

class Mapping(unittest.TestCase):
    def test_exact(self):self.assertEqual(runtime.check_mapping(Driver(),recovery.MAPPING,'4')['cuda_uuid_hex'],recovery.MAPPING['cuda_uuid_hex'])
    def test_wrong_mask(self):
        with self.assertRaises(AssertionError):runtime.check_mapping(Driver(),recovery.MAPPING,recovery.MAPPING['nvml_uuid'])
    def test_wrong_pci(self):
        with self.assertRaises(AssertionError):runtime.check_mapping(Driver(pci='0000:A2:00.0'),recovery.MAPPING,'4')
    def test_wrong_uuid(self):
        with self.assertRaises(AssertionError):runtime.check_mapping(Driver(uuid='00'*16),recovery.MAPPING,'4')
    def test_driver_failure(self):
        with self.assertRaises(AssertionError):runtime.check_mapping(Driver(rc=100),recovery.MAPPING,'4')

class Dispatch(unittest.TestCase):
    def stage(self,name,remaining=100,seconds=3600):
        argv=['/usr/bin/python3','-B',str(recovery.E/('a1_audit.py' if name=='audit' else 'a1_native.py'))]
        env={'CUDA_VISIBLE_DEVICES':'' if name=='audit' else recovery.MAPPING['nvml_uuid'],'OTHER':'unchanged'}
        return recovery.adjusted_stage(name,argv,env,seconds,lambda *args:args,lambda:recovery.ABSOLUTE_DEADLINE-remaining),argv,env
    def test_tx(self):
        (name,argv,env,seconds),oldargv,oldenv=self.stage('tx')
        self.assertEqual(argv[2],str(recovery.HERE/'runtime.py'));self.assertEqual(env,{'CUDA_VISIBLE_DEVICES':'4','OTHER':'unchanged'})
        self.assertEqual(oldargv[2],str(recovery.E/'a1_native.py'));self.assertEqual(oldenv['CUDA_VISIBLE_DEVICES'],recovery.MAPPING['nvml_uuid'])
    def test_rx(self):self.assertEqual(self.stage('rx')[0][2]['CUDA_VISIBLE_DEVICES'],'4')
    def test_audit_unchanged(self):
        actual,argv,env=self.stage('audit');self.assertEqual(actual[1],argv);self.assertEqual(actual[2],env)
    def test_deadline_clamp(self):self.assertEqual(self.stage('tx',remaining=35)[0][3],35)
    def test_stage_clamp(self):self.assertEqual(self.stage('rx',remaining=500,seconds=100)[0][3],100)
    def test_expired(self):
        with self.assertRaises(AssertionError):self.stage('rx',remaining=0)
    def test_bad_stage(self):
        with self.assertRaises(AssertionError):self.stage('other')
    def test_bad_native_entry(self):
        with self.assertRaises(AssertionError):recovery.adjusted_stage('tx',['python','-B','other.py'],{'CUDA_VISIBLE_DEVICES':recovery.MAPPING['nvml_uuid']},10,lambda *a:a,lambda:0)
    def test_bad_admission_mask(self):
        with self.assertRaises(AssertionError):recovery.adjusted_stage('rx',['python','-B',str(recovery.E/'a1_native.py')],{'CUDA_VISIBLE_DEVICES':'5'},10,lambda *a:a,lambda:0)

if __name__=='__main__':unittest.main(verbosity=2)

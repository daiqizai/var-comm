import copy,unittest
from migration import check_registration,check_migrated_identity
from contracts import PRO_HOST,RUNTIME_KEYS

class MigrationTests(unittest.TestCase):
    def setUp(self):
        self.r=dict(required_host=PRO_HOST,calibration_only=True,source_reference_calls=1000,
            cross_device_feature_reuse=False,cross_device_scalar_reuse=False,no_formal_data=True,
            original_registration={'sha256':'ef5dd3f162bea7fb226a4b0d6b8be465b91d783a5807f7d028252d2198635bf5'},
            caps=dict(model_load=1,source_reference_dino=1000,reconstruction_dino=5061),max_seconds=14400,
            budget={'GPU_closed_seconds':44777.58979978971},cuda_admission={'pci':'pro'})
        self.metadata={'metrics':['dinov2_vitl14'],'metric_selection_usage':'new','weight':'fixed','transform':'fixed'}
        self.runtime={k:'fixed'for k in RUNTIME_KEYS};self.runtime.update(host=PRO_HOST,device=self.r['cuda_admission'])
        self.previous=copy.deepcopy(self.runtime);self.previous.update(host='H800',device={'pci':'old'},DINO_L_metadata=copy.deepcopy(self.metadata))
        self.previous['DINO_L_metadata']['metric_selection_usage']='old'
    def test_registered_hardware_only(self):
        self.assertTrue(check_migrated_identity(self.previous,self.metadata,self.runtime,self.r))
    def test_foreign_cache_rejected(self):
        for field in ('cross_device_feature_reuse','cross_device_scalar_reuse'):
            r=copy.deepcopy(self.r);r[field]=True
            with self.assertRaises(AssertionError):check_registration(r,PRO_HOST)
    def test_model_transform_mismatch_rejected(self):
        for field in ('weight','transform','metrics'):
            m=copy.deepcopy(self.metadata);m[field]='changed'
            with self.assertRaises(AssertionError):check_migrated_identity(self.previous,m,self.runtime,self.r)
    def test_other_runtime_change_rejected(self):
        for field in set(RUNTIME_KEYS)-{'host','device'}:
            rt=copy.deepcopy(self.runtime);rt[field]='changed'
            with self.assertRaises(AssertionError):check_migrated_identity(self.previous,self.metadata,rt,self.r)
    def test_wrong_device_host_rejected(self):
        for field in ('host','device'):
            rt=copy.deepcopy(self.runtime);rt[field]='wrong'
            with self.assertRaises(AssertionError):check_migrated_identity(self.previous,self.metadata,rt,self.r)
    def test_old_reference_caps_rejected(self):
        r=copy.deepcopy(self.r);r['source_reference_calls']=0
        with self.assertRaises(AssertionError):check_registration(r,PRO_HOST)
    def test_budget_rejected(self):
        r=copy.deepcopy(self.r);r['budget']['GPU_closed_seconds']=96*3600
        with self.assertRaises(AssertionError):check_registration(r,PRO_HOST)
    def test_population_cap_rejected(self):
        r=copy.deepcopy(self.r);r['caps']['reconstruction_dino']=5062
        with self.assertRaises(AssertionError):check_registration(r,PRO_HOST)

if __name__=='__main__':unittest.main()

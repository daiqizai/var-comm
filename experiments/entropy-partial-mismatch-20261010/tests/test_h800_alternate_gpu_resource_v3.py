import copy,os,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import h800_alternate_gpu_resource_v3 as r


def snapshot(index=3,util=61,free=21017):
    return dict(gpu=dict(index=index,utilization_percent=util,free_MiB=free),cpu=dict(effective_cpu_cores=2,
        allowed_cpus=[7,8],host_busy_percent=50,available_memory_bytes=3*r.GiB),personal_filesystem_free_bytes=11*r.GiB)
def own(n=0):return dict(initialized=bool(n),pid=os.getpid(),identity_verified=True,actual_identity='exact-current-device',reserved_bytes=n*r.GiB)


class ResourceTests(unittest.TestCase):
    def test_GPU3_explicit_shared61_allowed_but80_rejected(self):
        r.assess_snapshot(snapshot(),own(),own(),True)
        with self.assertRaises(r.ResourceBusy):r.assess_snapshot(snapshot(util=80),own(),own(),True)
    def test_other_H800_retains50(self):
        for index in (0,1,2):
            r.assess_snapshot(snapshot(index,50),own(),own(),True)
            with self.assertRaises(r.ResourceBusy):r.assess_snapshot(snapshot(index,51),own(),own(),True)
    def test_original_memory_margins_and_samePID_credit(self):
        with self.assertRaises(r.ResourceBusy):r.assess_snapshot(snapshot(free=20479),own(),own(),True)
        s,_=r.assess_snapshot(snapshot(util=99,free=8192),own(12),own(13),False)
        self.assertEqual(s['resource_guard']['credited_own_reserved_bytes'],12*r.GiB)
        with self.assertRaises(r.ResourceBusy):r.assess_snapshot(snapshot(free=4095),own(16),own(16),False)
        bad=own(12);bad['pid']=-1
        with self.assertRaises(RuntimeError):r.assess_snapshot(snapshot(free=8192),bad,own(12),False)
    def test_no_external_credit_when_uninitialized(self):
        with self.assertRaises(r.ResourceBusy):r.assess_snapshot(snapshot(free=8192),own(),own(),False)
        self.assertEqual(r.policy(3)['own_allocator_cap_bytes'],16*r.GiB)


if __name__=='__main__':unittest.main()

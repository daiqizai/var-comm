"""Synthetic grid integrity and exact frozen numeric-call comparison."""
import ast,copy,pathlib,sys,unittest
from contracts import validate_plan,validate_closure,validate_metric_identity,POPULATIONS,POLICY_SHA,METRICS


def fixture(pop):
    count,noises=POPULATIONS[pop]
    sources=[dict(source_id=pop+'/'+str(i),source_index=i,reference_sha256=('%064x'%(i+1)))for i in range(count)]
    frames=[dict(source_id=s['source_id'],source_index=s['source_index'],population=pop,
      snr_db=snr,noise_seed=n,method='ADA_PREDCLASS_FIXEDFRAME',reference_sha256=s['reference_sha256'],
      pixel_sha256='f'*64,received_prefix=32,reuse_old=False,accepted=True,image={'synthetic':True},
      source_status='ACCEPTED_SOURCE',old_values={})for s in sources for snr in(4,10,19)for n in noises]
    return dict(population=pop,frozen_policy={'sha256':POLICY_SHA},sources=sources,frames=frames)


class FormalMetrics(unittest.TestCase):
    def test_full_common500_grid(self):self.assertEqual(validate_plan(fixture('common500'),'common500'),4500)
    def test_full_kodak_grid(self):self.assertEqual(validate_plan(fixture('kodak24'),'kodak24'),216)
    def test_duplicate_or_missing_event(self):
        p=fixture('kodak24');p['frames'][-1]=copy.deepcopy(p['frames'][0])
        with self.assertRaises(AssertionError):validate_plan(p,'kodak24')
        p=fixture('kodak24');p['frames'].pop()
        with self.assertRaises(AssertionError):validate_plan(p,'kodak24')
    def test_wrong_noise_or_policy(self):
        p=fixture('common500');p['frames'][0]['noise_seed']=4101
        with self.assertRaises(AssertionError):validate_plan(p,'common500')
        p=fixture('kodak24');p['frozen_policy']['sha256']='0'*64
        with self.assertRaises(AssertionError):validate_plan(p,'kodak24')
    def test_source_identity_mismatch(self):
        for field,value in(('source_id','wrong'),('reference_sha256','0'*64),('method','another')):
            p=fixture('kodak24');p['frames'][0][field]=value
            with self.assertRaises(AssertionError):validate_plan(p,'kodak24')
    def test_gray_keeps_denominator(self):
        p=fixture('kodak24');j=p['frames'][0]
        j.update(image=None,source_status='FIXED_GRAY_PHY_REJECT',accepted=False,received_prefix=0)
        self.assertEqual(validate_plan(p,'kodak24'),216)
        j['accepted']=True
        with self.assertRaises(AssertionError):validate_plan(p,'kodak24')
    def test_reuse_requires_all_metrics_and_identity(self):
        p=fixture('kodak24');j=p['frames'][0]
        j.update(reuse_old=True,old_values={m:0.25 for m in METRICS},old_metric_identity_verified=True)
        self.assertEqual(validate_plan(p,'kodak24'),216)
        j['old_metric_identity_verified']=False
        with self.assertRaises(AssertionError):validate_plan(p,'kodak24')
        j['old_metric_identity_verified']=True;j['old_values'].pop('dreamsim')
        with self.assertRaises(AssertionError):validate_plan(p,'kodak24')
    def test_populations_remain_separate(self):
        with self.assertRaises(AssertionError):validate_plan(fixture('common500'),'kodak24')
    def test_source_decode_reject_keeps_denominator(self):
        p=fixture('kodak24');p['frames'][0].update(source_status='FIXED_GRAY_SOURCE_REJECT',accepted=False)
        self.assertEqual(validate_plan(p,'kodak24'),216)
        p['frames'][0]['image']=None
        with self.assertRaises(AssertionError):validate_plan(p,'kodak24')
    def test_nonfinite_or_boolean_cached_scores_rejected(self):
        for value in (float('nan'),float('inf'),True):
            p=fixture('kodak24');j=p['frames'][0]
            j.update(reuse_old=True,old_values={m:0.25 for m in METRICS},old_metric_identity_verified=True)
            j['old_values']['dreamsim']=value
            with self.assertRaises(AssertionError):validate_plan(p,'kodak24')
    def test_closed_science_and_audit_required(self):
        w=dict(actual_wait=True,returncode=0,process={'pid':1},parent_process={'pid':2},metadata_error=None)
        c=dict(status='COMPLETE',ledger={'unresolved':0})
        a=dict(status='PASS',violations=[],process={'pid':3},parent_process={'pid':4})
        validate_closure(w,c,lambda p:True,a)
        for pid in (1,2,3,4):
            with self.assertRaises(AssertionError):validate_closure(w,c,lambda p:p['pid']!=pid,a)
        for field,value in (('returncode',False),('returncode',1),('actual_wait',False),('timed_out',True),('terminated',True),('killed',True),('metadata_error','error')):
            bad=dict(w);bad[field]=value
            with self.assertRaises(AssertionError):validate_closure(bad,c,lambda p:True,a)
    def test_unresolved_failed_completion_or_audit_rejected(self):
        w=dict(actual_wait=True,returncode=0,process={},parent_process={})
        for c in (dict(status='RUNNING'),dict(status='COMPLETE',ledger={'unresolved':1}),dict(status='COMPLETE_PHY_AWAITING_SOURCE_RX',packet_ledger={'unresolved':1})):
            with self.assertRaises(AssertionError):validate_closure(w,c,lambda p:True)
        for a in (dict(status='FAIL',violations=[],process={},parent_process={}),dict(status='PASS',violations=['mismatch'],process={},parent_process={})):
            with self.assertRaises(AssertionError):validate_closure(w,dict(status='COMPLETE'),lambda p:True,a)
    def test_cache_requires_exact_metric_and_device(self):
        cfg={'path':'synthetic/config','sha256':'a'*64};loader={'path':'synthetic/loader','sha256':'b'*64}
        rr=dict(torch='same-torch',cuda='same-cuda',pci='0000:A3:00.0')
        identity=dict(config=cfg,loader=loader,torch=rr['torch'],cuda=rr['cuda'],pci=rr['pci'].lower(),fp32=True,tf32=False,batch=1,threads=6,interop=2)
        validate_metric_identity(identity,rr,cfg,loader)
        for field,value in (('torch','different'),('cuda','different'),('pci','0000:63:00.0'),('fp32',False),('tf32',True),('batch',2),('threads',1),('interop',1),('config',{}),('loader',{})):
            bad=copy.deepcopy(identity);bad[field]=value
            with self.assertRaises(AssertionError):validate_metric_identity(bad,rr,cfg,loader)
    def test_uncached_clip_source_uses_same_feature_path(self):
        root=pathlib.Path(__file__).resolve().parent;tree=ast.parse((root/'worker.py').read_bytes())
        branches=[n for n in ast.walk(tree)if isinstance(n,ast.If)and ast.unparse(n.test)=="s['source_id'] in refs"]
        self.assertEqual(len(branches),1)
        self.assertEqual(ast.unparse(branches[0].orelse[0]),"fx = feature(x, s['reference_sha256'])")
    def test_original_numeric_calls_unchanged(self):
        root=pathlib.Path(__file__).resolve().parent
        old=ast.parse((root/'a_metrics_reference.py').read_bytes());new=ast.parse((root/'worker.py').read_bytes())
        selected={'psnr','lp_pair','network.embed','network.encode_image','met.load_clip','met.load_dreamsim',
         'met.preprocess_dreamsim','met.preprocess_clip','met.validate_images','F.cosine_similarity',
         'four.psnr_projection','four.lpips_projection','four.bound_assets'}
        def calls(tree):
            return sorted(ast.dump(n,include_attributes=False)for n in ast.walk(tree)
              if isinstance(n,ast.Call)and ast.unparse(n.func)in selected)
        self.assertEqual(calls(old),calls(new));self.assertGreater(len(calls(new)),10)
    def test_no_model_import_for_contracts(self):
        self.assertFalse(any(x.split('.')[0]in('torch','torchvision','tensorflow','transformers','diffusers','timm')for x in sys.modules))


if __name__=='__main__':unittest.main()

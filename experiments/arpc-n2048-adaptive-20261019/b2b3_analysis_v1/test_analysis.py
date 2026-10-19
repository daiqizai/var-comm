"""Synthetic CPU qualification only; production prepare/run functions are never called."""
import ast, builtins, copy, hashlib, importlib.abc, json, math, os, sys, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import analysis_common as c
import native
import dino_worker
import score

ORIGINAL=(Path(__file__).resolve().parent.parent/'cal100_dino.py').read_bytes()
FIXTURE=Path(tempfile.mkdtemp(prefix='b2b3_analysis_synthetic_')).resolve()
CODE_ALLOWED={p.resolve() for p in Path(__file__).resolve().parent.glob('*.py')}
class NoModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','tensorflow','sionna','transformers','lpips','PIL'}:
            raise AssertionError('Synthetic qualification attempted model import: '+fullname)
sys.meta_path.insert(0,NoModels())
def guard(event,args):
    if event=='subprocess.Popen':raise AssertionError('No subprocess/science in synthetic tests')
    if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
        p=Path(os.fsdecode(args[0])).resolve()
        if str(p).startswith('/mnt/') and not p.is_relative_to(FIXTURE) and p not in CODE_ALLOWED:
            raise AssertionError('No production PFS assets in synthetic tests: '+str(p))
sys.addaudithook(guard)

def fixture(name):
    p=FIXTURE/name;p.parent.mkdir(parents=True,exist_ok=True);return p

def job(data=b'\x80\x00',codec=0,endpoint=4,length=9):
    p=fixture('wire_'+hashlib.sha256(data+bytes([codec,endpoint,length])).hexdigest()+'.bin')
    if not p.exists():p.write_bytes(data)
    return dict(key=c.protocol.input_key(codec,endpoint,length,data),codec=codec,endpoint=endpoint,
        bit_length=length,input=c.pin(p),generation_seed=0)

class IdentityTests(unittest.TestCase):
    def test_actual_input_key(self):
        j=job();self.assertEqual(c.exact_key(j),(0,4,9,hashlib.sha256(b'\x80\x00').hexdigest()))
    def test_endpoint_is_part_of_identity(self):
        self.assertNotEqual(c.exact_key(job(endpoint=4)),c.exact_key(job(endpoint=5)))
    def test_codec_is_part_of_identity(self):
        self.assertNotEqual(c.exact_key(job()),c.exact_key(job(codec=1,length=16)))
    def test_length_is_part_of_identity(self):
        self.assertNotEqual(c.exact_key(job(length=9)),c.exact_key(job(length=16)))
    def test_changed_wire_rejected(self):
        j=job();j['input']=dict(j['input'],sha256='0'*64)
        with self.assertRaises(AssertionError):c.exact_key(j)
    def test_generation_seed_rejected(self):
        j=job();j['generation_seed']=1
        with self.assertRaises(AssertionError):c.exact_key(j)
    def test_public_jobs_omit_sources(self):
        j=job();v=native.public_job(j,j['input'],0)
        self.assertFalse({'source_id','source','truth','TX_CDF','source_index'}&set(v))
    def test_public_jobs_reject_extra_truth(self):
        j=job();j['source_id']='forbidden'
        with self.assertRaises(AssertionError):native.public_job(j,j['input'],0)
    def test_arpc_paid_endpoint_actual(self):
        j=job(codec=1,endpoint=3,length=16);v=native.public_job(j,j['input'],1)
        self.assertEqual(v['paid_profile'],3);self.assertEqual(v['path'],j['input']['path'])
    def test_image_hash_and_shape(self):
        a=np.linspace(0,1,3*256*256,dtype=np.float32).reshape(3,256,256);p=fixture('image.npz')
        np.savez(p,image=a);b,h=c.image(c.pin(p),'image')
        self.assertTrue(np.array_equal(a,b));self.assertEqual(h,hashlib.sha256(a.tobytes()).hexdigest())
    def test_gray_is_exact_point_five(self):
        a,h=c.image(None,None);self.assertTrue((a==.5).all());self.assertEqual(a.dtype,np.float32)
    def test_nonfinite_image_fails(self):
        a=np.zeros((3,256,256),np.float32);a[0,0,0]=np.nan;p=fixture('nan.npz');np.savez(p,image=a)
        with self.assertRaises(AssertionError):c.image(c.pin(p),'image')
    def test_uint8_image_fails(self):
        p=fixture('wrong_dtype.npz');np.savez(p,image=np.zeros((3,256,256),np.uint8))
        with self.assertRaises(AssertionError):c.image(c.pin(p),'image')
    def test_output_exclusive(self):
        p=fixture('exclusive.json');c.put(p,{'synthetic':True})
        with self.assertRaises(FileExistsError):c.put(p,{'synthetic':False})

def synthetic_rows():
    return [dict(codec=codec,snr_db=snr,policy_id=policy,source_id=f's{s:04d}',
        noise_seed=seed,dino=(s%7)/10+(seed-4102)/100)
        for codec,snr,policy in c.transport_plan.B2_CELLS for s in range(1000)
        for seed in c.transport_plan.SEEDS['B2']]

class SelectionTests(unittest.TestCase):
    def test_exact_27000_nine_cells(self):
        rows=synthetic_rows();src,means=c.source_means(rows)
        self.assertEqual(len(rows),27000);self.assertEqual(len(src),9000);self.assertEqual(len(means),9)
        self.assertTrue(all(x['source_mean_count']==1000 for x in means))
    def test_three_noise_equal_source_weight(self):
        src,means=c.source_means(synthetic_rows());self.assertAlmostEqual(src[0]['DINO_source_mean'],0)
        expected=sum((s%7)/10 for s in range(1000))/1000
        for x in means:self.assertAlmostEqual(x['calibration_mean_DINO'],expected,14)
    def test_failure_denominator_retained(self):
        rows=synthetic_rows()
        for r in rows:r['dino']=-.3 if r['source_id']=='s0000' else .6
        _,means=c.source_means(rows)
        for x in means:self.assertAlmostEqual(x['calibration_mean_DINO'],(.6*999-.3)/1000,13)
    def test_missing_noise_fails(self):
        rows=synthetic_rows();rows.pop()
        with self.assertRaises(AssertionError):c.source_means(rows)
    def test_duplicate_noise_fails(self):
        rows=synthetic_rows();rows.append(rows[0])
        with self.assertRaises(AssertionError):c.source_means(rows)
    def test_nonfinite_score_fails(self):
        rows=synthetic_rows();rows[0]['dino']=float('nan')
        with self.assertRaises(AssertionError):c.source_means(rows)
    def test_exact_tie_preserves_original_rule(self):
        _,means=c.source_means(synthetic_rows());selected=c.transport_plan.select_four(means)
        self.assertEqual([s['policy_id'] for s in selected],['nest_t24_q2_r1-2','nest_t24_q2_r2-3','arpc_t7_q2_r2-3','arpc_t7_q4_r2-3'])
    def test_five_nest10_candidates_all_present(self):
        _,means=c.source_means(synthetic_rows());self.assertEqual(sum(x['codec']==0 and x['snr_db']==10 for x in means),5)
    def test_metadata_translation_only_label(self):
        a={'weights':'x','flags':[1,2],'metric_selection_usage':'cal100'}
        b=dict(a,metric_selection_usage='cal1000')
        self.assertEqual(c.dino_identity(a),c.dino_identity(b));b['weights']='y';self.assertNotEqual(c.dino_identity(a),c.dino_identity(b))

class AdapterTests(unittest.TestCase):
    def test_exact_original_sha(self):
        self.assertEqual(hashlib.sha256(ORIGINAL).hexdigest(),dino_worker.ORIGINAL_SHA)
    def test_adapter_compiles_and_reverses(self):
        text=dino_worker.adapt(ORIGINAL);compile(text,'synthetic_adapter','exec')
        for old,new in reversed(dino_worker.EDITS):text=text.replace(new,old,1)
        self.assertEqual(text,ORIGINAL.decode())
    def test_source_change_rejected(self):
        with self.assertRaises(AssertionError):dino_worker.adapt(ORIGINAL+b'\n')
    def test_metric_call_is_unchanged(self):
        before,after=ast.parse(ORIGINAL),ast.parse(dino_worker.adapt(ORIGINAL))
        def selected(t):return [ast.dump(n,include_attributes=False) for n in ast.walk(t) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='score']
        self.assertEqual(selected(before),selected(after));self.assertEqual(len(selected(before)),1)
    def test_adaptation_has_no_new_metrics(self):
        t=dino_worker.adapt(ORIGINAL)
        self.assertIn("requested=('dinov2_vitl14',)",t);self.assertIn("reference_forward",Path(__file__).with_name('score.py').read_text())

class ClosureTests(unittest.TestCase):
    def records(self):
        child={'pid':9001,'start_ticks':111,'host':'fixture-host'};parent={'pid':9000,'start_ticks':100,'host':'fixture-host'}
        w=dict(process=child,returncode=0,timed_out=False,terminated=False,killed=False,metadata_error=None)
        la=dict(process=child,parent_process=parent)
        ob=dict(status='CLOSED_ON_EXECUTION_HOST',actual_wait={'path':'wait'},launch={'path':'launch'},host='fixture-host',child=child,parent=parent,child_gone=True,parent_gone=True,observed_unix=20,not_before_unix=19)
        return {'ob':ob,'wait':w,'launch':la}
    def check(self,records):
        with patch.object(c,'load',side_effect=lambda d:records[d['path']]):return c.check_observation({'path':'ob'})
    def test_cross_host_observation_pass(self):self.assertEqual(self.check(self.records())['host'],'fixture-host')
    def test_wrong_process_generation_fails(self):
        r=self.records();r['wait']['process']=dict(r['wait']['process'],start_ticks=112)
        with self.assertRaises(AssertionError):self.check(r)
    def test_parent_live_fails(self):
        r=self.records();r['ob']['parent_gone']=False
        with self.assertRaises(AssertionError):self.check(r)
    def test_host_mismatch_fails(self):
        r=self.records();r['ob']['host']='other'
        with self.assertRaises(AssertionError):self.check(r)
    def test_boolean_returncode_rejected(self):
        r=self.records();r['wait']['returncode']=False
        with self.assertRaises(AssertionError):self.check(r)
    def test_timeout_rejected(self):
        r=self.records();r['wait']['timed_out']=True
        with self.assertRaises(AssertionError):self.check(r)
    def test_early_observation_rejected(self):
        r=self.records();r['ob']['observed_unix']=18
        with self.assertRaises(AssertionError):self.check(r)

class TargetedIdentityTests(unittest.TestCase):
    """New review fixes only; selected separately from the original 35 tests."""
    def runtime(self):
        return dict(pid=1,gpu_name='gpuA',gpu_uuid='a',started_unix=1,python='frozen-python',torch='frozen-torch',
            native_arguments=dict(cache_dir='/old/cache',seed=0,bf16=0),cfg=3,tau=.5,top_k=900,top_p=.97)
    def test_runtime_device_cache_only_changes_allowed(self):
        a=self.runtime();b=copy.deepcopy(a);b.update(pid=2,gpu_name='gpuB',gpu_uuid='b',started_unix=9);b['native_arguments']['cache_dir']='/new/cache'
        self.assertEqual(native.canonical_arpc_runtime(a),native.canonical_arpc_runtime(b))
    def test_runtime_sampling_change_detected(self):
        a=self.runtime();b=copy.deepcopy(a);b['top_p']=.95
        self.assertNotEqual(native.canonical_arpc_runtime(a),native.canonical_arpc_runtime(b))
    def test_runtime_precision_change_detected(self):
        a=self.runtime();b=copy.deepcopy(a);b['native_arguments']['bf16']=1
        self.assertNotEqual(native.canonical_arpc_runtime(a),native.canonical_arpc_runtime(b))
    def test_unknown_cache_origin_rejected(self):
        v=dict(source={'codec':0},cache={'origin':'UNREGISTERED_RUN'})
        with self.assertRaises(AssertionError):native.require_member(v,{'proofs':{}},{})
    def test_known_origin_must_have_exact_member(self):
        v=dict(source={'codec':1},cache={'origin':'a2b_rx_v1','image':{'evidence':{'key':'changed'}}})
        with self.assertRaises(AssertionError):native.require_member(v,{'proofs':{'a2b_rx_v1':{}}},{'a2b_rx_v1':{'{"key":"original"}'}})
    def test_B3_model_routes_changed_rejected(self):
        a=dict(model_routes={'sha256':'one'},static={'cdf':'one'},weights=['one']);b=copy.deepcopy(a);b['model_routes']['sha256']='two'
        with self.assertRaises(AssertionError):native.same_native_registration(a,b)
    def test_B3_static_changed_rejected(self):
        a=dict(model_routes={'sha256':'one'},static={'cdf':'one'},weights=['one']);b=copy.deepcopy(a);b['static']['cdf']='two'
        with self.assertRaises(AssertionError):native.same_native_registration(a,b)
    def test_B3_weights_changed_rejected(self):
        a=dict(model_routes={'sha256':'one'},static={'cdf':'one'},weights=['one']);b=copy.deepcopy(a);b['weights']=['two']
        with self.assertRaises(AssertionError):native.same_native_registration(a,b)
    def metadatas(self):
        m=dict(DINO_L_metadata=dict(weights='frozen',preprocess=[224,224],numeric='float32',metric_selection_usage='prior'),numeric_flags={'tf32':False})
        return m,copy.deepcopy(m)
    def test_dino_population_label_only_allowed(self):
        a,b=self.metadatas();b['DINO_L_metadata']['metric_selection_usage']='current'
        score.compatible_dino(a,b,{'tf32':False},{})
    def test_dino_weights_changed_rejected(self):
        a,b=self.metadatas();b['DINO_L_metadata']['weights']='other'
        with self.assertRaises(AssertionError):score.compatible_dino(a,b,{'tf32':False},{})
    def test_dino_preprocess_changed_rejected(self):
        a,b=self.metadatas();b['DINO_L_metadata']['preprocess']=[256,256]
        with self.assertRaises(AssertionError):score.compatible_dino(a,b,{'tf32':False},{})
    def test_dino_numeric_changed_rejected(self):
        a,b=self.metadatas();b['numeric_flags']['tf32']=True
        with self.assertRaises(AssertionError):score.compatible_dino(a,b,{'tf32':False},{})
    def test_dino_runtime_changed_rejected(self):
        a,b=self.metadatas();a['runtime_identity']={'expected_runtime':{'torch':'old'}}
        with self.assertRaises(AssertionError):score.compatible_dino(a,b,{'tf32':False},{'torch':'new'})
    def test_unbound_pair_suite_rejected(self):
        with self.assertRaises(AssertionError):score.check_pair({'metric_identity':'unknown'},'sid','source','image',.5,{'known'})
    def test_cal100_scalar_changed_rejected(self):
        a={k:'same' for k in ('source_id','source_index','codec','endpoint','source','image','array_key','image_float_sha256')};a['dino']=.1;b=copy.deepcopy(a);b['dino']=.2
        with self.assertRaises(AssertionError):score.check_cached_state(a,b)
    def probe_fixture(self):
        p=fixture('runtime_identity/'+self._testMethodName+'/binary');p.write_bytes(b'binary')
        m=fixture('runtime_identity/'+self._testMethodName+'/overlay/pkg.dist-info/METADATA');m.write_bytes(b'metadata');rr=m.with_name('RECORD');rr.write_bytes(b'record')
        policy=dict(base_python='/usr/bin/python3',target=str(m.parent.parent),preserve_base_torch=True,packages=[])
        v=dict(executable='/usr/bin/python3',executable_binary=c.pin(p),python='frozen-python',torch='frozen-torch',torch_version_file=c.pin(p),torch_init=c.pin(p),
            torch_build={'__version__':'frozen-torch'},overlay=policy['target'],model_imports=[],packages=[dict(name='flash-attn',version='2.8.3.post2',METADATA=c.pin(m),RECORD=c.pin(rr))])
        return v,policy,p,m
    def test_probe_exact_metadata_pass(self):
        v,p,_,_=self.probe_fixture();native.validate_runtime_metadata(v,self.runtime(),p)
    def test_probe_other_interpreter_version_rejected(self):
        v,p,_,_=self.probe_fixture();v['python']='other-python'
        with self.assertRaises(AssertionError):native.validate_runtime_metadata(v,self.runtime(),p)
    def test_probe_changed_binary_rejected(self):
        v,p,b,_=self.probe_fixture();b.write_bytes(b'changed')
        with self.assertRaises(AssertionError):native.validate_runtime_metadata(v,self.runtime(),p)
    def test_probe_changed_overlay_rejected(self):
        v,p,_,m=self.probe_fixture();m.write_bytes(b'changed')
        with self.assertRaises(AssertionError):native.validate_runtime_metadata(v,self.runtime(),p)
    def test_probe_model_import_rejected(self):
        v,p,_,_=self.probe_fixture();v['model_imports']=['torch']
        with self.assertRaises(AssertionError):native.validate_runtime_metadata(v,self.runtime(),p)
    def test_probe_program_imports_stdlib_only(self):
        tree=ast.parse(native.METADATA_PROBE);modules={a.name.split('.')[0] for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names}
        self.assertFalse(modules&{'torch','numpy','transformers','sionna'})
        compile(native.METADATA_PROBE,'metadata-only','exec')

if __name__=='__main__':unittest.main(verbosity=2)

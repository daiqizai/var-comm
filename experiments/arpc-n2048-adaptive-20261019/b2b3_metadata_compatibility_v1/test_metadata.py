"""New targeted synthetic cases only; no original tests, prepare or science."""
import ast, copy, hashlib, importlib.abc, json, os, sys, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
import metadata as m
import lineage as ln
import runner

FIXTURE=Path(tempfile.mkdtemp(prefix='b2b3_metadata_targeted_')).resolve()
ORIGINAL_SOURCE=Path(__file__).resolve().parent.parent/'b2b3_analysis_v1/native.py'
ORIGINAL_BYTES=ORIGINAL_SOURCE.read_bytes()
assert len(ORIGINAL_BYTES)==34279 and hashlib.sha256(ORIGINAL_BYTES).hexdigest()==m.ORIGINAL['native.py'][1]
ALLOWED={p.resolve()for p in Path(__file__).resolve().parent.glob('*')if p.is_file()}
ALLOWED.add(ORIGINAL_SOURCE.resolve())
BLOCKED=[]
class NoModels(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0]in {'torch','tensorflow','sionna','transformers','diffusers','lpips','PIL','numpy'}:
            BLOCKED.append(fullname);raise AssertionError('No models or tensors in metadata tests')
sys.meta_path.insert(0,NoModels())
def guard(event,args):
    if event=='subprocess.Popen':raise AssertionError('No subprocess in synthetic qualification')
    if event=='open'and isinstance(args[0],(str,bytes,os.PathLike)):
        p=Path(os.fsdecode(args[0])).resolve()
        if str(p).startswith('/mnt/')and not p.is_relative_to(FIXTURE)and p not in ALLOWED:
            raise AssertionError('No production assets: '+str(p))
sys.addaudithook(guard)

def file(name,data):
    p=FIXTURE/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data);return p
def pair(label='T1B_NEW_RX'):
    bits=[1,0,1,1,0,0,1,0,1];profile=dict(family='PARTIAL',m=6,K=17)
    key=m.digest(dict(kind='ACTUAL_SOURCE_INPUT',payload=bits,**profile))
    s=dict(population='cal1000',source_id='synthetic',codec=0,endpoint=16,m=6,K=17,bit_length=len(bits),
        bits_sha256=hashlib.sha256(bytes(bits)).hexdigest(),object_sha256=hashlib.sha256(ln.packed(bits)).hexdigest())
    archive=dict(path='/synthetic/closed.npz',bytes=7,sha256='a'*64)
    im=dict(evidence=dict(path='/synthetic/evidence.json',bytes=8,sha256='b'*64),image_archive=archive,image_sha256='c'*64,origin='ACTUAL_NEW')
    if label=='T1C_NEW_RX':im.update(image_key='image',image_slot=None)
    ev=dict(source_input_key=key,image_archive=archive,image_sha256='c'*64,evidence=dict(
        receiver_truth_used=False,TX_CDF_used=False,source_decode_called=True,render_called=True,canonical=True,
        source_status='ARITHMETIC_SOURCE_DECODED',m=6,K=17))
    rx=dict(rx_profile=profile,header=dict(header_ok=True),body=dict(crc_accepted=True,parser_accepted=True,payload=bits))
    return s,im,key,ev,rx

class PairTests(unittest.TestCase):
    def test_t1b_saved_shape_without_inline_identity(self):
        s,im,k,e,r=pair();ln.validate_pair(s,im,'T1B_NEW_RX',k,e,r);self.assertNotIn('original_visual_identity',im)
    def test_t1c_saved_image_key_not_assumed_in_evidence(self):
        s,im,k,e,r=pair('T1C_NEW_RX');ln.validate_pair(s,im,'T1C_NEW_RX',k,e,r);self.assertNotIn('image_key',e)
    def test_unknown_schema_rejected(self):
        s,im,k,e,r=pair();im['free_truth']='x'
        with self.assertRaises(AssertionError):ln.validate_pair(s,im,'T1B_NEW_RX',k,e,r)
    def test_unknown_origin_rejected(self):
        s,im,k,e,r=pair();im['origin']='ASSUMED_CACHE'
        with self.assertRaises(AssertionError):ln.validate_pair(s,im,'T1B_NEW_RX',k,e,r)
    def test_changed_actual_payload_rejected(self):
        s,im,k,e,r=pair();r['body']['payload'][0]=0
        with self.assertRaises(AssertionError):ln.validate_pair(s,im,'T1B_NEW_RX',k,e,r)
    def test_changed_paid_endpoint_rejected(self):
        s,im,k,e,r=pair();s['K']+=1
        with self.assertRaises(AssertionError):ln.validate_pair(s,im,'T1B_NEW_RX',k,e,r)
    def test_truth_or_tx_cdf_rejected(self):
        for field in ('receiver_truth_used','TX_CDF_used'):
            s,im,k,e,r=pair();e['evidence'][field]=True
            with self.assertRaises(AssertionError):ln.validate_pair(s,im,'T1B_NEW_RX',k,e,r)
    def test_changed_member_archive_rejected(self):
        s,im,k,e,r=pair();e['image_archive']=dict(e['image_archive'],sha256='d'*64)
        with self.assertRaises(AssertionError):ln.validate_pair(s,im,'T1B_NEW_RX',k,e,r)
    def test_wrong_t1c_array_key_rejected(self):
        s,im,k,e,r=pair('T1C_NEW_RX');im['image_key']='other'
        with self.assertRaises(AssertionError):ln.validate_pair(s,im,'T1C_NEW_RX',k,e,r)
    def test_zero_padding_and_bits_hash_are_distinct(self):
        s,im,k,e,r=pair();self.assertEqual(ln.packed(r['body']['payload']),b'\xb2\x80')
        s['bits_sha256']=s['object_sha256']
        with self.assertRaises(AssertionError):ln.validate_pair(s,im,'T1B_NEW_RX',k,e,r)
    def test_no_inline_identity_is_fabricated(self):
        s,im,k,e,r=pair();c=dict(kind='B0_EXACT_INPUT',source=s,cache={q:s[q]for q in ('population','source_id','codec','endpoint')})
        c['cache'].update(origin='ORIGINAL_ENTROPY_RX',image=im);before=copy.deepcopy(c['cache'])
        key=ln.cache_key(s,c['cache']);bound=dict(origin='T1B_NEW_RX',per_image_evidence=im['evidence'],image_archive=im['image_archive'],image_sha256=im['image_sha256'])
        ln.validate_candidate(c,{},dict(members={key:bound}),dict(path='/proof.json',bytes=1,sha256='e'*64))
        self.assertEqual(c['cache'],before);self.assertIn('original_entropy_metadata_revision',c)
    def test_unbound_cache_never_becomes_missing(self):
        s,im,*_=pair();c=dict(source=s,cache={q:s[q]for q in ('population','source_id','codec','endpoint')})
        c['cache'].update(origin='ORIGINAL_ENTROPY_RX',image=im)
        with self.assertRaises(AssertionError):ln.validate_candidate(c,{},dict(members={}),{})

class NativeIdentityTests(unittest.TestCase):
    def fixture(self):
        routes=dict(native_qualification=dict(model_state_identity={'VAR':'v','Dc':'d'},numeric_settings=dict(runtime={'python':'3.10','torch':'2.11','threads':6})),numeric_flags={'tf32':False})
        v=dict(models=routes['native_qualification']['model_state_identity'],runtime=routes['native_qualification']['numeric_settings']['runtime'],numeric_flags=routes['numeric_flags'],sampling='unconditional_class1000_greedy_no_CFG',training_updates=0,source_mathematics='Unchanged SHA-bound original function/class definitions')
        return copy.deepcopy(v),routes
    def test_exact_actual_runtime_passes(self):
        v,r=self.fixture();ln.actual_identity(v,r)
    def test_changed_model_rejected(self):
        v,r=self.fixture();v['models']['VAR']='other'
        with self.assertRaises(AssertionError):ln.actual_identity(v,r)
    def test_changed_numeric_runtime_rejected(self):
        v,r=self.fixture();v['runtime']['threads']=2
        with self.assertRaises(AssertionError):ln.actual_identity(v,r)
    def test_changed_sampling_rejected(self):
        v,r=self.fixture();v['sampling']='alternate'
        with self.assertRaises(AssertionError):ln.actual_identity(v,r)

class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.root=FIXTURE/self.id().rsplit('.',1)[-1];self.root.mkdir()
    def test_exclusive_receipt_never_overwrites(self):
        p=self.root/'claim.json';m.put(p,{'a':1});before=p.read_bytes()
        with self.assertRaises(FileExistsError):m.put(p,{'a':2})
        self.assertEqual(before,p.read_bytes())
    def test_residue_requires_exact_set_and_bytes(self):
        al=self.root/'runtime';ao=self.root/'output';r=al/'B2';o=ao/'B2';r.mkdir(parents=True);o.mkdir(parents=True)
        vals={}
        for n in m.RESIDUE:
            p=r/n;p.write_text('{}');d=m.pin(p);vals[n]=(d['bytes'],d['sha256'])
        with patch.multiple(m,AL=al,AO=ao,RESIDUE=vals):
            m.check_residue(r,o,True);(r/'unexpected').write_text('x')
            with self.assertRaises(AssertionError):m.check_residue(r,o,True)
    def test_residue_output_must_be_empty(self):
        al=self.root/'runtime';ao=self.root/'output';r=al/'B2';o=ao/'B2';r.mkdir(parents=True);o.mkdir(parents=True)
        vals={}
        for n in m.RESIDUE:
            p=r/n;p.write_text('{}');d=m.pin(p);vals[n]=(d['bytes'],d['sha256'])
        (o/'registration.json').write_text('{}')
        with patch.multiple(m,AL=al,AO=ao,RESIDUE=vals):
            with self.assertRaises(AssertionError):m.check_residue(r,o,True)
    def test_existing_lineage_preserved_exactly(self):
        al=self.root/'runtime';p=al/'B2/native_identity_compatibility.json';p.parent.mkdir(parents=True);p.write_bytes(b'{ "a" : 1 }\n')
        d=m.pin(p);vals={'native_identity_compatibility.json':(d['bytes'],d['sha256'])}
        with patch.multiple(runner,AL=al,RESIDUE=vals):
            runner.lineage_action('B2',al/'B2',{'a':1});self.assertEqual(m.pin(p),d)
            with self.assertRaises(AssertionError):runner.lineage_action('B2',al/'B2',{'a':2})
    def test_b3_uses_new_exclusive_directories(self):
        al=self.root/'runtime';ao=self.root/'output'
        with patch.multiple(runner,AL=al,AO=ao):
            runner.directory_action('B3',al/'B3',ao/'B3')
            with self.assertRaises(AssertionError):runner.directory_action('B3',al/'B3',ao/'B3')
    def test_transform_roundtrips_and_only_two_functions_change(self):
        tree,proof=m.transformed(ORIGINAL_BYTES.decode().replace('\r\n','\n'))
        self.assertTrue(proof['inverse_exact']);self.assertTrue(proof['all_other_functions_AST_equal'])
        self.assertEqual([n.name for n in tree.body],['verified_candidate','prepare'])
        compile(tree,'synthetic-only','exec')
    def test_unexpected_source_text_rejected(self):
        source=ORIGINAL_BYTES.decode().replace('\r\n','\n').replace("ident=im['original_visual_identity']","ident=im.get('original_visual_identity')")
        with self.assertRaises(AssertionError):m.transformed(source)
    def test_unknown_failure_remains_fatal(self):
        al=self.root/'runtime';control=self.root/'control';p=al/'B2/nest/rx/failure.json';p.parent.mkdir(parents=True);p.write_text('{}')
        with patch.multiple(m,AL=al,CONTROL=control):
            with self.assertRaises(AssertionError):m.reject_unknown_failure('B2')
    def test_metadata_prepare_blocks_model_import(self):
        with self.assertRaises(AssertionError):runner.scientific_import_guard('import',('torch',None))

class ClosureTests(unittest.TestCase):
    def test_hostless_history_requires_pinned_diagnosis_host(self):
        d=dict(status='METADATA_DIAGNOSIS_PASS_NO_PIXELS',scientific_calls=0,pixel_files_opened=0,model_imports=[],
            source_files_changed=False,historical_no_host_PID_schema_preserved=True,observer_host='main')
        m.require_prepare_host(d,'main')
        with self.assertRaises(AssertionError):m.require_prepare_host(d,'leo')
    def fixture(self):
        child=dict(pid=111,start_ticks=12,state='S',host='main');parent=dict(pid=112,start_ticks=13,state='S',host='main')
        w=dict(process=child,returncode=0,seconds=3,metadata_error=None,terminated=False,killed=False,timed_out=False)
        launch=dict(process=child,parent_process=parent,argv=['py','runner'],started_unix=10)
        comp=dict(process=child,parent_process=parent,argv=launch['argv'],finished_unix=12)
        obs=dict(status='CLOSED_ON_PREPARE_HOST',host='main',child=child,parent=parent,child_gone=True,parent_gone=True,observed_unix=15)
        return obs,w,launch,comp
    def test_remote_host_consumes_observation_not_local_proc(self):
        x=self.fixture()
        with patch.object(m,'gone',side_effect=AssertionError('must not use remote local proc')):
            m.assert_closed_observation(*x,'leo')
    def test_same_host_rechecks_real_proc(self):
        with patch.object(m,'gone',return_value=False):
            with self.assertRaises(AssertionError):m.assert_closed_observation(*self.fixture(),'main')
    def test_false_wait_zero_not_integer_rejected(self):
        o,w,l,c=self.fixture();w['returncode']=False
        with self.assertRaises(AssertionError):m.assert_closed_observation(o,w,l,c,'leo')
    def test_changed_wrapper_argv_rejected(self):
        o,w,l,c=self.fixture();l['argv']=['other']
        with self.assertRaises(AssertionError):m.assert_closed_observation(o,w,l,c,'leo')
    def test_pid_generation_mismatch_rejected(self):
        o,w,l,c=self.fixture();o=copy.deepcopy(o);o['child']['start_ticks']=999
        with self.assertRaises(AssertionError):m.assert_closed_observation(o,w,l,c,'leo')
    def test_premature_closure_rejected(self):
        o,w,l,c=self.fixture();o['observed_unix']=11
        with self.assertRaises(AssertionError):m.assert_closed_observation(o,w,l,c,'leo')

if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__]);n=suite.countTestCases()
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    print(json.dumps(dict(targeted_tests=n,blocked_model_imports=BLOCKED,new_model_calls=0,new_channel_calls=0,new_scores=0,original_56_tests_rerun=False)))
    raise SystemExit(0 if result.wasSuccessful()else 1)

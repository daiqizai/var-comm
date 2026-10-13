"""Independent engineering-only R0 entry review; imports no real backend/model."""
from pathlib import Path
import ast,contextlib,hashlib,importlib.util,io,json,sys,time,types,unittest
from unittest.mock import patch
import numpy as np
W=Path(__file__).resolve().parents[2];E=W/'experiments/layered-mismatch-raw-20261013'
O=W/'results/layered_mismatch_raw_20261013/r0_supplement_review_v1'
sys.path.insert(0,str(E))
import r0_bler as b
import test_r0_bler as bt
import test_r0_clean as ct


class FakeTensor:
    def __init__(self,a):self.a=np.asarray(a)
    def __getitem__(self,key):return FakeTensor(self.a[key])
    def detach(self):return self
    def cpu(self):return self
    def numpy(self):return self.a


class Interfaces(unittest.TestCase):
    def test_original_real_runtime_TX_and_RX_wrapper_shapes_all_modulations(self):
        # Invoke only wrapper methods on an unconstructed object with fake modules.
        for q in (2,4,6):
            g=bt.group(q);events=[];payload=np.arange(24,dtype=np.uint8)%2
            runtime=object.__new__(b.RealRuntime)
            runtime.backend=types.SimpleNamespace(torch=types.SimpleNamespace(float32=np.float32,
                as_tensor=lambda a,**kw:np.asarray(a,dtype=np.float32)))
            runtime.legacy=object()
            def transmit(backend,legacy,actual,n,actualq,counters,group,session):
                events.append(('tx',actual.shape,n,actualq,counters,group,session))
                return FakeTensor(np.ones((1,60,2),np.float32)),dict(k=40,n=n,q=actualq,crc_bits=16,actual_E=[120.])
            def receive(backend,legacy,actual,k,n,actualq,snr,counters,group,session):
                events.append(('rx',actual.shape,k,n,actualq,snr,counters,group,session))
                decoded=np.zeros((1,40),np.uint8)
                return decoded[:,:-16],np.array([True],bool),decoded
            runtime.packet=types.SimpleNamespace(transmit_packet=transmit,receive_packet=receive)
            wave,energy=runtime.transmit(payload,g,19)
            hard,accepted,decoded=runtime.receive(wave,g,7,19)
            self.assertEqual(wave.shape,(60,2));self.assertEqual(energy,120.)
            self.assertEqual(hard.shape,(24,));self.assertIs(accepted,True);self.assertEqual(decoded.shape,(40,))
            self.assertEqual([e[0] for e in events],['tx','rx'])
            self.assertTrue(all(e[-1]==b.BODY_SESSION and e[-2]==0 for e in events))

    def test_shared_timing_lock_descriptor_is_inherited_by_each_scientific_child(self):
        for name in ('r0_bler.py','r0_clean.py'):
            tree=ast.parse((E/name).read_text(encoding='utf8'))
            calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
                   and isinstance(n.func.value,ast.Name) and n.func.value.id=='subprocess' and n.func.attr=='Popen']
            self.assertEqual(len(calls),1,name)
            self.assertIn('pass_fds',{k.arg for k in calls[0].keywords},name)

    def test_original_metric_cache_API_is_used_without_extra_source_scoring(self):
        tree=ast.parse((E/'r0_clean.py').read_text(encoding='utf8'))
        score=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='score_stage')
        calls=[n for n in ast.walk(score) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
        refs=[n for n in calls if n.func.attr=='prepare_reference']
        scores=[n for n in calls if n.func.attr=='score']
        self.assertEqual(len(refs),1);self.assertEqual(len(scores),1)
        self.assertEqual(ast.unparse(next(k.value for k in scores[0].keywords if k.arg=='prepared')),'prepared')
        self.assertEqual(ct.m.CAPS['score']['source_embedding'],200)


O.mkdir(parents=True,exist_ok=True)
stream=io.StringIO()
suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(bt.SupplementTests),
    unittest.defaultTestLoader.loadTestsFromTestCase(ct.Tests),unittest.defaultTestLoader.loadTestsFromTestCase(Interfaces)])
result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
text=stream.getvalue();print(text)
assert result.wasSuccessful(), 'Do not seal a passing review when any test fails'
bindings={name:hashlib.sha256((E/name).read_bytes()).hexdigest() for name in
          ['r0_bler.py','test_r0_bler.py','r0_clean.py','test_r0_clean.py']}
receipt=dict(status='PASS_INDEPENDENT_R0_ENTRY_REVIEW',actual_exit_code=0,tests_run=result.testsRun,
    source_bindings=bindings,reviewer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    torch_imported='torch' in sys.modules,real_backend_constructions=0,real_packet_calls=0,
    real_model_calls=0,real_source_channel_calls=0,bootstrap_calls=0,
    synthetic_arrays_only=True,created_unix=time.time(),
    verified=[
        'Frozen exact missing body grid1389 at4/7/10;64 trials per cell; existing672 body points and header points are not rerun.',
        'Real original adapter transmit/receive and CRC1D/2D signatures were read; synthetic wrappers cover q2/q4/q6 without LDPC or channel calls.',
        'Es2 convention uses original modulation and per-component variance10^(-SNR/10); no per-frame normalization; actual energy is recorded.',
        'Correct/reject/undetected form exclusive outcomes; software failure stays unresolved and cannot be rewritten as a CRC rejection.',
        'Reservation precedes TX; completed exact-identity reuse consumes no new random input/TX/RX; unresolved attempts cannot automatically resume.',
        'Clean new5400 and reused3400 form exact44x200 source-endpoint grid; no original result is overwritten.',
        'Original render_received receives only the clean raster prefix; exactly5400 renders/54000 prior scales/5400 decoder forwards are counted.',
        'Only DINO-L is loaded; original prepare_reference/score and identical image preprocessing are retained; source200 and reconstruction5400 counted separately.',
        'Timing v6 uses O6/owner.lock exclusive; supplemental owners/children retain shared lock until actual wait, avoiding check-then-launch overlap.',
        'Pre-model WAIT consumes zero model/packet calls; original code and timingv6 scientific files remain untouched.'
    ],
    original_findings=[dict(id='TIMING_RUNTIME_LOCATION',status='FIXED',detail='Use actual TMP six_method_timing_v6, not O6/RUN.'),
        dict(id='TIMING_CHECK_LAUNCH_RACE',status='FIXED',detail='O6 shared lock held by owner and inherited child FD, original timing exclusive lock unchanged.')],
    readiness_limits=['Registration drafts cannot run; root must freeze reviewed registration with actual resource/deadline assignment.',
        'Scientific numerical qualification is inherited where explicitly bound; no new real-model/packet qualification was run by this review.',
        'Pure ranking requires authenticated actual exits/output pins and the full source/689-profile domain; successful CPU tests do not provide those results.'])
for name,data in [('review.json',(json.dumps(receipt,indent=2,ensure_ascii=False)+'\n').encode()),
                  ('tests.txt',text.encode()),('review_source.py',Path(__file__).read_bytes())]:
    p=O/name
    with p.open('xb') as f:f.write(data)

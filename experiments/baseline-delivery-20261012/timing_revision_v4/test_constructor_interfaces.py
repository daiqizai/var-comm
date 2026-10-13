"""Synthetic CPU interface checks; no Torch import, real model or PHY call."""
import ast
from contextlib import nullcontext
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import numpy as np
from constructor_compat import adapt_build,OLD,NEW

ROOT=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
BASE=ROOT/'experiments/baseline-delivery-20261012'
TMP=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/baseline_strength_20261011')


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module


def extract_method(path,cls,method,namespace):
    source=Path(path).read_text();tree=ast.parse(source)
    c=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls)
    f=next(n for n in c.body if isinstance(n,ast.FunctionDef) and n.name==method)
    exec(compile(ast.get_source_segment(source,f),str(path),'exec'),namespace)
    return namespace[method]


endpoints=load(BASE/'timing_endpoints.py','_v4_test_endpoints')
class Meter:
    def call(self,name,fn):return fn()
class InvalidStream(Exception):pass


class Interfaces(unittest.TestCase):
    def test_original_vendor_constructor_attribute_and_correct_repair(self):
        class Module: pass
        namespace=dict(nn=SimpleNamespace(Module=Module,MSELoss=lambda **kw:object()),
            create_encoder=lambda **kw:object(),create_decoder=lambda **kw:object(),
            Distortion=lambda args:object(),Channel=lambda args,config:object())
        namespace['torch']=SimpleNamespace(nn=namespace['nn'])
        source=(TMP/'vendor/SwinJSCC/net/network.py').read_text();c=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='SwinJSCC')
        # Execute the authoritative constructor with inert stand-ins only.
        exec(compile(ast.get_source_segment(source,c),'<vendor-constructor-stubs>','exec'),namespace)
        args=SimpleNamespace(model='SAandRA',multiple_snr='1,4,10,13',C='6,13')
        cfg=SimpleNamespace(encoder_kwargs={},decoder_kwargs={},logger=None,pass_channel=True,downsample=4)
        native=namespace['SwinJSCC'](args,cfg)
        self.assertEqual(native.multiple_snr,[1,4,10,13]);self.assertFalse(hasattr(native,'args'))
        with self.assertRaises(AttributeError):exec(OLD,dict(s=SimpleNamespace(native=native)))
        exec(NEW,dict(s=SimpleNamespace(native=native)))
        self.assertEqual(native.multiple_snr,list(range(1,20)))

    def test_runtime_repair_has_exact_inverse_AST(self):
        runtime=load(BASE/'timing_runtime.py','_v4_test_runtime')
        changed,proof=adapt_build(runtime)
        self.assertTrue(proof['inverse_AST_equal']);self.assertTrue(proof['explicit_SNR_TX_RX_calls_unchanged'])
        self.assertEqual(changed.__name__,'build')

    def test_Swin_actual_encode_features_passes_explicit_19(self):
        seen=[];selfobj=SimpleNamespace(native=SimpleNamespace(encoder=lambda *a:seen.append(a) or 'features'))
        ns={'MODEL':'SwinJSCC_w/_SAandRA','WIDTH':320}
        f=extract_method(ROOT/'experiments/external-comparison-20261004/swin_model.py','SwinCodec','encode_features',ns)
        image=SimpleNamespace(shape=(1,3,256,256))
        self.assertEqual(f(selfobj,image,19,6),'features');self.assertEqual(seen[0][1:3],(19.,6))

    def test_Swin_actual_receive_data_passes_explicit_19(self):
        seen=[]
        class Output:
            def clamp(self,*args):seen.append(('clamp',args));return 'reconstruction'
        selfobj=SimpleNamespace(native=SimpleNamespace(decoder=lambda *a:seen.append(a) or Output()))
        ns={'MODEL':'SwinJSCC_w/_SAandRA','unpack_data':lambda *a:'received_only_features'}
        f=extract_method(ROOT/'experiments/external-comparison-20261004/swin_model.py','SwinCodec','receive_data',ns)
        self.assertEqual(f(selfobj,'y',19,'paid_power','paid_indices'),'reconstruction')
        self.assertEqual(seen[0][0:2],('received_only_features',19.))

    def make_entropy(self,actual,decode=lambda *a:dict(received_tokens=np.arange(3))):
        native=SimpleNamespace(torch=SimpleNamespace(no_grad=nullcontext));seen=[]
        codec=SimpleNamespace(native=native,decode=lambda *a:seen.append(('decode',a)) or decode(*a))
        rt=SimpleNamespace(qualified=True,profiles={str(i):{} for i in range(360)},receive=lambda *a,**k:actual)
        endpoint=endpoints.EntropyEndpoint(native,codec,rt,None,lambda *a:seen.append(('render',a)) or np.zeros((3,256,256),np.float32),{},True,None,None,InvalidStream)
        return endpoint,seen

    def test_entropy_rejected_header_has_fixed_output_no_source_decode(self):
        obj,seen=self.make_entropy(dict(header=dict(header_ok=False),body=None,rx_profile=None))
        image,actual=obj.receive(np.zeros((1024,2)),4,7,Meter(),'frame')
        self.assertTrue((image==.5).all());self.assertEqual(seen,[])

    def test_entropy_success_uses_only_received_profile_payload(self):
        profile=dict(family='EC_VAR_PARTIAL',m=7,K=25)
        obj,seen=self.make_entropy(dict(header=dict(header_ok=True),body=dict(crc_accepted=True,parser_accepted=True,payload=[1,0,1]),rx_profile=profile))
        _,actual=obj.receive(np.zeros((1024,2)),10,7,Meter(),'frame')
        self.assertEqual(actual['status'],'ARITHMETIC_SOURCE_DECODED')
        self.assertEqual(seen[0][1][0],'EC_VAR_PARTIAL');self.assertEqual(seen[0][1][2:],(7,25))
        self.assertEqual(seen[1][1][2:],(7,25))

    def test_entropy_noncanonical_stream_is_fixed_failure(self):
        def bad(*args):raise InvalidStream('noncanonical')
        obj,seen=self.make_entropy(dict(header=dict(header_ok=True),body=dict(crc_accepted=True,parser_accepted=True,payload=[1,0]),rx_profile=dict(family='EC_VAR_WHOLE',m=7,K=0)),bad)
        image,actual=obj.receive(np.zeros((1024,2)),19,7,Meter(),'frame')
        self.assertTrue((image==.5).all());self.assertEqual(actual['status'],'SOURCE_PARSE_REJECT')
        self.assertEqual(len(seen),1)

    def test_raw_received_state_drives_render_and_preserves_KEEP(self):
        offsets=[0,1,5,14,30,55,91,155,255,424,680]
        state=dict(kind='tokens',prefix=[list(range(offsets[i],offsets[i+1])) for i in range(4)],partial_values=[31,32],m=4,K=2)
        parsed=dict(receiver_state=state,source_status='KEEP_ACTUAL_HARD_TOKENS')
        actual=dict(parsed,header=dict(header_ok=True),body=dict(crc_accept=False))
        seen=[];native=SimpleNamespace(torch=SimpleNamespace(no_grad=nullcontext))
        rt=SimpleNamespace(cat=SimpleNamespace(header_map=lambda:{i:i for i in range(433)}),receiver=SimpleNamespace(receive=lambda *a,**k:actual),original=SimpleNamespace(present_actual=lambda *a:parsed))
        obj=endpoints.RawEndpoint(native,rt,lambda *a:seen.append(a) or np.zeros((3,256,256),np.float32),{},offsets,None)
        _,result=obj.receive(np.zeros((1024,2)),4,7,Meter(),'frame')
        self.assertEqual(result['status'],'KEEP_ACTUAL_HARD_TOKENS');self.assertEqual(seen[0][2:],(4,2));self.assertEqual(len(seen[0][1]),32)

    def test_raw_header_failure_skips_render(self):
        state=dict(kind='gray',prefix=[],partial_values=[],m=0,K=0);parsed=dict(receiver_state=state,source_status='HEADER_REJECT_GRAY')
        native=SimpleNamespace(torch=SimpleNamespace(no_grad=nullcontext));seen=[]
        rt=SimpleNamespace(cat=SimpleNamespace(header_map=lambda:{i:i for i in range(433)}),receiver=SimpleNamespace(receive=lambda *a,**k:dict(parsed,header=dict(header_ok=False),body=None)),original=SimpleNamespace(present_actual=lambda *a:parsed))
        obj=endpoints.RawEndpoint(native,rt,lambda *a:seen.append(a),{},[0],None)
        image,result=obj.receive(np.zeros((1024,2)),4,7,Meter(),'frame')
        self.assertTrue((image==.5).all());self.assertEqual(seen,[])

    def test_fresh_entropy_fallback_attempts_are_counted_and_not_sliced(self):
        calls=[]
        def encode(family,tokens,modes):
            calls.append((family,modes));m=modes[0];n=12 if m==7 else 8
            return {m:dict(bits=np.ones(n,np.uint8),arithmetic_bits=n)}
        profiles={str(m):dict(family='EC_VAR_WHOLE',m=m,K=0,q=2,nominal_rate='2/3',source_capacity_bits=10) for m in [7,6]}
        point=dict(target_m=7,target_K=0,q=2,nominal_rate='2/3',source_capacity_bits=10)
        p,bits,attempts=endpoints.fresh_entropy_choice(SimpleNamespace(encode=encode),SimpleNamespace(fallback_endpoints=lambda *a:((7,0),(6,0))),profiles,np.arange(680),point,False)
        self.assertEqual(p['m'],6);self.assertEqual(len(bits),8);self.assertEqual(len(calls),2)
        self.assertEqual([r['fits'] for r in attempts],[False,True])

    def test_RX_method_signatures_have_no_TX_truth_arguments(self):
        import inspect
        for cls in [endpoints.RawEndpoint,endpoints.EntropyEndpoint,endpoints.ContinuousEndpoint,endpoints.SwinEndpoint]:
            self.assertEqual(list(inspect.signature(cls.receive).parameters),['self','observed','snr','counter','meter','event_id'])


if __name__=='__main__':unittest.main(verbosity=2)

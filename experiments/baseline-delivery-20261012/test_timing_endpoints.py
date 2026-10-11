"""Synthetic receive-isolation/fallback tests, no models or channel calls."""
from contextlib import nullcontext
from types import SimpleNamespace as NS
import unittest
import numpy as np
from timing_endpoints import EntropyEndpoint, RawEndpoint, fresh_entropy_choice, Meter


class InvalidStream(ValueError):
    pass


class TestEndpoints(unittest.TestCase):
    def entropy(self, actual, decode):
        native=NS(torch=NS(no_grad=nullcontext))
        codec=NS(native=native, decode=decode)
        runtime=NS(qualified=True, profiles={str(i):i for i in range(360)},
                   receive=lambda *args, **kw: actual)
        calls=[]
        endpoint=EntropyEndpoint(native,codec,runtime,None,
            lambda native,tokens,m,K: calls.append((tokens.tolist(),m,K)) or np.zeros((3,256,256),np.float32),
            {'4':{'target_m':9,'target_K':192}},True,None,None,InvalidStream)
        return endpoint,calls

    def test_rx_uses_actual_header_not_target_policy_or_tx_tokens(self):
        actual=dict(header={'header_ok':True},body=dict(crc_accepted=True,parser_accepted=True,payload=[1,0,1]),
                    rx_profile=dict(family='EC_VAR_PARTIAL',m=4,K=6),status='RECEIVED')
        calls=[]
        def decode(family,bits,m,K):
            calls.append((family,bits.tolist(),m,K))
            return {'received_tokens':np.array([37,41])}
        endpoint,render=self.entropy(actual,decode)
        endpoint.receive(np.zeros((1024,2)),4,9,Meter(lambda:None),'test')
        self.assertEqual(calls,[('EC_VAR_PARTIAL',[1,0,1],4,6)])
        self.assertEqual(render,[([37,41],4,6)])

    def test_entropy_crc_reject_skips_source_decoder(self):
        actual=dict(header={'header_ok':True},body=dict(crc_accepted=False,parser_accepted=False),
                    rx_profile=None,status='BODY_REJECT')
        endpoint,render=self.entropy(actual,lambda *a: self.fail('Rejected payload reached source decoder'))
        image,state=endpoint.receive(None,4,9,Meter(lambda:None),'test')
        self.assertTrue(np.all(image==.5));self.assertEqual(render,[])

    def test_noncanonical_source_is_fixed_gray_without_retry(self):
        actual=dict(header={'header_ok':True},body=dict(crc_accepted=True,parser_accepted=True,payload=[1,0]),
                    rx_profile=dict(family='EC_VAR_PARTIAL',m=4,K=6),status='RECEIVED')
        calls=[]
        def decode(*args):
            calls.append(1);raise InvalidStream('noncanonical')
        endpoint,render=self.entropy(actual,decode)
        image,state=endpoint.receive(None,4,9,Meter(lambda:None),'test')
        self.assertEqual(calls,[1]);self.assertTrue(np.all(image==.5))
        self.assertEqual(state['status'],'SOURCE_PARSE_REJECT');self.assertEqual(render,[])

    def test_failed_longer_source_attempts_are_fresh_and_retained(self):
        calls=[]
        class Codec:
            def encode_endpoints(self,tokens,endpoints):
                calls.append(tuple(endpoints));return {(5,6):dict(bits=np.ones(9,np.uint8),arithmetic_bits=9)}
            def encode(self,family,tokens,modes):
                calls.append(modes);return {5:dict(bits=np.ones(5,np.uint8),arithmetic_bits=5)}
        plan=NS(fallback_endpoints=lambda m,k: [(5,6),(5,0)])
        profiles={str(k):dict(family='EC_VAR_PARTIAL' if k else 'EC_VAR_WHOLE',m=5,K=k,q=2,
                             nominal_rate='1/2',source_capacity_bits=7) for k in [6,0]}
        point=dict(target_m=5,target_K=6,q=2,nominal_rate='1/2',source_capacity_bits=7)
        p,bits,attempts=fresh_entropy_choice(Codec(),plan,profiles,np.zeros(680,np.int64),point,True)
        self.assertEqual(calls,[((5,6),),(5,)])
        self.assertEqual([r['fits'] for r in attempts],[False,True]);self.assertEqual(p['K'],0)

    def test_raw_crc_reject_preserves_actual_hard_token_keep(self):
        state=dict(kind='tokens',prefix=[[11],[12]],partial_values=[13],m=2,K=1)
        actual=dict(header={'header_ok':True},body={'crc_accepted':False},receiver_state=state,source_status='KEEP')
        native=NS(torch=NS(no_grad=nullcontext))
        runtime=NS(cat=NS(header_map=lambda:dict.fromkeys(range(433))),
                   receiver=NS(receive=lambda *a,**k:actual),
                   original=NS(present_actual=lambda *a:{'receiver_state':state}))
        rendered=[]
        endpoint=RawEndpoint(native,runtime,
            lambda n,t,m,k: rendered.append((t.tolist(),m,k)) or np.zeros((3,256,256),np.float32),
            {},[0,1,2],None)
        endpoint.receive(None,4,9,Meter(lambda:None),'test')
        self.assertEqual(rendered,[([11,12,13],2,1)])


if __name__=='__main__':
    unittest.main()

"""No Torch/Sionna imports: numerical mapping and prepaid callback contracts."""
import copy
import math
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock
import numpy as np
import main_raw64_plan_only as p
import main_raw64_packet_adapter as a
from test_main_raw64_plan_only import assets,Encoder

class Tensor(np.ndarray):
    @property
    def device(self):return 'cpu'
    def long(self):return self.astype(np.int64).view(Tensor)
    def to(self,dtype):return self.astype(dtype).view(Tensor)
    def cpu(self):return self
    def numpy(self):return np.asarray(self)
    def square(self):return (self*self).view(Tensor)
    def sum(self,*args,**kw):return np.asarray(super().sum(*args,**kw)).view(Tensor)

class Torch:
    float32=np.float32
    @staticmethod
    def tensor(v,dtype=None,device=None):return np.asarray(v,dtype=dtype).view(Tensor)
    as_tensor=tensor
    @staticmethod
    def stack(v,dim):return np.stack(v,axis=dim).view(Tensor)
    @staticmethod
    def logsumexp(v,dim):
        v=np.asarray(v,dtype=np.float64);m=v.max(axis=dim,keepdims=True)
        return (m+np.log(np.exp(v-m).sum(axis=dim,keepdims=True))).squeeze(axis=dim).view(Tensor)

class FakeBackend:
    device='cpu';torch=Torch()
    def __init__(self,old):self.identity=old;self.identity64=p.new_identity(old,p.sha(p.__file__));self.decodes=0;self.mode='correct'
    def plan(self,k,n,num_bits_per_symbol):return p.encoder_layout(Encoder,k,n,num_bits_per_symbol,self.identity64)
    def encode(self,info,n,q):
        self.info=np.array(info,dtype=np.uint8);return Torch.tensor(np.resize(info,(len(info),n)),np.float32)
    def decode(self,logits,k,n,q):
        self.decodes+=1;full=self.info.copy()
        if self.mode=='crc_fail':full[0,0]^=1
        elif self.mode=='undetected':
            full[0,0]^=1;full=self.phy.append_crc_batch(full[:,:-16])
        return Torch.tensor(full,np.float32)

class PacketTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.paths=assets();old=p.read(cls.paths['old_plan']);cls.identity=old['backend_identity']
        cls.common=p.load(cls.paths['original_common'],'uep_common')
        legacyroot=Path(cls.paths['original_planner']).parent
        cls.phy=p.load(legacyroot/'uep_phy.py','uep_phy')
        cls.lookup=p.load(legacyroot/'phy_lookup.py','main64_original_lookup')
        cls.original=p.load(cls.paths['original_planner'],'profiles');cls.core=p.load(cls.paths['action_core'],'original_main_finite_action_core')
        b=FakeBackend(cls.identity);b.qualified=False;b.identity=b.identity64
        planner=cls.core.MainPlanner(cls.original.Planner(b));row=planner.candidate(1,0,p.MCS('64QAM','1/3'),'nominal')
        cls.catalogue=p.append_catalogue(p.read(cls.paths['old_profiles']),p.read(cls.paths['old_aliases']),[row])
        cls.key=row['groups'][0]['phy_key'];cls.row=row
    def context(self):
        backend=FakeBackend(self.identity);backend.phy=self.phy
        class Header:
            def __init__(s):s.pid=None;s.calls=0
            def transmit(s,pid):s.pid=pid;return np.ones((68,2))
            def receive(s,wave,snr,book):
                s.calls+=1;ok=str(s.pid) in book
                return dict(header_crc_ok=True,header_fields_legal=ok,header_ok=ok,profile_id=s.pid if ok else None,score=1.)
        plan=dict(snrs_db=[1,4,7,10,13,19],qualification={'body_phy_keys':[self.key],'header_profile_ids':[0,269,270,271,4095],
            'mode':'4_noiseless_4_awgn60','noise_namespace':a.QUALIFICATION_NAMESPACE},
            proxy={'cells':[dict(phy_key=self.key,snr_db=13)],'random_namespace':a.PROXY_NAMESPACE})
        return a.Adapter(backend,self.phy,self.common,self.lookup,Header(),self.catalogue,plan)
    def event(self,phase='qualification',index=0):return dict(phase=phase,kind='body',phy_key=self.key,snr_db=60 if phase=='qualification' else 13,index=index,blocks=8 if phase=='qualification' else 256,event_id='event/'+str(index),noiseless=phase=='qualification' and index<4)
    def test_all64_symbols_power_labels_and_logit_signs(self):
        bits=((np.arange(64)[:,None]>>np.arange(5,-1,-1))&1).astype(np.float32).reshape(1,-1)
        wave=a.modulate(Torch.tensor(bits),6,Torch,None)
        expected=np.array([-7,-5,-1,-3,7,5,1,3])/math.sqrt(21)
        np.testing.assert_allclose(wave[0,:,0],np.repeat(expected,8),rtol=1e-6)
        np.testing.assert_allclose(wave[0,:,1],np.tile(expected,8),rtol=1e-6)
        self.assertAlmostEqual(float(np.square(wave).sum()/64),2.,places=6)
        logits=a.demap(wave,60,6,Torch,None);np.testing.assert_array_equal(logits>0,bits==1)
    def test_exact_logsumexp_demap_matches_independent_constellation_likelihood(self):
        y=Torch.tensor([[[.17,-.83],[1.3,.2]]],np.float32);snr=4
        actual=np.asarray(a.demap(y,snr,6,Torch,None));pam=np.array([-7,-5,-1,-3,7,5,1,3])/math.sqrt(21)
        expected=[]
        for value in y.ravel():
            probs=np.exp(-.5*10**(snr/10)*(float(value)-pam)**2)
            for bit in (2,1,0):expected.append(np.log(probs[[i for i in range(8) if i>>bit&1]].sum()/probs[[i for i in range(8) if not i>>bit&1]].sum()))
        np.testing.assert_allclose(actual,[expected],atol=2e-6)
    def test_q2_q4_all_packet_operations_are_original_delegates(self):
        legacy=SimpleNamespace(modulate_torch=mock.Mock(return_value='m'),demap_torch=mock.Mock(return_value='d'),transmit_packet=mock.Mock(return_value='t'),receive_packet=mock.Mock(return_value='r'))
        for q in (2,4):
            self.assertEqual(a.modulate(None,q,None,legacy),'m');self.assertEqual(a.demap(None,13,q,None,legacy),'d')
            self.assertEqual(a.transmit_packet(None,legacy,None,80,q,[1]),'t');self.assertEqual(a.receive_packet(None,legacy,None,28,80,q,13,[1]),'r')
        self.assertEqual(legacy.receive_packet.call_count,2)
    def test_exact_original_scrambling_sessions_no_H_length_or_padding(self):
        ctx=self.context();req,cb=ctx.prepare(self.event());self.assertEqual(ctx.backend.decodes,0)
        self.assertEqual(ctx.backend.info.shape,(1,28));self.assertEqual(req['session'],'qualification');self.assertEqual(req['counter'],0)
        result=cb();self.assertTrue(result['correct']);self.assertEqual(result['actual_hard_sha256'],self.common.array_sha(ctx.backend.info[0,:-16]))
        self.assertNotIn('hard_payload',result);self.assertNotIn('decoded_bits',result)
        self.assertEqual(ctx.backend.decodes,1)
        with self.assertRaisesRegex(ValueError,'single-use'):cb()
        self.assertEqual(ctx.backend.decodes,1)
    def test_proxy_uses_exact_registered_new_payload_counter_noise(self):
        ctx=self.context();event=self.event('proxy',7);req,cb=ctx.prepare(event)
        g=ctx.groups[self.key];payload,counter=a.proxy_payloads(self.common,g,13,7)
        noise=self.common.rng(a.PROXY_NAMESPACE,'noise',self.key,13,7).standard_normal((g['symbols'],2)).astype(np.float32)
        self.assertEqual(req['payload_sha256'],self.common.array_sha(payload[0]));self.assertEqual(req['counter'],counter)
        self.assertEqual(req['noise_sha256'],self.common.array_sha(noise));self.assertEqual(req['session'],'UEP_BLER_public_counter_v1')
        self.assertEqual(ctx.backend.decodes,0);cb();self.assertEqual(ctx.backend.decodes,1)
        old,counters=self.lookup.random_payloads(dict(g,q=6,kind='body'),13,7,1)
        self.assertNotEqual(counter,int(counters[0]))
    def test_four_noiseless_four_high_SNR_without_extra_callback(self):
        ctx=self.context()
        for i in range(8):
            req,cb=ctx.prepare(self.event(index=i));self.assertEqual(req['noiseless'],i<4)
            self.assertEqual(req['noise_sha256'] is None,i<4);self.assertEqual(ctx.backend.decodes,i);cb()
        self.assertEqual(ctx.backend.decodes,8)
        event=self.event(index=4);event['noiseless']=True
        with self.assertRaises(ValueError):ctx.prepare(event)
    def test_C_R_U_and_failed_CRC_hard_bits_retained(self):
        for mode,expected in [('correct',(True,False,False)),('crc_fail',(False,True,False)),('undetected',(False,False,True))]:
            ctx=self.context();ctx.backend.mode=mode;req,cb=ctx.prepare(self.event('proxy'));result=cb()
            self.assertEqual(tuple(result[x] for x in ('correct','rejected','undetected')),expected)
            self.assertEqual(len(result['actual_hard_sha256']),64);self.assertEqual(len(result['decoded_sha256']),64)
            self.assertNotIn('hard_payload',result);self.assertNotIn('decoded_bits',result)
            if mode!='correct':self.assertNotEqual(result['actual_hard_sha256'],self.common.array_sha(ctx.backend.info[0,:-16]))
    def test_noiseless_qualification_failure_is_not_counted_correct(self):
        ctx=self.context();ctx.backend.mode='crc_fail';_,cb=ctx.prepare(self.event())
        with self.assertRaisesRegex(ValueError,'roundtrip'):cb()
        self.assertEqual(ctx.backend.decodes,1)
        with self.assertRaises(ValueError):cb()
    def test_header_old_new_known_and_unknown_and_prepaid_callback(self):
        for pid in (0,269,270,271,4095):
            ctx=self.context();event=dict(phase='qualification',kind='header',phy_key=a.HEADER_KEY,profile_id=pid,snr_db=60,blocks=1,index=0,event_id='header/'+str(pid))
            req,cb=ctx.prepare(event);self.assertEqual(ctx.header.calls,0);result=cb()
            self.assertEqual(result['header_ok'],pid<=270);self.assertEqual(result['profile_id'],pid if pid<=270 else None)
            self.assertEqual(req['actual_E'],136.);self.assertEqual(ctx.header.calls,1)
    def test_unknown_or_unregistered_point_rejected_before_encode_decode(self):
        ctx=self.context()
        for updates in (dict(phy_key='missing'),dict(index=8),dict(blocks=256),dict(phase='development'),dict(snr_db=19)):
            e=self.event();e.update(updates)
            with self.assertRaises((ValueError,KeyError)):ctx.prepare(e)
        self.assertEqual(ctx.backend.decodes,0)
    def test_catalogue_rejects_changed_received_domain_body_or_KEEP(self):
        for mutate in (lambda c:c['profiles'][270].update(receiver_rule='DROP'),lambda c:c['profiles'][0].update(profile_id=270),lambda c:c['profiles'][270]['groups'][0].update(source_bits=25)):
            c=copy.deepcopy(self.catalogue);mutate(c);c['catalogue_digest']=p.identity(c['profiles'])
            with self.assertRaises(ValueError):a.validate_catalogue(c,self.identity)
    def test_runtime_plan_mismatch_stops_before_callback(self):
        ctx=self.context()
        with mock.patch.object(ctx.backend,'plan',return_value={}):
            with self.assertRaisesRegex(ValueError,'sealed metadata'):ctx.prepare(self.event())
        self.assertEqual(ctx.backend.decodes,0)
    def test_actual433_catalogue_mixed_backend_IDs_and_exact_old_prefix(self):
        path=Path(os.environ.get('MAIN64_TEST_CATALOGUE',str(Path(__file__).absolute().parent.parent/'assets/catalogue.json')))
        self.assertEqual(p.sha(path),'af772451d33a5750d77a57ba7f296b851cdff2baf098cd96896141818b3d2f03')
        cat=p.read(path);groups=a.validate_catalogue(cat,self.identity)
        self.assertEqual((len(groups),len(cat['profiles']),len(cat['aliases'])),(433,433,523))
        self.assertEqual(cat['profiles'][:270],p.read(self.paths['old_profiles']))
        self.assertEqual(cat['aliases'][:329],p.read(self.paths['old_aliases']))
        self.assertEqual({r['groups'][0]['modulation'] for r in cat['profiles'][270:]},{'64QAM'})
        self.assertEqual(max(r['token_count'] for r in cat['profiles'][270:]),397)
    def test_exact_original_CPU_numerical_flags(self):
        torch=SimpleNamespace(set_num_threads=mock.Mock(),set_num_interop_threads=mock.Mock(),
            use_deterministic_algorithms=mock.Mock(),backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
                cudnn=SimpleNamespace(allow_tf32=True,benchmark=True)))
        a.runtime_flags(torch)
        torch.set_num_threads.assert_called_once_with(2);torch.set_num_interop_threads.assert_called_once_with(1)
        torch.use_deterministic_algorithms.assert_called_once_with(True)
        self.assertFalse(torch.backends.cuda.matmul.allow_tf32);self.assertFalse(torch.backends.cudnn.allow_tf32)
        self.assertFalse(torch.backends.cudnn.benchmark)

if __name__=='__main__':unittest.main()

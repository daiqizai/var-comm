"""CPU checks for the registered N512 digital protocol and receive boundaries."""
import inspect
import os
from pathlib import Path
import unittest
from unittest.mock import patch

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import torch
import digital
import digital_protocol as phy


class ProtocolTests(unittest.TestCase):
    def test_real_native_noiseless_roundtrip(self):
        actions=phy.legal_actions()
        self.assertEqual([a.name for a in actions],['QPSK/N512/m4','QPSK/N512/m5',
            '16QAM/N512/m4','16QAM/N512/m5','16QAM/N512/m6'])
        rng=np.random.default_rng(912)
        scales=[rng.integers(0,4096,p*p) for p in phy.SIZES]
        for action in actions:
            with self.subTest(action=action.name):
                wave,ledger=phy.transmit(scales,321,action)
                event=phy.receive(wave,19,action.phy)
                self.assertEqual(wave.shape,(512,2))
                self.assertTrue(event['header_ok'] and event['body_crc_ok'])
                self.assertEqual((event['decoded_label'],event['decoded_mode']),(321,action.m))
                for a,b in zip(event['prefix'],scales[:action.m]):
                    np.testing.assert_array_equal(a,b)
                self.assertEqual(ledger['N_header'],68)
                self.assertEqual(ledger['N_body'],444)
                self.assertEqual(ledger['header_coded_bits'],136)
                self.assertEqual(ledger['coded_bits'],ledger['body_coded_bits']+136)
                self.assertLessEqual(ledger['effective_code_rate'],.9)
                if action.phy=='QPSK':self.assertEqual(ledger['E'],1024.)
                else:self.assertEqual(ledger['E'],float(np.square(wave,dtype=np.float64).sum()))

    def test_paid_header_modes_and_no_truth_rx_argument(self):
        self.assertEqual(set(inspect.signature(phy.receive).parameters),{'y','snr','phy'})
        for m in range(4,8):
            bits=np.concatenate((phy.indices_to_bits([10],10),phy.indices_to_bits([m-4],2)))
            decoded,ok,_=phy.decode_packet(phy.encode_packet(bits,68)['symbols'],12,19,'QPSK')
            self.assertTrue(ok)
            self.assertEqual(4+int(phy.bits_to_indices(decoded[10:12],2)[0]),m)
        for args in [(6,'QPSK'),(7,'QPSK'),(7,'16QAM')]:
            with self.assertRaises(ValueError):phy.Action(*args)
        with self.assertRaises(ValueError):phy.Action(4,'QPSK',N=1024)

    def test_body_crc_failure_retains_received_candidate(self):
        header=np.concatenate((phy.indices_to_bits([777],10),phy.indices_to_bits([0],2)))
        candidate=np.zeros(phy.raw_bits(4),dtype=np.uint8)
        with patch.object(phy,'decode_packet',side_effect=[(header,True,0.),(candidate,False,0.)]):
            event=phy.receive(np.ones((512,2)),7,'QPSK')
        self.assertTrue(event['header_ok'])
        self.assertFalse(event['body_crc_ok'])
        self.assertEqual(event['decoded_label'],777)
        self.assertEqual(event['decoded_mode'],4)
        self.assertEqual(event['trusted_prefix_scales'],0)
        self.assertEqual(event['hard_candidate_prefix_scales'],4)
        self.assertTrue(event['raw_candidate_used_after_crc_failure'])
        for i,tokens in enumerate(event['prefix']):np.testing.assert_array_equal(tokens,np.zeros(phy.SIZES[i]**2))
        with patch.object(phy,'decode_packet',return_value=(header,False,0.)):
            failure=phy.receive(np.ones((512,2)),7,'QPSK')
        self.assertFalse(failure['header_ok'])
        self.assertIsNone(failure['decoded_label'])
        self.assertEqual(failure['prefix'],[])


class ReceiverTests(unittest.TestCase):
    def test_received_only_cache_and_unconditional_class(self):
        calls=[]
        def complete(vae,var,prefix,label,device):
            calls.append((tuple(tuple(x) for x in prefix),label))
            return torch.full((1,32,16,16),label/2000.)
        prefix=[np.zeros(p*p,dtype=np.int64) for p in phy.SIZES[:4]]
        event=dict(header_ok=True,prefix=prefix,decoded_label=17)
        loaded=dict(vae=None,var=None,device=torch.device('cpu'))
        cache=digital.ReceivedCache()
        with patch.object(digital,'complete_latent',side_effect=complete), \
             patch.object(digital,'prefix_latent',return_value=torch.zeros(1,32,16,16)), \
             patch.object(digital.assets,'check',return_value=None):
            first=digital.render_received(event,loaded,cache)
            repeated=digital.render_received(event,loaded,cache)
            changed=digital.render_received({**event,'decoded_label':99},loaded,cache)
            failed=digital.render_received({'header_ok':False},loaded,cache)
        self.assertEqual([label for _,label in calls],[17,1000,99])
        self.assertGreaterEqual(cache.hits,3)
        self.assertTrue(torch.equal(first['U'],changed['U']))
        self.assertFalse(torch.equal(first['C'],changed['C']))
        self.assertTrue(torch.equal(first['C'],repeated['C']))
        self.assertTrue(all(value is None for value in failed.values()))

    def test_gray_failure_has_no_actual_latent(self):
        loaded=dict(device=torch.device('cpu'),dino=None,
            lpips=lambda x,y:(x-y).square().mean((1,2,3)))
        record=dict(pixels=np.zeros((3,256,256),dtype=np.uint8))
        def features(model,x):
            return torch.stack((torch.ones(len(x)),x.mean((1,2,3)),x.std((1,2,3))),dim=1)
        with patch.object(digital,'dino_features',side_effect=features),patch.object(digital.assets,'check',return_value=None):
            scored=digital.quality_outputs(loaded,record,{'C':(None,'Dc'),'U':(None,'D0')},images=True)
        self.assertEqual(scored['C'][0],scored['U'][0])
        np.testing.assert_array_equal(scored['C'][1],np.full((3,256,256),.5))
        np.testing.assert_array_equal(scored['C'][1],scored['U'][1])
        fields=digital.latent_fields(torch.ones(32,16,16),None)
        self.assertFalse(fields['latent_valid'])
        self.assertFalse(fields['decoder_applied'])
        self.assertEqual(fields['latent_sq_err_final'],'')
        self.assertEqual(fields['zero_erasure_proxy_sq_error'],8192.)

    def test_same_latent_decoder_and_same_action_class_outputs(self):
        choices={'C':{'action_m':5},'U':{'action_m':4}}
        latents={m:dict(C=torch.full((1,32,16,16),m+1.),U=torch.full((1,32,16,16),m+2.),
                      prefix=torch.full((1,32,16,16),float(m))) for m in (4,5)}
        received={m:({'header_ok':True},latents[m],{'waveform_sha256':f'wave{m}',
            'observation_sha256':f'y{m}'}) for m in (4,5)}
        outputs,info=digital.selected_outputs('QPSK',choices,received)
        self.assertEqual(len(outputs),8)
        self.assertIs(outputs['D_C_QPSK'][0],outputs['D_C_QPSK_D0'][0])
        self.assertIs(outputs['D_U_QPSK'][0],outputs['D_U_QPSK_D0'][0])
        self.assertIs(outputs['D_C_QPSK_at_U_action'][0],latents[4]['C'])
        self.assertIs(outputs['D_U_QPSK_at_C_action'][0],latents[5]['U'])
        self.assertIs(info['D_C_QPSK_at_U_action'][2],info['D_U_QPSK'][2])
        self.assertIs(info['D_U_QPSK_at_C_action'][2],info['D_C_QPSK'][2])


class CalibrationTests(unittest.TestCase):
    def test_reliability_reference_quality_cap_and_ties(self):
        candidates=[dict(phy_family='QPSK',snr_db=1,class_condition='U',action_m=m,
            action_id=f'QPSK/N512/m{m}',failure_fraction=.1,psnr_db=19.5 if m==4 else 20.,
            lpips_alex=.1 if m==4 else .2) for m in (4,5)]
        selected=digital.reliability_then_quality(candidates,'QPSK',1,'U')
        self.assertEqual(selected['reliability_reference_m'],5)
        self.assertEqual(selected['action_m'],5)
        candidates[0]['psnr_db']=19.8
        self.assertEqual(digital.reliability_then_quality(candidates,'QPSK',1,'U')['action_m'],4)
        candidates[0].update(failure_fraction=.05,lpips_alex=.2)
        self.assertEqual(digital.reliability_then_quality(candidates,'QPSK',1,'U')['action_m'],4)
        candidates[0]['failure_fraction']=.1
        self.assertEqual(digital.reliability_then_quality(candidates,'QPSK',1,'U')['action_m'],4)

    def test_complete_source_noise_grid_and_preprocessing(self):
        records=[dict(image_id=f's{i}',preprocessing_id=f'p{i}') for i in range(3)]
        methods=['D_U_QPSK','D_C_QPSK'];seeds=[2001,2002,2003]
        rows=[dict(source_id=r['image_id'],source_index=i,preprocessing_id=r['preprocessing_id'],
            snr_db=s,noise_seed=n,method=m) for i,r in enumerate(records) for s in digital.SNRS for n in seeds for m in methods]
        scopes=[(m,) for m in methods]
        digital.validate_grid(rows,records,seeds,scopes,('method',))
        for wrong in (rows[:-1],rows+[rows[0]], [{**rows[0],'preprocessing_id':'wrong'}]+rows[1:]):
            with self.assertRaises(RuntimeError):digital.validate_grid(wrong,records,seeds,scopes,('method',))


if __name__=='__main__':
    unittest.main(verbosity=2)

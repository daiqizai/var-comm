"""RX independence tests using the frozen arithmetic core and a fake renderer.

No header/LDPC decoder, GPU model, real source dataset or channel is run.
"""
import copy
import hashlib
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np

HERE = Path(__file__).resolve().parent
if (HERE.parent/'phy_codec').exists():
    sys.path.insert(0, str(HERE.parent/'phy_codec'))
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import h_payload_rx as rx
import h64_phy as phy
import h64_source as codec
from h64_catalog import all_buckets, catalogue, digest, SIZES, token_count
from test_h64_core import FakeBackend, Provider, load_primitives


class ReceiverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cat = catalogue(all_buckets(FakeBackend()))
        cls.primitives = load_primitives()

    def setUp(self):
        self.registry, self.rendered = [], []
        self.scales = [(np.arange(s*s, dtype=np.int64)+17*i) % 4096 for i,s in enumerate(SIZES)]
        def render(tokens, m, K):
            self.rendered.append((tokens.copy(),m,K))
            return np.full((3,256,256), (int(tokens.sum())%200+20)/255, dtype=np.float32)
        self.receiver = rx.Receiver(self.cat, primitives=self.primitives,
            provider_factory=lambda:Provider(self.registry), render_tokens=render)

    def profile(self, mode='raw', m=6, K=0, q=6):
        return next(p for p in self.cat['profiles'] if (p['mode'],p['m'],p['K'],p['q'],p['nominal_rate'])
                    == (mode,m,K,q,'1/2'))

    def trace(self, profile, payload=None, decoded=None):
        if decoded is None: decoded=phy.pack_body(payload,profile)
        body=dict(**phy.parse_body(decoded,profile), decoded_bits=decoded.tolist(),
                  profile_key=profile['profile_key'],phy_key=self.receiver.phy_keys[profile['profile_id']])
        return dict(rx=dict(header=dict(profile_id=profile['profile_id'],header_ok=True,header_crc_ok=True,
            header_fields_legal=True,score=0.),body=body,status=body['status']),rx_profile=copy.deepcopy(profile),
            tx={'profile_id':'deliberately wrong'},evaluation_only={'do_not_read':True},
            received_raw_tokens=['never trust convenience tokens'],gray='never trust CPU gray flag')

    def reconstruct(self, trace):return self.receiver.reconstruct(rx.receiver_view(trace))

    def test_only_rx_fields_are_read_and_actual_raw_tokens_rendered(self):
        p=self.profile(q=4);bits=phy.raw_payload(self.scales,6);bits[0]^=1
        data=self.trace(p,bits)
        class PoisonTrace(dict):
            def __getitem__(self,key):
                if key not in ('rx','rx_profile'):raise AssertionError('TX/truth accessed: '+key)
                return super().__getitem__(key)
        out=self.reconstruct(PoisonTrace(data))
        np.testing.assert_array_equal(self.rendered[0][0],phy.raw_tokens(bits,p))
        self.assertEqual(out['summary']['received_profile_id'],p['profile_id'])
        self.assertFalse(out['summary']['gray']);self.assertEqual(len(self.registry),0)
        self.assertNotEqual(int(out['received_tokens'][0]),int(self.scales[0][0]))

    def test_wrong_accepted_partial_profile_controls_length_and_positions(self):
        p=self.profile(m=7,K=81);bits=phy.raw_payload(self.scales,7,81)
        out=self.reconstruct(self.trace(p,bits))
        self.assertEqual(self.rendered[0][1:],(7,81))
        self.assertEqual(len(out['received_tokens']),236)
        np.testing.assert_array_equal(out['received_tokens'][-81:],self.scales[7][:81])

    def test_header_reject_gray_skips_all_model_calls(self):
        trace=dict(rx=dict(header=dict(header_ok=False,profile_id=None,header_crc_ok=False,
                                      header_fields_legal=False),body=None,status='HEADER_REJECT'),rx_profile=None)
        out=self.reconstruct(trace)
        np.testing.assert_array_equal(out['image'],np.full((3,256,256),.5,np.float32))
        self.assertEqual(self.rendered,[]);self.assertEqual(self.registry,[])

    def test_crc_and_padding_reject_have_no_model_or_source_decode(self):
        p=self.profile();valid=phy.pack_body(phy.raw_payload(self.scales,6),p)
        bad_crc=valid.copy();bad_crc[-1]^=1
        bad_pad=valid[:-16].copy();bad_pad[-1]=1;bad_pad=phy.append_crc(bad_pad)
        for bits,reason in ((bad_crc,'CRC_REJECT'),(bad_pad,'NONZERO_KNOWN_PADDING')):
            out=self.reconstruct(self.trace(p,decoded=bits))
            self.assertTrue(out['summary']['gray']);self.assertEqual(out['summary']['reason'],reason)
        self.assertEqual(self.rendered,[]);self.assertEqual(self.registry,[])

    def test_independent_arithmetic_roundtrip_and_each_call_fresh_state(self):
        encoded=codec.encode_prefixes(self.scales,lambda:Provider([]),self.primitives,(6,))
        trace=self.trace(self.profile('arithmetic'),encoded[6]['bits'])
        first=self.reconstruct(trace);second=self.reconstruct(trace)
        self.assertEqual(len(self.registry),2)
        self.assertIsNot(self.registry[0],self.registry[1])
        self.assertEqual(first['summary']['source_status'],'ARITHMETIC_SOURCE_DECODED')
        self.assertEqual(first['summary']['canonical_details']['zero_extension_reads'],30)
        np.testing.assert_array_equal(first['received_tokens'],np.concatenate(self.scales[:6]))
        np.testing.assert_array_equal(first['image'],second['image'])

    def test_valid_wrong_arithmetic_stream_is_rendered_not_repaired(self):
        encoded=codec.encode_prefixes(self.scales,lambda:Provider([]),self.primitives,(6,))
        bits=encoded[6]['bits'].copy();bits[0]^=1
        out=self.reconstruct(self.trace(self.profile('arithmetic'),bits))
        self.assertTrue(out['summary']['arithmetic_canonical']);self.assertFalse(out['summary']['gray'])
        self.assertNotEqual(int(out['received_tokens'][0]),int(self.scales[0][0]))
        np.testing.assert_array_equal(self.registry[0].history[0],out['received_tokens'][:1])
        self.assertEqual(len(self.rendered),1)

    def test_crc_valid_noncanonical_stream_is_protocol_gray_not_clean_proxy(self):
        encoded=codec.encode_prefixes(self.scales,lambda:Provider([]),self.primitives,(6,))
        bits=np.append(encoded[6]['bits'],0).astype(np.uint8)
        out=self.reconstruct(self.trace(self.profile('arithmetic'),bits))
        self.assertTrue(out['summary']['body_crc_accepted']);self.assertTrue(out['summary']['body_parser_accepted'])
        self.assertEqual(out['summary']['source_status'],'ARITHMETIC_SOURCE_INVALID_GRAY')
        self.assertFalse(out['summary']['arithmetic_canonical']);self.assertTrue(out['summary']['source_decode_complete'])
        self.assertEqual(self.rendered,[])

    def test_runtime_error_is_not_hidden_as_channel_gray(self):
        encoded=codec.encode_prefixes(self.scales,lambda:Provider([]),self.primitives,(6,))
        trace=self.trace(self.profile('arithmetic'),encoded[6]['bits'])
        with patch.object(codec,'decode_prefix',side_effect=RuntimeError('GPU resource fault')):
            with self.assertRaisesRegex(RuntimeError,'GPU resource'):self.reconstruct(trace)
        self.assertEqual(self.rendered,[])

    def test_renderer_failure_is_not_hidden(self):
        p=self.profile();trace=self.trace(p,phy.raw_payload(self.scales,6))
        self.receiver.render_tokens=lambda *_:np.full((3,256,256),np.nan,np.float32)
        with self.assertRaisesRegex(rx.TraceIntegrityError,'finite'):self.reconstruct(trace)

    def test_forged_profile_parse_or_layout_raises_instead_of_gray(self):
        p=self.profile();trace=self.trace(p,phy.raw_payload(self.scales,6))
        bad=copy.deepcopy(trace);bad['rx_profile']['m']=7
        with self.assertRaisesRegex(rx.TraceIntegrityError,'profile differs'):self.reconstruct(bad)
        bad=copy.deepcopy(trace);bad['rx']['body']['payload'][0]^=1
        with self.assertRaisesRegex(rx.TraceIntegrityError,'parse differs'):self.reconstruct(bad)
        bad=copy.deepcopy(trace);bad['rx']['body']['phy_key']='wrong'
        with self.assertRaisesRegex(rx.TraceIntegrityError,'layout'):self.reconstruct(bad)
        bad=copy.deepcopy(trace);bad['rx']['header']['profile_id']=4095
        with self.assertRaisesRegex(rx.TraceIntegrityError,'unavailable'):self.reconstruct(bad)
        self.assertEqual(self.rendered,[])

    def test_image_then_registered_target_psnr_and_target_tamper(self):
        p=self.profile();out=self.reconstruct(self.trace(p,phy.raw_payload(self.scales,6)))
        before=out['image'].copy();target=np.full((3,256,256),73,dtype=np.uint8)
        targetsha=hashlib.sha256(target.tobytes()).hexdigest()
        scored=rx.score_reconstruction(out,target,expected_preprocessing_sha256=targetsha)
        expected=float(np.mean(np.square(before.astype(np.float64)-(target.astype(np.float32)/255.).astype(np.float64))))
        self.assertEqual(scored['mse'],expected);self.assertEqual(scored['psnr_db'],float(-10*np.log10(expected)))
        np.testing.assert_array_equal(out['image'],before)
        target[0,0,0]^=1
        with self.assertRaisesRegex(rx.TraceIntegrityError,'preprocessing'):rx.score_reconstruction(out,target,expected_preprocessing_sha256=targetsha)

    def test_evaluation_truth_only_classifies_after_image_exists(self):
        p=self.profile();out=self.reconstruct(self.trace(p,phy.raw_payload(self.scales,6)))
        before=out['image'].copy()
        diag=rx.evaluation_diagnostics(out,dict(header_correct=False,parsed_wire_matches_transmission=False,accepted_wire_mismatch=True))
        self.assertTrue(diag['reconstructed_after_accepted_wire_mismatch'])
        np.testing.assert_array_equal(out['image'],before)
        bad=copy.deepcopy(out);bad['summary']['source_decode_complete']=False
        with self.assertRaisesRegex(rx.TraceIntegrityError,'pending'):rx.evaluation_diagnostics(bad,{})

    def test_mutated_image_cannot_receive_old_provenance_score(self):
        p=self.profile();out=self.reconstruct(self.trace(p,phy.raw_payload(self.scales,6)))
        target=np.full((3,256,256),3,np.uint8);targetsha=hashlib.sha256(target.tobytes()).hexdigest()
        out['image'][0,0,0]=.01
        with self.assertRaisesRegex(rx.TraceIntegrityError,'RGB changed'):rx.score_reconstruction(out,target,expected_preprocessing_sha256=targetsha)


if __name__=='__main__':unittest.main()

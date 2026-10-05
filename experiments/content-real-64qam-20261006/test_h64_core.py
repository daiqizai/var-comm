import ast
from contextlib import AbstractContextManager
from pathlib import Path
import sys
import types
import unittest
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from h64_catalog import all_buckets, bucket, catalogue, canonical_states, make_profile, normalize_state, token_count
from h64_source import CODEC_PROTOCOL, InvalidCodeStream, reference_primitives, roundtrip, decode_prefix
from h64_phy import (append_crc, pack_body, parse_body, raw_payload, raw_tokens, pam, modulate, demap,
                     scramble_mask, exact_uncoded_ser, exact_ser_from_decision_regions, receive_frame,
                     receive_body, transmit_body, integer_bits, logsumexp)


def load_primitives():
    # Execute only the exact frozen pure arithmetic definitions, not Torch/model imports.
    source = HERE.parents[1]/'historical_eval_20261003/source/src/var_comm'
    if not source.exists():
        import os
        source = Path(os.environ['H64_REFERENCE_VAR_COMM'])
    entropy_tree = ast.parse((source/'entropy.py').read_text(encoding='utf-8'))
    keep = [x for x in entropy_tree.body if isinstance(x, (ast.Assign, ast.FunctionDef))]
    entropy = types.ModuleType('frozen_test_entropy'); entropy.__dict__['np'] = np
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(source/'entropy.py'), 'exec'), entropy.__dict__)
    whole_tree = ast.parse((source/'whole_entropy.py').read_text(encoding='utf-8'))
    keep = [x for x in whole_tree.body if isinstance(x, ast.ClassDef) and x.name in ('ArithmeticEncoder', 'ArithmeticDecoder')]
    whole = types.ModuleType('frozen_test_whole'); whole.__dict__.update(entropy.__dict__)
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(source/'whole_entropy.py'), 'exec'), whole.__dict__)
    return reference_primitives(whole, entropy)


class Provider(AbstractContextManager):
    def __init__(self, registry, conditional=False):
        self.registry=registry; self.history=[]; self.conditional=conditional; registry.append(self)
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def cdf(self):
        from h64_catalog import SIZES
        count=np.full(4096, 4096, dtype=np.int64)
        if self.conditional:
            target=sum(int(x.sum()) for x in self.history) % 4096
            donor=(target+1) % 4096
            count[target]+=2048; count[donor]-=2048
        cdf=np.concatenate(([0], count.cumsum()))
        return np.tile(cdf, (SIZES[len(self.history)]**2, 1))
    def advance(self, values): self.history.append(np.asarray(values).copy())


class RecordingLedger:
    def __init__(self): self.events=[]; self.configs={}; self.cache={}
    def register_configuration(self, key, value): self.configs[key]=value
    def decode_once(self, phase, event_id, kind, key, request, decode):
        if event_id in self.cache: return self.cache[event_id]
        self.events.append(dict(phase=phase, event_id=event_id, kind=kind, status='RESERVED'))
        result=decode(); self.events[-1]['status']='COMPLETE'; self.cache[event_id]=result
        return result


class FakeBackend:
    """A transparent test code, never a replacement physical backend."""
    def __init__(self): self.calls=0; self.last_k=None; self.raise_decode=False
    def plan(self, k, n, q): return dict(k=k, n=n, q=q, testing_only=True,layout_id='TEST_ONLY')
    def encode(self, bits, n, q):
        return np.pad(np.asarray(bits), ((0,0),(0,n-bits.shape[1])))
    def decode(self, logits, k, n, q):
        self.calls+=1; self.last_k=k
        if self.raise_decode: raise RuntimeError('test decoder failure')
        return (np.asarray(logits)[:,:k] > 0).astype(np.uint8)


class FakeHeader:
    def __init__(self, pid, ok=True): self.pid=pid; self.ok=ok
    def receive(self, y, snr, codebook):
        return dict(profile_id=self.pid if self.ok else None, header_ok=self.ok,
                    header_crc_ok=self.ok, header_fields_legal=self.pid in codebook)


class ResourceTests(unittest.TestCase):
    def test_all_integer_states_are_unique_and_untruncated(self):
        states=list(canonical_states())
        self.assertEqual([token_count(*x) for x in states], list(range(1,681)))
        self.assertEqual(normalize_state(7,100), (8,0))
        self.assertEqual(normalize_state(0,1), (1,0))
    def test_actual_paid_capacity_and_raw64_maxima(self):
        cat=catalogue(all_buckets())
        expected={'1/2':(7,81,236),'2/3':(8,61,316),'3/4':(8,101,356),'5/6':(8,140,395)}
        for rate, (m,K,count) in expected.items():
            rows=[p for p in cat['profiles'] if p['q']==6 and p['nominal_rate']==rate and p['mode']=='raw']
            self.assertEqual(len(rows), count)
            self.assertEqual((rows[-1]['m'], rows[-1]['K']), (m,K))
            self.assertEqual({token_count(p['m'],p['K']) for p in rows}, set(range(1,count+1)))
        self.assertLess(cat['profile_count'],4096)
        self.assertFalse(any(p['mode']=='raw' and p['m']>=9 for p in cat['profiles']))
    def test_partial_control_does_not_replace_whole_four_arm_tags(self):
        cat=catalogue(all_buckets())
        names={f for p in cat['profiles'] for f in p['families']}
        self.assertTrue({'H16-R','H16-A','H64-R','H64-A'}.issubset(names))
        for p in cat['profiles']:
            if any(f in ('H16-R','H16-A','H64-R','H64-A') for f in p['families']):
                self.assertIn(p['m'], (6,7,8,9)); self.assertEqual(p['K'],0)
    def test_unavailable_bucket_is_explicit_and_not_replaced(self):
        rows=all_buckets(); rows[4].update(admission='UNAVAILABLE', rejection='test')
        cat=catalogue(rows)
        self.assertEqual(len(cat['buckets']),8)
        self.assertFalse(any(p['q']==6 and p['nominal_rate']=='1/2' for p in cat['profiles']))


class CodecTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.primitives=load_primitives()
    def setUp(self):
        from h64_catalog import SIZES
        self.scales=[(np.arange(s*s, dtype=np.int64)+17*i)%4096 for i,s in enumerate(SIZES)]
    def test_independent_roundtrip_and_bounded_flush(self):
        registry=[]
        enc, receipts=roundtrip(self.scales, lambda:Provider(registry, True), self.primitives)
        self.assertEqual(len(registry),5)
        for m in (6,7,8,9):
            self.assertEqual(receipts[m]['zero_extension_reads'],30)
            self.assertGreaterEqual(receipts[m]['flush_bits'],2)
        self.assertEqual(CODEC_PROTOCOL['class_condition'],1000)
        self.assertFalse(CODEC_PROTOCOL['cfg'])
    def test_noncanonical_tail_and_shortening_rejected(self):
        reg=[]
        enc,_=roundtrip(self.scales, lambda:Provider(reg), self.primitives, (6,))
        bits=enc[6]['bits']
        for altered in (bits[:-1], np.append(bits,0), bits[:1]):
            with self.assertRaises(InvalidCodeStream):
                decode_prefix(altered,6,lambda:Provider([]),self.primitives)
    def test_valid_wrong_token_stream_is_not_truth_corrected(self):
        enc,_=roundtrip(self.scales, lambda:Provider([]), self.primitives, (6,))
        bits=enc[6]['bits'].copy(); bits[0]^=1
        result=decode_prefix(bits,6,lambda:Provider([]),self.primitives)
        self.assertTrue(result['canonical'])
        self.assertNotEqual(int(result['scales'][0][0]),int(self.scales[0][0]))
    def test_fixed_uniform_length_matches_old_core(self):
        enc,_=roundtrip(self.scales,lambda:Provider([]),self.primitives,(6,))
        self.assertEqual(enc[6]['arithmetic_bits'],12*token_count(6)+2)


class FrameTests(unittest.TestCase):
    def setUp(self):
        self.profile=next(p for p in catalogue(all_buckets())['profiles'] if p['q']==6 and p['m']==1 and p['K']==0 and p['nominal_rate']=='1/2' and p['mode']=='raw')
    def test_crc_l_and_padding_are_separate(self):
        payload=integer_bits(123,12); info=pack_body(payload,self.profile)
        self.assertEqual(len(info),self.profile['k'])
        self.assertEqual(parse_body(info,self.profile)['payload'],payload.tolist())
        bad=info.copy();bad[-1]^=1
        self.assertEqual(parse_body(bad,self.profile)['status'],'CRC_REJECT')
        for location in (1,13+12):
            altered=info[:-16].copy();altered[location]^=1
            result=parse_body(append_crc(altered),self.profile)
            self.assertTrue(result['crc_accepted']);self.assertEqual(result['status'],'DECODE_INVALID')
    def test_wrong_but_crc_valid_payload_retained(self):
        result=parse_body(pack_body(integer_bits(234,12),self.profile),self.profile)
        self.assertEqual(raw_tokens(result['payload'],self.profile).tolist(),[234])
    def test_raster_partial_uses_exact_prefix_and_count(self):
        from h64_catalog import SIZES
        scales=[np.arange(s*s,dtype=np.int64) for s in SIZES]
        p=next(p for p in catalogue(all_buckets())['profiles'] if p['q']==6 and p['m']==7 and p['K']==81 and p['mode']=='raw')
        bits=raw_payload(scales,7,81)
        self.assertEqual(len(bits),2832)
        tokens=raw_tokens(bits,p)
        np.testing.assert_array_equal(tokens[-81:],scales[7][:81])
    def test_meter_before_failure_and_no_direct_tx_profile_in_frame_api(self):
        backend=FakeBackend();ledger=RecordingLedger()
        wave,_=transmit_body(backend,integer_bits(99,12),self.profile,11)
        backend.raise_decode=True
        with self.assertRaises(RuntimeError):
            receive_body(backend,wave,self.profile,19,11,ledger,'qualification','packet')
        self.assertEqual(ledger.events[0]['status'],'RESERVED')
        self.assertEqual(backend.calls,1)
    def test_frame_uses_decoded_header_profile_and_header_failure_skips_body(self):
        backend=FakeBackend();ledger=RecordingLedger()
        wave,_=transmit_body(backend,integer_bits(99,12),self.profile,11)
        codebook={self.profile['profile_id']:self.profile}
        frame=np.concatenate((np.zeros((68,2)),wave))
        result=receive_frame(backend,FakeHeader(self.profile['profile_id']),frame,19,11,codebook,ledger,'qualification','frame')
        self.assertEqual(result['body']['payload'],integer_bits(99,12).tolist())
        self.assertEqual([e['kind'] for e in ledger.events],['header','body'])
        ledger2=RecordingLedger()
        result=receive_frame(backend,FakeHeader(self.profile['profile_id'],False),frame,19,11,codebook,ledger2,'qualification','badheader')
        self.assertEqual(result['status'],'HEADER_REJECT');self.assertEqual(len(ledger2.events),1)
    def test_wrong_legal_header_uses_rx_layout_without_tx_repair(self):
        backend=FakeBackend();ledger=RecordingLedger()
        rx_profile=next(p for p in catalogue(all_buckets())['profiles'] if p['q']==4 and p['m']==6 and p['mode']=='raw')
        wave,_=transmit_body(backend,integer_bits(99,12),self.profile,11)
        frame=np.concatenate((np.zeros((68,2)),wave))
        codebook={self.profile['profile_id']:self.profile,rx_profile['profile_id']:rx_profile}
        result=receive_frame(backend,FakeHeader(rx_profile['profile_id']),frame,19,11,codebook,ledger,'qualification','wronglegal')
        self.assertEqual(backend.last_k,rx_profile['k'])
        self.assertEqual(result['body']['profile_key'],rx_profile['profile_key'])
        self.assertNotEqual(result['body']['profile_key'],self.profile['profile_key'])
    def test_profile_layout_mismatch_fails_before_packet_decode(self):
        p=dict(self.profile,layout_id='WRONG_REGISTERED_LAYOUT')
        backend=FakeBackend();ledger=RecordingLedger()
        with self.assertRaisesRegex(ValueError,'layout differ'):
            receive_body(backend,np.zeros((956,2)),p,19,11,ledger,'qualification','wronglayout')
        self.assertEqual(backend.calls,0);self.assertEqual(ledger.events,[])


class MappingTests(unittest.TestCase):
    def test_gray_mapping_power_and_uniform_ser(self):
        for q in (4,6):
            levels=pam(q)
            self.assertAlmostEqual(float(np.mean(levels**2)),1,places=14)
            labels=np.argsort(levels)
            self.assertTrue(all(bin(int(a)^int(b)).count('1')==1 for a,b in zip(labels[:-1],labels[1:])))
            for snr in (0,7,13,16,19):
                self.assertAlmostEqual(exact_uncoded_ser(snr,q),exact_ser_from_decision_regions(snr,q),places=13)
    def test_noiseless_all_labels_and_actual_energy_varies(self):
        for q in (4,6):
            bits=np.concatenate([integer_bits(i,q) for i in range(1<<q)])[None]
            wave=modulate(bits,q)
            np.testing.assert_array_equal((demap(wave,19,q)>0).astype(np.uint8),bits)
            self.assertAlmostEqual(float(np.square(wave.astype(np.float64)).sum())/(1<<q),2,places=6)
            self.assertGreater(np.ptp(np.square(wave).sum(-1)),0)
    def test_llrs_match_full_constellation_sum(self):
        for q in (4,6):
            bits=np.stack([integer_bits(i,q) for i in range(1<<q)])
            constellation=modulate(bits,q)[:,0,:].astype(np.float64)
            y=np.array([[[.11,-.37],[1.9,-1.4]]],dtype=np.float64)
            metric=-.5*10**1.3*np.square(y[0,:,None,:]-constellation[None,:,:]).sum(-1)
            expected=[]
            for pos in range(q):
                mask=bits[:,pos].astype(bool)
                expected.append(logsumexp(metric[:,mask],-1)-logsumexp(metric[:,~mask],-1))
            np.testing.assert_allclose(demap(y,13,q),np.stack(expected,-1).reshape(1,-1),rtol=2e-6,atol=3e-6)
    def test_scramble_public_counter_only_reproducible(self):
        np.testing.assert_array_equal(scramble_mask(5736,7),scramble_mask(5736,7))
        self.assertFalse(np.array_equal(scramble_mask(5736,7),scramble_mask(5736,8)))


if __name__=='__main__': unittest.main()

"""Synthetic CPU validation of actual source decode and H receipt parsing."""
import argparse
import hashlib
from pathlib import Path
import tempfile
import unittest
import numpy as np
import t1_entropy_core as c
from t1_codec_runtime import SourceCodec,InvalidSourceStream
from t1_source_check import load_h_source

ROOT=None
class RuntimeTests(unittest.TestCase):
    def setup_static(self,d):
        entropy,_,_=c.load_integer_codec(ROOT)
        counts=np.ones((10,4096),dtype=np.int64)*5
        cdf=entropy.probability_cdf(np.log(counts+.5))
        p=d/'model.npz';np.savez(p,counts=counts,cdf=cdf)
        done=d/'static.json';c.write(done,dict(status='T1_STATIC_TRAIN20K_CDF_COMPLETE',source_count=20000,
            source_role='original_train20k',calibration_used_for_fitting=False,outputs={str(p):c.sha(p)}))
        return SourceCodec(ROOT,static_completion=done)

    def test_actual_decoder_recovers_all_prefixes_without_truth(self):
        with tempfile.TemporaryDirectory() as temp:
            codec=self.setup_static(Path(temp)); tokens=np.arange(680)%4096
            tx=codec.encode('EC_STATIC_WHOLE',tokens,modes=(4,5,6,7,8,9))
            for m,row in tx.items():
                rx=codec.decode('EC_STATIC_WHOLE',row['bits'],m)
                self.assertTrue(np.array_equal(rx['received_tokens'],tokens[:c.OFFSETS[m]]))
                self.assertFalse(rx['receiver_truth_used']);self.assertEqual(rx['zero_extension_reads'],30)

    def test_malformed_and_noncanonical_streams_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            codec=self.setup_static(Path(temp)); row=codec.encode('EC_STATIC_WHOLE',np.arange(680),modes=(5,))[5]
            for bits in [[],[2,0],np.append(row['bits'],0),row['bits'][:20]]:
                with self.assertRaises(InvalidSourceStream):codec.decode('EC_STATIC_WHOLE',bits,5)

    def fixture_h(self,d):
        sid='synthetic-source';tokens=np.arange(680,dtype=np.int64)
        pixels=np.zeros((3,256,256),dtype=np.uint8);pre=hashlib.sha256(pixels.tobytes()).hexdigest()
        tsha=hashlib.sha256(b'int64:680\0'+tokens.astype('<i8').tobytes()).hexdigest()
        a=d/'tokens.npz';np.savez(a,tokens=tokens,pixels=pixels)
        ap=d/'asset.json';c.write(ap,dict(source_id=sid,source_index=0,tokens_sha256=tsha,preprocessing_id=pre,
            archive=str(a),outputs={str(a):c.sha(a)}))
        bits=d/'bits.npz';np.savez(bits,**{f'm{m}_bits':np.array([0,1],dtype=np.uint8) for m in (6,7,8,9)})
        cp=d/'codec.json';c.write(cp,dict(status='H_FULL1000_SOURCE_CODEC_SOURCE_COMPLETE',source_index=0,
            source_id=sid,independent_roundtrip=True,registration_sha256='synthetic-reg',tokens_sha256=tsha,
            preprocessing_id=pre,source_assets_checkpoint=dict(path=str(ap),sha256=c.sha(ap)),
            outputs={str(bits):c.sha(bits)},lengths=[dict(m=m,raw_bits=int(12*c.OFFSETS[m]),arithmetic_bits=2,
            flush_bits=2,zero_extension_reads=30) for m in (6,7,8,9)]))
        done=dict(registration_sha256='synthetic-reg',records=[dict(source_index=0,source_id=sid,
            checkpoint=str(cp),sha256=c.sha(cp))],outputs={str(cp):c.sha(cp),str(bits):c.sha(bits)})
        cal=dict(source_ids=[sid],preprocessing_ids=[pre])
        return done,cal,a

    def test_receipt_chain_source_arrays_and_existing_lengths(self):
        with tempfile.TemporaryDirectory() as temp:
            done,cal,_=self.fixture_h(Path(temp));tokens,streams,rows,cp,bindings=load_h_source(done,0,cal)
            self.assertEqual(set(streams),{6,7,8,9});self.assertEqual(len(tokens),680)
            self.assertEqual(len(bindings),4);self.assertEqual(len(rows),4)

    def test_changed_source_array_fails_before_reuse(self):
        with tempfile.TemporaryDirectory() as temp:
            done,cal,array=self.fixture_h(Path(temp));array.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'SHA256'):load_h_source(done,0,cal)

def main():
    global ROOT
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    ROOT=Path(a.root);r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(RuntimeTests))
    c.write(a.out,dict(status='PASS' if r.wasSuccessful() else 'FAIL',synthetic_only=True,tests_run=r.testsRun,
        failures=len(r.failures),errors=len(r.errors),new_real_model_calls=0,new_real_channel_decodes=0))
    return 0 if r.wasSuccessful() else 1

if __name__=='__main__':raise SystemExit(main())

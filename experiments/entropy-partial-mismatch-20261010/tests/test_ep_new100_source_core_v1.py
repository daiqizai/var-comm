"""Synthetic pixels, fake encoder/provider and bounded source IO only."""
from contextlib import nullcontext
import copy
import hashlib
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock
import numpy as np
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import ep_new100_source_core_v1 as c
import ep_source_codec as partial


class Ledger:
    def __init__(self):self.calls={k:0 for k in c.CAPS}
    def call(self,kind,fn,**unused):
        c.require(self.calls[kind]<c.CAPS[kind],'Fixture cap exhausted');self.calls[kind]+=1;return fn()


class SourceCoreTests(unittest.TestCase):
    def fixture(self,folder):
        pixels=np.zeros((3,256,256),np.uint8);pixels[:,0,0]=1
        archive=Path(folder)/'pixels.npz'
        with archive.open('xb') as f:np.savez(f,pixels=pixels)
        pin=c.descriptor(archive);pin['bytes']=archive.stat().st_size
        record=dict(source_index=0,source_id='synthetic-only',canonical_source_id='fixture-only',evaluation_class_index=0,
            pixels_archive=pin,preprocessing_id=c.pixel_sha(pixels),horizontal_flip_pixel_sha256=c.pixel_sha(np.ascontiguousarray(pixels[:,:,::-1])))
        ledger=Ledger();backend=types.SimpleNamespace(source_index=0,current=None,traces=[],tx_cdfs={},received={},
            t=types.SimpleNamespace(no_grad=nullcontext),native=object(),boundary=lambda:None,
            g=types.SimpleNamespace(image_sha=lambda x:hashlib.sha256(x.tobytes()).hexdigest()))
        def encode(tokens,endpoints):
            self.assertEqual(tokens.shape,(680,))
            for scale in range(10):
                ledger.call('prior_scale',lambda:None)
                backend.traces.append(dict(source=backend.source_index,role='TX',m=10,scale=scale,cdf_sha256='a'*64))
            return {(m,K):dict(bits=np.array([0,1,m%2,K%2],np.uint8),m=m,K=K,arithmetic_bits=4) for m,K in endpoints}
        codec=types.SimpleNamespace(encode_endpoints=encode)
        encoder=mock.Mock(return_value=(np.arange(680,dtype=np.int64),np.zeros((32,16,16),np.float32)))
        return record,backend,codec,ledger,encoder

    def test_one_encoder_one_TX_ten_priors_zero_RX_or_render(self):
        with tempfile.TemporaryDirectory() as folder:
            record,b,codec,ledger,encoder=self.fixture(folder)
            result=c.encode_source(record,b,codec,partial,ledger,encoder,Path,Path(folder)/'source')
            encoder.assert_called_once();self.assertEqual(ledger.calls,dict.fromkeys(c.CAPS,0)|dict(encoder=1,source_tx=1,prior_scale=10))
            self.assertEqual(result['actual_new_encoder'],1);self.assertEqual(result['independent_RX_calls'],0)
            with np.load(result['source_assets']['path'],allow_pickle=False) as z:
                self.assertEqual(set(z.files),{'pixels','tokens','F'});self.assertEqual(z['tokens'].shape,(680,))
            with np.load(result['archive']['path'],allow_pickle=False) as z:self.assertEqual(len(z.files),24)
            self.assertEqual(b.traces,[]);self.assertEqual(b.received,{})

    def test_changed_closed_pixels_rejected_before_encoder(self):
        with tempfile.TemporaryDirectory() as folder:
            record,b,codec,ledger,encoder=self.fixture(folder);record['preprocessing_id']='f'*64
            with self.assertRaisesRegex(RuntimeError,'content-gate hashes'):
                c.encode_source(record,b,codec,partial,ledger,encoder,Path,Path(folder)/'source')
            encoder.assert_not_called();self.assertEqual(sum(ledger.calls.values()),0)

    def test_latent_dtype_or_token_domain_rejected_without_TX(self):
        with tempfile.TemporaryDirectory() as folder:
            record,b,codec,ledger,encoder=self.fixture(folder)
            encoder.return_value=(np.full(680,4096,np.int64),np.zeros((32,16,16),np.float32))
            with self.assertRaisesRegex(RuntimeError,'encoder output malformed'):
                c.encode_source(record,b,codec,partial,ledger,encoder,Path,Path(folder)/'source')
            self.assertEqual(ledger.calls['encoder'],1);self.assertEqual(ledger.calls['source_tx'],0)

    def test_same_source_directory_never_replayed(self):
        with tempfile.TemporaryDirectory() as folder:
            record,b,codec,ledger,encoder=self.fixture(folder);out=Path(folder)/'source';out.mkdir()
            with self.assertRaises(FileExistsError):c.encode_source(record,b,codec,partial,ledger,encoder,Path,out)
            encoder.assert_not_called()

    def test_source_scope_and_total_budget(self):
        self.assertEqual(c.CAPS,dict(model_load=1,encoder=100,source_tx=100,source_rx=0,var_render=0,prior_scale=1000,decoder_forward=0))
        with tempfile.TemporaryDirectory() as folder:
            record,b,codec,ledger,encoder=self.fixture(folder);record['source_index']=100
            with self.assertRaisesRegex(RuntimeError,'outside fixed scope'):
                c.encode_source(record,b,codec,partial,ledger,encoder,Path,Path(folder)/'source')
            encoder.assert_not_called()

    def test_encoder_AST_loader_does_not_execute_module_top_level(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'original.py';path.write_text("raise RuntimeError('top-level must not execute')\ndef encode_flat(native,pixels):\n    require(True,'valid')\n    return native,pixels\n")
            with mock.patch.object(c,'sha',return_value=c.ENCODER_SHA):fn,proof=c.encoder_projection(path)
            self.assertEqual(fn(1,2),(1,2));self.assertFalse(proof['AST_body_changed'])
            self.assertFalse(proof['original_module_top_level_executed'])
            self.assertEqual(proof['loaded_globals'],['require'])

    def test_missing_encoder_global_dependency_refused_before_execution(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'original.py';path.write_text('def encode_flat(native,pixels):\n    return unknown_global(native,pixels)\n')
            with mock.patch.object(c,'sha',return_value=c.ENCODER_SHA),self.assertRaisesRegex(RuntimeError,'global dependencies'):
                c.encoder_projection(path)

    def test_changed_encoder_source_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'original.py';path.write_text('def encode_flat(native,pixels): pass\n')
            with self.assertRaisesRegex(RuntimeError,'source bytes changed'):c.encoder_projection(path)


if __name__=='__main__':unittest.main()

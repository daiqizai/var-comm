"""CPU contracts; real CUDA parity is a separate mandatory qualification."""
import hashlib
import inspect
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import source_quality as q
import quality_driver as driver

try:
    import torch
except ImportError:
    torch = None


class Contracts(unittest.TestCase):
    def test_full_partial_normalizes_and_terminal_is_exact(self):
        self.assertEqual(q.canonical_state(8, 169), (9, 0))
        self.assertEqual(q.canonical_state(9, 256), (10, 0))
        self.assertEqual(q.state_id(6, 32), 'm6_K32')
        for args in [(True, 0), (4, 1.0), (4, -1), (4, 26), (10, 1), (3, 0)]:
            with self.assertRaises(RuntimeError): q.canonical_state(*args)

    def test_registered_state_ids_are_unambiguous(self):
        values = [dict(state_id='m6_K32', m=6, K=32), dict(state_id='m4_K0', m=4, K=0)]
        self.assertEqual(q.validate_states(values)[0]['state_id'], 'm4_K0')
        for bad in [values*2, [dict(state_id='m4_K25', m=4, K=25)],
                    [dict(state_id='m4_K0', m=4.1, K=0)]]:
            with self.assertRaises(RuntimeError): q.validate_states(bad)

    def test_entropy_uses_original_rounding_then_spatial_index(self):
        old = SimpleNamespace(ENTROPY_DECIMALS=6,
            entropy_values=lambda _: np.array([.2, .20000001, .6, .6, .1]))
        np.testing.assert_array_equal(q.entropy_order(old, np.zeros((5, 4096))), [2, 3, 0, 1, 4])

    def test_token_partition_never_pads_or_clips(self):
        values = np.arange(680, dtype=np.int64)
        self.assertEqual([len(x) for x in q.split_tokens(values)], [s*s for s in q.SIZES])
        for bad in [values[:-1], values.astype(float), np.full(680, 4096)]:
            with self.assertRaises(RuntimeError): q.split_tokens(bad)

    def test_actual_payload_key_changes_with_every_accepted_input(self):
        prefix = q.split_tokens(np.arange(680, dtype=np.int64))[:4]
        args = (prefix, np.array([2, 3]), np.array([101, 102]), {'model': 'a'})
        key = q.accepted_payload_key(*args)
        changed = [x.copy() for x in prefix]; changed[0][0] = 17
        variants = [(changed, args[1], args[2], args[3]),
                    (prefix, np.array([2, 4]), args[2], args[3]),
                    (prefix, args[1], np.array([101, 103]), args[3]),
                    (prefix, args[1], args[2], {'model': 'b'})]
        for variant in variants: self.assertNotEqual(key, q.accepted_payload_key(*variant))
        self.assertEqual(key, q.accepted_payload_key(*args))

    def test_actual_receiver_api_has_no_source_truth_or_labels(self):
        self.assertEqual(list(inspect.signature(q.ReceivedRenderer.render_received).parameters),
            ['self', 'prefix_tokens', 'partial_values', 'm', 'K', 'receiver', 'cache'])

    def test_float_image_domain_matches_existing_published_fingerprints(self):
        image = np.full((3, 256, 256), .5, dtype=np.float32)
        self.assertEqual(q.rgb_sha(image), hashlib.sha256(b'float32:3,256,256:RGB\0'+image.tobytes()).hexdigest())
        for bad in [image.astype(np.float16), image[:, :32], image*3]:
            with self.assertRaises(RuntimeError): q.rgb_sha(bad)

    def test_reuse_rejects_development_conditional_models_or_population_changes(self):
        renderer = {'models': 'model-sha', 'generation': q.GENERATION}
        good = dict(population='calibration', label_conditioned=False,
                    renderer_identity=renderer, source_ids=['a', 'b'], metric=q.PRIMARY_METRIC)
        q.validate_reuse_registration(good, renderer, ['a', 'b'])
        for delta in [dict(population='development'), dict(label_conditioned=True),
                      dict(renderer_identity={'models': 'other'}), dict(source_ids=['b', 'a']),
                      dict(metric='clip_image_cosine')]:
            with self.assertRaises(RuntimeError): q.validate_reuse_registration(dict(good, **delta), renderer, ['a', 'b'])

    def test_checkpoint_hash_and_archive_reference_are_enforced(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp); image = np.full((3, 256, 256), .5, np.float32)
            archive = tmp/'images.npz'
            np.savez_compressed(archive, images=image[None], source_rgb=image,
                                row_ids=np.array(['row']), image_slots=np.array([0]))
            row = dict(row_id='row', state_id='m4_K0', receiver='VAR', image_sha256=q.rgb_sha(image), reference_sha256=q.rgb_sha(image))
            cp = dict(binding='a', rows=[row], float_reconstructions=dict(path=str(archive), sha256=q.sha(archive)))
            cp['payload_sha256'] = q.identity(cp)
            path = tmp/'cp.json'; q.write(path, cp)
            self.assertEqual(q._read_checkpoint(path, 'a')[2][('m4_K0', 'VAR')][0], row)
            with self.assertRaises(RuntimeError): q._read_checkpoint(path, 'b')
            cp['rows'][0]['image_sha256'] = 'wrong'; cp['payload_sha256'] = q.identity({k:v for k,v in cp.items() if k!='payload_sha256'})
            q.write(path, cp)
            with self.assertRaises(RuntimeError): q._read_checkpoint(path, 'a')

    def test_safe_pause_checks_scope_before_model_rendering(self):
        class Tensor:
            def cpu(self): return self
            def numpy(self): return np.arange(680, dtype=np.int64)
        # Shared small image is safe: a source's copied record is not mutated.
        image = np.zeros((3,256,256), np.uint8)
        records = [dict(image_id=str(i), preprocessing_id='p', pixels=image) for i in range(1000)]
        calls=[]
        native = SimpleNamespace(loaded={'identity': {'models': {}}}, flags={}, frozen=lambda: calls.append('frozen'),
                                 data=lambda role: (calls.append(role) or dict(records=records,T=[Tensor()]*1000,bindings={})))
        qualification=dict(status='REAL_SOURCE_QUALITY_PARITY_PASS', source_role='calibration',
            frozen_identity=native.loaded['identity'], source_sha256=q.sha(q.__file__))
        with tempfile.TemporaryDirectory() as tmp:
            states=Path(tmp)/'states.json'; q.write(states, dict(registered_before_quality=True, development_read=False,
                                                    states=[dict(state_id='m4_K0',m=4,K=0)]))
            out=Path(tmp)/'out'
            with patch.object(q,'SourceRenderer',side_effect=AssertionError('must not render on STOP')):
                self.assertIsNone(q.run_quality(native,states,out,lambda *_:[],{}, qualification=qualification,
                                                 source_limit=300,stop_requested=lambda:True))
            self.assertEqual(calls,['calibration','frozen'])
            self.assertEqual(q.read(out/'status.json')['status'],'PAUSED_AT_SOURCE_BOUNDARY')
            self.assertNotIn('source_index',records[0])

    def test_metric_loader_interop_wrapper_only_allows_same_value_and_restores(self):
        calls=[]
        setter=lambda value:calls.append(value)
        fake=SimpleNamespace(set_num_interop_threads=setter,get_num_interop_threads=lambda:2)
        with driver.verify_repeated_interop(fake):
            fake.set_num_interop_threads(2)
            with self.assertRaises(RuntimeError):fake.set_num_interop_threads(6)
        self.assertIs(fake.set_num_interop_threads,setter)
        self.assertEqual(calls,[])
        with self.assertRaises(ValueError):
            with driver.verify_repeated_interop(fake):raise ValueError('test cleanup')
        self.assertIs(fake.set_num_interop_threads,setter)


@unittest.skipIf(torch is None, 'Torch is unavailable locally; run on the native environment for tensor tests')
class TensorContracts(unittest.TestCase):
    def fixture(self):
        class Quantizer:
            def __init__(self):
                self.embedding = torch.nn.Embedding(4096, 2)
                with torch.no_grad(): self.embedding.weight.copy_(torch.arange(8192).reshape(4096, 2)+10)
                self.calls=[]
            def get_next_autoregressive_input(self,m,n,latent,embedded):
                self.calls.append(embedded.clone())
                return latent+embedded.sum(), None
        renderer=object.__new__(q.ReceivedRenderer)
        renderer.torch=torch; renderer.device='cpu'; renderer.vae=SimpleNamespace(quantize=Quantizer())
        return renderer

    def test_unknown_partial_positions_are_zero_vectors_not_token_zero(self):
        renderer=self.fixture(); latent=torch.zeros(1,2,16,16)
        renderer.direct_partial(latent,4,np.array([2,7]),np.array([4,5]))
        vectors=renderer.vae.quantize.calls[0].reshape(1,2,25)
        np.testing.assert_array_equal(vectors[0,:,2].detach(),renderer.vae.quantize.embedding.weight[4].detach())
        mask=torch.ones(25,dtype=torch.bool);mask[[2,7]]=False
        self.assertTrue(torch.equal(vectors[:,:,mask],torch.zeros(1,2,23)))
        self.assertTrue(torch.all(renderer.vae.quantize.embedding.weight[0]!=0))

    def test_empty_partial_skips_entire_quantizer_contribution(self):
        renderer=self.fixture(); latent=torch.ones(1,2,16,16)
        result=renderer.direct_partial(latent,4,np.array([],int),np.array([],int))
        self.assertTrue(torch.equal(result,latent));self.assertEqual(renderer.vae.quantize.calls,[])
        self.assertNotEqual(result.data_ptr(),latent.data_ptr())


if __name__=='__main__': unittest.main()

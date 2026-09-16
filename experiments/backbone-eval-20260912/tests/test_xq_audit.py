"""Independent CPU audit with tiny *untrained* instances of official source.

This checks source/API compatibility and algebra, NOT pretrained image quality or
checkpoint validity. The released architecture/weights are not allocated/read.
Run separately from old-VAR import tests because both upstreams use ``models``.
"""
import contextlib
import importlib
import io
import os
from pathlib import Path
import sys
import unittest
import warnings
from unittest.mock import patch

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'vendor/xqgan-source'))
import torch

torch.set_num_threads(2)


class TinySourceAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            from tokenizer.tokenizer_image import xqgan_model as xm
            from tokenizer.tokenizer_image.dino_enc import dinov2 as de
            from tokenizer.tokenizer_image.dino_enc import vision_transformer as custom_vit
            from models.var import VAR
            from models.helpers import sample_with_top_k_top_p_
            import timm
            import dist
        from xq_adapter import ModelAdapter, SCALES
        cls.xm, cls.de, cls.custom_vit, cls.timm = xm, de, custom_vit, timm
        cls.source_factory = timm.create_model
        cls.scales = SCALES
        setattr(dist, '__device', 'cpu')

        def factory(name, *args, **kwargs):
            # Same registry/factory route as adapter; only random test width/depth
            # shrunk to keep this review a small CPU test, never a model evaluation.
            kwargs.update(pretrained=False, embed_dim=32, depth=1, num_heads=4)
            return cls.source_factory(name, *args, **kwargs)

        args = xm.ModelArgs(codebook_size=4096, codebook_embed_dim=32,
            v_patch_nums=list(SCALES), enc_type='dinov2', dec_type='dinov2',
            semantic_guide='dinov2', detail_guide='none', num_latent_tokens=121,
            encoder_model='vit_base_patch14_dinov2.lvd142m',
            decoder_model='vit_base_patch14_dinov2.lvd142m', abs_pos_embed=True,
            share_quant_resi=4, product_quant=2, half_sem=True, test_model=True)
        torch.manual_seed(735)
        with patch.object(xm, 'create_model', factory), patch.object(de, 'create_model', factory), \
                patch.object(torch.distributed, 'get_rank', return_value=0), \
                patch.object(torch.distributed, 'get_world_size', return_value=1), \
                warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
            warnings.simplefilter('ignore')
            cls.vae = xm.VQModel(args).eval().requires_grad_(False)
            cls.var = VAR(vae_local=cls.vae, num_classes=1000, depth=2, embed_dim=32,
                num_heads=4, drop_path_rate=0, norm_eps=1e-6, shared_aln=False,
                cond_drop_rate=0, attn_l2_norm=True, patch_nums=SCALES,
                flash_if_available=False, fused_if_available=False, p_drop=0).eval().requires_grad_(False)
            cls.var.init_weights()
        cls.adapter = ModelAdapter.__new__(ModelAdapter)
        cls.adapter.device = torch.device('cpu')
        cls.adapter.vae, cls.adapter.var = cls.vae, cls.var
        cls.adapter.sampler = sample_with_top_k_top_p_
        cls.adapter.last_audit = {}
        cls.source = torch.randn(1, 3, 256, 256).clamp(-1, 1)
        cls.labels = torch.tensor([17])
        with torch.inference_mode(), warnings.catch_warnings():
            warnings.simplefilter('ignore')
            cls.tokens = cls.adapter.encode(cls.source)

    def test_factory_routes_to_custom_vit_required_by_four_dimensional_pos_embed(self):
        from timm.models import model_entrypoint
        self.assertTrue(model_entrypoint('vit_base_patch14_dinov2').__module__.endswith('dino_enc.vision_transformer'))
        self.assertIsInstance(self.vae.encoder.model, self.custom_vit.VisionTransformer)
        self.assertIsInstance(self.vae.decoder.model, self.custom_vit.VisionTransformer)
        with torch.inference_mode():
            self.assertEqual(tuple(self.vae.encoder.model._pos_embed(torch.zeros(1, 11, 11, 32)).shape), (1, 122, 32))
        self.assertFalse(torch.cuda.is_initialized())

    def test_real_upstream_shape_and_both_product_branches(self):
        from xq_adapter import token_metadata
        for branch in self.tokens:
            self.assertEqual([list(t.shape) for t in branch], [[1, p*p] for p in self.scales])
        self.assertEqual(len(self.tokens), 2)
        for m, bits in ((8, 2424), (9, 3960), (10, 6864)):
            self.assertEqual(token_metadata(self.tokens, m)['raw_bits'], bits)

    def test_native_full_roundtrip_against_source_not_self_comparison(self):
        with warnings.catch_warnings(), torch.inference_mode():
            warnings.simplefilter('ignore')
            result = self.adapter.decode(self.tokens, 10)
            native = self.vae.img_to_reconstructed_img(self.source, last_one=True).add(1).mul(.5)
        self.assertLessEqual(float((result-native).abs().max()), 2e-5)

    def test_natural_prefix_matches_native_intermediate_reconstruction(self):
        with warnings.catch_warnings(), torch.inference_mode():
            warnings.simplefilter('ignore')
            native = self.vae.img_to_reconstructed_img(self.source, last_one=False)
            for m in (8, 9):
                prefix = [branch[:m] for branch in self.tokens]
                result = self.adapter.decode(prefix, m)
                self.assertLessEqual(float((result - native[m-1].add(1).mul(.5)).abs().max()), 2e-5)

    def test_zero_prefix_sampling_exact_match_official_algorithm(self):
        # No source tokens at m=0: same RNG/head/PQ sampling path must produce the
        # same image as the unmodified official method at the same seed/cfg.
        with warnings.catch_warnings(), torch.inference_mode():
            warnings.simplefilter('ignore')
            reference = self.var.autoregressive_infer_cfg(B=1, label_B=self.labels, g_seed=41,
                cfg=3.25, top_k=750, top_p=.95, more_smooth=False, joint_sample=False)
            observed = self.adapter.complete([[], []], self.labels, 0, mode='sample', seed=41, cfg=3.25)
        self.assertTrue(torch.equal(reference, observed), float((reference-observed).abs().max()))

    def test_conditional_argmax_replays_and_preserves_both_prefixes(self):
        from xq_adapter import token_hash
        for m in (8, 9):
            prefix = [[t.clone() for t in branch[:m]] for branch in self.tokens]
            before = token_hash(prefix)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                first = self.adapter.complete(prefix, self.labels, m, mode='argmax', cfg=1.0)
                second = self.adapter.complete(prefix, self.labels, m, mode='argmax', cfg=1.0)
            self.assertTrue(torch.equal(first, second))
            self.assertEqual(token_hash(prefix), before)
            self.assertEqual(self.adapter.last_audit['input_hash'], before)
            self.assertEqual(self.adapter.last_audit['used_prefix_hash'], before)
            self.assertEqual(self.adapter.last_audit['received_true_suffix_indices'], 0)
            self.assertTrue(self.adapter.last_audit['prefix_preserved'])

    def test_full_received_identity(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            expected = self.adapter.decode(self.tokens, 10)
            actual = self.adapter.complete(self.tokens, self.labels, 10, mode='argmax', cfg=1)
        self.assertTrue(torch.equal(expected, actual))
        self.assertTrue(self.adapter.last_audit['full_token_identity'])

    def test_strict_complete_tiny_state_restoration_enforced(self):
        state = self.vae.state_dict()
        self.vae.load_state_dict(state, strict=True)
        self.var.load_state_dict(self.var.state_dict(), strict=True)
        missing = {key: value for key, value in state.items() if key != 'encoder.model.pos_embed'}
        with self.assertRaisesRegex(RuntimeError, 'Missing key'):
            self.vae.load_state_dict(missing, strict=True)


class RestrictedCheckpointCompatibilityAudit(unittest.TestCase):
    def test_reviewed_metadata_and_tensors_roundtrip_via_private_restricted_module(self):
        import argparse
        import tempfile
        from ruamel.yaml.comments import CommentedSeq
        from ruamel.yaml.scalarfloat import ScalarFloat
        from checkpoint_io import _module
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'tiny-approved.pt'
            original = {'model': {'w': torch.arange(4)},
                        'args': argparse.Namespace(scales=CommentedSeq([1, 1, 2]), lr=ScalarFloat(0.1))}
            torch.save(original, path)
            # Private module exercised on a tiny synthetic fixture only. Production
            # entrypoint continues requiring exact public artifact size and hash.
            restored = torch.load(path, map_location='cpu', mmap=True, weights_only=False,
                                  pickle_module=_module())
            self.assertTrue(torch.equal(restored['model']['w'], original['model']['w']))
            self.assertEqual(restored['args'].scales, [1, 1, 2])
            self.assertEqual(float(restored['args'].lr), 0.1)
            self.assertFalse(torch.cuda.is_initialized())

    def test_forbidden_globals_rejected_without_invocation(self):
        import pickle
        from checkpoint_io import _module
        # GLOBAL + STOP only; even an accidentally unrestricted loader would merely
        # resolve a function, never call it. No REDUCE/shell/eval command is run.
        for payload in (b'cos\nsystem\n.', b'cposix\nsystem\n.', b'cbuiltins\neval\n.'):
            with self.subTest(payload=payload), self.assertRaises(pickle.UnpicklingError):
                _module().Unpickler(io.BytesIO(payload)).load()

    def test_public_entrypoint_rejects_unpinned_files_before_deserializing(self):
        import tempfile
        from checkpoint_io import load_xq_checkpoint
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'unlisted.pt'
            path.write_bytes(b'not an approved checkpoint')
            with patch.object(torch, 'load') as loader:
                with self.assertRaisesRegex(RuntimeError, 'Only the two exact'):
                    load_xq_checkpoint(path)
                loader.assert_not_called()


if __name__ == '__main__':
    unittest.main()

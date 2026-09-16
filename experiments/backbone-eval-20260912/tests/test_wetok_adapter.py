"""CPU-only engineering tests; no checkpoints, images, or GPU required."""
import importlib.util
from pathlib import Path
import unittest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('wetok_adapter', ROOT/'scripts/wetok_adapter.py')
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


class WeTokAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.quant_code = adapter._import_file(
            '_test_wetok_lfq', ROOT/'vendor/wetok'/adapter.SOURCE_FILES['quantizer'])

    def test_pinned_sources(self):
        for key, rel in adapter.SOURCE_FILES.items():
            self.assertEqual(adapter._sha256(ROOT/'vendor/wetok'/rel), adapter.SOURCE_SHA256[key])
        self.assertEqual(adapter._sha256(ROOT/'vendor/wetok/configs/WeToK/Inference/ImageNet_downsample16_imagenet.yaml'), adapter.CONFIG_SHA256)

    def test_discrete_full_budget_and_group_layout(self):
        q = self.quant_code.LFQ(dim=32, codebook_size=256, num_codebooks=4).eval()
        rng = torch.Generator().manual_seed(231)
        h = torch.randn((2, 32, 16, 16), generator=rng)
        with torch.inference_mode():
            quantized, _, indices = q(h)
            self.assertEqual(list(indices.shape), [2048])
            grid = (2, 16, 16, 4)
            reconstructed = q.decode(indices.reshape(grid)).permute(0, 3, 1, 2)
            error = float((quantized-reconstructed).abs().max())
            self.assertLessEqual(error, 1e-5)
            self.assertTrue(torch.equal(reconstructed, torch.where(h > 0, 1., -1.)))
            fixture = adapter.WeTokAdapter.__new__(adapter.WeTokAdapter)
            fixture.num_codebooks = 4
            fixture.codebook_size = 256
            fixture.bits_per_index = 8
            tokens = {'indices':indices, 'index_grid_shape':grid, 'latent_shape':tuple(h.shape),
                      'official_quantized_vs_index_roundtrip_max_abs':error}
            meta = fixture.token_metadata(tokens)
            self.assertEqual(meta['raw_bits'],8192)
            self.assertEqual(meta['index_count'],1024)
            self.assertEqual(meta['index_count_total'],2048)
            self.assertEqual(meta['shapes'],[[2048]])
            self.assertEqual(meta['logical_index_shape'],[2,16,16,4])
            self.assertLessEqual(meta['index_max'],255)
            self.assertGreaterEqual(meta['index_min'],0)
            with self.assertRaises(ValueError):
                fixture.token_metadata(tokens,prefix_scales=8)
            with self.assertRaises(ValueError):
                fixture.decode(tokens,prefix_scales=8)

    def test_ema_mapping_and_missing_weight_fail_closed(self):
        codec = nn.Module()
        codec.encoder = nn.Linear(3,4)
        codec.decoder = nn.Linear(4,3)
        state = {'model_ema.'+name.replace('.',''):torch.ones_like(value)
                 for name,value in codec.state_dict().items()}
        selected, ema_keys = adapter._select_ema_state(codec,state)
        self.assertEqual(len(ema_keys),4)
        codec.load_state_dict(selected,strict=True)
        self.assertTrue(all(torch.all(p==1) for p in codec.parameters()))
        state.pop('model_ema.encoderweight')
        with self.assertRaises(RuntimeError):
            adapter._select_ema_state(codec,state)

    def test_meta_architecture_parameter_count(self):
        module = adapter._import_file(
            '_test_wetok_codec', ROOT/'vendor/wetok'/adapter.SOURCE_FILES['model'])
        with torch.device('meta'):
            enc = module.Encoder(**adapter.EXPECTED_DDCONFIG)
            dec = module.Decoder(**adapter.EXPECTED_DDCONFIG)
        count = sum(x.numel() for x in enc.parameters()) + sum(x.numel() for x in dec.parameters())
        self.assertGreater(count,100_000_000)
        # Only structure inspected, no actual model forward/weights allocated.
        print('WeTok codec inference parameter count:',count)

if __name__ == '__main__':
    unittest.main(verbosity=2)

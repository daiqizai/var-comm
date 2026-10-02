"""CPU contract checks only; synthetic stubs are never scientific measurements."""
import copy
import importlib.util
import socket
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import metric_models as mm

HAS_TORCH = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("torchvision") is not None


class AssetContracts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def file(self, name="model.py", content=b"registered source"):
        p = self.root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content)
        return p, {"path": str(p), "sha256": mm.sha256_file(p)}

    def tree(self):
        p, spec = self.file()
        return {"path": str(self.root), "revision": "a" * 40, "files": {p.name: spec["sha256"]}}

    def test_file_sha_and_tamper(self):
        p, spec = self.file()
        self.assertEqual(mm.verify_file(spec), p)
        p.write_bytes(b"modified")
        with self.assertRaisesRegex(ValueError, "mismatch"):
            mm.verify_file(spec)

    def test_missing_and_wrong_registered_digest(self):
        _, spec = self.file()
        with self.assertRaises(ValueError):
            mm.verify_file({**spec, "sha256": ""})
        with self.assertRaises(ValueError):
            mm.verify_file(spec, expected_sha="0" * 64)

    def test_full_source_manifest(self):
        spec = self.tree()
        self.assertEqual(mm.verify_tree(spec), self.root)
        self.file("nested/unregistered.py")
        with self.assertRaisesRegex(ValueError, "Unregistered"):
            mm.verify_tree(spec)

    def test_wrong_source_revision(self):
        spec = self.tree()
        with self.assertRaises(ValueError):
            mm.verify_tree(spec, expected_revision="b" * 40)
        spec["revision"] = "main"
        with self.assertRaises(ValueError):
            mm.verify_tree(spec)

    def test_asset_escape(self):
        spec = self.tree()
        nested = self.root / "nested"
        nested.mkdir()
        spec["path"] = str(nested)
        spec["files"] = {"../model.py": next(iter(spec["files"].values()))}
        with self.assertRaisesRegex(ValueError, "escapes"):
            mm.verify_tree(spec)

    def test_offline_guard_restored(self):
        original = socket.create_connection
        with mm.no_network():
            with self.assertRaisesRegex(RuntimeError, "disabled"):
                socket.create_connection(("example.com", 443))
        self.assertIs(socket.create_connection, original)

    def test_asset_plan_has_official_clip_checksum(self):
        plan = mm.official_asset_plan()
        self.assertIn(mm.CLIP_SHA256, plan["clip"]["weights_url"])
        self.assertIn("v0.2.0-checkpoints", plan["dreamsim"]["weights_url"])
        self.assertEqual(len(mm.SOURCE_REVISIONS), 6)
        self.assertIn("dinov2_vitl14", mm.MANDATORY)
        self.assertIn("dinov2_vitl14_pretrain.pth", plan["dinov2_vitl14"]["weights_url"])

    def test_dinov2_import_namespace_does_not_replace_retained_source(self):
        local = types.SimpleNamespace(__file__=str(self.root / "dinov2" / "models.py"))
        wrong = types.SimpleNamespace(__file__=str(self.root.parent / "another_tree" / "models.py"))
        with mock.patch.dict(mm.sys.modules, {"dinov2.metric_contract_test": local}):
            mm.verify_dinov2_namespace(self.root)
        with mock.patch.dict(mm.sys.modules, {"dinov2.metric_contract_test": wrong}):
            with self.assertRaisesRegex(RuntimeError, "different source"):
                mm.verify_dinov2_namespace(self.root)

    def test_dinov2_architecture_rejects_other_sizes_and_registers(self):
        class DinoVisionTransformer:
            def __init__(self):
                self.embed_dim = 1024
                self.cls_token = types.SimpleNamespace(shape=(1, 1, 1024))
                self.patch_embed = types.SimpleNamespace(patch_size=(14, 14))
                self.blocks = [None] * 24
                self.num_register_tokens = 0
                self.register_tokens = None
        mm.validate_dinov2_vitl14_architecture(DinoVisionTransformer(), DinoVisionTransformer)
        for field, wrong in (("embed_dim", 384), ("blocks", [None]*12), ("num_register_tokens", 4), ("register_tokens", "present")):
            model = DinoVisionTransformer()
            setattr(model, field, wrong)
            with self.subTest(field=field), self.assertRaises(ValueError):
                mm.validate_dinov2_vitl14_architecture(model, DinoVisionTransformer)
        model = DinoVisionTransformer()
        model.patch_embed.patch_size = 16
        with self.assertRaises(ValueError):
            mm.validate_dinov2_vitl14_architecture(model, DinoVisionTransformer)


@unittest.skipUnless(HAS_TORCH, "torch and torchvision are required; run these CPU tests in the metric environment")
class TensorContracts(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        torch.manual_seed(137)
        self.x = torch.rand(2, 3, 256, 256)
        self.y = self.x.flip(-1)

    def evaluator(self):
        torch = self.torch
        class ClipStub(torch.nn.Module):
            def encode_image(self, x):
                return torch.cat((x.mean((2, 3)), x.square().mean((2, 3))), dim=-1)
        class DistsStub(torch.nn.Module):
            def forward(self, x, y, require_grad=False, batch_average=False):
                if require_grad or batch_average:
                    raise AssertionError("Evaluation must retain separate paired observations")
                return (x-y).abs().mean((1, 2, 3)).squeeze()
        class ClassifierStub(torch.nn.Module):
            def forward(self, x):
                logits = torch.zeros((len(x), 1000), device=x.device)
                logits[:, 7] = 1
                return logits
        class DreamStub(torch.nn.Module):
            def embed(self, x):
                f = torch.cat((x.mean((2, 3)), x.square().mean((2, 3))), dim=-1)
                return torch.nn.functional.normalize(f-f.mean(-1, keepdim=True), dim=-1)
        class DinoLargeStub(torch.nn.Module):
            def forward_features(self, x):
                f = torch.cat((x.mean((2, 3)), x.square().mean((2, 3))), dim=-1)
                return {"x_norm_clstoken": f.repeat(1, 171)[:, :1024]}
        e = mm.MetricEvaluator.__new__(mm.MetricEvaluator)
        e.device = torch.device("cpu")
        e.models = {"clip": ClipStub(), "dists": DistsStub(), "resnet50": ClassifierStub(), "dinov2_vitl14": DinoLargeStub(), "dreamsim": DreamStub(), "ms_ssim": lambda x, y, **kwargs: 1 - (x-y).square().mean((1, 2, 3))}
        e.metadata = {"synthetic": True, "scientific_result": False}
        return e

    def test_images_are_not_quantized(self):
        x = self.torch.full((1, 3, 256, 256), 0.123456)
        out = mm.validate_images(x)
        self.assertEqual(out[0, 0, 0, 0].item(), x[0, 0, 0, 0].item())
        prep = mm.preprocess_clip(x)
        expected = (0.123456-mm.CLIP_MEAN[0])/mm.CLIP_STD[0]
        self.assertAlmostEqual(prep[0, 0, 112, 112].item(), expected, places=6)

    def test_reject_bad_range_and_nonfinite(self):
        for value in (-0.01, 1.01, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                mm.validate_images(self.torch.full((1, 3, 16, 16), value))

    def test_reject_bad_type_channels_and_empty(self):
        for x in (self.x.byte(), self.x[0], self.x[:, :1], self.x[:0]):
            with self.subTest(shape=x.shape), self.assertRaises(ValueError):
                mm.validate_images(x)

    def test_resnet_official_transform(self):
        from torchvision.models import ResNet50_Weights
        expected = ResNet50_Weights.IMAGENET1K_V2.transforms(antialias=True)(self.x)
        self.assertTrue(self.torch.equal(mm.preprocess_resnet50(self.x), expected))

    def test_dinov2_preprocess_exact_retained_s14_formula(self):
        torch = self.torch
        resized = torch.nn.functional.interpolate(self.x.float(), size=(224, 224), mode="bicubic", align_corners=False)
        mean = resized.new_tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
        std = resized.new_tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
        self.assertTrue(torch.equal(mm.preprocess_dinov2_vitl14(self.x), (resized-mean)/std))
        smoothed = torch.nn.functional.interpolate(self.x, size=(224, 224), mode="bicubic", align_corners=False, antialias=True)
        self.assertFalse(torch.equal(resized, smoothed))

    def test_dinov2_feature_dimensions_and_nonzero_are_checked(self):
        torch = self.torch
        for features in (torch.ones(2, 384), torch.zeros(2, 1024), torch.full((2, 1024), float("nan"))):
            model = types.SimpleNamespace(forward_features=lambda x: {"x_norm_clstoken": features})
            with self.assertRaises(ValueError):
                mm.dinov2_vitl14_features(model, self.x)

    def test_dreamsim_geometry_differs_from_clip_crop(self):
        x = self.x[:, :, :, :128]
        self.assertEqual(mm.preprocess_clip(x).shape, (2, 3, 224, 224))
        self.assertEqual(mm.preprocess_dreamsim(x).shape, (2, 3, 224, 224))

    def test_identity_metrics_and_classifier_two_denominators(self):
        rows = self.evaluator().score(self.x, self.x, [7, 8])
        for row in rows:
            self.assertAlmostEqual(row["clip_image_cosine"], 1, places=5)
            self.assertAlmostEqual(row["dinov2_vitl14_cosine"], 1, places=5)
            self.assertNotIn("dino_cosine", row)
            self.assertAlmostEqual(row["dists"], 0, places=5)
            self.assertAlmostEqual(row["dreamsim"], 0, places=5)
            self.assertAlmostEqual(row["ms_ssim"], 1, places=5)
            self.assertTrue(row["resnet50_top1_source_prediction"])
        self.assertTrue(rows[0]["resnet50_top1_label"])
        self.assertFalse(rows[1]["resnet50_top1_label"])
        self.assertFalse(rows[1]["resnet50_source_top1_label"])

    def test_reference_cache_matches_uncached(self):
        e = self.evaluator()
        ref = e.prepare_reference(self.x)
        self.assertEqual(e.score(self.x, self.y, [7, 8], prepared=ref), e.score(self.x, self.y, [7, 8]))
        changed = self.x.clone()
        changed[0, 0, 0, 0] += 1e-6
        with self.assertRaisesRegex(ValueError, "cache"):
            e.score(changed, self.y, [7, 8], prepared=ref)
        e.metadata["another_checkpoint"] = True
        with self.assertRaisesRegex(ValueError, "cache"):
            e.score(self.x, self.y, [7, 8], prepared=ref)

    def test_expand_reference_does_not_infer_original_again(self):
        torch = self.torch
        e = self.evaluator()
        source = self.x[:1]
        cached = e.prepare_reference(source)
        with mock.patch.object(e.models["clip"], "encode_image", side_effect=AssertionError("original rerun")), \
                mock.patch.object(e.models["resnet50"], "forward", side_effect=AssertionError("original rerun")), \
                mock.patch.object(e.models["dreamsim"], "embed", side_effect=AssertionError("original rerun")), \
                mock.patch.object(e.models["dinov2_vitl14"], "forward_features", side_effect=AssertionError("original rerun")):
            repeated_source, repeated_cache = e.expand_reference(source, cached, 3)
        self.assertEqual(repeated_source.shape, (3, 3, 256, 256))
        self.assertEqual(repeated_cache["batch"], 3)
        for name in ("clip", "resnet50", "dreamsim", "dinov2_vitl14"):
            self.assertTrue(torch.equal(repeated_cache[name][0], cached[name][0]))
            self.assertTrue(torch.equal(repeated_cache[name][1], cached[name][0]))
        previous_pixel = source[0, 0, 0, 0].item()
        previous_feature = cached["clip"][0, 0].item()
        with torch.inference_mode():
            repeated_source[0, 0, 0, 0] += 0.01
            repeated_cache["clip"][0, 0] += 0.01
        self.assertEqual(source[0, 0, 0, 0].item(), previous_pixel)
        self.assertEqual(cached["clip"][0, 0].item(), previous_feature)

    def test_expand_reference_rejects_invalid_count_and_stale_caches(self):
        e = self.evaluator()
        source = self.x[:1]
        cached = e.prepare_reference(source)
        for count in (0, -1, True, 1.5):
            with self.subTest(count=count), self.assertRaises(ValueError):
                e.expand_reference(source, cached, count)
        for invalid in (dict(cached, digest="0"*64), dict(cached, evaluator_identity="0"*64), dict(cached, batch=2)):
            with self.assertRaisesRegex(ValueError, "cache"):
                e.expand_reference(source, invalid, 2)
        with self.assertRaisesRegex(ValueError, "one"):
            e.expand_reference(self.x, e.prepare_reference(self.x), 2)
        invalid = dict(cached, clip=cached["clip"].repeat(2, 1))
        with self.assertRaisesRegex(ValueError, "feature cache"):
            e.expand_reference(source, invalid, 2)

    def test_expanded_reference_batch_matches_scalar_score(self):
        torch = self.torch
        e = self.evaluator()
        source = self.x[:1]
        cached = e.prepare_reference(source)
        repeat, expanded = e.expand_reference(source, cached, 2)
        batch_rows = e.score(repeat, self.y, [7, 8], prepared=expanded)
        scalar = [e.score(source, self.y[i:i+1], [label], prepared=cached)[0] for i, label in enumerate((7, 8))]
        for actual, expected in zip(batch_rows, scalar):
            self.assertEqual(set(actual), set(expected))
            for key in actual:
                if key in ("clip_image_cosine", "dists", "dinov2_vitl14_cosine", "dreamsim", "ms_ssim"):
                    self.assertLessEqual(abs(actual[key] - expected[key]), 2e-5)
                else:
                    self.assertEqual(actual[key], expected[key])
        single, single_cache = e.expand_reference(source, cached, 1)
        self.assertTrue(torch.equal(single, source))
        self.assertEqual(e.score(single, self.y[:1], [7], prepared=single_cache), scalar[:1])

    def test_label_conditioned_flag_is_preserved(self):
        rows = self.evaluator().score(self.x, self.y, [7, 8], label_conditioned=True)
        self.assertTrue(all(row["label_conditioned"] for row in rows))
        with self.assertRaises(ValueError):
            self.evaluator().score(self.x, self.y, [7, 8], label_conditioned="D_C")

    def test_invalid_labels_are_rejected(self):
        for labels in ([7.0, 8.0], [True, False], [-1, 0], [1, 1000], [7]):
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                self.evaluator().score(self.x, self.y, labels)

    def test_single_image_official_dists_scalar_is_handled(self):
        rows = self.evaluator().score(self.x[:1], self.y[:1], [7])
        self.assertEqual(len(rows), 1)
        self.assertGreater(rows[0]["dists"], 0)

    def test_registered_resolution_and_pair_shape(self):
        for y in (self.y[:, :, :224, :224], self.y[:1]):
            with self.assertRaisesRegex(ValueError, "shape"):
                self.evaluator().score(self.x, y, [7, 8])

    def test_nonfinite_metric_output_rejected(self):
        e = self.evaluator()
        e.models["ms_ssim"] = lambda x, y, **kw: self.torch.full((len(x),), float("nan"))
        with self.assertRaisesRegex(ValueError, "Invalid metric output"):
            e.score(self.x, self.y, [7, 8])

    def test_freeze_disables_training_and_grad(self):
        model = self.torch.nn.Linear(3, 2).double().train()
        model = mm._freeze(model, "cpu")
        self.assertFalse(model.training)
        self.assertFalse(any(p.requires_grad for p in model.parameters()))
        self.assertEqual(next(model.parameters()).dtype, self.torch.float32)

    def test_missing_mandatory_and_explicit_optional_status(self):
        with self.assertRaisesRegex(ValueError, "Required metric assets"):
            mm.MetricEvaluator({}, device="cpu")
        e = mm.MetricEvaluator({}, device="cpu", requested=("dreamsim", "ms_ssim"))
        self.assertEqual(e.metadata["metrics"]["dreamsim"]["status"], "UNAVAILABLE_NOT_REGISTERED")

    def test_dreamsim_checkpoint_default_cpu_preserves_load_arguments(self):
        # No checkpoint or GPU is touched: this inspects only torch.load calls.
        torch = self.torch
        calls = []
        def original(*args, **kwargs):
            calls.append((args, kwargs))
            return "checkpoint"
        with mock.patch.object(torch, "load", original):
            with mm.cpu_default_checkpoint_loads():
                self.assertEqual(torch.load("tagged.pth", weights_only=True), "checkpoint")
                torch.load("tagged.pth", "explicit_positional", weights_only=False)
                torch.load("tagged.pth", map_location=None, weights_only=True, mmap=True)
                torch.load(f="tagged.pth", weights_only=True)
            self.assertIs(torch.load, original)
            with self.assertRaisesRegex(RuntimeError, "fixture failure"):
                with mm.cpu_default_checkpoint_loads():
                    raise RuntimeError("fixture failure")
            self.assertIs(torch.load, original)
        self.assertEqual(calls[0], (("tagged.pth",), {"weights_only": True, "map_location": "cpu"}))
        self.assertEqual(calls[1], (("tagged.pth", "explicit_positional"), {"weights_only": False}))
        self.assertEqual(calls[2], (("tagged.pth",), {"map_location": None, "weights_only": True, "mmap": True}))
        self.assertEqual(calls[3], ((), {"f": "tagged.pth", "weights_only": True, "map_location": "cpu"}))


if __name__ == "__main__":
    unittest.main(verbosity=2)

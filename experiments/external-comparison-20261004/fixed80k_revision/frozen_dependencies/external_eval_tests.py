"""CPU engineering fixtures; these never generate a scientific result."""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
from external_eval_common import *


class ExternalEvaluationTests(unittest.TestCase):
    def test_registered_scope_contains_1800_pairs_and_3600_unique_rows(self):
        frames = [spec for i in range(100) for spec in frame_specs(i)]
        ids = [rid for i in range(100) for rid in expected_ids(i)]
        self.assertEqual(len(frames), 1800)
        self.assertEqual(len(set(frame_key(f) for f in frames)), 1800)
        self.assertEqual(len(ids), 3600)
        self.assertEqual(len(set(ids)), 3600)
        self.assertEqual({s["snr_db"] for s in frames}, {1, 7, 13})
        self.assertEqual({s["N"] for s in frames}, {1024, 2048})

    def test_truncated_probe_is_never_a_main_result(self):
        full = dict(header_accepted=True, diagnostic_probe=False, complete_author_schedule=True,
            model_parameters_unchanged=True, parameter_gradients_accumulated=False,
            schedule_mode="actual_data_cbr", NFE=258, t_start=258)
        validate_full_sampler(full, True)
        for changes in ({"diagnostic_probe": True}, {"NFE": 2},
                        {"complete_author_schedule": False}, {"schedule_mode": "author_fixed_1_over_48"},
                        {"parameter_gradients_accumulated": True}, {"header_accepted": False}):
            with self.assertRaises(RuntimeError): validate_full_sampler(dict(full, **changes), True)
        validate_full_sampler(dict(header_accepted=False, NFE=0, fallback="fixed_gray_0.5"), False)

    def test_false_accept_audit_never_changes_receiver_decision(self):
        rx = SimpleNamespace(accepted=True, power=.5, indices=(1, 7))
        exact = audit_received_context(.5, (1, 7), rx)
        self.assertTrue(exact["offline_metadata_exact"])
        wrong = audit_received_context(.6, (1, 7), rx)
        self.assertTrue(wrong["offline_false_accept"])
        self.assertFalse(wrong["audit_used_to_control_receiver"])
        self.assertTrue(rx.accepted)
        rejected = SimpleNamespace(accepted=False, power=None, indices=None)
        self.assertFalse(audit_received_context(.5, (1, 7), rejected)["offline_false_accept"])

    def _frame(self, root):
        target = np.full((3, 256, 256), .25, dtype=np.float32)
        source = dict(image_id="engineering_fixture", rgb=target)
        spec = frame_specs(0)[0]
        images = np.full((2, 3, 256, 256), .5, dtype=np.float32)
        observed = np.zeros((1024, 2), dtype=np.float64)
        transmitted = np.ones_like(observed)
        archive = root / "frame.npz"
        digest = atomic_npz(archive, images=images, observed=observed, transmitted=transmitted)
        rows = [dict(**spec, method=m, replay_row_id=row_id(spec, m), image_sha256=rgb_sha(images[0]),
                     observed_sha256=array_sha(observed), header_accepted=False,
                     audit_used_to_control_receiver=False, TX_seconds=.1, RX_seconds=.2) for m in METHODS]
        value = dict(binding="engineering", frame=spec, source_id=source["image_id"],
            reference_sha256=rgb_sha(target), rows=rows, archive=str(archive), archive_sha256=digest,
            observed_sha256=array_sha(observed), transmitted_sha256=array_sha(transmitted),
            hifi_receipt=dict(header_accepted=False, NFE=0, fallback="fixed_gray_0.5"))
        value["payload_sha256"] = identity(value)
        return value, source, spec

    def test_frame_receipt_checks_pixels_shared_waveform_and_timing(self):
        with tempfile.TemporaryDirectory() as temporary:
            value, source, spec = self._frame(Path(temporary))
            validate_frame_receipt(value, "engineering", spec, source)
            for field, bad in (("observed_sha256", "0" * 64), ("RX_seconds", -1),
                               ("method", METHODS[0]), ("audit_used_to_control_receiver", True)):
                changed = copy.deepcopy(value)
                changed["rows"][1][field] = bad
                changed["payload_sha256"] = identity({k: v for k, v in changed.items() if k != "payload_sha256"})
                with self.assertRaises(RuntimeError): validate_frame_receipt(changed, "engineering", spec, source)

    def test_frame_resume_detects_actual_archive_corruption(self):
        with tempfile.TemporaryDirectory() as temporary:
            value, source, spec = self._frame(Path(temporary))
            with Path(value["archive"]).open("ab") as stream: stream.write(b"corruption")
            with self.assertRaises(RuntimeError): validate_frame_receipt(value, "engineering", spec, source)

    def test_source_cache_preserves_all_rows_when_gray_outputs_deduplicate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = np.full((3, 256, 256), .25, dtype=np.float32)
            image = np.full_like(target, .5)
            rows = [dict(replay_row_id=rid, image_sha256=rgb_sha(image), reference_sha256=rgb_sha(target))
                    for rid in expected_ids(0)]
            archive = root / "source.npz"
            digest = atomic_npz(archive, images=image[None], source_rgb=target,
                row_ids=np.asarray(expected_ids(0)), image_slots=np.zeros(36, dtype=np.int64))
            value = dict(binding="engineering", source_index=0, rows=rows, frame_bindings={},
                float_reconstructions=dict(path=str(archive), sha256=digest, image_slots=[0] * 36))
            value["payload_sha256"] = identity(value)
            validate_source(value, "engineering", 0)
            changed = copy.deepcopy(value); changed["rows"].pop()
            changed["payload_sha256"] = identity({k: v for k, v in changed.items() if k != "payload_sha256"})
            with self.assertRaises(RuntimeError): validate_source(changed, "engineering", 0)

    def test_atomic_registration_rejects_changed_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "registration.json"
            seal(path, {"budget": 1024}); seal(path, {"budget": 1024})
            with self.assertRaises(RuntimeError): seal(path, {"budget": 2048})
            self.assertFalse(path.with_name(path.name + ".tmp").exists())


if __name__ == "__main__":
    unittest.main()

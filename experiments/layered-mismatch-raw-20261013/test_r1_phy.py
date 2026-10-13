"""Synthetic CPU only: no actual PHY backend, source image, or model invocation."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import r1_phy as r1
import grouped_wire as wire
import contract
from test_grouped_wire import profile, source_scales as synthetic_scales


def synthetic_source():
    return dict(source_id="SYNTHETIC_ONLY", source_index=631, selection_slot=7,
        preprocessing_id="p" * 64, tokens_sha256="t" * 64,
        archive={"sha256": "a" * 64}, checkpoint={"sha256": "c" * 64})


def nomination(G=1, variants=("KEEP", "DROP")):
    return dict(snr_hat=7, physical_key="synthetic_wire", profile_id=5 if G == 1 else 433, G=G,
        logical_receivers=[dict(nomination_index=0, arm="S" if G == 1 else "U", rule=v) for v in variants])


class StubRuntime:
    def __init__(self, journal, fail=False, crc=False):
        self.journal, self.fail, self.crc = journal, fail, crc
        self.tx_calls, self.rx_calls = [], []

    def transmit(self, n, scales, counter):
        self.tx_calls.append((copy.deepcopy(n), counter))
        wave = np.ones((1024, 2), dtype=np.float64)
        return wave, {"SYNTHETIC": True, "waveform_sha256": wire.array_sha(wave), "actual_E": 2048.0}, [np.zeros(12, dtype=np.uint8)]

    def receive(self, G, received, true_snr, counter, tag):
        self.rx_calls.append(dict(G=G, true_snr=true_snr, counter=counter, received=received.copy()))
        def header():
            if self.fail:
                raise RuntimeError("synthetic unknown packet failure")
            return {"header_ok": True}
        self.journal.call(dict(event_id=tag + ".header", kind="header"), {"SYNTHETIC": True}, header)
        state = dict(kind="tokens", prefix=[[2048]], partial_values=[], m=1, K=0, order="raster")
        if G == 1:
            return dict(received_sha256=wire.array_sha(received), header_ok=True, body_crc_accept=self.crc,
                receiver_state=state, rx_profile_id=5, body={"hard_payload": [1] + [0] * 11})
        return dict(received_sha256=wire.array_sha(received), state=state, header_ok=True)

    def audit(self, actual, n, payloads):
        return {"SYNTHETIC_POST_RX_ONLY": True}


class PlanTests(unittest.TestCase):
    def rows(self):
        single = profile(pid=5, G=1)
        single.update(wire_key="single", candidate_id="canonical_single")
        grouped = profile(pid=433)
        grouped.update(wire_key="two", candidate_id="canonical_two", family_memberships=["U", "EEP"])
        rows = []
        for hat in (7, 10):
            for arm in ("U", "S", "EEP", "S_nom", "S_back3"):
                p = grouped if arm in ("U", "EEP") else single
                rows.append(dict(snr_hat=hat, arm=arm, profile_id=p["profile_id"], physical_key=p["wire_key"], candidate_id=p["candidate_id"],
                    receiver_variants=["KEEP", "DROP"] if arm == "S" else (["DROP"] if p["G"] == 2 else ["KEEP"])))
        return rows, [single, grouped]

    def test_alias_nominations_share_physical_frame_but_retain_views(self):
        rows, profiles = self.rows()
        groups = r1.nomination_groups(rows, profiles)
        self.assertEqual(len(groups), 4)
        self.assertEqual(sum(len(g["logical_receivers"]) for g in groups), 6)
        self.assertEqual(len([g for g in groups if g["G"] == 1]), 2)
        for g in (g for g in groups if g["G"] == 1):
            self.assertEqual({v["rule"] for v in g["logical_receivers"]}, {"KEEP", "DROP"})
            keep = next(v for v in g["logical_receivers"] if v["rule"] == "KEEP")
            drop = next(v for v in g["logical_receivers"] if v["rule"] == "DROP")
            self.assertEqual(keep["arms"], ["S", "S_back3", "S_nom"])
            self.assertEqual(len(keep["nomination_indices"]), 3)
            self.assertEqual(drop["arms"], ["S"])

    def test_wrong_physical_id_or_family_is_rejected(self):
        rows, profiles = self.rows()
        rows[0]["physical_key"] = "not-the-wire"
        with self.assertRaises(ValueError):
            r1.nomination_groups(rows, profiles)
        rows, profiles = self.rows()
        profiles[1]["family_memberships"] = ["EEP"]
        with self.assertRaises(ValueError):
            r1.nomination_groups(rows, profiles)

    def test_unknown_registration_rejected_before_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "not_registered.json"
            p.write_text(json.dumps({"schema": r1.SCHEMA, "status": "PROPOSED_NOT_EXECUTED"}))
            with patch.object(r1, "RealRuntime") as runtime:
                with self.assertRaises(ValueError):
                    r1.run(r1.pin(p))
                runtime.assert_not_called()

    def test_public_counter_uses_new_selection_slot_not_old1000_index(self):
        source = synthetic_source()
        ident = r1.input_identity(source, nomination(), 4, 4101, {"stub": True})
        self.assertEqual(ident["public_counter"], contract.public_counter(7, 7, 4101, "cal200"))
        self.assertNotEqual(ident["public_counter"], source["source_index"] * 3)


class SourceTests(unittest.TestCase):
    def make_source(self, root):
        a = root / "s.npz"
        tokens = np.arange(680, dtype=np.int64)
        pixels = np.zeros((3, 256, 256), dtype=np.uint8)
        np.savez(a, tokens=tokens, pixels=pixels)
        record = dict(source_index=631, source_id="SYNTHETIC", selection_slot=7,
            preprocessing_id=hashlib.sha256(pixels.tobytes()).hexdigest(),
            tokens_sha256=hashlib.sha256(b"int64:680\0" + tokens.astype("<i8").tobytes()).hexdigest(), archive=r1.pin(a))
        cp = dict(status="H_FULL1000_SOURCE_ASSET_READY", original_calibration_index=631,
            **{k: record[k] for k in ("source_index", "source_id", "preprocessing_id", "tokens_sha256")},
            archive="/historical/calibration/s.npz", outputs={"/historical/calibration/s.npz": record["archive"]["sha256"]})
        p = root / "cp.json"
        p.write_text(json.dumps(cp))
        record["checkpoint"] = r1.pin(p)
        return record

    def test_calibration_source_schema_validates_both_arrays(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = self.make_source(Path(tmp))
            scales = r1.source_scales(record)
            self.assertEqual([len(s) for s in scales], [s * s for s in wire.SIZES])
            np.testing.assert_array_equal(np.concatenate(scales), np.arange(680))

    def test_changed_pixels_cannot_pass_even_with_resealed_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = self.make_source(Path(tmp))
            # Make checkpoint and archive match but retain original pixel identity.
            np.savez(record["archive"]["path"], tokens=np.arange(680, dtype=np.int64),
                     pixels=np.ones((3, 256, 256), dtype=np.uint8))
            record["archive"] = r1.pin(record["archive"]["path"])
            cp = r1.readpin(record["checkpoint"])
            cp["outputs"][cp["archive"]] = record["archive"]["sha256"]
            Path(record["checkpoint"]["path"]).write_text(json.dumps(cp))
            record["checkpoint"] = r1.pin(record["checkpoint"]["path"])
            with self.assertRaisesRegex(ValueError, "pixels hash"):
                r1.source_scales(record)

    def test_common500_checkpoint_is_not_accepted_as_calibration(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = self.make_source(Path(tmp))
            cp = r1.readpin(record["checkpoint"])
            cp["status"] = "COMMON500_FROZEN_SOURCE_ASSET_READY_V1"
            Path(record["checkpoint"]["path"]).write_text(json.dumps(cp))
            record["checkpoint"] = r1.pin(record["checkpoint"]["path"])
            with self.assertRaises(ValueError):
                r1.source_scales(record)


class ExecutionTests(unittest.TestCase):
    def test_single_KEEP_DROP_share_one_actual_event(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal = r1.Journal(Path(tmp) / "journal")
            runtime = StubRuntime(journal)
            p, result, reused = r1.execute_frame(synthetic_source(), synthetic_scales(), nomination(),
                4, 4101, runtime, journal, {"runtime": "SYNTHETIC"})
            self.assertEqual(len(runtime.tx_calls), 1)
            self.assertEqual(len(runtime.rx_calls), 1)
            self.assertFalse(reused)
            self.assertEqual(result["receiver_views"]["KEEP"]["prefix"], [[2048]])
            self.assertEqual(result["receiver_views"]["DROP"]["kind"], "gray")
            self.assertEqual(result["actual_RX"]["receiver_state"]["prefix"], [[2048]])
            self.assertEqual(journal.snapshot()["physical_complete"], 1)

    def test_complete_identical_input_reuses_without_TX_channel_or_RX(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal = r1.Journal(Path(tmp) / "journal")
            runtime = StubRuntime(journal)
            args = (synthetic_source(), synthetic_scales(), nomination(), 4, 4101, runtime, journal, {"rt": "SYNTHETIC"})
            first = r1.execute_frame(*args)
            with patch.object(contract, "standard_noise", side_effect=AssertionError("No second channel")):
                second = r1.execute_frame(*args)
            self.assertTrue(second[2])
            self.assertEqual(first[:2], second[:2])
            self.assertEqual(len(runtime.tx_calls), 1)

    def test_true_SNR_variants_share_standard_noise_but_scale_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal = r1.Journal(Path(tmp) / "journal")
            runtime = StubRuntime(journal)
            results = [r1.execute_frame(synthetic_source(), synthetic_scales(), nomination(), true,
                4101, runtime, journal, {"rt": "SYNTHETIC"})[1] for true in (4, 7, 10)]
            self.assertEqual(len({r["observation_identity"]["standard_noise_sha256"] for r in results}), 1)
            self.assertEqual(len({r["observation_identity"]["received_sha256"] for r in results}), 3)
            self.assertEqual(len({c["counter"] for c in runtime.rx_calls}), 1)
            self.assertEqual([c["true_snr"] for c in runtime.rx_calls], [4, 7, 10])

    def test_changed_runtime_or_pixel_identity_never_hits_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal = r1.Journal(Path(tmp) / "journal")
            runtime = StubRuntime(journal)
            first = synthetic_source()
            second = dict(first, preprocessing_id="CHANGED")
            for source, rt in ((first, "A"), (first, "B"), (second, "B")):
                self.assertFalse(r1.execute_frame(source, synthetic_scales(), nomination(), 4, 4101,
                    runtime, journal, {"rt": rt})[2])
            self.assertEqual(len(runtime.tx_calls), 3)

    def test_failed_slot_retains_reserved_frame_and_packet_and_cannot_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal = r1.Journal(Path(tmp) / "journal")
            runtime = StubRuntime(journal, fail=True)
            args = (synthetic_source(), synthetic_scales(), nomination(), 4, 4101, runtime, journal, {"rt": "A"})
            with self.assertRaisesRegex(RuntimeError, "unknown"):
                r1.execute_frame(*args)
            self.assertEqual(journal.snapshot()["physical_reserved"], 1)
            self.assertEqual(journal.snapshot()["physical_complete"], 0)
            self.assertEqual(journal.snapshot()["unresolved_packets"], 1)
            with self.assertRaisesRegex(ValueError, "cannot replay"):
                r1.execute_frame(*args)
            with self.assertRaisesRegex(ValueError, "blocks new"):
                r1.execute_frame(synthetic_source(), synthetic_scales(), nomination(), 7, 4101,
                    runtime, journal, {"rt": "A"})
            self.assertEqual(len(runtime.tx_calls), 1)
            self.assertEqual(len(runtime.rx_calls), 1)

    def test_frame_and_packet_caps_reserve_before_callback(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal = r1.Journal(Path(tmp) / "j", frame_cap=1, packet_cap=1)
            key, _ = journal.reserve_frame({"case": 1})
            self.assertEqual(journal.call({"event_id": "one"}, {}, lambda: {"ok": 1}), {"ok": 1})
            with self.assertRaisesRegex(ValueError, "cap"):
                journal.call({"event_id": "two"}, {}, lambda: self.fail("No actual decode permitted"))
            journal.complete_frame(key, {"status": "COMPLETE", "input_identity": {"case": 1}})
            with self.assertRaisesRegex(ValueError, "allowance"):
                journal.reserve_frame({"case": 2})

    def test_group_hooks_count_RX_only_and_preserve_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal = r1.Journal(Path(tmp) / "j")
            journal.reserve_frame({"case": 1})
            for direction in ("tx", "rx"):
                base = dict(direction=direction, packet="header", public_counter=0)
                journal.grouped_hook(dict(stage="begin", **base))
                journal.grouped_hook(dict(stage="done", **base))
            self.assertEqual(journal.snapshot()["packet_decodes_complete"], 1)
            base = dict(direction="rx", packet="G1", public_counter=0)
            journal.grouped_hook(dict(stage="begin", **base))
            journal.grouped_hook(dict(stage="failed", exception_type="SYNTHETIC", **base))
            self.assertEqual(journal.snapshot()["unresolved_packets"], 1)

    def test_group_variant_cannot_sneak_in_KEEP(self):
        with self.assertRaises(ValueError):
            r1.receiver_views({"state": {}}, "grouped", ["KEEP"])

    def test_no_existing_journal_is_reopened_or_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            r1.Journal(Path(tmp) / "j")
            with self.assertRaisesRegex(ValueError, "Fresh"):
                r1.Journal(Path(tmp) / "j")


if __name__ == "__main__":
    output = None
    if "--report" in sys.argv:
        i = sys.argv.index("--report")
        output = Path(sys.argv[i + 1])
        del sys.argv[i:i + 2]
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__]))
    if output:
        report = dict(status="PASS" if result.wasSuccessful() else "FAIL", tests_run=result.testsRun,
            failures=len(result.failures), errors=len(result.errors), scope="SYNTHETIC_CPU_ENGINEERING_ONLY",
            real_PHY_runtime_constructed=False, real_packet_encodes=0, real_packet_decodes=0,
            real_source_channel_calls=0, visual_model_calls=0, R1_registration_created=False,
            scientific_launch="NOT_RUN", scientific_qualification="NOT_RUN", torch_imported="torch" in sys.modules,
            source_sha256={p.name: r1.sha(p) for p in (Path(__file__), Path(r1.__file__), Path(wire.__file__), Path(contract.__file__))})
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
    raise SystemExit(0 if result.wasSuccessful() else 1)

"""Synthetic CPU engineering tests. No real encoder/decoder/channel/model calls."""
import copy
import hashlib
import inspect
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grouped_wire as wire


def profile(pid=433, q1=2, q2=6, j=1, m=2, K=1, idle=0, G=2):
    """Synthetic metadata is not usable as an actual Sionna admission record."""
    count = wire.PREFIX[m] + K
    tokens = [wire.PREFIX[j], count - wire.PREFIX[j]] if G == 2 else [count]
    uses = [256, 700 - idle] if G == 2 else [956 - idle]
    qs = [q1, q2] if G == 2 else [q1]
    groups = []
    for t, nuses, q in zip(tokens, uses, qs):
        k, n = 12 * t + 16, nuses * q
        layout = dict(k=k, n=n, bg="SYNTHETIC", z=1, k_ldpc=k, k_filler=0,
            n_cb=n, n_cb_comp=n, mother_bits=n, puncturing_bits=0,
            puncturing_first_2Z=0, shortening_bits=0, decoder_config={"SYNTHETIC": True},
            implementation={"SYNTHETIC": True}, interleaver_sha256="0" * 64)
        layout["layout_id"] = wire.identity(layout)
        g = dict(source_bits=12 * t, crc_bits=16, tail_bits=0, information_bits=k,
            transmitted_bits=n, symbols=nuses, modulation={2: "QPSK", 4: "16QAM", 6: "64QAM"}[q], layout=layout)
        g["phy_key"] = wire.identity(g)
        groups.append(g)
    return dict(profile_id=pid, N=1024, G=G, j=j if G == 2 else None, m=m, K=K,
        order="raster", header_symbols=68, header_phy_key=wire.HEADER_KEY,
        idle_symbols=idle, groups=groups, receiver_rule="DROP" if G == 2 else wire.SINGLE_KEEP_RULE)


def source_scales():
    return [(np.arange(s * s, dtype=np.int64) + i * 100) % 4096 for i, s in enumerate(wire.SIZES)]


class StubWireBackend:
    def __init__(self, row, payloads=None, accepts=(True, True), header_ok=True, received_id=None):
        self.row = row
        self.payloads = payloads or wire.serialize_groups(source_scales(), row)
        self.accepts = accepts
        self.header_ok = header_ok
        self.received_id = row["profile_id"] if received_id is None else received_id
        self.calls = []

    def require_catalogue(self, catalog):
        self.calls.append(("check", catalog.digest))

    def header_transmit(self, pid, counter):
        self.calls.append(("header_tx", pid, counter))
        return np.ones((68, 2))

    def header_receive(self, samples, snr, counter, mapping):
        self.calls.append(("header_rx", samples.copy(), snr, counter))
        legal = str(self.received_id) in mapping
        return dict(profile_id=self.received_id if self.header_ok and legal else None,
            header_crc_ok=self.header_ok, header_fields_legal=legal, header_ok=self.header_ok and legal, score=0.0)

    def packet_transmit(self, payload, group, index, counter):
        self.calls.append(("packet_tx", wire.Q[group["modulation"]], index, counter, payload.copy()))
        # Deliberately nonconstant synthetic energy; do not normalize each frame.
        return np.full((group["symbols"], 2), index + 1.0), {"synthetic": True}

    def packet_receive(self, samples, group, index, snr, counter):
        self.calls.append(("packet_rx", wire.Q[group["modulation"]], index, snr, counter, samples.copy()))
        return self.payloads[index], bool(self.accepts[index])


class TensorStub:
    def __init__(self, value):
        self.value = np.asarray(value)

    def __getitem__(self, key):
        return TensorStub(self.value[key])

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class ActualFacadeTests(unittest.TestCase):
    def facade(self, q=6):
        calls, ledger = [], []
        p = profile(q1=q, q2=q)
        layouts = {(g["information_bits"], g["transmitted_bits"], q): g["layout"] for g in p["groups"]}
        torch = SimpleNamespace(float32="SYNTHETIC", as_tensor=lambda value, **kw: TensorStub(value))
        backend = SimpleNamespace(torch=torch, device="SYNTHETIC", plan=lambda k, n, q: layouts[(k, n, q)])
        legacy = SimpleNamespace(crc_accept_batch=lambda bits: True)
        def tx(b, l, data, n, got_q, counters, index, session):
            calls.append(("tx", got_q, counters, index, session))
            return TensorStub(np.ones((1, n // got_q, 2))), {"synthetic": True}
        def rx(b, l, received, k, n, got_q, snr, counters, index, session):
            calls.append(("rx", got_q, snr, counters, index, session))
            return np.zeros((1, k - 16), dtype=np.uint8), np.array([True]), np.zeros((1, k), dtype=np.uint8)
        packet = SimpleNamespace(Q=wire.Q.copy(), HEADER_KEY=wire.HEADER_KEY, transmit_packet=tx, receive_packet=rx)
        header = SimpleNamespace(transmit=lambda pid: np.ones((68, 2)),
            receive=lambda samples, snr, mapping: dict(profile_id=p["profile_id"], header_ok=True,
                header_crc_ok=True, header_fields_legal=True))
        facade = wire.ActualPhyBackend(backend, legacy, packet, header, ledger.append)
        return p, facade, calls, ledger

    def test_all_three_modulations_forward_exactly_to_raw64_adapter(self):
        for q in (2, 4, 6):
            p, facade, calls, ledger = self.facade(q)
            catalog = wire.Catalogue([p])
            facade.verify_catalogue(catalog)
            payload = wire.serialize_groups(source_scales(), p)
            wire.transmit(p["profile_id"], payload, 90, catalog, facade)
            wire.receive(np.zeros((1024, 2)), 7, 90, catalog, facade)
            self.assertEqual([c[1] for c in calls], [q, q, q, q])
            self.assertEqual(len(ledger), 12)  # 3 packets TX + 3 RX, begin/done.
            self.assertEqual([e["stage"] for e in ledger], ["begin", "done"] * 6)
            self.assertTrue(all(c[-1] == wire.BODY_SESSION for c in calls))

    def test_plan_check_does_not_encode_or_decode(self):
        p, facade, calls, ledger = self.facade()
        catalog = wire.Catalogue([p])
        with self.assertRaises(ValueError):
            facade.require_catalogue(catalog)
        facade.verify_catalogue(catalog)
        facade.require_catalogue(catalog)
        self.assertEqual(calls, [])
        self.assertEqual(ledger, [])

    def test_ledger_failure_prevents_actual_call(self):
        p, facade, calls, ledger = self.facade()
        def stop(event):
            raise RuntimeError("No registered packet allowance")
        facade.packet_ledger = stop
        with self.assertRaisesRegex(RuntimeError, "allowance"):
            facade.packet_transmit(np.zeros(p["groups"][0]["source_bits"]), p["groups"][0], 0, 8)
        self.assertEqual(calls, [])

    def test_failed_actual_stub_call_has_no_retry(self):
        p, facade, calls, ledger = self.facade()
        def fail(*args):
            calls.append("called-once")
            raise RuntimeError("Synthetic infrastructure failure")
        facade.raw64_packet.transmit_packet = fail
        with self.assertRaisesRegex(RuntimeError, "infrastructure"):
            facade.packet_transmit(np.zeros(p["groups"][0]["source_bits"]), p["groups"][0], 0, 8)
        self.assertEqual(calls, ["called-once"])
        self.assertEqual([r["stage"] for r in ledger], ["begin", "failed"])

    def test_decoded_crc_inconsistency_is_error_not_gray(self):
        p, facade, calls, ledger = self.facade()
        facade.legacy_phy.crc_accept_batch = lambda b: False
        with self.assertRaisesRegex(ValueError, "contradiction"):
            facade.packet_receive(np.zeros((p["groups"][0]["symbols"], 2)), p["groups"][0], 0, 7, 3)


class GroupedWireTests(unittest.TestCase):
    def setUp(self):
        self.row = profile()
        self.catalog = wire.Catalogue([self.row])
        self.payload = wire.serialize_groups(source_scales(), self.row)

    def receive(self, accepts=(True, True), payloads=None, header_ok=True):
        backend = StubWireBackend(self.row, payloads, accepts, header_ok)
        event = wire.receive(np.arange(2048).reshape(1024, 2) / 1024, 4, 91, self.catalog, backend)
        return event, backend

    def test_raster_serialization_includes_first_K(self):
        values = wire.bits_to_indices(np.concatenate(self.payload))
        expected = np.concatenate((source_scales()[0], source_scales()[1], source_scales()[2][:1]))
        np.testing.assert_array_equal(values, expected)
        self.assertEqual(len(self.payload[0]), 12 * wire.PREFIX[self.row["j"]])

    def test_full_prefix_retains_actual_values(self):
        event, _ = self.receive()
        self.assertEqual(event["receive_status"], "FULL_PREFIX")
        self.assertEqual(event["state"]["partial_values"], [200])
        self.assertEqual(event["state"]["accepted_groups"], 2)

    def test_base_only_is_exact_j_complete_scales(self):
        event, _ = self.receive((True, False))
        self.assertEqual(event["receive_status"], "BASE_ONLY")
        self.assertEqual((event["state"]["m"], event["state"]["K"]), (1, 0))
        self.assertEqual(event["state"]["prefix"], [[0]])
        self.assertIsNone(event["groups"][1]["accepted_payload"])

    def test_failed_G1_never_uses_good_G2_but_decodes_it(self):
        event, backend = self.receive((False, True))
        self.assertEqual(event["state"], wire.gray_state())
        self.assertTrue(event["groups"][1]["crc_accept"])
        self.assertEqual(len([c for c in backend.calls if c[0] == "packet_rx"]), 2)

    def test_both_failed_still_in_denominator(self):
        event, _ = self.receive((False, False))
        self.assertEqual(event["receive_status"], "G1_REJECT")
        self.assertEqual(len(event["groups"]), 2)
        self.assertTrue(all(g["crc_reject"] for g in event["groups"]))

    def test_header_crc_fail_calls_no_body_and_is_gray(self):
        event, backend = self.receive(header_ok=False)
        self.assertEqual(event["state"], wire.gray_state())
        self.assertEqual(event["groups"], [])
        self.assertFalse(any(c[0] == "packet_rx" for c in backend.calls))

    def test_unknown_id_is_rejected_by_header_fields(self):
        backend = StubWireBackend(self.row, received_id=999)
        event = wire.receive(np.zeros((1024, 2)), 10, 3, self.catalog, backend)
        self.assertTrue(event["header_crc_ok"])
        self.assertFalse(event["header_fields_legal"])
        self.assertEqual(event["groups"], [])

    def test_valid_wrong_header_decodes_received_layout(self):
        wrong = profile(pid=434, q1=4, q2=6, m=2, K=2)
        catalog = wire.Catalogue([self.row, wrong])
        backend = StubWireBackend(wrong)
        event = wire.receive(np.zeros((1024, 2)), 4, 4, catalog, backend)
        self.assertEqual(event["state"]["K"], 2)
        self.assertEqual(event["groups"][0]["q"], 4)
        before = copy.deepcopy(event)
        truth = wire.audit_after_receiver(event, 433, self.payload)
        self.assertTrue(truth["header_false_accept"])
        self.assertFalse(truth["body_comparison_available"])
        self.assertIsNone(truth["undetected_body_errors"])
        self.assertEqual(event, before)

    def test_valid_wrong_legacy_single_header_uses_actual_single_packet(self):
        old = profile(pid=7, q1=6, G=1)
        catalog = wire.Catalogue([old, self.row])
        backend = StubWireBackend(old, accepts=(True,))
        event = wire.receive(np.zeros((1024, 2)), 4, 4, catalog, backend)
        self.assertEqual(event["receive_status"], "SINGLE_FORMAT_ACCEPT")
        self.assertEqual(len(event["groups"]), 1)
        self.assertEqual(event["state"]["m"], 2)

    def test_wrong_legacy_single_crc_failure_still_DROP_not_KEEP(self):
        old = profile(pid=7, G=1)
        catalog = wire.Catalogue([old, self.row])
        backend = StubWireBackend(old, accepts=(False,))
        event = wire.receive(np.zeros((1024, 2)), 4, 4, catalog, backend)
        self.assertEqual(event["state"], wire.gray_state())
        self.assertEqual(event["receive_status"], "SINGLE_FORMAT_DROP_CRC_REJECT")
        self.assertEqual(event["received_profile_metadata_rule"], wire.SINGLE_KEEP_RULE)

    def test_undetected_error_is_not_corrected(self):
        damaged = [a.copy() for a in self.payload]
        damaged[0][0] ^= 1
        event, _ = self.receive(payloads=damaged)
        self.assertEqual(event["state"]["prefix"][0], [2048])
        before = copy.deepcopy(event)
        truth = wire.audit_after_receiver(event, 433, self.payload)
        self.assertEqual(truth["undetected_body_errors"], 1)
        self.assertEqual(event, before)

    def test_energy_and_known_padding_are_counted_without_normalization(self):
        row = profile(idle=5)
        catalog = wire.Catalogue([row])
        backend = StubWireBackend(row)
        wave, meta = wire.transmit(433, self.payload, 90, catalog, backend)
        self.assertEqual(wave.shape, (1024, 2))
        self.assertEqual(meta["header_uses"] + meta["body_uses"] + meta["idle_uses"], 1024)
        self.assertEqual(meta["actual_E"], 68 * 2 + 256 * 2 + 695 * 8 + 5 * 2)
        self.assertEqual(meta["idle_actual_E"], 10)
        np.testing.assert_array_equal(wave[-5:], np.ones((5, 2)))

    def test_true_snr_and_public_counter_reach_both_packets(self):
        event, backend = self.receive()
        packets = [c for c in backend.calls if c[0] == "packet_rx"]
        self.assertEqual([(c[3], c[4]) for c in packets], [(4, 91), (4, 91)])
        self.assertEqual([c[1] for c in packets], [2, 6])

    def test_receiver_signature_has_no_truth_inputs(self):
        self.assertEqual(tuple(inspect.signature(wire.receive).parameters),
                         ("noisy_samples", "true_snr", "public_counter", "catalog", "backend"))
        self.assertNotIn("true_snr", inspect.signature(wire.transmit).parameters)

    def test_catalogue_is_immutable_from_inputs_and_readers(self):
        self.row["groups"][0]["modulation"] = "64QAM"
        row = self.catalog.row(433)
        row["groups"][0]["modulation"] = "64QAM"
        self.assertEqual(self.catalog.row(433)["groups"][0]["modulation"], "QPSK")

    def test_duplicate_and_overflow_ids_fail(self):
        with self.assertRaisesRegex(ValueError, "collide"):
            wire.Catalogue([self.row, self.row])
        with self.assertRaises(ValueError):
            wire.Catalogue([profile(pid=4096)])
        with self.assertRaises(ValueError):
            wire.Catalogue([profile(pid=None)])

    def test_unknown_modulation_cannot_fall_through_to_16QAM(self):
        self.row["groups"][1]["modulation"] = "256QAM"
        with self.assertRaisesRegex(ValueError, "mapping"):
            wire.Catalogue([self.row])

    def test_bad_budget_rate_layout_and_order_rejected(self):
        for mutation in (lambda p: p.update(idle_symbols=1), lambda p: p.update(order="entropy"),
                         lambda p: p["groups"][0]["layout"].update(z=99),
                         lambda p: p.update(K=9)):
            row = copy.deepcopy(self.row)
            mutation(row)
            with self.assertRaises(ValueError):
                wire.Catalogue([row])

    def test_source_float_and_nonbinary_received_data_rejected(self):
        with self.assertRaises(ValueError):
            wire.indices_to_bits([0.5])
        with self.assertRaises(ValueError):
            wire.bits_to_indices([2] * 12)
        with self.assertRaises(ValueError):
            wire.receive(np.zeros((1023, 2)), 4, 1, self.catalog, StubWireBackend(self.row))


if __name__ == "__main__":
    output = None
    if "--report" in sys.argv:
        pos = sys.argv.index("--report")
        output = Path(sys.argv[pos + 1])
        del sys.argv[pos:pos + 2]
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__]))
    if output:
        report = dict(status="PASS" if result.wasSuccessful() else "FAIL", scope="SYNTHETIC_CPU_ENGINEERING_ONLY",
            tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors),
            real_encoder_calls=0, real_decoder_calls=0, channel_calls=0, visual_model_calls=0,
            scientific_qualification="NOT_RUN", scientific_launch="NOT_RUN",
            source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (Path(__file__), Path(wire.__file__))},
            limitations=["Actual runtime/module SHA binding and backend plans must be checked by a finite registered owner.",
                          "No real header/LDPC/QAM roundtrip, BLER, source data or GPU model was invoked.",
                          "New grouped receiver uses mixed public domain and DROP; old single baselines keep raw433 domain and KEEP."])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf8")
    raise SystemExit(0 if result.wasSuccessful() else 1)

"""Synthetic body-only supplement tests; never import Torch or real PHY."""
import argparse
import copy
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import sys

import numpy as np
import r0_bler as b


def group(q=6):
    k, n, z, filler = 40, 60 * q, 2, 4
    layout = dict(k=k, n=n, k_ldpc=k+filler, k_filler=filler, shortening_bits=filler,
        z=z, bg="SYNTHETIC_ONLY", n_cb_comp=n+8, mother_bits=n+filler+12,
        puncturing_first_2Z=2*z, puncturing_remaining=8, puncturing_bits=12,
        repetition_bits=0, implementation={"synthetic":True},
        power_protocol="fixed_constellation_average_Es2")
    layout["layout_id"] = b.digest(layout)
    value = dict(source_bits=k-16, information_bits=k, transmitted_bits=n, symbols=60,
        crc_bits=16, tail_bits=0, modulation={2:"QPSK",4:"16QAM",6:"64QAM"}[q],
        modulation_padding_bits=0, layout=layout)
    value["phy_key"] = b.digest(dict(backend=layout["implementation"], source_bits=k-16,
        CRC=16, layout=layout, modulation=value["modulation"], modulation_padding_bits=0))
    return value


class StubRuntime:
    def __init__(self, outcome="correct", fail=False):
        self.outcome, self.fail = outcome, fail
        self.calls = []
        self.last_payload = None

    def transmit(self, payload, public_group, public_counter):
        self.calls.append("tx"); self.last_payload = payload.copy()
        wave = np.ones((public_group["symbols"], 2), dtype=np.float32)
        return wave, float(np.square(wave).sum())

    def receive(self, received_samples, public_group, true_snr, public_counter):
        self.calls.append("rx")
        if self.fail:
            raise RuntimeError("Synthetic engineering failure")
        self.observed = received_samples.copy()
        hard = self.last_payload.copy()
        if self.outcome == "undetected":
            hard[0] ^= 1
        accepted = self.outcome != "rejected"
        decoded = np.concatenate([hard, np.full(16, int(accepted), np.uint8)])
        return hard, accepted, decoded

    def crc_check(self, decoded):
        return bool(decoded[-1])


class SupplementTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
        self.ledger = b.Ledger(self.root/"ledger.sqlite", "synthetic", cap=128)
        self.g = group(); self.runtime = StubRuntime()

    def tearDown(self):
        self.ledger.close(); self.tmp.cleanup()

    def call(self, trial=0, runtime=None, snr=4):
        return b.execute_trial(self.g, snr, trial, runtime or self.runtime, self.ledger, {"synthetic":True})

    def test_dimensions_q2_q4_q6_and_resolved_layout(self):
        for q in (2,4,6):
            self.assertEqual(b.validate_group(group(q)), (q,40,60*q))
        changed = group(); changed["layout"]["z"] += 1
        with self.assertRaises(ValueError): b.validate_group(changed)

    def test_randomness_fixed_and_domain_separated(self):
        a,n,c = b.random_inputs(self.g,4,0)
        a2,n2,c2 = b.random_inputs(self.g,4,0)
        self.assertTrue(np.array_equal(a,a2) and np.array_equal(n,n2) and c==c2)
        identities = {(b.array_sha(b.random_inputs(self.g,s,i)[0]), b.array_sha(b.random_inputs(self.g,s,i)[1]),
            b.random_inputs(self.g,s,i)[2]) for s in b.SNRS for i in range(64)}
        self.assertEqual(len(identities),192)
        with self.assertRaises(ValueError): b.random_inputs(self.g,13,0)
        with self.assertRaises(ValueError): b.random_inputs(self.g,4,64)

    def test_Es2_noise_and_post_receiver_truth(self):
        row,reused = self.call()
        _,noise,_ = b.random_inputs(self.g,4,0)
        np.testing.assert_array_equal(self.runtime.observed, np.ones_like(noise)+noise*np.float32(10**(-4/20)))
        self.assertEqual(row["observed_mean_Es"],2)
        self.assertTrue(row["correct"] and not reused)
        self.assertEqual(row["header_calls"],0)

    def test_reject_and_undetected_are_distinct_and_keep_hard_bits(self):
        rejected,_ = self.call(0,StubRuntime("rejected"))
        undetected,_ = self.call(1,StubRuntime("undetected"))
        self.assertTrue(rejected["rejected"] and not rejected["correct"])
        self.assertTrue(undetected["undetected"] and undetected["crc_accept"] and not undetected["payload_equal"])

    def test_decode_signature_has_no_truth(self):
        args=list(inspect.signature(b.RealRuntime.receive).parameters)
        self.assertEqual(args,["self","received_samples","public_group","true_snr","public_counter"])

    def test_complete_identity_reuse_before_randomness_tx_rx(self):
        original,_ = self.call()
        with patch.object(b,"random_inputs",side_effect=AssertionError("must not rerun")):
            reused,flag=self.call()
        self.assertEqual(original,reused);self.assertTrue(flag)
        self.assertEqual(self.runtime.calls,["tx","rx"])

    def test_complete_ledger_reopens_with_identical_identity_only(self):
        self.call(); self.ledger.close()
        self.ledger=b.Ledger(self.root/"ledger.sqlite","synthetic",cap=128)
        result,flag=self.call();self.assertTrue(flag)
        with self.assertRaises(ValueError):b.Ledger(self.root/"ledger.sqlite","different",cap=128)

    def test_engineering_failure_reservation_is_not_CRC_reject_and_cannot_replay(self):
        runtime=StubRuntime(fail=True)
        with self.assertRaises(RuntimeError):self.call(runtime=runtime)
        self.assertEqual(self.ledger.snapshot()["unresolved"],1)
        self.assertEqual(self.ledger.snapshot()["complete"],0)
        with self.assertRaises(ValueError):self.call(runtime=runtime)
        with self.assertRaises(ValueError):self.call(1,runtime)
        self.assertEqual(runtime.calls,["tx","rx"])
        with self.assertRaises(ValueError):b.Ledger(self.root/"ledger.sqlite","synthetic",cap=128)

    def test_budget_reserved_before_any_transmit(self):
        self.ledger.cap=1;self.call()
        with self.assertRaises(ValueError):self.call(1)
        self.assertEqual(self.runtime.calls,["tx","rx"])

    def test_bad_wave_and_CRC_engineering_errors_do_not_complete(self):
        runtime=StubRuntime()
        runtime.transmit=lambda *a:(np.ones((60,2),np.float32), 12.)
        with self.assertRaises(ValueError):self.call(runtime=runtime)
        self.assertEqual(runtime.calls,[])
        self.assertEqual(self.ledger.snapshot()["complete"],0)

    def test_full64_point_and_Wilson_keep_nonzero_zeroerror_upper_bound(self):
        records=[self.call(i)[0] for i in range(64)]
        row=b.aggregate(self.g,4,records)
        self.assertEqual((row["n_blocks"],row["n_correct"],row["n_reject"],row["n_undetected"]),(64,64,0,0))
        self.assertGreater(row["p_reject_ci"][1],0)
        self.assertLess(row["p_correct_ci"][0],1)
        with self.assertRaises(ValueError):b.aggregate(self.g,4,records[:63])

    def test_duplicate_trial_cannot_make_complete_point(self):
        row=self.call()[0]
        with self.assertRaises(ValueError):b.aggregate(self.g,4,[row]*64)

    def test_draft_never_constructs_runtime(self):
        p=b.save(self.root/"draft.json",dict(schema=b.SCHEMA,status="DRAFT_NOT_REGISTERED"))
        with patch.object(b,"RealRuntime") as runtime:
            with self.assertRaises(ValueError):b.validate(p)
            runtime.assert_not_called()

    def test_timing_dispatch_without_real_wait_blocks(self):
        d=self.root/"timing";d.mkdir()
        run=self.root/"actual_six_method_timing_v6"
        self.assertTrue(b.timing_idle(d,run))
        run.mkdir();self.assertFalse(b.timing_idle(d,run));run.rmdir()
        b.save(d/"dispatch.json",{"identity":{"pid":99999999}})
        self.assertFalse(b.timing_idle(d,run))
        b.save(d/"actual_wait.json",{"actual_child_waited":True,"exit_code":0})
        self.assertTrue(b.timing_idle(d,run))

    def test_timing_shared_owner_lock_block_is_WAIT_without_calls(self):
        d=self.root/"timing_lock";d.mkdir();calls=[]
        def blocked(file, mode):
            calls.append(mode);raise BlockingIOError("Synthetic exclusive timing owner")
        stub=SimpleNamespace(LOCK_SH=1,LOCK_NB=4,flock=blocked)
        self.assertIsNone(b.timing_shared_lock(d,stub))
        self.assertEqual(calls,[5])
        def success(file,mode):calls.append(mode)
        stub.flock=success
        handle=b.timing_shared_lock(d,stub)
        self.assertFalse(handle.closed);handle.close()


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--report");a=parser.parse_args()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SupplementTests))
    report=dict(status="PASS" if result.wasSuccessful() else "FAIL",tests_run=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),scope="SYNTHETIC_CPU_ONLY",
        real_packet_encodes=0,real_packet_decodes=0,real_PHY_runtime_constructed=False,
        scientific_registration_created=False,scientific_launch="NOT_RUN",torch_imported="torch" in sys.modules,
        source_sha256={Path(__file__).name:b.sha(__file__),"r0_bler.py":b.sha(b.__file__)})
    if a.report:
        p=Path(a.report);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(report,indent=2)+"\n")
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__=="__main__":main()

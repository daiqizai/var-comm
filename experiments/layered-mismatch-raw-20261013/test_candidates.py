"""Synthetic CPU tests: no Torch, models, encoder, decoder, packets or channel."""
import copy
import unittest
import candidates as c


class Unsupported(Exception):
    pass


class FakeBackend:
    def __init__(self):
        self.calls = []

    def plan(self, k, n, q):
        self.calls.append((k, n, q))
        return dict(k=k, n=n, k_ldpc=k+7, k_filler=7, repetition_bits=0, n_cb_comp=n+200,
                    implementation={"implementation": "TEST_ONLY"}, mother_bits=n+300)


class CandidateTests(unittest.TestCase):
    def test_quarter_grid_and_inherited_endpoint(self):
        endpoints = c.endpoint_grid([dict(m=8, K=1)])
        self.assertEqual(len(endpoints), 13)
        self.assertIn((6, 16), endpoints)
        self.assertIn((7, 75), endpoints)
        self.assertIn((8, 42), endpoints)
        self.assertIn((8, 84), endpoints)
        self.assertIn((8, 126), endpoints)

    def test_j_no_empty_second_group(self):
        specs = list(c.specifications([(6, 0), (7, 0)]))
        self.assertTrue(all(s["j"] <= s["m"] and s["k2"] > 16 for s in specs))
        self.assertFalse(any(s["j"] == s["m"] for s in specs))

    def test_exact_round_ties(self):
        self.assertEqual(c.nearest_even(5, 2), 2)
        self.assertEqual(c.nearest_even(7, 2), 4)
        for n in range(1, 100):
            self.assertEqual(c.nearest_even(n, 7), round(n/7))

    def test_eep_includes_both_crc(self):
        specs = [s for s in c.specifications([(6, 16)]) if s["family"] == "EEP"]
        self.assertEqual(len(specs), 6)
        for s in specs:
            self.assertEqual(s["N1"], round(956*s["k1"]/(s["k1"]+s["k2"])))
            self.assertEqual(s["q1"], s["q2"])

    def test_all_modulations_and_q_order(self):
        specs = list(c.specifications([(6, 16)]))
        self.assertEqual({(s["q1"],s["q2"]) for s in specs}, {(2,2),(2,4),(2,6),(4,4),(4,6),(6,6)})
        self.assertTrue(all(s["N1"]+s["N2"] == 956 for s in specs))

    def test_actual_backend_admission_cached(self):
        backend = FakeBackend(); planner = c.LayoutPlanner(backend, (Unsupported,))
        self.assertEqual(planner.plan(100, 200, 2), planner.plan(100, 200, 2))
        self.assertEqual(len(backend.calls), 1)

    def test_unknown_failure_not_disguised_as_illegal(self):
        class Broken:
            def plan(self, *args): raise RuntimeError("engineering failure")
        with self.assertRaises(RuntimeError): c.LayoutPlanner(Broken(), (Unsupported,)).plan(100,200,2)

    def test_known_rejection_kept_verbatim(self):
        class Reject:
            def plan(self, *args): raise Unsupported("real constraint")
        value = c.LayoutPlanner(Reject(), (Unsupported,)).plan(100,200,2)
        self.assertEqual(value["reason"], "real constraint")
        self.assertEqual(value["status"], "UNSUPPORTED_BY_FROZEN_BACKEND")

    def test_rate_filter_and_no_arbitrary_lower_bound(self):
        b = FakeBackend(); profiles, rows = c.generate(c.LayoutPlanner(b, (Unsupported,)), [(6,16)])
        self.assertTrue(any(r["status"] == "REJECT_MAX_RATE_5_OVER_6" for r in rows))
        self.assertTrue(all(6*k <= 5*n for k,n,q in b.calls))
        self.assertTrue(any(k/n < 1/3 for k,n,q in b.calls))
        self.assertTrue(all(p["receiver_rule"] == "DROP" for p in profiles))

    def test_deduplicated_physical_ids_and_old_preserved(self):
        profiles, _ = c.generate(c.LayoutPlanner(FakeBackend(), (Unsupported,)), [(6,16)])
        old = [dict(profile_id=0, wire_key="old", nested={"unchanged":True})]
        before = copy.deepcopy(old); value = c.public_catalogue(old, profiles)
        self.assertEqual(old, before); self.assertEqual(value["profiles"][0], before[0])
        ids = [p["profile_id"] for p in value["profiles"]]
        self.assertEqual(ids, list(range(len(ids))))

    def test_overflow_does_not_truncate_or_reassign_old(self):
        old = [dict(profile_id=0, wire_key="old")]
        new = [dict(profile_id=None, wire_key=str(i)) for i in range(4096)]
        value = c.public_catalogue(old, new)
        self.assertEqual(value["profile_count"], 4097)
        self.assertFalse(value["full_superset_fits_12bit"])
        self.assertTrue(all(p["profile_id"] is None for p in value["profiles"][1:]))
        self.assertEqual(value["profiles"][0]["profile_id"], 0)

    def test_group_physical_identity_uses_actual_backend(self):
        layout = FakeBackend().plan(100, 200, 2); g = c.group(layout, 2, 100)
        self.assertEqual(g["phy_key"], c.identity(dict(backend=layout["implementation"], source_bits=84,
             CRC=16, layout=layout, modulation="QPSK", modulation_padding_bits=0)))


if __name__ == "__main__": unittest.main(verbosity=2)

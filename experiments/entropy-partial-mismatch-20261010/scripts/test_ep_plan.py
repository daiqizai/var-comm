import unittest
from ep_plan import candidates, fallback_endpoints, full_shortlist, ks, select_stream, token_count


class PlanTests(unittest.TestCase):
    def test_grids_and_whole_permissions(self):
        rows = candidates()
        self.assertEqual(len(rows), 144)
        self.assertEqual(sum(r['target_K'] == 0 for r in rows), 36)
        self.assertEqual(ks(8), (42,84,126))
        self.assertEqual(ks(9), (64,128,192))
        self.assertEqual(min(r['source_capacity_bits'] for r in rows), 927)
        self.assertEqual(max(r['source_capacity_bits'] for r in rows), 4751)

    def test_whole_is_unchanged(self):
        self.assertEqual(fallback_endpoints(9,0), ((9,0),(8,0),(7,0),(6,0),(5,0),(4,0)))

    def test_partial_can_fall_back_to_whole(self):
        row = next(r for r in candidates() if r['candidate_id'] == 'm8_K42_q2_r1-2')
        path = fallback_endpoints(8,42)
        sizes = [token_count(*p) for p in path]
        self.assertEqual(sizes, sorted(set(sizes), reverse=True))
        selected = select_stream({(8,42):1000,(8,0):900}, row)
        self.assertEqual((selected['m'], selected['K']), (8,0))
        with self.assertRaisesRegex(ValueError, 'Missing actual'):
            select_stream({(8,0):900}, row)

    def test_actual_flush_length_not_token_estimate(self):
        row = candidates()[0]
        path = fallback_endpoints(row['target_m'], row['target_K'])
        lengths = {p:row['source_capacity_bits']+1 for p in path}
        lengths[path[-1]] = 722
        self.assertEqual(select_stream(lengths,row)['m'],4)
        lengths[path[-1]] = row['source_capacity_bits']+1
        with self.assertRaisesRegex(ValueError, 'No actual'):
            select_stream(lengths,row)

    def test_anchor_never_dropped(self):
        ids = [r['candidate_id'] for r in candidates()]
        winner = 'm9_q6_r5-6'
        result = full_shortlist(ids,winner)
        self.assertIn(winner,result)
        self.assertEqual(len(result),4)
        self.assertEqual(len(full_shortlist(ids,ids[0])),3)
        with self.assertRaisesRegex(ValueError, 'Complete finite'):
            full_shortlist(ids[:-1],winner)


if __name__ == '__main__':
    unittest.main()

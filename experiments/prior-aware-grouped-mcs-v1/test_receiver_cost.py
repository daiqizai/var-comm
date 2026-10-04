"""No CUDA/model calls: timing boundaries and deterministic repetitions."""
import unittest
import receiver_cost as r


class TimingTests(unittest.TestCase):
    def test_warmup_is_separate_and_hash_runs_outside_timed_interval(self):
        events=[];ticks=iter(range(8));n=[0]
        def run():events.append('compute');n[0]+=1;return 'same'
        def clock():events.append('clock');return next(ticks)
        def fingerprint(value):events.append('hash');return value
        result=r.timing_samples(run,lambda:events.append('sync'),lambda:events.append('guard'),fingerprint,clock)
        self.assertEqual(n[0],4);self.assertEqual(result['measured_repetitions'],3)
        self.assertEqual(result['warmup_total_seconds'],1);self.assertEqual(result['seconds'],[1,1,1])
        self.assertEqual(events[:7],['guard','sync','clock','compute','sync','clock','hash'])

    def test_uncached_repeated_outputs_must_match(self):
        n=[0]
        def run():n[0]+=1;return n[0]
        with self.assertRaises(RuntimeError):r.timing_samples(run,lambda:None,lambda:None,lambda v:v)


if __name__=='__main__':unittest.main()

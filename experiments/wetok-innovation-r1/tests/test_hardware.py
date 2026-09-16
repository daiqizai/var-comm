from collections import Counter
from pathlib import Path
import sys
import unittest
from unittest import mock

EXPERIMENT = Path(__file__).resolve().parents[1]
REFERENCE = EXPERIMENT.parent / 'wetok-comm-v2-20260912'
sys.path[:0] = [str(EXPERIMENT / 'src'), str(REFERENCE / 'src')]

from innovation_comm import hardware


class HardwareContextTests(unittest.TestCase):
    def test_receiver_order_balanced_over_all_frames_and_each_snr(self):
        methods = [f'receiver__{index}' for index in range(6)]
        counts = Counter()
        for index in range(100):
            for snr_position in range(7):
                for seed_position in range(3):
                    frame = (index * 7 + snr_position) * 3 + seed_position
                    counts.update((snr_position, name, position) for position, name in enumerate(hardware.receiver_order(methods, frame)))
        self.assertEqual(set(counts.values()), {50})
        self.assertEqual(len(counts), 7 * 6 * 6)

    def test_gpu_query_is_read_only_and_rejects_ambiguous_hardware(self):
        sample = 'GPU-fixture, Test GPU, 86, 79, 375, 2520, Active, Not Active\n'
        with mock.patch.object(hardware.subprocess, 'check_output', return_value=sample) as query:
            record = hardware.gpu_telemetry()
            self.assertEqual(record['temperature.gpu'], '86')
            self.assertEqual(record['clocks_event_reasons.sw_thermal_slowdown'], 'Active')
            arguments = query.call_args.args[0]
            self.assertTrue(any(argument.startswith('--query-gpu=') for argument in arguments))
            self.assertFalse(any(argument in ('-pl', '-ac', '-lgc', '-pm') for argument in arguments))
        with mock.patch.object(hardware.subprocess, 'check_output', return_value=sample + sample):
            with self.assertRaises(RuntimeError):
                hardware.gpu_telemetry()


if __name__ == '__main__':
    unittest.main()

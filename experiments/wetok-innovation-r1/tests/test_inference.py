from pathlib import Path
import sys
import unittest
from unittest import mock

import torch

EXPERIMENT = Path(__file__).resolve().parents[1]
BASE = EXPERIMENT.parent / 'wetok-comm-v2-20260912'
sys.path[:0] = [str(EXPERIMENT / 'src'), str(BASE / 'src')]

from innovation_comm.inference import receive_for_image, verify_pruned_result
from innovation_comm.model import InnovationSystem
from wetok_comm.geometry_candidate import GeometryJSCC
from wetok_comm.native import indices_to_features
from wetok_comm.training import module_sha256


class ImageInferenceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.backends.mha.set_fastpath_enabled(False)
        torch.manual_seed(741)
        self.parent = GeometryJSCC('single_pass', dict(width=24, heads=2, encoder_layers=1, receiver_layers=1,
            channel_positions=204, channel_features=30)).eval().requires_grad_(False)
        with torch.no_grad():
            self.parent.receiver.output.weight.normal_(0, .1)
        self.fusion = {'ratio_maximum': 10000., 'gate_hidden': 16, 'gate_initial_bias': -2.}
        self.source = indices_to_features(torch.randint(256, (2, 16, 16, 4), dtype=torch.uint8))
        self.snrs = torch.tensor([1., 19.])
        self.received = self.parent.transmit(self.source, self.snrs) + torch.randn(2, 3060, 2) * torch.pow(10., self.snrs / 10).rsqrt()[:, None, None]

    def test_no_history_image_is_bit_identical_with_only_one_fine_read(self):
        system = InnovationSystem(self.parent, 'multiscale_no_history', self.fusion).eval().requires_grad_(False)
        before = module_sha256(system)
        with torch.no_grad():
            original = system.receive(self.received, self.snrs)
            with mock.patch.object(system.receiver, 'read', wraps=system.receiver.read) as reader:
                efficient = receive_for_image(system, self.received, self.snrs)
                self.assertEqual(reader.call_count, 1)
                self.assertEqual(reader.call_args.args[3], 16)
            for key in ('logits', 'native_fq', 'receiver_features'):
                self.assertTrue(torch.equal(original[key], efficient[key]))
            self.assertEqual(verify_pruned_result(system, efficient, self.received, self.snrs), 0.)
        self.assertEqual(module_sha256(system), before)

    def test_conditional_reads_are_not_pruned_and_training_is_rejected(self):
        system = InnovationSystem(self.parent, 'multiscale_state_history', self.fusion).eval()
        with self.assertRaises(RuntimeError):
            receive_for_image(system, self.received, self.snrs)
        with torch.no_grad(), mock.patch.object(system.receiver, 'read', wraps=system.receiver.read) as reader:
            receive_for_image(system, self.received, self.snrs)
            self.assertEqual(reader.call_count, 3)
            self.assertEqual([call.args[3] for call in reader.call_args_list], [4, 8, 16])


if __name__ == '__main__':
    unittest.main()

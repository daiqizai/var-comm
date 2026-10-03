"""CPU checks for the qualification gate; no vendor model or GPU needed."""
import importlib.util
import unittest

TORCH=importlib.util.find_spec('torch') is not None


@unittest.skipUnless(TORCH,'Requires native Torch test environment')
class GateTests(unittest.TestCase):
    def test_full_gradient_passes_and_hidden_detach_fails(self):
        import torch
        from hifi_qualification import finite_difference
        class Operator:
            def __init__(self,detached=False): self.detached=detached
            def encode(self,x): return x+2*(x.detach() if self.detached else x)
            def forward(self,x): return x/.7
            def transpose(self,x): return x*.7
            def decode(self,x): return torch.tanh(x*.25)
        image=torch.full((1,3,4,4),.2)
        measurement={'ofdm_sig':torch.zeros_like(image),'x_mse':torch.zeros_like(image)}
        for objective in ('waveform','confirming_rgb'):
            self.assertTrue(finite_difference(Operator(),image,measurement,objective)['passed'])
            with self.assertRaises(RuntimeError): finite_difference(Operator(True),image,measurement,objective)

    def test_parameter_values_flags_and_gradients_are_guarded(self):
        import torch
        from hifi_qualification import parameter_guard,assert_unchanged
        model=torch.nn.Linear(3,2)
        initial=parameter_guard(model)
        assert_unchanged(model,initial)
        model.weight.requires_grad_(False)
        with self.assertRaises(RuntimeError): assert_unchanged(model,initial)
        model.weight.requires_grad_(True)
        model.weight.grad=torch.ones_like(model.weight)
        with self.assertRaises(RuntimeError): assert_unchanged(model,initial)
        model.weight.grad=None
        model.weight.data.add_(1)
        with self.assertRaises(RuntimeError): assert_unchanged(model,initial)


if __name__=='__main__': unittest.main()

"""Process-local attention variant and inclusive CUDA execution accounting."""

from collections import Counter, defaultdict
from contextlib import AbstractContextManager
import time

import torch


class AttentionExecution(AbstractContextManager):
    def __init__(self, model, variant):
        if variant not in ("original", "attention_direct"):
            raise ValueError("only the paired registered execution variants are allowed")
        self.variant = variant
        self.modules = [module for module in model.unet.modules() if module.__class__.__name__ == "AttentionBlock"]
        self.saved = []
        if not self.modules:
            raise RuntimeError("the author U-Net has no recognized attention blocks")

    def __enter__(self):
        if self.variant == "attention_direct":
            for module in self.modules:
                self.saved.append((module, module.forward))

                def direct(values, _module=module):
                    return _module._forward(values)

                module.forward = direct
        return self

    def __exit__(self, error_type, error, trace):
        for module, forward in self.saved:
            module.forward = forward


class ExecutionRecorder(AbstractContextManager):
    def __init__(self, model, capture=True):
        self.model, self.capture = model, capture
        self.counts = Counter()
        self.events, self.cpu_seconds = defaultdict(list), defaultdict(float)
        self.saved, self.snapshots = [], {}
        self.current_step, self.gradient_depth, self.unet_depth = -1, 0, 0
        self.capture_steps = set()
        self.last_outer_gradient = None

    def call(self, category, callback):
        self.counts[category] += 1
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        started = time.perf_counter()
        result = callback()
        self.cpu_seconds[category] += time.perf_counter() - started
        end.record()
        self.events[category].append((start, end))
        return result

    def replace(self, instance, attribute, wrapper):
        original = getattr(instance, attribute)
        self.saved.append((instance, attribute, original))
        setattr(instance, attribute, wrapper)

    def __enter__(self):
        original_unet = self.model.unet.forward

        def unet_forward(*arguments, **keywords):
            self.unet_depth += 1
            try:
                return self.call("unet_forward", lambda: original_unet(*arguments, **keywords))
            finally:
                self.unet_depth -= 1

        self.replace(self.model.unet, "forward", unet_forward)
        for module in self.model.unet.modules():
            if module.__class__.__name__ != "AttentionBlock":
                continue
            original = module._forward

            def attention(values, _original=original):
                category = "attention_initial_forward" if self.unet_depth else "attention_backward_recompute"
                return self.call(category, lambda: _original(values))

            self.replace(module, "_forward", attention)
        for instance, attribute, category in ((self.model, "setup_receive", "setup_receive"),
                                               (self.model.loss, "forward", "consistency_loss_forward"),
                                               (self.model.operator, "encode", "operator_encode_forward"),
                                               (self.model.operator, "decode", "operator_decode_forward")):
            original = getattr(instance, attribute)

            def wrapped(*arguments, _original=original, _category=category, **keywords):
                return self.call(_category, lambda: _original(*arguments, **keywords))

            self.replace(instance, attribute, wrapped)
        original_conditioner = self.model.conditioner

        def conditioner(*arguments, **keywords):
            self.current_step = int(arguments[1])
            schedule, state = arguments[2], arguments[3]
            self.capture_steps = {0, len(schedule.seq) // 2, len(schedule.seq) - 2}
            if self.capture and self.current_step in self.capture_steps:
                self.snapshots[self.current_step] = {"state": state.detach().cpu().numpy().copy(),
                    "cpu_rng": torch.get_rng_state().clone(), "cuda_rng": torch.cuda.get_rng_state(self.model.device).clone()}
            return self.call("posterior_step_inclusive", lambda: original_conditioner(*arguments, **keywords))

        self.replace(self.model, "conditioner", conditioner)
        original_gradient = torch.autograd.grad

        def gradient(*arguments, **keywords):
            outer = self.gradient_depth == 0
            self.gradient_depth += 1
            try:
                if outer:
                    result = self.call("input_gradient_inclusive", lambda: original_gradient(*arguments, **keywords))
                    self.last_outer_gradient = result[0].detach().cpu().numpy().copy()
                    if self.capture and self.current_step in self.snapshots:
                        self.snapshots[self.current_step]["gradient"] = self.last_outer_gradient
                else:
                    targets = keywords.get("inputs", arguments[1] if len(arguments) > 1 else ())
                    if isinstance(targets, torch.Tensor):
                        targets = (targets,)
                    self.counts["checkpoint_inner_autograd_calls"] += 1
                    self.counts["checkpoint_requested_parameter_tensors"] += sum(isinstance(value, torch.nn.Parameter) for value in targets)
                    self.counts["checkpoint_requested_parameter_elements"] += sum(value.numel() for value in targets if isinstance(value, torch.nn.Parameter))
                    result = original_gradient(*arguments, **keywords)
                return result
            finally:
                self.gradient_depth -= 1

        self.replace(torch.autograd, "grad", gradient)
        return self

    def summary(self):
        torch.cuda.synchronize()
        return {"calls": dict(self.counts), "GPU_seconds_inclusive": {category: sum(start.elapsed_time(end) for start, end in events) / 1000
                for category, events in self.events.items()}, "CPU_call_seconds_inclusive": dict(self.cpu_seconds),
                "do_not_sum_nested_categories": True,
                "attention_recompute_is_part_of_input_gradient_not_another_sampler": True}

    def __exit__(self, error_type, error, trace):
        for instance, attribute, original in reversed(self.saved):
            setattr(instance, attribute, original)

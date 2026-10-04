"""Frozen ImageNet ADM + author HiFi-DiffCom, adapted to paid-header Swin.

Build Swin first, then construct this receiver in a dedicated inference worker.
The caller supplies only a CRC-decoded RxContext and measured data I/Q. It must
reuse that exact observation for the Swin and HiFi arms. No generative training,
quality-based parameter selection, global no_grad, or vendor edits occur here.
"""
import ast
import hashlib
import logging
import math
import os
from pathlib import Path
import sys
import time
from types import ModuleType

import numpy as np
import torch
import yaml

from hifi_swin_operator import SwinHiFiOperator, strict_decoded_context
from hifi_swin_schedule import SCHEDULE_MODES, apply_registered_schedule


ADM_SHA256 = "a37c32fffd316cd494cf3f35b339936debdc1576dad13fe57c42399a5dbc78b1"


class _QuietScheduleInitialization:
    # The author's initialization prints its original C2 start. Suppress that
    # intermediate log and report the actual adapted start below instead.
    def info(self, *args, **kwargs):
        pass


def _sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _author_config_type(source):
    """Use the qualified wrapper's minimal Config import, not ADJSCC loading."""
    source = Path(source).resolve()
    for name in ("guided_diffusion", "conditioning_method"):
        module = sys.modules.get(name)
        if module is not None:
            module_path = Path(module.__file__).resolve()
            if source not in module_path.parents:
                raise RuntimeError(f"conflicting {name} module; use a dedicated inference worker")
    sys.path.insert(0, str(source))
    # Swin's top-level utils.py collides with DiffCom's utils package. As in the
    # established adapter, install the latter only after Swin has been built.
    package = ModuleType("utils")
    package.__path__ = [str(source / "utils")]
    sys.modules["utils"] = package
    utility = ModuleType("utils.util")
    utility.__file__ = str(source / "utils/util.py")
    utility.__dict__["os"] = os
    tree = ast.parse(Path(utility.__file__).read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Config"]
    if len(nodes) != 1:
        raise RuntimeError("pinned author Config definition not found")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), utility.__file__, "exec"), utility.__dict__)
    sys.modules["utils.util"] = utility
    package.util = utility
    return utility.Config


class FrozenHiFiSwinReceiver:
    def __init__(self, swin_native, diffcom_vendor, diffusion_checkpoint, *,
                 expected_adm_sha256=ADM_SHA256, schedule_mode="actual_data_cbr"):
        if schedule_mode not in SCHEDULE_MODES:
            raise ValueError("unregistered HiFi schedule mode")
        self.schedule_mode = schedule_mode
        self.model = swin_native
        if self.model.training:
            raise ValueError("load the selected frozen Swin checkpoint in eval mode first")
        self.device = next(self.model.parameters()).device
        if next(self.model.parameters()).dtype != torch.float32:
            raise ValueError("the registered Swin/ADM posterior uses float32")
        self.source = Path(diffcom_vendor).resolve()
        self.adm_checkpoint = Path(diffusion_checkpoint).resolve()
        actual_sha = _sha256(self.adm_checkpoint)
        if actual_sha != expected_adm_sha256:
            raise ValueError("ImageNet ADM checkpoint does not match the registered SHA256")
        config_type = _author_config_type(self.source)
        specification = yaml.safe_load((self.source / "configs/diffcom.yaml").read_text())
        specification.update(conditioning_method="hifi_diffcom", operator_name="swinjscc_paid_header",
                             channel_type="awgn", model_name="256x256_diffusion_uncond",
                             testset_name="registered_calibration_or_development",
                             CSNR=7., model_path=str(self.adm_checkpoint), N=1.0,
                             device=self.device, sigma=math.sqrt(.5 / 10 ** .7), batch_size=1)
        self.config = config_type(specification)
        self.logger = logging.getLogger("frozen_hifi_swin_registered_adaptation")
        from guided_diffusion.script_util import (create_model_and_diffusion,
            model_and_diffusion_defaults, args_to_dict)
        from guided_diffusion.noise_schedule import NoiseSchedule
        from conditioning_method.diffcom import get_conditioning_method, ConsistencyLoss
        from utils import utils_model

        arguments = utils_model.create_argparser({"model_path": str(self.adm_checkpoint),
            "num_channels": 256, "num_res_blocks": 2, "attention_resolutions": "8,16,32"}).parse_args([])
        self.unet, self.diffusion = create_model_and_diffusion(
            **args_to_dict(arguments, model_and_diffusion_defaults().keys()))
        self.unet.load_state_dict(torch.load(self.adm_checkpoint, map_location="cpu"), strict=True)
        self.unet.to(self.device).eval()
        # Preserve custom checkpoint autograd flags. No optimizer is constructed;
        # image gradients use autograd.grad and must never populate parameter.grad.
        self.schedule_type = NoiseSchedule
        self.conditioner = get_conditioning_method("hifi_diffcom").conditioning
        self.loss = ConsistencyLoss(self.config, self.device)
        self._parameters = tuple(p for m in (self.model, self.unet) for p in m.parameters())
        self._versions = tuple(p._version for p in self._parameters)
        self._grad_flags = tuple(p.requires_grad for p in self._parameters)
        self._assert_fixed()
        self.provenance = {"method": "HiFi-DiffCom+SwinJSCC_registered_adaptation",
                           "diffusion_checkpoint_sha256": actual_sha,
                           "diffusion_model": "ImageNet_ADM_256x256_unconditional",
                           "prior_finetuning_steps": 0,
                           "operator_mask": "fixed_CRC_decoded_paid_header",
                           "operator_gain": "fixed_CRC_decoded_float32_power",
                           "confirming_decoder": "same_clamped_RGB_as_paired_Swin_receiver",
                           "upstream_input_detaches_removed_in_new_operator_only": 3,
                           "schedule_mode": schedule_mode,
                           "sampling_parameters": specification["diffcom_series"]["hifi_diffcom"]}

    def _assert_fixed(self):
        if tuple(p._version for p in self._parameters) != self._versions:
            raise RuntimeError("posterior inference changed fixed model parameters")
        if tuple(p.requires_grad for p in self._parameters) != self._grad_flags:
            raise RuntimeError("posterior inference changed checkpoint autograd flags")
        if any(p.grad is not None for p in self._parameters):
            raise RuntimeError("posterior inference accumulated model parameter gradients")

    def receive_frame(self, observed_frame, N, snr_db, *, sampling_seed=23, step_limit=None):
        """Preferred physical entry: decode the paid header from this waveform.

        The frame has N actual complex observations, including the header. Only
        public N/SNR/seed are supplied separately; no transmitter metadata is an
        argument. The project PHY must be configured before calling this method.
        """
        from swin_protocol import decode_header, layout
        spec = layout(N)
        observed = np.asarray(observed_frame, dtype=np.float64)
        if observed.shape != (N, 2) or not np.isfinite(observed).all():
            raise ValueError("received physical frame must contain N finite I/Q pairs")
        context = decode_header(observed[spec["data_N"]:], snr_db, N)
        return self.receive(observed[:spec["data_N"]], snr_db, context,
                            sampling_seed=sampling_seed, step_limit=step_limit)

    def receive(self, observed_data, snr_db, rx_context, *, sampling_seed=23, step_limit=None):
        """Return [3,256,256] numpy RGB and a resource/runtime receipt.

        rx_context comes from swin_protocol's real header decode. step_limit is
        exclusively a qualification probe and is explicitly marked in receipts.
        """
        if not rx_context.accepted:
            return torch.full((3, 256, 256), .5).numpy(), {
                **self.provenance, "N_total": int(rx_context.N), "NFE": 0,
                "header_accepted": False, "fallback": "fixed_gray_0.5",
                "complete_author_schedule": False, "model_parameters_unchanged": True}
        if not torch.is_grad_enabled() or torch.is_inference_mode_enabled():
            raise RuntimeError("HiFi image posterior must run outside no_grad/inference_mode")
        self._assert_fixed()
        context = strict_decoded_context(rx_context.N, snr_db,
            {"power": rx_context.power, "indices": rx_context.indices,
             "rate": rx_context.channels}, crc_accepted=bool(rx_context.accepted))
        operator = SwinHiFiOperator(self.model, context)
        self.config.CSNR = float(snr_db)
        self.config.sigma = math.sqrt(.5 * 10 ** (-float(snr_db) / 10))
        is_cuda = self.device.type == "cuda"
        if is_cuda:
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        start = time.perf_counter()
        measurement = operator.measurement(observed_data)
        schedule = self.schedule_type(self.config, _QuietScheduleInitialization(), self.device)
        schedule_receipt = apply_registered_schedule(schedule, self.config, context.data_n, self.schedule_mode)
        self.logger.info("Registered HiFi schedule: mode=%s data_N=%s CBR=%.12g DSNR=%.12g t_start=%s",
            self.schedule_mode, context.data_n, schedule_receipt["schedule_effective_cbr"],
            schedule_receipt["schedule_dsnr"], schedule.t_start)
        sequence = schedule.seq
        if not len(sequence):
            raise RuntimeError("author noise schedule contains no reverse steps")
        if step_limit is not None and (type(step_limit) is not int or step_limit < 1):
            raise ValueError("diagnostic step_limit must be a positive integer")
        limit = len(sequence) if step_limit is None else min(step_limit, len(sequence))
        cuda_devices = [self.device.index if self.device.index is not None else torch.cuda.current_device()] if is_cuda else []
        with torch.random.fork_rng(devices=cuda_devices):
            torch.random.default_generator.manual_seed(int(sampling_seed))
            if is_cuda:
                torch.cuda.default_generators[cuda_devices[0]].manual_seed(int(sampling_seed))
            state = (schedule.sqrt_alphas_cumprod[schedule.t_start] * (2 * measurement["x_mse"] - 1) +
                     schedule.sqrt_1m_alphas_cumprod[schedule.t_start] * torch.randn_like(measurement["x_mse"]))
            for index in range(limit):
                _pred, _cof, state, _channel, loss = self.conditioner(
                    self.config, index, schedule, state, None, None, measurement,
                    self.unet, self.diffusion, operator, self.loss,
                    last_timestep=(sequence[index] == sequence[-1]))
                if not torch.isfinite(state).all():
                    raise FloatingPointError("author reverse sampler returned nonfinite pixels")
        image = (state.detach() / 2 + .5).clamp(0, 1)[0].cpu().numpy()
        if is_cuda:
            torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - start
        self._assert_fixed()
        receipt = {**self.provenance, **context.ledger(), **schedule_receipt,
                   "header_accepted": True, "fallback": "", "sampling_seed": int(sampling_seed),
                   "t_start": int(schedule.t_start), "NFE": limit,
                   "complete_author_schedule": limit == len(sequence),
                   "diagnostic_probe": step_limit is not None,
                   "RX_seconds": elapsed,
                   "latent_consistency": float(loss["ofdm_sig"].detach()),
                   "confirming_consistency": float(loss["x_mse"].detach()),
                   "model_parameters_unchanged": True, "parameter_gradients_accumulated": False}
        if is_cuda:
            receipt.update(max_memory_allocated_bytes=torch.cuda.max_memory_allocated(self.device),
                           max_memory_reserved_bytes=torch.cuda.max_memory_reserved(self.device))
        return image, receipt

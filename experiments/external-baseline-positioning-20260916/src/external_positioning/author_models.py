"""Execute pinned upstream neural models without unrelated metric/codec imports."""

import ast
from abc import ABC, abstractmethod
import hashlib
import importlib
import logging
import math
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import torch
from torch import nn
import yaml

EXPERIMENT = Path(__file__).resolve().parents[2]


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def tensor_source(pixels, device):
    return torch.from_numpy(np.asarray(pixels).copy()).to(device).float().div(255)[None]


class SwinAuthor:
    def __init__(self, checkpoint, device="cuda:0"):
        if not torch.__version__.startswith("1.12.1"):
            raise RuntimeError("registered Swin evaluation requires the isolated torch1.12.1 environment")
        self.device = torch.device(device)
        source = EXPERIMENT / "vendor/SwinJSCC"
        sys.path.insert(0, str(source))
        from net.network import SwinJSCC

        model_name = "SwinJSCC_w/_SAandRA"
        common = {"model": model_name, "img_size": (256, 256), "C": None, "window_size": 8,
                  "mlp_ratio": 4., "qkv_bias": True, "qk_scale": None, "norm_layer": nn.LayerNorm, "patch_norm": True}
        encoder = {**common, "patch_size": 2, "in_chans": 3, "embed_dims": [128, 192, 256, 320],
                   "depths": [2, 2, 6, 2], "num_heads": [4, 6, 8, 10]}
        decoder = {**common, "embed_dims": [320, 256, 192, 128], "depths": [2, 6, 2, 2], "num_heads": [10, 8, 6, 4]}
        arguments = SimpleNamespace(model=model_name, channel_type="awgn", multiple_snr="1,4,7,10,13", C="32,64,96", distortion_metric="MSE")
        config = SimpleNamespace(encoder_kwargs=encoder, decoder_kwargs=decoder, logger=None, device=self.device,
                                 pass_channel=True, downsample=4, norm=False)
        self.model = SwinJSCC(arguments, config).to(self.device)
        self.model.load_state_dict(torch.load(checkpoint, map_location="cpu"), strict=True)
        self.model.eval().requires_grad_(False)
        self.model.encoder.update_resolution(256, 256)
        self.model.decoder.update_resolution(16, 16)
        self.model.H = self.model.W = 256
        self.parameters = sum(parameter.numel() for parameter in self.model.parameters())
        self.width = 320

    @torch.no_grad()
    def transmit(self, pixels, snr_db, rate):
        image = tensor_source(pixels, self.device)
        features, mask = self.model.encoder(image, float(snr_db), int(rate), self.model.model)
        if tuple(features.shape) != (1, 256, self.width) or not torch.equal(mask, mask[:, :1].expand_as(mask)):
            raise RuntimeError("Swin learned rate mask is not the audited channel-wise layout")
        indices = torch.where(mask[0, 0] > 0)[0].cpu().numpy().tolist()
        if len(indices) != rate:
            raise RuntimeError("Swin active-channel count differs from requested trained rate")
        power = features.square().sum() / mask.sum()
        normalized = features / torch.sqrt(power * 2)
        flat = normalized.flatten()
        half = flat.numel() // 2
        active = torch.where(mask.flatten()[:half] > 0)[0]
        if not torch.equal(mask.flatten()[:half], mask.flatten()[half:]):
            raise RuntimeError("Swin complex pairing does not preserve active I/Q pairs")
        native = torch.stack((flat[:half][active], flat[half:][active]), dim=1)
        signal = native.cpu().numpy().astype(np.float64) * math.sqrt(2)
        return signal, {"power": float(power), "indices": indices, "width": self.width, "rate": rate}

    @torch.no_grad()
    def receive(self, observed, snr_db, metadata):
        observations = torch.as_tensor(np.asarray(observed) / math.sqrt(2), device=self.device, dtype=torch.float32)
        full = torch.zeros(256 * self.width, device=self.device)
        channel_mask = torch.zeros(self.width, device=self.device, dtype=torch.bool)
        channel_mask[metadata["indices"]] = True
        active = torch.where(channel_mask.repeat(128))[0]
        half = full.numel() // 2
        if len(active) != len(observations):
            raise ValueError("received Swin metadata does not match data length")
        full[active] = observations[:, 0]
        full[half + active] = observations[:, 1]
        features = full.reshape(1, 256, self.width) * torch.sqrt(full.new_tensor(metadata["power"] * 2))
        reconstructed = self.model.decoder(features, float(snr_db), self.model.model).clamp(0, 1)
        return reconstructed[0].cpu().numpy()

    @torch.no_grad()
    def native_reference(self, pixels, snr_db, rate, standard_noise):
        image = tensor_source(pixels, self.device)
        features, mask = self.model.encoder(image, float(snr_db), int(rate), self.model.model)
        power = features.square().sum() / mask.sum()
        half = features.numel() // 2
        active = torch.where(mask.flatten()[:half] > 0)[0]
        noise = torch.as_tensor(standard_noise, device=self.device, dtype=torch.float32)
        original = self.model.channel.gaussian_noise_layer

        def supplied_noise(channel_input, std, name=None):
            values = torch.zeros_like(channel_input)
            values[active] = torch.complex(noise[:, 0], noise[:, 1]) * float(std)
            return channel_input + values

        self.model.channel.gaussian_noise_layer = supplied_noise
        try:
            noisy = self.model.channel.forward(features, float(snr_db), power) * mask
            return self.model.decoder(noisy, float(snr_db), self.model.model).clamp(0, 1)[0].cpu().numpy()
        finally:
            self.model.channel.gaussian_noise_layer = original


def load_diffcom_modules():
    source = EXPERIMENT / "vendor/diffcom_code"
    sys.path.insert(0, str(source))
    package = ModuleType("utils")
    package.__path__ = [str(source / "utils")]
    sys.modules["utils"] = package
    utility = ModuleType("utils.util")
    utility.__file__ = str(source / "utils/util.py")
    utility.__dict__["os"] = os
    tree = ast.parse(Path(utility.__file__).read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Config"]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), utility.__file__, "exec"), utility.__dict__)
    sys.modules["utils.util"] = utility
    package.util = utility
    from _djscc.network import ADJSCC
    from channel.channel import Channel

    filename = source / "guided_diffusion/measurement.py"
    tree = ast.parse(filename.read_text())
    names = {"register_operator", "get_operator", "NonlinearOperator", "shuffle", "de_shuffle", "ChannelWrapper", "DeepJSCC"}
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    module = ModuleType("author_awgn_measurement")
    module.__file__ = str(filename)
    module.__dict__.update(torch=torch, nn=nn, ABC=ABC, abstractmethod=abstractmethod, ADJSCC=ADJSCC,
                           Channel=Channel, Config=utility.Config, __OPERATOR__={})
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(filename), "exec"), module.__dict__)
    return module, utility.Config


class DiffComAuthor:
    def __init__(self, checkpoint, channels, diffusion_checkpoint=None, device="cuda:0"):
        self.device = torch.device(device)
        source = EXPERIMENT / "vendor/diffcom_code"
        measurement, config_type = load_diffcom_modules()
        specification = yaml.safe_load((source / "configs/diffcom.yaml").read_text())
        specification.update(conditioning_method="hifi_diffcom", operator_name="djscc", channel_type="awgn",
                             model_name="256x256_diffusion_uncond", testset_name="original_calibration_or_development",
                             djscc={"channel_num": channels, "jscc_model_path": str(checkpoint)}, CSNR=10.,
                             model_path=str(diffusion_checkpoint) if diffusion_checkpoint else "", N=1.0,
                             device=self.device, sigma=math.sqrt(.5 / 10), batch_size=1)
        self.config = config_type(specification)
        self.logger = logging.getLogger("frozen_author_diffcom")
        self.operator = measurement.DeepJSCC(self.config, self.logger, self.device)
        self.operator.model.to(self.device).eval().requires_grad_(False)
        self.channels = channels
        self.real_length = channels * 64 * 64
        self.parameters = sum(parameter.numel() for parameter in self.operator.model.parameters())
        self.diffusion_parameters = 0
        self.unet = None
        if diffusion_checkpoint is not None:
            from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults, args_to_dict
            from guided_diffusion.noise_schedule import NoiseSchedule
            from conditioning_method.diffcom import get_conditioning_method, ConsistencyLoss
            from utils import utils_model

            arguments = utils_model.create_argparser({"model_path": str(diffusion_checkpoint), "num_channels": 256,
                                                     "num_res_blocks": 2, "attention_resolutions": "8,16,32"}).parse_args([])
            self.unet, self.diffusion = create_model_and_diffusion(**args_to_dict(arguments, model_and_diffusion_defaults().keys()))
            self.unet.load_state_dict(torch.load(diffusion_checkpoint, map_location="cpu"), strict=True)
            self.unet.to(self.device).eval()
            self.operator.model.requires_grad_(True)
            self.diffusion_parameters = sum(parameter.numel() for parameter in self.unet.parameters())
            self.schedule_type = NoiseSchedule
            self.conditioner = get_conditioning_method("hifi_diffcom").conditioning
            self.loss = ConsistencyLoss(self.config, self.device)
            self.fixed_parameter_versions = tuple(parameter._version for model in (self.operator.model, self.unet) for parameter in model.parameters())

    def permutation(self, frame_seed):
        generator = torch.Generator(device="cpu").manual_seed(int(frame_seed))
        return torch.randperm(self.real_length, generator=generator)

    @torch.no_grad()
    def transmit(self, pixels, snr_db, frame_seed):
        self.config.CSNR = float(snr_db)
        features = self.operator.encode(tensor_source(pixels, self.device))
        permutation = self.permutation(frame_seed)
        shuffled = features[..., permutation]
        power = shuffled.square().sum() / torch.ones_like(shuffled).sum()
        normalized = shuffled / torch.sqrt(power * 2)
        signal = normalized.reshape(-1, 2).cpu().numpy().astype(np.float64) * math.sqrt(2)
        return signal, {"power": float(power), "frame_seed": int(frame_seed), "channels": self.channels}

    def setup_receive(self, observed, snr_db, metadata):
        self.config.CSNR = float(snr_db)
        self.config.sigma = math.sqrt(1 / (2 * 10 ** (float(snr_db) / 10)))
        self.operator.s_shape = (1, self.channels, 64, 64)
        self.operator.channel.s_shape = (1, self.real_length)
        self.operator.channel.CSNR = float(snr_db)
        self.operator.channel.channel.chan_param = float(snr_db)
        self.operator.channel.shuffled_indices = self.permutation(metadata["frame_seed"])
        self.operator.channel.avg_pwr = torch.tensor(metadata["power"], device=self.device, dtype=torch.float32)
        received = torch.as_tensor(np.asarray(observed) / math.sqrt(2), device=self.device, dtype=torch.float32).reshape(1, -1)
        with torch.no_grad():
            deinterleaved = self.operator.transpose(received)
            base = self.operator.decode(deinterleaved)
        return {"x_mse": base.detach(), "ofdm_sig": received.detach(), "s_hat": deinterleaved.detach(),
                "cof_est": None, "cof_gt": None, "channel_usage": len(observed)}

    @torch.no_grad()
    def base_receive(self, observed, snr_db, frame_seed):
        measurement = self.setup_receive(observed, snr_db, {"power": 1.0, "frame_seed": frame_seed})
        return measurement["x_mse"].clamp(0, 1)[0].cpu().numpy()

    @torch.no_grad()
    def native_reference(self, pixels, snr_db, frame_seed, standard_noise):
        self.config.CSNR = float(snr_db)
        self.operator.channel.CSNR = float(snr_db)
        channel = self.operator.channel.channel
        channel.chan_param = float(snr_db)
        original = channel.gaussian_noise_layer
        noise = torch.as_tensor(standard_noise, device=self.device, dtype=torch.float32)

        def supplied_noise(channel_input, std, name=None):
            return channel_input + torch.complex(noise[:, 0], noise[:, 1]).reshape_as(channel_input) * float(std)

        channel.gaussian_noise_layer = supplied_noise
        torch.manual_seed(int(frame_seed))
        try:
            measurement = self.operator.observe_and_transpose(tensor_source(pixels, self.device))
            return measurement["x_mse"].clamp(0, 1)[0].cpu().numpy()
        finally:
            channel.gaussian_noise_layer = original

    def hifi_receive(self, observed, snr_db, metadata, sampling_seed=23, step_limit=None):
        if self.unet is None:
            raise RuntimeError("diffusion checkpoint not loaded")
        measurement = self.setup_receive(observed, snr_db, metadata)
        schedule = self.schedule_type(self.config, self.logger, self.device)
        torch.manual_seed(sampling_seed)
        torch.cuda.manual_seed_all(sampling_seed)
        state = (schedule.sqrt_alphas_cumprod[schedule.t_start] * (2 * measurement["x_mse"] - 1) +
                 schedule.sqrt_1m_alphas_cumprod[schedule.t_start] * torch.randn_like(measurement["x_mse"]))
        sequence = schedule.seq
        limit = len(sequence) if step_limit is None else min(step_limit, len(sequence))
        for index in range(limit):
            _predicted, _coefficient, state, _channel_state, loss = self.conditioner(
                self.config, index, schedule, state, None, None, measurement, self.unet, self.diffusion,
                self.operator, self.loss, last_timestep=(sequence[index] == sequence[-1]))
            if not torch.isfinite(state).all():
                raise FloatingPointError("author reverse sampler produced a nonfinite state")
        image = (state.detach() / 2 + .5).clamp(0, 1)[0].cpu().numpy()
        current_versions = tuple(parameter._version for model in (self.operator.model, self.unet) for parameter in model.parameters())
        if current_versions != self.fixed_parameter_versions:
            raise RuntimeError("inference changed a supposedly fixed model parameter")
        if any(parameter.grad is not None for model in (self.operator.model, self.unet) for parameter in model.parameters()):
            raise RuntimeError("image posterior sampling accumulated model-parameter gradients")
        return image, {"t_start": int(schedule.t_start), "NFE": limit, "complete_author_schedule": step_limit is None,
                       "sampling_seed": sampling_seed, "latent_consistency": float(loss["ofdm_sig"].detach()),
                       "confirming_consistency": float(loss["x_mse"].detach()), "model_parameters_unchanged": True,
                       "upstream_checkpoint_autograd_flags_retained": True}

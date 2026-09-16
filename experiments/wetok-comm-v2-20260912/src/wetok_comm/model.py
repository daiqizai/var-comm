"""Global native-Fq transmission and matched single/multiscale receivers."""

import torch
from torch import nn
from torch.nn import functional

from .native import hard_native_st


class SNRModulation(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, width * 2))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, hidden, snrs):
        gain, shift = self.network(snrs[:, None] / 20).chunk(2, -1)
        return hidden * (1 + gain[:, None]) + shift[:, None]


class CommunicationEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        width, heads, layers = config['width'], config['heads'], config['encoder_layers']
        self.positions, self.features = config['channel_positions'], config['channel_features']
        spatial, unused = torch.linalg.qr(torch.randn(256, 256))
        channel, unused = torch.linalg.qr(torch.randn(self.features, self.features))
        self.spatial_mixing = nn.Parameter(spatial[:, :self.positions].transpose(0, 1).contiguous())
        self.channel_lift = nn.Linear(32, self.features, bias=False)
        self.channel_lift.weight.data.copy_(channel[:, :32])
        self.input_projection = nn.Linear(32, width)
        self.position = nn.Parameter(torch.randn(1, 256, width) * .02)
        self.snr = SNRModulation(width)
        self.blocks = nn.ModuleList([nn.TransformerEncoderLayer(width, heads, 4 * width, dropout=0.,
            activation='gelu', batch_first=True, norm_first=True) for layer in range(layers)])
        self.output = nn.Linear(width, self.features)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, source, snrs):
        values = source.flatten(2).transpose(1, 2)
        hidden = self.snr(self.input_projection(values) + self.position, snrs)
        for block in self.blocks:
            hidden = block(hidden)
        values = self.channel_lift(values) + self.output(hidden)
        symbols = torch.einsum('mn,bnc->bmc', self.spatial_mixing, values).reshape(len(source), -1, 2)
        return symbols / symbols.square().mean((1, 2), keepdim=True).clamp_min(1e-8).sqrt()


class CommunicationReceiver(nn.Module):
    def __init__(self, encoder, config):
        super().__init__()
        width, heads = config['width'], config['heads']
        self.positions, self.features = config['channel_positions'], config['channel_features']
        self.spatial_inverse = nn.Parameter(encoder.spatial_mixing.detach().t().clone() * (256 / self.positions))
        self.channel_inverse = nn.Linear(self.features, 32, bias=False)
        self.channel_inverse.weight.data.copy_(encoder.channel_lift.weight.detach().t())
        self.memory_input = nn.Linear(self.features, width)
        self.memory_position = nn.Parameter(torch.randn(1, self.positions, width) * .02)
        self.memory_snr = SNRModulation(width)
        self.memory_block = nn.TransformerEncoderLayer(width, heads, 4 * width, dropout=0., activation='gelu',
                                                       batch_first=True, norm_first=True)
        self.query_input = nn.Linear(32, width)
        self.context_input = nn.Sequential(nn.LayerNorm(32), nn.Linear(32, width), nn.GELU(), nn.Linear(width, width))
        self.query_position = nn.Parameter(torch.randn(1, width, 16, 16) * .02)
        self.scale_condition = nn.Sequential(nn.Linear(2, 64), nn.SiLU(), nn.Linear(64, width))
        self.query_snr = SNRModulation(width)
        self.blocks = nn.ModuleList([nn.TransformerDecoderLayer(width, heads, 4 * width, dropout=0., activation='gelu',
            batch_first=True, norm_first=True) for layer in range(config['receiver_layers'])])
        self.output = nn.Linear(width, 32)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def prepare(self, received, snrs):
        values = received.reshape(len(received), self.positions, self.features)
        guessed = self.channel_inverse(torch.einsum('nm,bmc->bnc', self.spatial_inverse, values))
        guessed = guessed.transpose(1, 2).reshape(len(received), 32, 16, 16)
        memory = self.memory_block(self.memory_snr(self.memory_input(values) + self.memory_position, snrs))
        return guessed, memory

    def read(self, guessed, memory, snrs, size, previous=None):
        base = functional.adaptive_avg_pool2d(guessed, size)
        context = base if previous is None else functional.interpolate(previous, size=(size, size), mode='bilinear', align_corners=False)
        positions = functional.adaptive_avg_pool2d(self.query_position, size).flatten(2).transpose(1, 2)
        scale = guessed.new_tensor([[size / 16, 16 / size]]).expand(len(guessed), -1)
        query = self.query_input(base.flatten(2).transpose(1, 2)) + positions
        query = query + self.context_input(context.flatten(2).transpose(1, 2)) + self.scale_condition(scale)[:, None]
        query = self.query_snr(query, snrs)
        for block in self.blocks:
            query = block(query, memory)
        return base + self.output(query).transpose(1, 2).reshape(len(guessed), 32, size, size)


class NativeWeTokJSCC(nn.Module):
    def __init__(self, variant, config):
        super().__init__()
        if variant not in ('single_pass', 'multiscale_no_history', 'multiscale_conditioned'):
            raise ValueError('unknown receiver comparison arm')
        self.variant = variant
        self.encoder = CommunicationEncoder(config)
        self.receiver = CommunicationReceiver(self.encoder, config)

    def transmit(self, source_fq, snrs):
        if tuple(source_fq.shape[1:]) != (32, 16, 16) or snrs.shape != (len(source_fq),):
            raise ValueError('TX accepts native Fq and nominal SNR only')
        return self.encoder(source_fq, snrs)

    def receive(self, received, snrs):
        if received.shape[1:] != (3060, 2) or snrs.shape != (len(received),):
            raise ValueError('RX accepts the paid waveform and nominal SNR only')
        guessed, memory = self.receiver.prepare(received, snrs)
        states = []
        if self.variant == 'single_pass':
            logits = self.receiver.read(guessed, memory, snrs, 16)
            expected = torch.tanh(logits / 2)
            states = [functional.adaptive_avg_pool2d(expected, size) for size in (4, 8)]
        else:
            previous = None
            for size in (4, 8, 16):
                context = previous if self.variant == 'multiscale_conditioned' else None
                logits = self.receiver.read(guessed, memory, snrs, size, context)
                if size < 16:
                    previous = torch.tanh(logits / 2)
                    states.append(previous)
        return {'logits': logits, 'native_fq': hard_native_st(logits), 'states': states}


def communicate(network, source_fq, snrs, standard_noise, noiseless=False):
    transmitted = network.transmit(source_fq, snrs)
    if standard_noise.shape != transmitted.shape:
        raise ValueError('noise must cover exactly the paid real coordinates')
    received = transmitted if noiseless else transmitted + standard_noise * torch.pow(10., snrs / 10).rsqrt()[:, None, None]
    result = network.receive(received, snrs)
    result.update(transmitted=transmitted, received=received)
    return result

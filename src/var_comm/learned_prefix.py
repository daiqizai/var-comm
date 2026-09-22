"""Global learned prefix transmission and differentiable hard next-scale reception."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as functional

from .next_scale_prior import PATCH_NUMS

PREFIX_LENGTHS = tuple(size ** 2 for size in PATCH_NUMS[:8])


def straight_through_embedding(logits, codebook):
    indices = logits.argmax(dim=-1)
    hard = functional.embedding(indices, codebook)
    if torch.is_grad_enabled() and logits.requires_grad:
        soft = logits.softmax(dim=-1) @ codebook
        hard = hard + (soft - soft.detach())
    return hard, indices


def empty_cache(var):
    return all(not block.attn.caching and block.attn.cached_k is None and block.attn.cached_v is None for block in var.blocks)


class FrozenVARStream:
    def __init__(self, vae, var, labels):
        if vae.training or var.training or var.cond_drop_rate != 0 or not empty_cache(var):
            raise RuntimeError('frozen source model or cache state changed')
        self.vae, self.var = vae, var
        self.condition = var.class_emb(labels)
        self.positions = var.lvl_embed(var.lvl_1L) + var.pos_1LC
        self.next_input = self.condition[:, None].expand(-1, var.first_l, -1) + var.pos_start + self.positions[:, :var.first_l]
        self.fhat = self.condition.new_zeros(len(labels), var.Cvae, 16, 16)
        self.scale, self.offset, self.processed = 0, 0, False
        for block in var.blocks:
            block.attn.kv_caching(True)

    def predict(self, need_logits=True):
        if self.processed or self.scale >= 10:
            raise RuntimeError('a scale was read twice or outside the schedule')
        hidden = self.next_input
        conditioned = self.var.shared_ada_lin(self.condition)
        for block in self.var.blocks:
            hidden = block(x=hidden, cond_BD=conditioned, attn_bias=None)
        self.processed = True
        return self.var.get_logits(hidden, self.condition).float() if need_logits else None

    def state_features(self):
        size = PATCH_NUMS[self.scale]
        return functional.interpolate(self.fhat, size=(size, size), mode='area').flatten(2).transpose(1, 2)

    def advance(self, embeddings):
        if not self.processed:
            raise RuntimeError('source scale was not processed')
        size = PATCH_NUMS[self.scale]
        values = embeddings.transpose(1, 2).reshape(len(embeddings), 32, size, size)
        if self.scale < 9:
            values = functional.interpolate(values, size=(16, 16), mode='bicubic')
        residual = self.vae.quantize.quant_resi[self.scale / 9](values)
        self.fhat = self.fhat + residual
        self.offset += size ** 2
        self.scale += 1
        if self.scale < 10:
            size = PATCH_NUMS[self.scale]
            next_features = functional.interpolate(self.fhat, size=(size, size), mode='area').flatten(2).transpose(1, 2)
            self.next_input = self.var.word_embed(next_features) + self.positions[:, self.offset:self.offset + size ** 2]
        self.processed = False

    def image(self):
        if self.scale != 10:
            raise RuntimeError('cannot decode an incomplete image state')
        return self.vae.decoder(self.vae.post_quant_conv(self.fhat)).clamp(-1, 1).add(1).mul(0.5)

    def close(self):
        for block in self.var.blocks:
            block.attn.kv_caching(False)


def render_predicted_prefix(prefix, labels, vae, var, codebook):
    stream = FrozenVARStream(vae, var, labels)
    suffix = []
    try:
        for embeddings in prefix:
            stream.predict(need_logits=False)
            stream.advance(embeddings)
        for _scale in (8, 9):
            embeddings, _indices = straight_through_embedding(stream.predict(), codebook)
            suffix.append(embeddings)
            stream.advance(embeddings)
        return stream.image(), suffix
    finally:
        stream.close()


class GlobalPrefixEncoder(nn.Module):
    def __init__(self, width=128, layers=2):
        super().__init__()
        basis, _upper = torch.linalg.qr(torch.randn(255, 255))
        self.mixing = nn.Parameter(basis[:, :187].transpose(0, 1).contiguous())
        self.channel_mixing = nn.Linear(32, 32, bias=False)
        nn.init.eye_(self.channel_mixing.weight)
        self.input_projection = nn.Linear(32, width)
        self.position = nn.Parameter(torch.randn(1, 255, width) * 0.02)
        self.blocks = nn.ModuleList([nn.TransformerEncoderLayer(width, 4, width * 4, dropout=0.0, activation='gelu',
                                                               batch_first=True, norm_first=True) for _layer in range(layers)])
        self.output = nn.Linear(width, 32)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, embeddings):
        hidden = self.input_projection(embeddings) + self.position
        for block in self.blocks:
            hidden = block(hidden)
        refined = self.channel_mixing(embeddings + self.output(hidden))
        symbols = torch.einsum('mn,bnc->bmc', self.mixing, refined).reshape(len(embeddings), 2992, 2)
        return symbols / symbols.square().mean(dim=(1, 2), keepdim=True).clamp_min(1e-8).sqrt()


class PrefixReader(nn.Module):
    def __init__(self, mixing, width=128, layers=2):
        super().__init__()
        self.backprojection = nn.Parameter(mixing.detach().transpose(0, 1).clone() * (255 / 187))
        self.memory_input = nn.Linear(32, width)
        self.memory_position = nn.Parameter(torch.randn(1, 187, width) * 0.02)
        self.query_input = nn.Linear(32, width)
        self.query_position = nn.Parameter(torch.randn(1, 255, width) * 0.02)
        self.class_embedding = nn.Embedding(1000, width)
        nn.init.normal_(self.class_embedding.weight, std=0.02)
        self.snr_embedding = nn.Sequential(nn.Linear(1, width), nn.SiLU(), nn.Linear(width, width))
        self.context_fusion = nn.Sequential(nn.LayerNorm(65), nn.Linear(65, width), nn.GELU(), nn.Linear(width, width))
        self.memory_block = nn.TransformerEncoderLayer(width, 4, width * 4, dropout=0.0, activation='gelu', batch_first=True, norm_first=True)
        self.blocks = nn.ModuleList([nn.TransformerDecoderLayer(width, 4, width * 4, dropout=0.0, activation='gelu',
                                                               batch_first=True, norm_first=True) for _layer in range(layers)])
        self.output = nn.Linear(width, 32)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.gate = nn.Sequential(nn.Linear(width + 1, 64), nn.GELU(), nn.Linear(64, 1))
        nn.init.constant_(self.gate[-1].bias, -2.2)
        self.precision_raw = nn.Parameter(torch.tensor(math.log(math.expm1(8.0))))

    def prepare(self, received, labels, snrs):
        values = received.reshape(len(received), 187, 32)
        guessed = torch.einsum('nm,bmc->bnc', self.backprojection, values)
        condition = self.class_embedding(labels) + self.snr_embedding(snrs[:, None] / 20)
        memory = self.memory_block(self.memory_input(values) + self.memory_position + condition[:, None])
        return guessed, memory, condition

    def codebook_logits(self, values, normalized_codebook):
        precision = functional.softplus(self.precision_raw).clamp_min(0.1)
        return (2 * values @ normalized_codebook.transpose(0, 1) - normalized_codebook.square().sum(-1)) * (precision / 32)

    def read(self, guessed, memory, condition, start, stop, context_logits, state, normalized_codebook):
        context_logp = context_logits.log_softmax(dim=-1)
        context_probability = context_logp.exp()
        expected = context_probability @ normalized_codebook
        entropy = -(context_probability * context_logp).sum(dim=-1, keepdim=True) / math.log(4096)
        features = self.context_fusion(torch.cat((expected, state, entropy), dim=-1))
        query = self.query_input(guessed[:, start:stop]) + self.query_position[:, start:stop] + condition[:, None] + features
        for block in self.blocks:
            query = block(query, memory)
        refined = guessed[:, start:stop] + self.output(query)
        channel_logp = self.codebook_logits(refined, normalized_codebook).log_softmax(dim=-1)
        gate = self.gate(torch.cat((query, entropy), dim=-1)).sigmoid()
        return (1 - gate) * channel_logp + gate * context_logp, gate


class PrefixJSCC(nn.Module):
    def __init__(self, codebook, mean, std, variant, width=128, layers=2):
        super().__init__()
        if variant not in ('parallel', 'next_scale'):
            raise ValueError('unknown learned receiver variant')
        self.variant = variant
        self.register_buffer('codebook', codebook.detach().float().clone())
        self.register_buffer('embedding_mean', mean.detach().float().clone())
        self.register_buffer('embedding_std', std.detach().float().clone())
        self.register_buffer('normalized_codebook', (codebook.detach().float() - mean) / std)
        self.encoder = GlobalPrefixEncoder(width, layers)
        self.reader = PrefixReader(self.encoder.mixing, width, layers)

    def transmit(self, indices):
        if indices.dtype != torch.long or indices.shape[1:] != (255,):
            raise ValueError('source must be the 255 discrete prefix indices')
        embeddings = functional.embedding(indices, self.codebook)
        return self.encoder((embeddings - self.embedding_mean) / self.embedding_std)

    def receive(self, received, labels, snrs, vae, var, teacher_tokens=None, teacher_mask=None):
        if not self.training and teacher_tokens is not None:
            raise RuntimeError('ground-truth context is forbidden at evaluation')
        guessed, memory, condition = self.reader.prepare(received, labels, snrs)
        gates, logits_by_scale, predicted, indices_by_scale = [], [], [], []
        if self.variant == 'parallel':
            if teacher_tokens is not None:
                raise RuntimeError('parallel receiver cannot use teacher context')
            context_logits = self.reader.codebook_logits(guessed, self.normalized_codebook)
            logits, gate = self.reader.read(guessed, memory, condition, 0, 255, context_logits, guessed, self.normalized_codebook)
            embeddings, indices = straight_through_embedding(logits, self.codebook)
            predicted = list(embeddings.split(PREFIX_LENGTHS, dim=1))
            image, suffix = render_predicted_prefix(predicted, labels, vae, var, self.codebook)
            return {'image': image, 'logits': logits, 'indices': indices, 'prefix_embeddings': predicted,
                    'suffix_embeddings': suffix, 'gates': gate, 'teacher_samples': 0}
        teacher_active = teacher_tokens is not None and bool(teacher_mask.any())
        teacher_embeddings = None
        if teacher_active:
            teacher_embeddings = functional.embedding(teacher_tokens, self.codebook).split(PREFIX_LENGTHS, dim=1)
        stream = FrozenVARStream(vae, var, labels)
        offset = 0
        try:
            for scale, length in enumerate(PREFIX_LENGTHS):
                prior_logits = stream.predict()
                state = stream.state_features() / self.embedding_std
                logits, gate = self.reader.read(guessed, memory, condition, offset, offset + length, prior_logits, state, self.normalized_codebook)
                embeddings, indices = straight_through_embedding(logits, self.codebook)
                predicted.append(embeddings)
                logits_by_scale.append(logits)
                gates.append(gate)
                indices_by_scale.append(indices)
                context = torch.where(teacher_mask[:, None, None], teacher_embeddings[scale], embeddings) if teacher_active else embeddings
                stream.advance(context)
                offset += length
            if teacher_active:
                stream.close()
                image, suffix = render_predicted_prefix(predicted, labels, vae, var, self.codebook)
            else:
                suffix = []
                for _scale in (8, 9):
                    embeddings, _indices = straight_through_embedding(stream.predict(), self.codebook)
                    suffix.append(embeddings)
                    stream.advance(embeddings)
                image = stream.image()
        finally:
            stream.close()
        return {'image': image, 'logits': torch.cat(logits_by_scale, dim=1), 'indices': torch.cat(indices_by_scale, dim=1),
                'prefix_embeddings': predicted, 'suffix_embeddings': suffix, 'gates': torch.cat(gates, dim=1),
                'teacher_samples': int(teacher_mask.sum()) if teacher_active else 0}

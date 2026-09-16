"""Inference-only adapter for the *paired* XQ-GAN MSVR10P2 + d17 VAR.

No training; no coordinate truncation; each residual stage contains TWO codebooks.
Upstream source is pinned and unmodified. Constructor-only pretrained downloads are
suppressed because the complete upstream state is loaded strictly immediately after.
"""
from __future__ import annotations
import contextlib
import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import patch
import torch
from checkpoint_io import load_xq_checkpoint

SCALES = (1, 1, 2, 3, 3, 4, 5, 6, 8, 11)
CODEBOOK_SIZE = 4096
PRODUCT_QUANT = 2

def token_metadata(tokens, prefix_scales):
    m = int(prefix_scales)
    if not 0 <= m <= len(SCALES) or len(tokens) != PRODUCT_QUANT:
        raise ValueError('Expected two product branches and an actual natural stage count')
    shapes, counts = [], []
    batch = None
    for branch in tokens:
        if len(branch) < m or len(branch) > len(SCALES):
            raise ValueError('Missing prefix or extra residual stages')
        sh, n = [], 0
        for si, t in enumerate(branch[:m]):
            if t.ndim != 2 or t.shape[1] != SCALES[si] ** 2:
                raise ValueError(f'Unexpected index shape at stage {si + 1}: {t.shape}')
            if t.dtype not in (torch.int32, torch.int64):
                raise TypeError('Indices must be integer, not latent feature values')
            if batch is None: batch = int(t.shape[0])
            if batch != t.shape[0] or t.numel() == 0:
                raise ValueError('Inconsistent batch')
            if int(t.min()) < 0 or int(t.max()) >= CODEBOOK_SIZE:
                raise ValueError('Each product branch uses [0,4096), NOT [0,8192)')
            sh.append(list(t.shape)); n += int(t.shape[1])
        shapes.append(sh); counts.append(n)
    return {'raw_bits': sum(counts) * 12, 'index_count': sum(counts),
            'shapes': shapes, 'indices_per_branch': counts, 'product_quant': 2,
            'codebook_size_per_branch': 4096, 'fixed_bits_per_index': 12,
            'prefix_scales': m, 'scale_table': list(SCALES),
            'raw_bits_excludes_class_and_channel': True}

def token_hash(tokens):
    h = hashlib.sha256()
    for branch in tokens:
        for t in branch:
            x = t.detach().cpu().contiguous()
            h.update(str(tuple(x.shape)).encode()); h.update(x.numpy().tobytes())
    return h.hexdigest()

def extract_state(payload):
    if isinstance(payload, dict):
        if payload and all(isinstance(v, torch.Tensor) for v in payload.values()): return payload
        for key in ('state_dict', 'model'):
            if key in payload: return extract_state(payload[key])
    raise ValueError('Unknown checkpoint state layout; refuse silent partial load')

def compare_states(a, b):
    # Memory-mapped checkpoint tensors remain on CPU; never load an optimizer/GPU.
    ka, kb = set(a), set(b)
    diffs = [k for k in sorted(ka & kb)
             if a[k].shape != b[k].shape or a[k].dtype != b[k].dtype or not torch.equal(a[k], b[k])]
    return {'same_keys': ka == kb, 'exactly_equal': ka == kb and not diffs,
            'mismatched_tensors': len(diffs), 'mismatched_key_examples': diffs[:12],
            'missing_in_other': sorted(ka - kb)[:12], 'extra_in_other': sorted(kb - ka)[:12]}

class ModelAdapter:
    def __init__(self, config, device):
        self.device = torch.device(device)
        source = Path(config['xq_source']).resolve()
        sys.path.insert(0, str(source))
        from tokenizer.tokenizer_image import xqgan_model as xm
        from tokenizer.tokenizer_image.dino_enc import dinov2 as de
        import timm
        import dist
        from models.var import VAR
        from models.helpers import sample_with_top_k_top_p_
        self.sampler = sample_with_top_k_top_p_
        setattr(dist, '__device', str(self.device))
        generator_path = Path(config['xq_generator_checkpoint']).resolve()
        standalone_path = Path(config['xq_tokenizer_checkpoint']).resolve()
        ckpt = load_xq_checkpoint(generator_path)
        if 'trainer' not in ckpt: raise ValueError('Not the released paired VAR training checkpoint')
        trainer = ckpt['trainer']
        vs = trainer['vae_local']; gs = trainer['var_wo_ddp']
        if gs['head.weight'].shape != (8192, 1088) or gs['pos_1LC'].shape != (1, 286, 1088):
            raise ValueError('Checkpoint is NOT VAR-d17-MSVR10P2-4096 at the declared schedule')
        standalone = load_xq_checkpoint(standalone_path)
        candidates = {}
        for key in ('model', 'ema', 'state_dict'):
            if isinstance(standalone, dict) and key in standalone:
                try: candidates[key] = compare_states(vs, extract_state(standalone[key]))
                except ValueError: pass
        if not candidates:
            candidates['root'] = compare_states(vs, extract_state(standalone))
        # Always evaluate the actual generator-paired tokenizer used by official inference.py.
        # Standalone differences are disclosed, never swapped into an incompatible generator.
        args = xm.ModelArgs(codebook_size=4096, codebook_embed_dim=32,
            v_patch_nums=list(SCALES), enc_type='dinov2', dec_type='dinov2',
            semantic_guide='dinov2', detail_guide='none', num_latent_tokens=121,
            encoder_model='vit_base_patch14_dinov2.lvd142m',
            decoder_model='vit_base_patch14_dinov2.lvd142m', abs_pos_embed=True,
            share_quant_resi=4, product_quant=2, half_sem=True, test_model=True)
        original_factory = timm.create_model
        def build_without_redundant_download(name, *a, **kw):
            kw['pretrained'] = False
            return original_factory(name, *a, **kw)
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(xm, 'create_model', build_without_redundant_download))
            stack.enter_context(patch.object(de, 'create_model', build_without_redundant_download))
            # Only training semantic-loss constructor queries DDP rank; no DDP job is created.
            stack.enter_context(patch.object(torch.distributed, 'get_rank', return_value=0))
            stack.enter_context(patch.object(torch.distributed, 'get_world_size', return_value=1))
            self.vae = xm.VQModel(args)
        self.vae.load_state_dict(vs, strict=True)
        self.vae = self.vae.to(self.device).eval().requires_grad_(False)
        self.var = VAR(vae_local=self.vae, num_classes=1000, depth=17, embed_dim=1088,
            num_heads=17, drop_path_rate=0.1*17/24, norm_eps=1e-6,
            shared_aln=False, cond_drop_rate=0.0, attn_l2_norm=True,
            patch_nums=SCALES, flash_if_available=False, fused_if_available=False, p_drop=0.0)
        self.var.load_state_dict(gs, strict=True)
        self.var = self.var.to(self.device).eval().requires_grad_(False)
        assert len(self.vae.quantizes) == 2
        assert all(q.vocab_size == 4096 for q in self.vae.quantizes)
        self.metadata = {
            'model': 'xq', 'tokenizer': 'MSVR10P2-4096', 'generator': 'VAR-d17-MSVR10P2-4096',
            'source_commit': '137869c6e60b1c48edc2a33f543a28b565a10632',
            'tokenizer_state_source': 'generator_checkpoint.trainer.vae_local (official inference)',
            'standalone_vs_paired_tokenizer': candidates,
            'paired_tokenizer_matches_standalone': any(x['exactly_equal'] for x in candidates.values()),
            'scale_table': list(SCALES), 'product_quant': 2, 'full_raw_bits': 6864,
            'codebook_size_per_branch': 4096, 'generator_logit_dim': 8192,
            'tokenizer_parameters': sum(p.numel() for p in self.vae.parameters()),
            'generator_parameters': sum(p.numel() for p in self.var.parameters()),
            'strict_checkpoint_load': True, 'checkpoint_deserialization': 'exact SHA-bound fixed-global whitelist, custom Unpickler for ruamel config metadata; no unrestricted pickle', 'precision': 'float32, no autocast, TF32 disabled',
            'dino_training_exposure': 'DINOv2 encoder/decoder + semantic guide; not independent semantic evidence',
            'constructor_only_overrides': ['pretrained=False before complete strict checkpoint restore',
                'single-process rank=0,world_size=1 for unused semantic-loss constructor'],
            'primary_completion': 'conditional argmax, no CFG mixing',
            'sample_completion': 'official ramp t=cfg*stage/9, top_k=750, top_p=.95; no best-of selection',
        }
        self.last_audit = {}
        del ckpt, trainer, vs, gs, standalone, candidates

    @torch.inference_mode()
    def encode(self, x):
        tokens = self.vae.img_to_idxBl(x)
        token_metadata(tokens, 10)
        return tokens

    token_metadata = staticmethod(token_metadata)

    @torch.inference_mode()
    def decode(self, tokens, prefix_scales):
        m = int(prefix_scales); token_metadata(tokens, m)
        if m == 0: raise ValueError('No zero-rate quality arm registered')
        B = tokens[0][0].shape[0]
        fhat = torch.zeros(B, 64, 11, 11, device=self.device, dtype=torch.float32)
        for si in range(m):
            pn = SCALES[si]
            parts = [q.embedding(tokens[k][si]).transpose(1,2).reshape(B,32,pn,pn)
                     for k,q in enumerate(self.vae.quantizes)]
            fhat,_ = self.vae.get_next_autoregressive_input(si, 10, fhat, torch.cat(parts,dim=1))
        self.last_audit = {'prefix_preserved': True, 'prefix_scales': m,
            'quant_residual_denominator_total_scales': 10, 'suffix': 'zero residual, not token index zero'}
        return self.vae.fhat_to_img(fhat).add(1).mul(.5)

    @torch.inference_mode()
    def complete(self, tokens, labels, prefix_scales, mode='argmax', seed=0, cfg=1.0):
        m = int(prefix_scales)
        if any(len(branch) != m for branch in tokens):
            raise ValueError('Receiver must receive ONLY the prefix; true suffix is forbidden')
        token_metadata(tokens, m)
        labels = torch.as_tensor(labels, dtype=torch.long, device=self.device).reshape(-1)
        if int(labels.min()) < 0 or int(labels.max()) >= 1000: raise ValueError('Invalid ImageNet class')
        if m == 10:
            rgb = self.decode(tokens, 10)
            self.last_audit.update({'full_token_identity': True, 'input_hash': token_hash(tokens)})
            return rgb
        if mode not in ('argmax', 'sample'): raise ValueError(mode)
        if mode == 'argmax' and cfg != 1.0: raise ValueError('Registered argmax has no CFG mixing')
        B = labels.shape[0]; var = self.var
        sampling = mode == 'sample'
        lab = torch.cat((labels, torch.full_like(labels, 1000))) if sampling else labels
        sos = cond = var.class_emb(lab)
        cond_shared = var.shared_ada_lin(cond)
        level_pos = var.lvl_embed(var.lvl_1L) + var.pos_1LC
        next_map = sos.unsqueeze(1).expand(lab.shape[0],var.first_l,-1) + var.pos_start + level_pos[:,:var.first_l]
        fhat = sos.new_zeros(B,64,11,11)
        rng = torch.Generator(device=self.device).manual_seed(int(seed))
        generated = [[], []]; cur = 0
        try:
            for block in var.blocks: block.attn.kv_caching(True)
            for si, pn in enumerate(SCALES):
                cur += pn*pn
                x = next_map
                for block in var.blocks: x = block(x=x, cond_BD=cond_shared, attn_bias=None)
                logits = var.get_logits(x, cond)
                if sampling:
                    t = float(cfg) * si / 9
                    logits = (1+t)*logits[:B] - t*logits[B:]
                if si < m:
                    indices = [tokens[k][si] for k in range(2)]
                elif sampling:
                    indices = [self.sampler(z, rng=rng, top_k=750, top_p=.95, num_samples=1)[:,:,0]
                               for z in logits.chunk(2,dim=-1)]
                else:
                    indices = [z.argmax(dim=-1) for z in logits.chunk(2,dim=-1)]
                for k in range(2): generated[k].append(indices[k].clone())
                parts = [self.vae.quantizes[k].embedding(indices[k]).transpose(1,2).reshape(B,32,pn,pn) for k in range(2)]
                fhat,next_map = self.vae.get_next_autoregressive_input(si,10,fhat,torch.cat(parts,dim=1))
                if si < 9:
                    next_map = var.word_embed(next_map.view(B,64,-1).transpose(1,2)) + level_pos[:,cur:cur+SCALES[si+1]**2]
                    if sampling: next_map = next_map.repeat(2,1,1)
        finally:
            for block in var.blocks: block.attn.kv_caching(False)
        preserved = all(torch.equal(tokens[k][s],generated[k][s]) for k in range(2) for s in range(m))
        if not preserved: raise RuntimeError('Completion altered a transmitted true token')
        self.last_audit = {'prefix_preserved': preserved, 'input_prefix_scales': m,
            'input_hash': token_hash(tokens), 'used_prefix_hash': token_hash([g[:m] for g in generated]),
            'generated_token_hash': token_hash(generated), 'input_raw_bits': token_metadata(tokens,m)['raw_bits'],
            'received_true_suffix_indices': 0, 'mode': mode, 'seed': int(seed), 'cfg_argument': cfg,
            'guidance_convention': 'official stage-dependent extra guidance' if sampling else 'conditional only/no CFG',
            'top_k': 750 if sampling else None, 'top_p': .95 if sampling else None,
            'true_class_given': True, 'class_side_bits_if_sent': 10, 'best_of_selection': False}
        return self.vae.fhat_to_img(fhat).add(1).mul(.5)

    @torch.inference_mode()
    def smoke_checks(self, source, tokens, labels):
        """Roundtrip against the unmodified upstream tokenizer, not just self-consistency."""
        restored = self.decode(tokens,10)
        native = self.vae.img_to_reconstructed_img(source,last_one=True).add(1).mul(.5)
        err = float((restored-native).abs().max())
        if err > 2e-5: raise RuntimeError(f'Upstream token roundtrip mismatch: {err}')
        return {'upstream_full_reconstruction_max_abs_error': err, 'indices': token_metadata(tokens,10)}

def build_adapter(config, device):
    return ModelAdapter(config, device)

"""Explicit new timing resources around the frozen endpoint mathematics."""
import ast
import importlib.util
import sys
import time
import types
from pathlib import Path
from contract import R, B, T, checked, read, save


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def replace_function(module, name, substitutions):
    """Only enumerated resource substitutions; inverse AST must be identical."""
    text = Path(module.__file__).read_text()
    node = next(n for n in ast.parse(text).body
                if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)
    source = ast.get_source_segment(text, node)
    changed = source
    for old, new in substitutions:
        assert changed.count(old) == 1, old
        changed = changed.replace(old, new)
    inverse = changed
    for old, new in reversed(substitutions):
        assert inverse.count(new) == 1
        inverse = inverse.replace(new, old)
    assert ast.dump(ast.parse(source)) == ast.dump(ast.parse(inverse))
    ns = dict(vars(module))
    exec(compile(changed, module.__file__ + '::timing_resources', 'exec'), ns)
    return ns[name]


def paths():
    dirs = ['t1b-m10-calibration-20261016',
            'entropy-partial-mismatch-20261010/scripts',
            'wcl-evidence-closure-20261009/scripts',
            'baseline-delivery-20261012', 'baseline-strength-20261011',
            'submission-closure-20261016']
    sys.path[:0] = [str(R / 'experiments' / p) for p in dirs]
    sys.path.insert(0, str(R / 'src'))


class Calls:
    def __init__(self):
        self.counts = {}

    def call(self, kind, *args, **kwargs):
        if isinstance(kind, dict):
            kind = kind['kind']
        self.counts[kind] = self.counts.get(kind, 0) + 1
        return args[-1]()

    def decode(self, phase, key, kind, request, fn):
        return self.call(kind, fn)


def flags(torch):
    torch.set_num_threads(6)
    torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision('highest')


def visual(r, out):
    paths()
    import tb_native
    calls = Calls()
    # Resource-only CPU affinity. The original 12 GiB allocator and all numerical
    # flags/model hashes remain unchanged for the NeST/P visual stack.
    build = replace_function(tb_native, 'build', [
        ('len(os.sched_getaffinity(0))==2', 'len(os.sched_getaffinity(0))==6')])
    n, render = build(read(checked(r['model_routes'])), calls, lambda: None,
                      out, r['resource']['gpu_pci'])
    return n, render, calls


def continuous(r, out, method):
    paths()
    import torch
    import baseline_runtime as base
    from timing_endpoints import ContinuousEndpoint, SwinEndpoint
    if method == 'P100000':
        n, _, calls = visual(r, out)
        ns, _ = base.p_math()
        state = torch.load(checked(r['P_checkpoint']), map_location='cpu',
                           weights_only=True)['model']
        model = ns['BudgetContinuous'](state['scale'].flatten(), 1024)
        model.load_state_dict(state, strict=True)
        model = model.cuda().eval().requires_grad_(False)
        return ContinuousEndpoint(n, model, n.loaded['decoder'], ns['seeded_noise']), calls
    flags(torch)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction((12 << 30) / torch.cuda.get_device_properties(0).total_memory)
    sm, sp, sr, _ = base.swin_modules()
    sp._PHY.ROOT = B / 'baseline_strength_20261011/native_build'
    sp._PHY.load_native()
    model = sm.build_official(base.TMP / 'vendor/SwinJSCC')
    state = torch.load(checked(r['Swin_checkpoint']), map_location='cpu',
                       weights_only=True)['model']
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False)
    model.native.multiple_snr = list(range(1, 20))
    return SwinEndpoint(torch, model, sr, sp), Calls()


def nest_codec(r, out):
    n, render, calls = visual(r, out)
    from ep_source_codec import PartialSourceCodec
    from t6_source_codec import SourceCodec10
    from h800_ep48_link_core_v1 import bind_static
    class Codec(PartialSourceCodec):
        def decode(self, family, payload, m, K=0):
            if m == 10:
                assert family == 'EC_VAR_WHOLE' and K == 0
                return SourceCodec10.decode(self, family, payload, 10)
            return super().decode(family, payload, m, K)
    codec = Codec(R, native=n)
    bind_static(codec, r['static'])
    return n, codec, render, calls


def arpc(r, out, role):
    import argparse
    import numpy as np
    cfg = r['arpc_public']
    sys.path[:0] = [cfg['upstream'], str(R / 'experiments/arpc-external-baseline-20261019')]
    import torch
    flags(torch)
    torch.manual_seed(0)
    np.random.seed(0)
    torch.cuda.set_device(0)
    props = torch.cuda.get_device_properties(0)
    torch.cuda.set_per_process_memory_fraction(min(.8, (70 << 30) / props.total_memory))
    import compression.sender as sender
    import compression.util as util
    import compression.reciever as receiver
    from tools.run_infinity import load_tokenizer, load_visual_tokenizer, load_transformer
    from a1_native import prefix_decoder
    from wire import pack, unpack, SIDES
    args = argparse.Namespace(
        pn='0.06M', model_path=cfg['weights_root']+'/var_codec.pth', vae_type=16,
        vae_path=cfg['weights_root']+'/vae_d16_reg.ckpt', model_type='infinity_2b',
        cfg_insertion_layer=0, add_lvl_embeding_only_first_block=1, use_bit_label=1,
        rope2d_each_sa_layer=1, rope2d_normalized_by_hw=2, use_scale_schedule_embedding=0,
        sampling_per_bits=1, text_channels=2048, apply_spatial_patchify=0,
        use_flex_attn=0, cache_dir=str(out/'runtime_cache'), enable_model_cache=0,
        checkpoint_type='torch', seed=0, bf16=0)
    import infinity.models.bsq_vae.vae as vm
    old = vm.load_cnn
    audit = {}
    def strict(model, state, prefix, expand=False, use_linear=False):
        result = old(model, state, prefix, expand, use_linear)
        missing = sorted(set(dict(model.named_parameters())) - set(result[2]))
        audit['vae_missing'] = missing
        assert not missing
        return result
    vm.load_cnn = strict
    tokenizer, textencoder = load_tokenizer(cfg['weights_root']+'/flan-t5-xl')
    vae = load_visual_tokenizer(args)
    prior = load_transformer(vae, args)
    schedule = [(1, s, s) for s in SIDES]
    assert [(h, w) for _, h, w in util.dynamic_resolution_h_w[1.0]['0.06M']['scales']] == [(s, s) for s in SIDES]
    q_schedule = [(*p, int((i+1)//((len(schedule)//3)+1)+2)) for i, p in enumerate(schedule)]
    processor = blip = None
    if role == 'tx':
        from transformers import BlipProcessor, BlipForConditionalGeneration
        processor = BlipProcessor.from_pretrained(cfg['weights_root']+'/blip-image-captioning-base', local_files_only=True)
        blip = BlipForConditionalGeneration.from_pretrained(cfg['weights_root']+'/blip-image-captioning-base', local_files_only=True).eval().cuda().requires_grad_(False)
    decode, adapter_hash = prefix_decoder(cfg['upstream'])
    from utils.arithmeticcoding import compress_to_bit_list
    original = receiver.decompress_from_bit_list
    def canonical(bits, num, probabilities):
        v = original(bits, num, probabilities)
        if compress_to_bit_list(v, probabilities) != bits:
            raise ValueError('Actual RX noncanonical source block')
        return v
    decode.__globals__['decompress_from_bit_list'] = canonical
    save(out/'native_identity.json', dict(
        arguments=vars(args), prefix_adapter_sha256=adapter_hash, **audit,
        model_dtypes={k: sorted({str(p.dtype) for p in m.parameters()})
                      for k, m in [('vae', vae), ('prior', prior), ('text', textencoder)]},
        source_truth_argument=None, cfg=3, tau=.5, generation_seed=0))
    return types.SimpleNamespace(**locals())

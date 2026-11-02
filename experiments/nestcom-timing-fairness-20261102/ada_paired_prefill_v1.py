"""One compiled, causal prefill route with independent arithmetic reception.

TX computes conditional probabilities once. RX recomputes a padded causal
prefill: future inputs are public zeros, never source tokens. Equal tensor
shapes avoid the floating-point kernel changes found in the previous audit.
This is a new numerical implementation, not a bit-exact old-stream replay or
the authors' released executable. Cold compilation is recorded separately.
"""
from pathlib import Path
import os, sys, time, json, hashlib, traceback, tempfile
import numpy as np

R = Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
sys.path[:0] = [str(R/'experiments/nestcom-perceptual-followup-20261030'),
               str(R/'experiments/adatoken-response-20261026')]
from followup_common import read, save, sha, checked, cr, pin, asha, Ledger


class PairedPrefill:
    def __init__(self, native):
        self.n = native
        self.t = t = native.torch
        p = native.prior
        # No KV history is needed for full causal prefills. Caches would add
        # state and extra copies without avoiding any computation in this route.
        for layer in p.layers:
            layer.attention.kv_cache = None
        p.freqs_cis = p.freqs_cis.to('cuda:0')
        self.pos = t.arange(256, device='cuda:0', dtype=t.int64)
        self.mask = t.ones((1, 1, 256, 256), device='cuda:0', dtype=t.bool).tril()
        self.calls = {1: 0, 2: 0}
        def forward(idx, cls):
            return p(idx, cls, input_pos=self.pos, mask=self.mask)[0]
        self.forward = t.compile(forward, backend='inductor', fullgraph=True,
                                 dynamic=False, mode='default')

    def logits(self, known, cls, branches=1):
        assert 0 <= len(known) <= 255 and branches in (1, 2)
        # Only the actually known prefix is inserted. All suffix placeholders
        # are independent of the source and invisible to earlier causal rows.
        values = np.zeros((branches, 255), dtype=np.int64)
        values[:, :len(known)] = np.asarray(known, dtype=np.int64)
        idx = self.t.as_tensor(values, device='cuda:0')
        classes = self.t.tensor([cls] if branches == 1 else [cls, 1000],
                                dtype=self.t.int64, device='cuda:0')
        with self.t.nn.attention.sdpa_kernel(self.t.nn.attention.SDPBackend.MATH):
            out = self.forward(idx, classes)
        self.calls[branches] += 1
        return out

    def tables(self, tokens, cls, cdf):
        logits = self.logits(tokens[:255], cls, 1)
        # One transfer for all conditional probabilities; no 256 CUDA syncs.
        logp = self.t.log_softmax(logits[0], dim=-1).float().cpu().numpy()
        return [cdf(row[None, :]) for row in logp]

    def decode(self, bits, prefix, cls, Encoder, Decoder, cdf, trace=False):
        from c_arithmetic import canonical
        dec = Decoder(bits); enc = Encoder(); tokens = []; hashes = []
        for i in range(prefix):
            logits = self.logits(tokens, cls, 1)
            table = self.n.cdf(logits[:, i:i+1], cdf)
            value = dec.decode(table); enc.encode(value, table)
            tokens.append(int(value[0]))
            if trace: hashes.append(asha(table))
        canonical(dec, enc, bits)
        return np.asarray(tokens, np.int64), hashes

    def render(self, received, cls, seed):
        t = self.t; tokens = list(map(int, received))
        with t.random.fork_rng(devices=[0]):
            t.manual_seed(seed)
            for i in range(len(tokens), 256):
                logits = self.logits(tokens, cls, 2)
                value = self.n.tail(logits[:, i:i+1])
                tokens.append(int(value.item()))
        values = np.asarray(tokens, np.int64)
        return self.n.render(values), values


def encode(route, tokens, cls, Encoder, cdf):
    tables = route.tables(tokens, cls, cdf); enc = Encoder(); streams = {}
    for i, token in enumerate(tokens):
        enc.encode(np.array([token], np.int64), tables[i]); streams[i+1] = enc.finish()
    return streams, tables


def tx(pixels, source_id, index, ctx, route):
    import timing_end_to_end_identity_v5 as original
    from c_predclass_protocol import container
    n = ctx['n']; g = ctx['g']; rt = ctx['rt']; Encoder, _, cdf = ctx['c']
    x = pixels.astype(np.float32)/np.float32(255)
    cls = ctx['classifier'].predict(x); tokens = n.tokenize(x)
    streams, _ = encode(route, tokens, cls, Encoder, cdf)
    profiles = [p for p in rt.profiles.values()
                if (p['q'], p['nominal_rate']) == (g['policy']['q'], g['policy']['rate'])]
    cap = profiles[0]['k']-39
    fits = [m for m in range(256, 0, -1) if len(streams[m]) <= cap]
    if fits:
        m = fits[0]; bits = streams[m]
        p = next(p for p in profiles if p['family']=='ADA_PREDICTED_CLASS_PREFIX' and p['m']==m)
        payload = container(cls, bits)
    else:
        m = 0; bits = np.empty(0, np.uint8)
        p = next(p for p in profiles if p['family']=='EXPLICIT_NULL_FIXEDFRAME')
        payload = np.zeros(2, np.uint8)
    counter, public = original.counters(g, index, p['profile_id'])
    wave, wire = rt.transmit(p['profile_id'], payload, counter)
    return wave, dict(counter=counter, public_base=public, source_id=source_id, source_index=index), dict(
        endpoint=[m, 0], predicted_class=cls, bits=bits, tokens=tokens,
        received_tokens=tokens[:m], capacity=cap, wire=wire)


def rx(observed, meta, ctx, route, event):
    from n512_phy import received_projection
    from c_predclass_protocol import generation_seed
    rt = ctx['rt']; g = ctx['g']; Encoder, Decoder, cdf = ctx['c']
    actual = rt.receive(observed, g['snr_db'], meta['counter'], rt.profiles,
                        ctx['calls'], event, phase='PAIRED_PREFILL_TIMING')
    p = received_projection(actual)
    if p['status'] != 'SOURCE_INPUT':
        return np.full((3,256,256), .5, np.float32), dict(status=p['reason'], success=False)
    bits = np.asarray(p['bits'], np.uint8); m = p['m']; cls = p['received_class']
    # Only the wire-decoded class, prefix length and bits enter reconstruction.
    tokens, _ = route.decode(bits, m, cls, Encoder, Decoder, cdf)
    image, complete = route.render(tokens, cls, generation_seed(bits, m, cls))
    return image, dict(status='ACCEPTED_SOURCE', success=True, received_tokens=tokens,
                       completed_tokens=complete, endpoint=[m,0], received_class=cls)


def main(request, expected):
    assert sha(request) == expected
    r = read(request); [checked(p) for p in r['code']]
    out = Path(r['output']); out.mkdir(exist_ok=False)
    os.sched_setaffinity(0, {38,40}); begin = time.time()
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        TORCHINDUCTOR_CACHE_DIR=str(out/'inductor'), TORCHINDUCTOR_COMPILE_THREADS='1',
        MAX_JOBS='1')
    phase = r['phase']; rows = []
    budgets = dict(compile_prefill=2, codec_qualification=2, tail_qualification=2,
                   timing_trial=160 if phase in ('tx','rx') else 0)
    ledger = Ledger(out/'calls.sqlite', budgets)
    try:
        def guard():
            assert time.time() < r['deadline_unix'] and time.time()-begin < r['seconds'], 'BOUNDED_DEADLINE'
        if phase == 'qualification':
            from c_predclass_native import load_native
            from c_arithmetic import load_codec
            import timing_end_to_end_identity_v5 as original
            calls = original.MemoryCalls()
            n = load_native(r['native'], out, calls); c = load_codec()[:3]; ctx = None
        else:
            assert cr(r['qualification'])['status'] == 'PASS'
            from timing_dependency_v4 import activate
            activate(r['sionna_package'])
            import timing_end_to_end_identity_v5 as original
            cfg = cr(r['config']); sources = cr(r['sources']); assert len(sources) == 32
            if phase == 'rx':
                inputs = cr(r['received_symbols']); assert len(inputs) == 160
                denied = {str(Path(s['archive']['path']).resolve()) for s in sources}
                def firewall(event, args):
                    if event == 'open' and isinstance(args[0], (str,bytes,os.PathLike)):
                        p = str(Path(os.fsdecode(args[0])).resolve())
                        if p in denied or '/witnesses/' in p:
                            raise PermissionError('RX source/TX truth prohibited: '+p)
                sys.addaudithook(firewall)
            ctx = original.setup(cfg, r['group'], out, phase, r['pci'])
            n = ctx['n']; c = ctx['c']
        t = n.torch; t.set_num_threads(2)
        assert t.get_num_threads() == t.get_num_interop_threads() == 2
        assert str(t.cuda.get_device_properties(0).uuid) == r['CUDA_UUID']
        route = PairedPrefill(n); Encoder, Decoder, cdf = c
        setup_start = time.time()
        with t.inference_mode():
            for branches in [1,2]:
                guard()
                def compile_call():
                    values = route.logits([], 0, branches); t.cuda.synchronize()
                    assert values.shape == (branches,256,16384) and bool(t.isfinite(values).all())
                ledger.call('compile_prefill', str(branches), compile_call)
            save(out/'setup.json', dict(seconds=time.time()-setup_start, torch=t.__version__,
                compile='inductor/default/fullgraph/static, 2 registered batch shapes',
                precision='FP32/math/TF32off', cpu_affinity=[38,40], threads=2, interop=2,
                padded_context=256, future_inputs='public zeros; strict causal mask',
                cold_compilation_excluded=True, included_in_resource_budget=True))
            if phase == 'qualification':
                qual = []
                for s in r['qualification_sources']:
                    guard(); tokens = np.load(checked(s['tokens']), allow_pickle=False)
                    assert tokens.shape == (256,) and asha(tokens) == s['token_sha256']
                    cls = s['predicted_class']
                    def check_codec():
                        streams, tables = encode(route, tokens, cls, Encoder, cdf)
                        recovered, hashes = route.decode(streams[256],256,cls,Encoder,Decoder,cdf,trace=True)
                        assert np.array_equal(recovered,tokens), 'TOKEN_MISMATCH'
                        assert hashes == [asha(v) for v in tables], 'CDF_MISMATCH'
                        return dict(source_id=s['source_id'], tokens=256, matching_CDFs=256,
                                    independently_terminated_candidates=256, full_bits=len(streams[256]))
                    q = ledger.call('codec_qualification',s['source_id'],check_codec)
                    def check_tail():
                        # A fixed 128-token known prefix exercises the CFG branch.
                        image, complete = route.render(tokens[:128],cls,2026110202)
                        assert np.array_equal(complete[:128],tokens[:128])
                        assert image.shape == (3,256,256) and image.dtype == np.float32
                        assert np.isfinite(image).all() and image.min() >= 0 and image.max() <= 1
                        return dict(prefix=128, tail=128, image_sha256=asha(image), input_role='codec qualification, not wireless or quality')
                    q['tail'] = ledger.call('tail_qualification',s['source_id'],check_tail)
                    qual.append(q); save(out/('source_'+str(s['source_index'])+'.json'),q)
                save(out/'qualification.json',dict(status='PASS',sources=qual,scope='new paired route; not old bitstream equivalence'))
            else:
                observations=[]; hashes={}; (out/'witnesses').mkdir() if phase=='tx' else None
                if phase=='tx': (out/'received_symbols').mkdir()
                for si,s in enumerate(sources):
                    if phase=='tx':
                        with np.load(checked(s['archive']),allow_pickle=False) as z: pixels=z['pixels'].copy()
                        assert hashlib.sha256(pixels.tobytes()).hexdigest() == s['preprocessing_id']
                    for repetition in range(5):
                        guard(); event=f'{si:02d}_{repetition}'
                        if phase=='rx':
                            inp=inputs[si*5+repetition]; assert inp['event']==event and inp['source_id']==s['source_id']
                            with np.load(checked(inp['symbols']),allow_pickle=False)as z: observed=z['observed'].copy()
                            assert asha(observed)==inp['observation_sha256']
                        def trial():
                            t.cuda.synchronize(); start=time.perf_counter()
                            value=tx(pixels,s['source_id'],s['source_index'],ctx,route) if phase=='tx' else rx(observed,inp['metadata'],ctx,route,event)
                            t.cuda.synchronize(); return value,time.perf_counter()-start
                        value,seconds=ledger.call('timing_trial',event,trial)
                        if phase=='tx':
                            wave,meta,witness=value; assert wave.shape==(1024,2) and np.isfinite(wave).all()
                            fingerprint=asha(wave); observed=original.noise(wave,meta,ctx)
                            path=out/'received_symbols'/(event+'.npz')
                            with path.open('xb') as f: np.savez(f,observed=observed)
                            observations.append(dict(event=event,source_id=s['source_id'],symbols=pin(path),observation_sha256=asha(observed),metadata=meta))
                            # TX verification remains outside timers and inaccessible to RX.
                            proof={k:v for k,v in witness.items() if k not in ('bits','tokens','received_tokens','wire')}
                            proof.update(bits_sha256=asha(witness['bits']),token_sha256=asha(witness['tokens']),
                                         received_tokens_sha256=asha(witness['received_tokens']),bits=len(witness['bits']),wave_sha256=fingerprint)
                            save(out/'witnesses'/(event+'.json'),proof)
                            extra=dict(endpoint=proof['endpoint'],predicted_class=proof['predicted_class'],bits=proof['bits'])
                        else:
                            image,proof=value; assert image.shape==(3,256,256) and image.dtype==np.float32 and np.isfinite(image).all()
                            fingerprint=asha(image)
                            extra={k:v for k,v in proof.items() if k not in ('received_tokens','completed_tokens')}
                            for key in ('received_tokens','completed_tokens'):
                                if key in proof: extra[key+'_sha256']=asha(proof[key])
                        if si in hashes: assert hashes[si]==fingerprint, 'NONDETERMINISTIC_OUTPUT'
                        else: hashes[si]=fingerprint
                        row=dict(source_id=s['source_id'],source_index=s['source_index'],event=event,repetition=repetition,
                                 phase='warmup'if repetition<2 else'measured',direction=phase,seconds=seconds,
                                 output_sha256=fingerprint,**extra)
                        rows.append(row); save(out/(event+'.json'),row)
                    print('PAIRED_PREFILL',phase,si+1,32,flush=True)
                save(out/'rows.json',rows)
                if phase=='tx': save(out/'received_symbols.json',observations)
        save(out/'completion.json',dict(status='COMPLETE',phase=phase,rows=len(rows),ledger=ledger.summary(),
             prior_prefills=route.calls,elapsed=time.time()-begin,training=0,quality_metrics=0,
             numerical_identity='new compiled causal-prefill route; original quality results immutable'))
    except BaseException:
        save(out/'failure.json',dict(traceback=traceback.format_exc(),ledger=ledger.summary(),rows=len(rows),automatic_retry=False))
        raise


if __name__=='__main__': main(*sys.argv[1:])

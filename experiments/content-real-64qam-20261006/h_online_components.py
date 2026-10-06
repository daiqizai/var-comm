"""Prepared H online component benchmark; no CLI, model loader, or PHY decoder.

The execution owner must first seal a dedicated exclusive-GPU registration.
Only original source pixels and actual already-received wire events are used.
This module never constructs a ledger/backend or spends packet quota.
"""
from __future__ import annotations
from collections import defaultdict
from contextlib import nullcontext, closing
import copy
import hashlib
import json
from pathlib import Path
import sqlite3
import statistics
import time
import numpy as np

FIXED = (0,25,50,75,4,21,24,29,33,41,52,60,64,87,92,95)
WARMUP = 1
REPEATS = 3
NOISE = 6201


def require(value, message):
    if not value:
        raise RuntimeError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def design(protocol, budget, population, schedule):
    """A bounded proposed execution interface, not an execution registration."""
    require(protocol['schema'] == 'H_CODEC_PROTOCOL_V1' and protocol['status'] == 'FROZEN_BEFORE_DATA',
            'Frozen H protocol required')
    require(protocol['timing']['preparation_timing_is_online'] is False
            and 'Fixed16' in protocol['timing']['online'], 'Registered fixed16 timing scope required')
    require(budget['branch'] == 'H' and budget['total_cap'] == 200000
            and budget['no_automatic_reserve_spending'] is True
            and 'timing' not in budget['phase_limits'], 'Original nontransferable packet budget required')
    require(population['stage'] == population['calibration_or_development'] == 'm1_development'
            and len(population['source_ids']) == len(set(population['source_ids'])) == 100,
            'Original development100 required')
    require(len(schedule) == 18 and {r['development_slot'] for r in schedule} == set(range(18))
            and all(r['phase'] == 'development' and r['status'] == 'FROZEN_POLICY_READY' for r in schedule),
            'Exactly the frozen eighteen ready H points; no policy search')
    return dict(status='PREPARED_REQUIRES_EXCLUSIVE_EXECUTION_REGISTRATION',
        source_indices=list(FIXED), source_ids=[population['source_ids'][i] for i in FIXED],
        schedule_sha256=digest(schedule), slots=list(range(18)), noise_seed=NOISE,
        warmup_repetitions=WARMUP, measured_repetitions=REPEATS, batch_size=1,
        component_cases=16*18, warmup_and_measured_calls=16*18*(WARMUP+REPEATS),
        new_packet_decodes=0, no_ledger_writer=True, reserve_spending=False,
        PHY_decode='Historical ledger event windows only; concurrent2x2 CPU and bookkeeping included, not isolated kernel latency',
        PHY_encode='NOT_MEASURED in this no-backend component benchmark',
        end_to_end_latency='NOT_MEASURED; do not add historical PHY windows to isolated GPU components',
        TX='fresh pixels->Encoder/VQ->registered payload choice, fresh probability history for each attempted m; raw packing included',
        RX='actual saved RX event->fresh independent canonical source recovery->VAR/Dc; no source tokens/labels or output memo',
        same_frame_probability_cache=False, cross_frame_cache=False,
        fallback='Every attempted prefix is actually encoded again in this conservative uncached implementation; no hypothetical optimized-latency claim',
        timing_rule_source='UEP receiver_cost timing_samples: guard then synchronize, one warmup and three measured repeats; fingerprints unchanged',
        warmup_excluded_from_summary=True, failed_frames_included=True,
        source_or_policy_selection=False, new_neural_metrics=False, automatic_launch=False)


class Meter:
    """Inclusive synchronized regions; atomic components and outer totals stay separate."""
    def __init__(self, synchronize, clock=time.perf_counter):
        self.synchronize, self.clock = synchronize, clock
        self.seconds, self.counts = defaultdict(float), defaultdict(int)

    def call(self, name, function):
        self.synchronize(); started = self.clock()
        try:
            return function()
        finally:
            self.synchronize(); elapsed = self.clock()-started
            require(np.isfinite(elapsed) and elapsed >= 0, 'Invalid synchronized component duration')
            self.seconds[name] += float(elapsed); self.counts[name] += 1


class Providers:
    """Fresh original IndependentProvider instances; preserve its measured counters."""
    def __init__(self, factory, meter, side):
        self.factory, self.meter, self.side = factory, meter, side
        self.instances, self.records = [], []

    def __call__(self):
        owner = self
        instance = self.factory()
        require(all(instance is not old for old in self.instances), 'A probability provider was reused')
        self.instances.append(instance)
        class Fresh:
            def __enter__(self):
                self.provider = owner.meter.call(owner.side+'_probability_setup', instance.__enter__)
                self.scales = 0
                return self
            def cdf(self):
                return self.provider.cdf()
            def advance(self, values):
                result = self.provider.advance(values); self.scales += 1
                return result
            def __exit__(self, *args):
                try:
                    return owner.meter.call(owner.side+'_probability_teardown', lambda:instance.__exit__(*args))
                finally:
                    model, cdf = self.provider.model_seconds, self.provider.cdf_seconds
                    require(np.isfinite(model) and model >= 0 and np.isfinite(cdf) and cdf >= 0,
                            'Original probability/CDF timing counters unavailable')
                    owner.records.append(dict(model_seconds=float(model), cdf_seconds=float(cdf),
                                              advanced_scales=self.scales))
        return Fresh()

    def summary(self):
        require(len(self.instances) == len(self.records), 'Probability context was not closed')
        return dict(provider_calls=len(self.records), advanced_scales=sum(r['advanced_scales'] for r in self.records),
                    probability_seconds=sum(r['model_seconds'] for r in self.records),
                    cdf_seconds=sum(r['cdf_seconds'] for r in self.records))


def measured_primitives(original, codec, meter, side):
    """Wrap the original integer methods without changing their state/bit operations."""
    class Encoder(original.encoder_type):
        def encode(self, *args, **kwargs):
            return meter.call(side+'_integer_encode', lambda:super(Encoder,self).encode(*args, **kwargs))
        def finish(self, *args, **kwargs):
            return meter.call(side+'_integer_finish', lambda:super(Encoder,self).finish(*args, **kwargs))
    class Decoder(original.decoder_type):
        def decode(self, *args, **kwargs):
            return meter.call(side+'_integer_decode', lambda:super(Decoder,self).decode(*args, **kwargs))
    return codec.ArithmeticPrimitives(Encoder, Decoder, original.cdf_function, original.validate_cdf)


class FreshArithmetic:
    """Lazy mapping used by the unmodified fallback selector; no stored source bits."""
    def __init__(self, scales, codec, providers, primitives):
        self.scales, self.codec, self.providers, self.primitives = scales, codec, providers, primitives
        self.requested = []
    def __contains__(self, m):
        return type(m) is int and m in (6,7,8,9)
    def __getitem__(self, m):
        require(m in self and m not in self.requested, 'A prefix stream was requested twice or outside m6..m9')
        self.requested.append(m)
        return self.codec.encode_prefixes(self.scales, self.providers, self.primitives, (m,))[m]['bits']


def native_encoder(native, pixels):
    """Exact original development100_source_driver Encoder/VQ expression, no cached tokens."""
    require(pixels.dtype == np.uint8 and pixels.shape == (3,256,256), 'Original uint8 RGB required')
    t, loaded = native.torch, native.loaded
    image = t.as_tensor(pixels[None], dtype=t.float32, device=loaded['device'])/127.5-1
    f = loaded['vae'].quant_conv(loaded['vae'].encoder(image))
    tokens = t.cat(loaded['vae'].quantize.f_to_idxBl_or_fhat(f,to_fhat=False),1)[0].cpu().numpy()
    require(tokens.shape == (680,) and np.issubdtype(tokens.dtype,np.integer)
            and np.all((tokens >= 0) & (tokens < 4096)), 'Original Encoder/VQ returned invalid tokens')
    return np.ascontiguousarray(tokens,dtype=np.int64)


def tx_once(pixels, entry, ctx, *, encoder, source_api, development_core, codec, primitives,
            provider_factory, synchronize, clock=time.perf_counter):
    """No channel/backend object is accepted by this TX source component."""
    meter = Meter(synchronize, clock); providers = Providers(provider_factory,meter,'TX')
    timed = measured_primitives(primitives,codec,meter,'TX')
    flat = meter.call('TX_visual_encoding', lambda:encoder(pixels.copy()))
    scales = source_api.split_tokens(flat)
    streams = FreshArithmetic(scales,codec,providers,timed)
    tx = meter.call('TX_source_total', lambda:development_core.select_payload(ctx,entry,scales,streams))
    require(tx['profile']['profile_id'] in {p['profile_id'] for p in ctx['catalogue']['profiles']},
            'Online payload choice outside frozen public catalogue')
    pstats = providers.summary()
    # Header/LDPC encoding is deliberately absent: no resident compatible PHY
    # backend has been admitted by this no-new-decoder component interface.
    return dict(tx=tx, tokens=flat.copy(), seconds=dict(meter.seconds), component_calls=dict(meter.counts),
                probability=pstats, attempted_arithmetic_prefixes=list(streams.requested),
                fingerprint=digest(dict(profile=tx['profile']['profile_key'], payload=development_core.phy.array_sha(tx['payload']),
                                        tokens=development_core.phy.array_sha(flat), attempts=tx['attempts'])),
                new_packet_decodes=0, cross_frame_cache=False)


def rx_once(view, catalogue, *, receiver_api, codec, primitives, provider_factory, render_tokens,
            synchronize, inference_context=nullcontext, clock=time.perf_counter):
    """Only the narrow RX view enters this function; no TX/source identifiers."""
    require(set(view) == {'rx','recorded_rx_profile'}, 'Only actual RX view is accepted')
    meter = Meter(synchronize, clock); providers = Providers(provider_factory,meter,'RX')
    timed = measured_primitives(primitives,codec,meter,'RX')
    receiver = receiver_api.Receiver(catalogue, primitives=timed, provider_factory=providers,
        render_tokens=lambda tokens,m,K:meter.call('RX_suffix_VAR_and_Dc', lambda:render_tokens(tokens,m,K)),
        inference_context=inference_context)
    reconstruction = meter.call('RX_source_and_image_total', lambda:receiver.reconstruct(copy.deepcopy(view)))
    summary = reconstruction['summary']
    return dict(image=reconstruction['image'], receiver_summary=summary, seconds=dict(meter.seconds),
                component_calls=dict(meter.counts), probability=providers.summary(),
                fingerprint=digest(dict(image_sha256=summary['image_sha256'],
                                        tokens=summary['actual_received_tokens_sha256'],
                                        status=summary['source_status'])),
                new_packet_decodes=0, final_output_cache=False)


def repetitions(call, *, synchronize, guard):
    """Original UEP one-warmup/three-repeat rule; no statistical sample inflation."""
    fingerprints = set(); warmups=[]; measured=[]
    for phase, count in (('warmup',WARMUP),('measured',REPEATS)):
        for repetition in range(count):
            guard(); synchronize(); value = call(); synchronize()
            require(value['new_packet_decodes'] == 0, 'Component benchmark cannot spend packet quota')
            fingerprints.add(value['fingerprint'])
            require(len(fingerprints) == 1, 'Repeated uncached computation changed output; diagnose without retry')
            item = {k:copy.deepcopy(v) for k,v in value.items() if k not in ('image','tokens','tx')}
            item.update(phase=phase,repetition=repetition)
            (warmups if phase == 'warmup' else measured).append(item)
    return dict(status='MEASURED_UNCACHED_COMPONENTS',warmups=warmups,measured=measured,
                measured_repetitions=3,warmup_repetitions=1,new_packet_decodes=0)


def summarize_repetitions(record):
    require(record['status'] == 'MEASURED_UNCACHED_COMPONENTS' and len(record['warmups']) == 1
            and len(record['measured']) == 3, 'Complete original timing repetitions required')
    samples=record['measured']; names=set().union(*(r['seconds'] for r in samples))
    return {name:dict(seconds=[r['seconds'].get(name,0.) for r in samples],
                     mean_seconds=statistics.mean(r['seconds'].get(name,0.) for r in samples),
                     median_seconds=statistics.median(r['seconds'].get(name,0.) for r in samples)) for name in sorted(names)}


def verify_tx_against_trace(result, trace, phy):
    """Evaluation-only parity after the timed TX computation; never drives choice."""
    tx, expected = result['tx'], trace['tx']; p=tx['profile']
    require(p['profile_id'] == expected['profile_id'] and p['profile_key'] == expected['profile_key']
            and tx['target_m'] == expected['target_m'] and tx['actual_m'] == expected['actual_m']
            and p['K'] == expected['K'] and p['mode'] == expected['mode']
            and tx['fell_back'] == expected['fell_back'] and tx['attempts'] == expected['attempts']
            and phy.array_sha(tx['payload']) == expected['payload_sha256'],
            'Fresh online TX differs from actual frozen transmitted payload/choice')
    return True


def historical_phy_windows(ledger_path, events):
    """Read completed event wall-clock intervals; never invoke BudgetLedger.

    These include SQLite reservation/completion bookkeeping and CPU contention.
    They are neither exclusive PHY kernel latency nor end-to-end frame time.
    """
    path=Path(ledger_path).absolute(); require(path.is_file(), 'Existing quiescent ledger required')
    rows=[]
    with closing(sqlite3.connect(path.as_uri()+'?mode=ro',uri=True)) as db:
        for event_id, expected in sorted(events.items()):
            found=db.execute('SELECT phase,kind,status,reserved_at,completed_at,result,result_sha,worker FROM events WHERE event_id=?',
                             (event_id,)).fetchall()
            require(len(found) == 1, 'Actual completed event missing/duplicated')
            phase,kind,status,start,end,result,checksum,worker=found[0]
            require(phase == 'development' and kind == expected['kind'] and status == 'COMPLETE'
                    and checksum == expected['result_sha256'] == digest(json.loads(result)),
                    'Ledger event differs from actual trace')
            require(isinstance(start,(float,int)) and isinstance(end,(float,int))
                    and np.isfinite(start) and np.isfinite(end) and end >= start, 'Missing/invalid historical timestamps')
            rows.append(dict(event_id=event_id,kind=kind,wall_seconds=float(end-start),worker=json.loads(worker),
                             timing_scope='CONCURRENT_CPU_EVENT_WINDOW_WITH_BOOKKEEPING',
                             exclusive_PHY_latency=False, callback_only=False, result_sha256=checksum))
    return dict(status='HISTORICAL_PHY_EVENT_WINDOWS_READ_ONLY',rows=rows,new_packet_decodes=0,
                standalone_PHY_measured=False,compatible_for_end_to_end_sum=False)


def native_components(native, primitives, catalogue, source_api, development_core, receiver_api, codec):
    """Injection adapter only. Model and code qualification belongs to the caller."""
    def provider():
        return source_api.IndependentProvider(native,primitives)
    def tx(pixels,entry,ctx):
        with native.torch.no_grad():
            return tx_once(pixels,entry,ctx,encoder=lambda x:native_encoder(native,x),
                source_api=source_api,development_core=development_core,codec=codec,primitives=primitives,
                provider_factory=provider,synchronize=native.torch.cuda.synchronize)
    def rx(view):
        return rx_once(view,catalogue,receiver_api=receiver_api,codec=codec,primitives=primitives,
            provider_factory=provider,render_tokens=lambda tokens,m,K:source_api.render_received(native,tokens,m,K),
            synchronize=native.torch.cuda.synchronize,inference_context=native.torch.no_grad)
    return dict(tx=tx,rx=rx)

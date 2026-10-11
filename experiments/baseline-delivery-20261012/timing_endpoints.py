"""Fresh online TX/RX paths for the frozen six-method timing experiment.

Dependencies are injected only after the separate owner's binding and resource
admission. Importing this module does not construct a model or a PHY decoder.
No source scores or cached streams are accepted by these endpoints.
"""
import hashlib
import time
import numpy as np


def array_sha(value):
    a = np.ascontiguousarray(value)
    return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()


class Meter:
    def __init__(self, synchronize):
        self.synchronize = synchronize
        self.seconds = {}

    def call(self, name, callback):
        self.synchronize(); started = time.perf_counter()
        try:
            return callback()
        finally:
            self.synchronize()
            elapsed = time.perf_counter()-started
            assert np.isfinite(elapsed) and elapsed >= 0
            self.seconds[name] = self.seconds.get(name, 0.)+elapsed


class PacketCharges:
    """Original PHY callback signature; charge actual calls in memory only."""
    def __init__(self, reservations):
        self.reservations = reservations
        self.events = []

    def call(self, event, request, callback):
        self.events.append(dict(event=event, request=request))
        return self.reservations.packet_call(callback)


def fresh_tokens(native, pixels):
    t = native.torch; loaded = native.loaded
    assert pixels.dtype == np.uint8 and pixels.shape == (3, 256, 256)
    with t.no_grad():
        x = t.as_tensor(pixels[None], dtype=t.float32, device=loaded['device'])/127.5-1
        f = loaded['vae'].quant_conv(loaded['vae'].encoder(x))
        a = t.cat(loaded['vae'].quantize.f_to_idxBl_or_fhat(f, to_fhat=False), 1)[0].cpu().numpy()
    assert a.shape == (680,) and np.issubdtype(a.dtype, np.integer)
    assert ((a >= 0) & (a < 4096)).all()
    return a.astype(np.int64, copy=False)


def fresh_entropy_choice(codec, plan, profiles, tokens, point, partial_allowed):
    """First fitting real terminated stream; every fallback attempt is online."""
    assert partial_allowed or point['target_K'] == 0
    attempts = []
    for m, K in plan.fallback_endpoints(point['target_m'], point['target_K']):
        family = 'EC_VAR_PARTIAL' if K else 'EC_VAR_WHOLE'
        matches = [p for p in profiles.values() if
                   (p['family'], p['m'], p['K'], p['q'], p['nominal_rate']) ==
                   (family, m, K, point['q'], point['nominal_rate'])]
        assert len(matches) == 1
        profile = matches[0]
        assert profile['source_capacity_bits'] == point['source_capacity_bits']
        # No slicing of a terminated stream and no reuse of a larger attempt's
        # probability state. Include every failed longer attempt in TX time.
        stream = (codec.encode_endpoints(tokens, [(m, K)])[(m, K)] if K else
                  codec.encode('EC_VAR_WHOLE', tokens, (m,))[m])
        bits = stream['bits']
        assert bits.dtype == np.uint8 and bits.ndim == 1 and len(bits) >= 2
        assert np.isin(bits, (0, 1)).all() and stream['arithmetic_bits'] == len(bits)
        fits = len(bits) <= profile['source_capacity_bits']
        attempts.append(dict(m=m, K=K, arithmetic_bits=len(bits), fits=fits))
        if fits:
            return profile, bits, attempts
    raise RuntimeError('No actual source stream fits; never fabricate a channel failure')


class EntropyEndpoint:
    def __init__(self, native, codec, runtime, plan, render, points, partial_allowed,
                 charges, standard_noise, invalid_source_stream):
        assert runtime.qualified
        assert len(runtime.profiles) == (360 if partial_allowed else 144)
        assert codec.native is native
        self.native, self.codec, self.runtime, self.plan = native, codec, runtime, plan
        self.render, self.points = render, points
        self.partial_allowed, self.charges = partial_allowed, charges
        self.standard_noise = standard_noise
        self.invalid_source_stream = invalid_source_stream
        self.t = native.torch

    def encode(self, pixels, snr, counter, meter):
        with self.t.no_grad():
            tokens = meter.call('visual_encoder_VQ', lambda: fresh_tokens(self.native, pixels))
            p, bits, attempts = meter.call('fresh_entropy_and_fallback', lambda:
                fresh_entropy_choice(self.codec, self.plan, self.runtime.profiles,
                                     tokens, self.points[str(snr)], self.partial_allowed))
            wave, tx = meter.call('paid_header_LDPC_modulation', lambda:
                                 self.runtime.transmit(p['profile_id'], bits, counter))
        return wave, dict(tokens=tokens, transmission=tx, attempts=attempts)

    def noise(self, wave, source_id, snr):
        return wave+self.standard_noise(source_id, 6201)*10**(-float(snr)/20)

    def receive(self, observed, snr, counter, meter, event_id):
        # Deliberately no TX metadata, tokens, target policy or stream arguments.
        with self.t.no_grad():
            actual = meter.call('header_LDPC_and_parse', lambda:
                self.runtime.receive(observed, snr, counter, self.runtime.profiles,
                                     self.charges, event_id, phase='timing'))
            body = actual['body']; p = actual['rx_profile']
            if (not actual['header']['header_ok'] or body is None or
                    not body['crc_accepted'] or not body['parser_accepted']):
                return np.full((3,256,256), .5, dtype=np.float32), actual
            assert p is not None
            try:
                decoded = meter.call('independent_entropy_decode', lambda:
                    self.codec.decode(p['family'], np.asarray(body['payload'], dtype=np.uint8), p['m'], p['K']))
            except self.invalid_source_stream as error:
                return np.full((3,256,256), .5, dtype=np.float32), dict(actual,
                    status='SOURCE_PARSE_REJECT', source_parse_error=str(error))
            image = meter.call('VAR_and_Dc', lambda:
                self.render(self.native, decoded['received_tokens'], p['m'], p['K']))
        return image, dict(actual, status='ARITHMETIC_SOURCE_DECODED')


class RawEndpoint:
    def __init__(self, native, runtime, render, points, prefix_offsets, noise):
        assert len(runtime.cat.header_map()) == 433
        self.native, self.runtime, self.render = native, runtime, render
        self.points, self.offsets, self.standard_noise = points, prefix_offsets, noise
        self.t = native.torch

    def encode(self, pixels, snr, counter, meter):
        runtime = self.runtime
        with self.t.no_grad():
            tokens = meter.call('visual_encoder_VQ', lambda: fresh_tokens(self.native, pixels))
            p = runtime.cat.entry(self.points[str(snr)]['profile_id'])
            scales = [tokens[self.offsets[i]:self.offsets[i+1]] for i in range(10)]
            payload = meter.call('raw12_serialization', lambda: runtime.original.serialize_raw(scales, p))
            wave, tx = meter.call('paid_header_LDPC_modulation', lambda:
                runtime.rx.transmit_frame(runtime.cat, runtime.backend, runtime.legacy,
                                         runtime.header, p['profile_id'], payload, counter))
        return wave, dict(tokens=tokens, transmission=tx, attempts=[])

    def noise(self, wave, source_id, snr):
        return wave+self.standard_noise(source_id, snr, 6201)*10**(-float(snr)/20)

    def receive(self, observed, snr, counter, meter, event_id):
        runtime = self.runtime
        actual = meter.call('header_LDPC_KEEP', lambda:
            runtime.receiver.receive(observed, float(snr), counter,
                                     phase='raw_holdout', event_prefix=event_id))
        actual = dict(actual, status=actual['source_status'])
        parsed = runtime.original.present_actual(actual['header'], actual['body'], runtime.cat)
        assert all(actual[k] == v for k, v in parsed.items())
        state = parsed['receiver_state']
        if state['kind'] == 'gray':
            return np.full((3,256,256), .5, dtype=np.float32), actual
        parts = [np.asarray(v, dtype=np.int64) for v in state['prefix']]
        tokens = np.concatenate(parts+[np.asarray(state['partial_values'], dtype=np.int64)])
        assert tokens.shape == (self.offsets[state['m']]+state['K'],)
        with self.t.no_grad():
            image = meter.call('VAR_and_Dc', lambda:
                self.render(self.native, tokens, state['m'], state['K']))
        return image, actual


class ContinuousEndpoint:
    def __init__(self, native, model, dc, seeded_noise):
        self.native, self.model, self.dc, self.seeded_noise = native, model, dc, seeded_noise
        self.t = native.torch

    def encode(self, pixels, snr, counter, meter):
        t = self.t; vae = self.native.loaded['vae']
        def encode():
            image = t.as_tensor(pixels[None], dtype=t.float32, device='cuda:0')/127.5-1
            return vae.quant_conv(vae.encoder(image))
        with t.no_grad():
            latent = meter.call('visual_encoder', encode)
            wave = meter.call('continuous_JSCC_encode', lambda: self.model.transmit(latent)[0].cpu().numpy())
        return wave, dict(tokens=None, transmission={}, attempts=[])

    def noise(self, wave, source_id, snr):
        noise = self.seeded_noise('VAR-CONTINUOUS-1024|'+source_id, 2001, (1024,2)).astype(np.float32)
        return wave+noise*np.float32(10**(-float(snr)/20))

    def receive(self, observed, snr, counter, meter, event_id):
        t = self.t
        with t.no_grad():
            latent = meter.call('continuous_JSCC_receive', lambda: self.model.receive(
                t.as_tensor(observed[None], device='cuda:0'),
                t.tensor([snr], dtype=t.float32, device='cuda:0')))
            image = meter.call('Dc', lambda: self.dc(latent)[0].cpu().numpy())
        return image, dict(status='CONTINUOUS_DECODED', header=dict(header_ok=True), body=None)


class SwinEndpoint:
    def __init__(self, torch, model, replay, protocol):
        self.t, self.model, self.replay, self.protocol = torch, model, replay, protocol

    def encode(self, pixels, snr, counter, meter):
        with self.t.no_grad():
            image = self.t.as_tensor(pixels[None], device='cuda:0', dtype=self.t.float32)/255
            wave = meter.call('Swin_encoder_and_paid_side_information', lambda:
                             self.replay.transmit(self.model, image, 1024, snr))
        return wave, dict(tokens=None, transmission={}, attempts=[])

    def noise(self, wave, source_id, snr):
        return wave+self.protocol.standard_noise(source_id, 2001, 1024, snr)/(10**(float(snr)/20))

    def receive(self, observed, snr, counter, meter, event_id):
        with self.t.no_grad():
            image, context = meter.call('paid_header_and_Swin_decoder', lambda:
                                       self.replay.receive(self.model, observed, 1024, snr))
        return image, dict(status='SWIN_DECODED' if context.accepted else 'HEADER_REJECT_GRAY',
                           header=dict(header_ok=context.accepted, crc_accepted=context.crc_accepted,
                                       fields_legal=context.fields_legal), body=None)


def fresh_case(endpoint, pixels, case, source_id, counter, reservations):
    reservations.begin(case)
    tx = Meter(endpoint.t.cuda.synchronize); rx = Meter(endpoint.t.cuda.synchronize)
    endpoint.t.cuda.synchronize(); start = time.perf_counter()
    wave, witness = tx.call('TX_total', lambda: endpoint.encode(pixels, case['snr_db'], counter, tx))
    channel_start = time.perf_counter()
    observed = endpoint.noise(wave, source_id, case['snr_db'])
    channel_time = time.perf_counter()-channel_start
    image, actual = rx.call('RX_total', lambda: endpoint.receive(
        observed, case['snr_db'], counter, rx, str(len(reservations.state['cases'])-1)))
    endpoint.t.cuda.synchronize(); outer = time.perf_counter()-start
    assert wave.shape == observed.shape == (1024,2) and np.isfinite(wave).all() and np.isfinite(observed).all()
    assert image.dtype == np.float32 and image.shape == (3,256,256)
    assert np.isfinite(image).all() and ((image >= 0) & (image <= 1)).all()
    assert outer+1e-6 >= channel_time+tx.seconds['TX_total']+rx.seconds['RX_total']
    identity = dict(fresh_visual_encoder=True, cached_tokens_or_streams=False, batch_size=1,
                    wave_sha256=array_sha(wave), observation_sha256=array_sha(observed),
                    reconstruction_sha256=array_sha(image), source_id=source_id,
                    status=actual['status'], header_accepted=actual['header']['header_ok'],
                    body_crc_accepted=(None if actual['body'] is None else
                        actual['body'].get('crc_accepted',actual['body'].get('crc_accept'))),
                    actual_energy=float(np.square(wave.astype(np.float64)).sum()),
                    source_attempts=witness['attempts'])
    if witness['tokens'] is not None:
        identity['fresh_tokens_sha256'] = hashlib.sha256(b'int64:680\0'+witness['tokens'].astype('<i8').tobytes()).hexdigest()
    timings = dict(tx_seconds=tx.seconds['TX_total'], rx_seconds=rx.seconds['RX_total'],
                   channel_seconds=channel_time, outer_seconds=outer,
                   TX_components=tx.seconds, RX_components=rx.seconds)
    reservations.finish(identity, timings)
    return identity, timings

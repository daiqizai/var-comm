"""A3 fresh development timing, reusing qualified RAW primitives and weights.

This is a new request, not a continuation or qualification of the old R2/R3
request. No historical scientific ledger is opened and no qualification packets
are sent. The prior completed R3 evidence is authenticated by the runner.
"""
from copy import deepcopy
from pathlib import Path
import sys
import numpy as np


class RawEndpoint:
    def __init__(self, r, baseline, bindings, calls, helper, method):
        self.r, self.calls, self.method = r, calls, method
        require, read, sha = helper.require, helper.read, helper.sha
        load, descriptor = helper.load, helper.descriptor
        private = read(r['private_sionna_manifest'])
        require(private['status'] == 'ACTUAL_ORIGINAL_SIONNA_PURE_PACKAGE_PRIVATE_COPY_VERIFIED_V1'
                and private['source'] == r['sionna_source']
                and private['private_parent'] == r['private_sionna_parent'], 'Existing private package required')
        for relative, row in private['entries'].items():
            require(row['source'] == str(Path(r['sionna_source']) / relative)
                    and row['copy'] == str(Path(r['private_sionna_parent']) / 'sionna' / relative)
                    and sha(row['source']) == row['sha256'] == sha(row['copy']), 'Private Sionna changed')
        require('sionna' not in sys.modules, 'One private Sionna namespace per process')
        sys.path.insert(0, r['private_sionna_parent'])
        self.native = helper.build_native(r, baseline, bindings)
        self.t = self.native.torch
        cfg = descriptor(r['raw_config']); a = cfg['adapter_config']
        self.planmod = load(a['plan_module'], 'main_raw64_plan_only', bindings)
        self.packet = load(cfg['adapter_module'], 'main_raw64_packet_adapter', bindings)
        load(a['original_common'], 'uep_common', bindings)
        load(a['original_planner'], 'profiles', bindings)
        self.legacy = load(a['original_phy'], 'uep_phy', bindings)
        oldmodule = load(a['original_backend'], '_timing_original_sionna_backend', bindings)
        old = oldmodule.SionnaBackend('cpu', a['legacy_qualification'])
        require(old.qualified and str(old.device) == 'cpu', 'Qualified original CPU backend identity required')
        self.backend = self.packet.Backend(old)
        self.header = self.legacy.Header(r['root'])
        self.original = load(cfg['original_receiver_module'], '_timing_original_raw_wire', bindings)
        self.rxmod = load(cfg['receiver_module'], '_timing_original_raw64_wire', bindings)
        self.catalogue = self.rxmod.MixedPublicCatalogue(read(a['catalogue']), read(cfg['profiles']), read(cfg['aliases']),
            legacy_backend_identity=self.backend.identity, original_receiver=self.original)
        self.noise_module = load(r['development_plan_module'], '_a3_original_development_noise', bindings)
        self.timing = load(r['timing_module'], '_timing_original_encoder_meter', bindings)
        self.source = load(r['source_driver'], '_timing_original_h_render', bindings)
        self.pair = load(r['pair_module'], '_timing_original_same_rx_renderer', bindings)
        self.renderer = self.pair.NativePairBackend(self.native, self.source.render_received, baseline['P_native_runtime'])
        def charge(event, request, callback):
            return calls.invoke('RAW_' + event['kind'], callback)
        self.receiver = self.rxmod.Receiver(self.catalogue, backend=self.backend, legacy_phy=self.legacy,
            header=self.header, charge=charge)
        self.last_actual, self.gray = None, False

    def point(self, snr):
        return self.r['points'][self.method][str(snr)]

    def encode(self, item, meter):
        tokens = meter.call('TX_Encoder_VQ', lambda: self.timing.native_encoder(self.native, item['pixels'].copy()))
        self.last_tokens = tokens
        point = self.point(self.snr); profile = self.catalogue.entry(point['profile_id'])
        payload = meter.call('TX_raw12', lambda: self.original.serialize_raw(self.source.split_tokens(tokens), profile))
        self.counter = self.noise_module.frame_counter(point['development_slot'], item['record']['source_index'], 6201)
        wave, _ = meter.call('TX_FEC_header_modulation', lambda: self.rxmod.transmit_frame(
            self.catalogue, self.backend, self.legacy, self.header, profile['profile_id'], payload, self.counter))
        return wave

    def receive(self, observed, snr, meter):
        actual = meter.call('RX_PHY_KEEP', lambda: self.receiver.receive(observed, float(snr), self.counter,
            phase='online_PHY', event_prefix=self.calls.case))
        self.last_actual, self.gray = actual, actual['gray']
        state = self.pair.received_view(actual)
        if self.gray:
            return np.full((3, 256, 256), .5, np.float32)
        decoder = self.native.loaded['decoder']; original = decoder.forward
        decoder.forward = lambda *a, **k: meter.call('RX_Dc', lambda: original(*a, **k))
        try:
            return meter.call('RX_VAR_completion_Dc', lambda: self.renderer.var(deepcopy(state)))
        finally:
            decoder.forward = original

    def noise(self, wave, item, snr):
        return wave + self.noise_module.standard_noise(item['record']['source_id'], snr, 6201) * 10 ** (-float(snr) / 20)

    def finish(self):
        self.native.frozen()

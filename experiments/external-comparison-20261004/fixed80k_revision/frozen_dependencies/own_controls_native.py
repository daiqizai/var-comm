"""Private aliases for unchanged frozen VAR primitives; no old main is called."""
from __future__ import annotations
from pathlib import Path
import os
import signal
import sys
import numpy as np
import own_controls_common as c


class Native:
    def __init__(self, root, signal_handler):
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        self.root = Path(root).resolve(); self.paths = c.locations(root); self.loaded = None
        sys.path.insert(0, str(self.root/'src'))
        replay_path = self.root/'experiments/unified-metrics-20261002/replay.py'
        registered = c.read(self.root/'outputs/UNIFIED-METRICS-20261002/supervisor_registration.json')
        c.require(registered['source_bindings'].get(str(replay_path)) == c.sha(replay_path), 'Frozen replay loader changed')
        self.replay = c.load_module('_own_controls_original_replay', replay_path)
        load = self.replay.load_file_module
        old = self.root/'experiments/extreme-bandwidth-20261001-N1024'
        self.assets = load('_own_controls_assets', old/'assets.py')
        oldphy = load('_own_controls_old_phy', old/'digital_protocol.py')
        digital = load('_own_controls_digital', old/'digital.py', {'assets':self.assets, 'digital_protocol':oldphy})
        self.common = load('_own_controls_original_common', self.paths['source']/'common.py',
                           {'assets':self.assets, 'digital':digital})
        self.phy = c.load_module('own_controls_phy', c.HERE/'own_controls_phy.py')
        self.receiver = load('_own_controls_receiver', self.paths['source']/'partial_receiver.py',
                             {'partial_phy':self.phy})
        self.m1 = load('_own_controls_m1', self.paths['source']/'m1_runner.py',
                       {'common':self.common, 'partial_phy':self.phy, 'partial_receiver':self.receiver})
        # The old primitive import installs its own stop handler. Restore this
        # isolated owner's handler before any model or population is loaded.
        signal.signal(signal.SIGTERM, signal_handler); signal.signal(signal.SIGINT, signal_handler)
        import torch
        self.torch = torch
        self.loaded = self.common.setup()
        original = c.read(self.paths['original_out']/'m1_calibration_registration.json')
        c.require(self.loaded['identity'] == original['identity'], 'Frozen Encoder/VAR/F/Dc/quality identities differ')
        c.require(str(self.loaded['device']) == 'cuda:0' and not torch.is_autocast_enabled()
                  and not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
                  and torch.are_deterministic_algorithms_enabled(), 'Original FP32 deterministic VAR backend required')
        self.flags = dict(torch=torch.__version__, cuda=torch.version.cuda, device=torch.cuda.get_device_name(0),
            matmul_tf32=torch.backends.cuda.matmul.allow_tf32, cudnn_tf32=torch.backends.cudnn.allow_tf32,
            deterministic=torch.are_deterministic_algorithms_enabled(), threads=torch.get_num_threads(),
            interop_threads=torch.get_num_interop_threads(), cudnn_benchmark=torch.backends.cudnn.benchmark,
            cudnn_deterministic=torch.backends.cudnn.deterministic, precision=torch.get_float32_matmul_precision())
        self._data = {}; self.proofs = {}; self.engine = None

    def data(self, role):
        c.require(role in ('calibration', 'development'), 'No training/holdout data access is available')
        if role not in self._data:
            data = self.common.data(role, self.loaded)
            registration, registration_bindings = c.original_registration(self.root, role)
            self.proofs[role] = c.data_proof(data, registration, role)
            self.proofs[role]['published_registration_bindings'] = registration_bindings
            self._data[role] = data
        return self._data[role]

    def actions(self):
        actions = self.phy.action_grid(2048, 'QPSK', orders=('entropy',), include_whole=True)
        c.require(len(actions) == 21 and sum(a.q == 0 for a in actions) == 5, 'Registered action grid differs')
        return actions

    def action(self, description):
        return self.phy.Action(**description)

    def score(self, data, index, action, snr, seed, memo, images=False):
        self.common.check()
        with self.torch.no_grad():
            return self.m1.score_one(self.loaded, data, index, action, snr, seed,
                                     memo['tx'], memo['render'], memo['metric'], images=images)

    @staticmethod
    def memo():
        return dict(tx=dict(waves={}, logits={}), render={}, metric={})

    def frozen(self):
        self.common.assert_frozen(self.loaded)

    def qualify(self):
        data = self.data('calibration'); proofs = []
        with self.torch.no_grad():
            for index in range(4):
                scales = self.common.split_tokens(data['T'][index].numpy()); logits = {}
                for action in self.actions():
                    self.common.check()
                    positions = self.m1.tx_positions(self.loaded, scales, action, logits)
                    wave, ledger = self.phy.transmit(scales, action, positions)
                    event = self.phy.receive(wave, 60, 2048, 'QPSK')
                    c.require(event['header_ok'] and event['prefix_crc_ok'] and event['body_crc_ok']
                              and event['action'] == action and ledger['E'] == 4096,
                              'Actual native N2048 paid PHY roundtrip failed')
                    c.require(all(np.array_equal(a,b) for a,b in zip(event['prefix'], scales[:action.m])),
                              'Noiseless prefix tokens differ')
                    result = self.receiver.complete_partial(self.loaded['vae'], self.loaded['var'], event, self.loaded['device'])
                    if action.q:
                        diagnostics = result['diagnostics']
                        c.require(diagnostics['known_tokens_fixed'] and diagnostics['partial_used']
                                  and np.array_equal(diagnostics['known_positions'], positions)
                                  and np.array_equal(diagnostics['known_values'], scales[action.m][positions]),
                                  'Independent TX/RX entropy order or clamping differs')
                    else:
                        old = self.common.legacy.complete_latent(self.loaded['vae'], self.loaded['var'],
                            event['prefix'], 1000, self.loaded['device']).float().cpu()
                        c.require(self.torch.equal(result['fhat'], old), 'q0 changed the original unconditional completion')
                    proofs.append(dict(source_index=index, source_id=data['records'][index]['image_id'],
                        action_id=c.action_id(action), E=ledger['E'], native_crc_roundtrip=True,
                        original_q0_exact=not bool(action.q), entropy_tx_rx_match=bool(action.q)))
        self.frozen()
        return dict(status='OWN_CONTROLS_NATIVE_QUALIFICATION_PASS', synthetic=False, engineering_only=True,
                    source_role='original_calibration', sources=4, cases=len(proofs), checks=proofs,
                    identity=self.loaded['identity'], numerical_runtime=self.flags,
                    training_updates=0, development_read=False, holdout_access=False)

    def replay_engine(self):
        if self.engine is None:
            # Keep the published scalar-field alias recovery exactly as used by
            # completed unified metrics. It changes no pixel or parity tolerance.
            compat_path = self.root/'outputs/METRIC-CONCURRENT-R4-20261002/runtime/replay_compat.py'
            source_publication = c.read(self.root/'outputs/METRIC-CONCURRENT-R4-20261002/extension_source_publication.json')
            c.published(source_publication)
            admitted = {**source_publication.get('runtime_source_bindings', {}),
                        **source_publication.get('source_bindings', {}), **source_publication.get('inputs', {})}
            c.require(admitted.get(str(compat_path)) == c.sha(compat_path), 'Published replay alias source is not bound')
            compat = c.load_module('_own_controls_legacy_alias', compat_path)
            self.compatibility = compat.install(self.replay, self.root)
            self.engine = self.replay.create_engine(self.root, studies=['N1024', 'M1'],
                loaded=self.loaded, data=self.data('development'))
        return self.engine

"""Independent H adapter to pinned Sionna; original backend stays unchanged."""
from __future__ import annotations
from functools import lru_cache
import hashlib
import inspect
from pathlib import Path
from h64_catalog import digest, require

DECODER = dict(cn_update='boxplus-phi', vn_update='sum', cn_schedule='flooding',
               num_iter=20, llr_max=20.0, hard_out=True, return_infobits=True,
               prune_pcm=True, harq_mode=False, precision='single', early_stop=False)
BIT_MAPPING = 'sequential_I_then_Q_bits;Gray_inverse_binary_PAM;16_axis[-3,-1,3,1]/sqrt5;64_axis[-7,-5,-1,-3,7,5,1,3]/sqrt21'
SCRAMBLING = 'H20261006_public_PCG64_SHA256_protocol_session_frame_group_after_rate_matching_v1'
POWER = 'fixed_constellation_average_Es2_no_frame_normalization'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class UnsupportedConfiguration(ValueError):
    """Actual backend constructor or RV0 layout rejects an enumerated resource."""


class H64Backend:
    def __init__(self, device='cpu', reference_qualification=None, legacy_backend_path=None):
        require(device == 'cpu', 'H packet decoding is CPU-only')
        import sionna
        import torch
        from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
        require(sionna.__version__ == '2.2.0', 'Pinned Sionna 2.2.0 required; do not upgrade')
        self.torch, self.Encoder, self.Decoder, self.device = torch, LDPC5GEncoder, LDPC5GDecoder, device
        self.identity = dict(implementation='Sionna_5G_LDPC_H64_ADAPTER_V1', version=sionna.__version__,
            encoder_source_sha256=sha(inspect.getfile(LDPC5GEncoder)),
            decoder_source_sha256=sha(inspect.getfile(LDPC5GDecoder)),
            decoder=DECODER, bit_mapping=BIT_MAPPING, scrambling=SCRAMBLING,
            power_protocol=POWER, precision='float32', cuda=torch.version.cuda, torch=torch.__version__,
            rate_matching='Sionna RV0; filler removal, first2Z and tail puncturing; supported n only')
        if legacy_backend_path:
            self.identity['legacy_adapter_source_sha256'] = sha(legacy_backend_path)
        if reference_qualification is not None:
            # This validates unchanged implementation facts, not new 64QAM qualification.
            old = reference_qualification['backend_identity']
            require(reference_qualification['status'] == 'PASS', 'Old backend qualification not PASS')
            for key in ('version', 'encoder_source_sha256', 'decoder_source_sha256', 'decoder',
                        'precision', 'cuda', 'torch', 'rate_matching'):
                require(old[key] == self.identity[key], 'Pinned backend core differs: '+key)
        self.qualified = False  # Only a new external H qualification may grant admission to science.

    @lru_cache(maxsize=8)
    def plan(self, k, n, num_bits_per_symbol):
        k, n, q = int(k), int(n), int(num_bits_per_symbol)
        if q not in (4, 6) or n % q or k < 29 or n < k:
            raise UnsupportedConfiguration('H single-codeblock dimensions/modulation invalid')
        try:
            e = self.Encoder(k, n, num_bits_per_symbol=q, device='cpu', precision='single')
        except (ValueError, AssertionError) as exc:
            raise UnsupportedConfiguration(type(exc).__name__+': '+str(exc)) from exc
        require(e.k == k and e.n == n and e.k_ldpc-e.k_filler == k, 'Encoder information dimensions differ')
        if n > e.n_cb_comp:
            raise UnsupportedConfiguration('Repetition outside pinned RV0 encoder is unsupported')
        interleaver = e.out_int.cpu().numpy()
        require(sorted(interleaver.tolist()) == list(range(n)), 'Output interleaver is not a permutation')
        layout = dict(k=k, n=n, q=q, k_ldpc=int(e.k_ldpc), k_filler=int(e.k_filler),
            n_cb=int(e.n_cb), n_cb_comp=int(e.n_cb_comp), bg=str(e._bg), z=int(e.z), code_blocks=1,
            mother_bits=int(e.n_ldpc), puncturing_bits=int(e.n_ldpc-e.k_filler-n),
            puncturing_first_2Z=int(2*e.z), puncturing_remaining=int(e.n_cb_comp-n),
            shortening_bits=int(e.k_filler), repetition_bits=0, modulation_padding_bits=0,
            interleaver_sha256=hashlib.sha256(interleaver.tobytes()).hexdigest(),
            decoder_config=DECODER, bit_mapping=BIT_MAPPING, scrambling=SCRAMBLING,
            power_protocol=POWER, actual_effective_rate=k/n, implementation=self.identity)
        layout['layout_id'] = digest(layout)
        return layout

    @lru_cache(maxsize=8)
    def codecs(self, k, n, q):
        self.plan(k, n, q)
        e = self.Encoder(int(k), int(n), num_bits_per_symbol=int(q), precision='single', device=self.device)
        d = self.Decoder(e, **{key: value for key, value in DECODER.items() if key != 'early_stop'}, device=self.device)
        e.eval(); d.eval()
        return e, d

    def encode(self, bits, n, q):
        t = self.torch
        bits = t.as_tensor(bits, dtype=t.float32, device=self.device)
        with t.inference_mode():
            return self.codecs(bits.shape[-1], n, q)[0](bits)

    def decode(self, logits, k, n, q):
        t = self.torch
        logits = t.as_tensor(logits, dtype=t.float32, device=self.device)
        with t.inference_mode():
            return self.codecs(k, n, q)[1](logits)

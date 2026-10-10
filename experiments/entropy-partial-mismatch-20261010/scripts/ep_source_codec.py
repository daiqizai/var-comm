"""Source-only EC_VAR_PARTIAL extension; no launch, PHY, model load or search.

Keep the original whole-source codec unchanged. The paid PHY must deliver the
actual family, m and K to decode(); target policy and source truth are not RX
arguments. Engineering tests with a fake CDF are not real-model qualification.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

LEGACY_SCRIPTS = Path(__file__).resolve().parents[2] / 'wcl-evidence-closure-20261009/scripts'
if str(LEGACY_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(LEGACY_SCRIPTS))
from t1_codec_runtime import SourceCodec, InvalidSourceStream, MODEL_STATES
from t1_entropy_core import OFFSETS, SIZES, CODE_SHA, require, source_tokens

PARTIAL_FAMILY = 'EC_VAR_PARTIAL'
WHOLE_FAMILY = 'EC_VAR_WHOLE'
SOURCE_SYNTAX = 'EC_VAR_PARTIAL_RASTER_INTEGER24_ARITHMETIC32_V1'
PROVIDER_SHA256 = 'b38f8451ca2d94a8ceb4381df3647e327dba38e3d4cf969aa15597bc47c342ce'


def quarter_counts(m):
    require(type(m) is int and 4 <= m <= 9, 'Registered complete prefix m4..m9 required')
    length = SIZES[m] ** 2
    return tuple(length * numerator // 4 for numerator in (1, 2, 3))


def endpoint(m, K):
    require(type(m) is int and type(K) is int and 4 <= m <= 9,
            'Integer registered m4..m9 endpoint required')
    require(K == 0 or K in quarter_counts(m), 'K must be zero or a registered floored quarter')
    return m, K


def token_count(m, K):
    endpoint(m, K)
    return int(OFFSETS[m]) + K


def all_endpoints():
    return tuple((m, K) for m in range(4, 10) for K in (0, *quarter_counts(m)))


def fallback_endpoints(target_m, target_K):
    """Whole policies retain their exact old whole-only fallback permissions."""
    endpoint(target_m, target_K)
    if target_K == 0:
        return tuple((m, 0) for m in range(target_m, 3, -1))
    ceiling = token_count(target_m, target_K)
    return tuple(reversed([x for x in all_endpoints() if token_count(*x) <= ceiling]))


def choose_encoded_length(lengths, target_m, target_K, source_capacity):
    """No model, quality, receiver state, raw substitution or PHY qualification."""
    require(type(source_capacity) is int and 2 <= source_capacity <= 8191,
            'Paid length-field source capacity required')
    attempts = []
    for m, K in fallback_endpoints(target_m, target_K):
        if (m, K) not in lengths:
            return dict(status='BLOCKED_MISSING_PREFIX', actual_m=None, actual_K=None,
                        missing_endpoint=[m, K], attempts=attempts)
        length = lengths[(m, K)]
        require(type(length) is int and length >= 2, 'Actual complete arithmetic stream length required')
        fits = length <= source_capacity
        attempts.append(dict(m=m, K=K, arithmetic_bits=length, fits=fits))
        if fits:
            return dict(status='SOURCE_LENGTH_FITS_LAYOUT_PENDING', actual_m=m, actual_K=K,
                        arithmetic_bits=length, missing_endpoint=None, attempts=attempts)
    return dict(status='TX_UNENCODABLE', actual_m=None, actual_K=None,
                missing_endpoint=None, attempts=attempts)


def partial_source_model_id():
    identity = dict(syntax=SOURCE_SYNTAX, models=MODEL_STATES, null_class=1000,
                    cdf_total=1 << 24, entropy_precision=32, ordering='raster', cfg=False,
                    integer_source_sha256=CODE_SHA, independent_provider_sha256=PROVIDER_SHA256,
                    partial_cdf='next_scale_full_integer_CDF_first_K_rows_given_received_complete_prefix')
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


class PartialSourceCodec(SourceCodec):
    """Uses the original checked native/CDF construction, with fresh RX contexts.

    Constructing with native=None loads only original NumPy arithmetic helpers.
    A real VAR encode/decode still requires the original bind_native() admission.
    Whole wire profiles remain EC_VAR_WHOLE (or legacy EC_STATIC_WHOLE), K=0.
    New partial wire profiles have EC_VAR_PARTIAL and one registered positive K.
    """

    def encode_endpoints(self, tokens, endpoints):
        """One fresh TX probability pass; finish is non-mutating, never a slice
        of an old terminated whole stream. No prepared source cache is consulted.
        This pass can produce whole endpoints for exact original-bit regression.
        """
        tokens = source_tokens(tokens)
        requested = tuple(endpoint(*p) for p in endpoints)
        require(requested and len(set(requested)) == len(requested), 'Nonempty unique endpoints required')
        requested = set(requested)
        last = max(m + int(K > 0) for m, K in requested)
        encoder = self.Encoder()
        result = {}

        def flush(m, K):
            bits = np.asarray(encoder.finish(), dtype=np.uint8)
            result[(m, K)] = dict(bits=bits, raw_bits=12 * token_count(m, K),
                                 arithmetic_bits=len(bits), flush_bits=len(bits) - len(encoder.bits),
                                 m=m, K=K, order='raster',
                                 source_family=PARTIAL_FAMILY if K else WHOLE_FAMILY)

        with self._inference(WHOLE_FAMILY), self._context(WHOLE_FAMILY) as provider:
            for scale in range(last):
                values = tokens[OFFSETS[scale]:OFFSETS[scale + 1]]
                cdf = self._cdf(WHOLE_FAMILY, provider, scale)
                start = 0
                for K in sorted(K for m, K in requested if m == scale and K > 0):
                    encoder.encode(values[start:K], cdf[start:K])
                    flush(scale, K)
                    start = K
                if scale + 1 < last or (scale + 1, 0) in requested:
                    encoder.encode(values[start:], cdf[start:])
                    provider.advance(values.copy())
                    if (scale + 1, 0) in requested:
                        flush(scale + 1, 0)
                # An unfinished final scale must not enter provider.advance().
        require(set(result) == requested, 'Every requested source endpoint must be encoded')
        return result

    def decode(self, family, payload_bits, m, K=0):
        """Only actually received wire fields and payload enter source recovery.

        Independent probability state advances with recovered complete scales.
        No truth-based correction/rejection, transmitted CDF or target policy.
        """
        endpoint(m, K)
        if family != PARTIAL_FAMILY:
            require(K == 0, 'Legacy whole source profile cannot declare partial tokens')
            return super().decode(family, payload_bits, m)
        require(K > 0, 'Partial wire family requires a paid positive K')
        a = np.asarray(payload_bits)
        if a.ndim != 1 or len(a) < 2 or not np.isin(a, (0, 1)).all():
            raise InvalidSourceStream('Invalid arithmetic bit vector')
        a = a.astype(np.uint8, copy=True)

        class Bounded(self.Decoder):
            def read_bit(inner):
                if inner.position >= len(inner.bits) + 30:
                    raise InvalidSourceStream('Arithmetic terminal lookahead exceeded')
                return super().read_bit()

        decoder, canonical, recovered = Bounded(a), self.Encoder(), []
        with self._inference(WHOLE_FAMILY), self._context(WHOLE_FAMILY) as provider:
            for scale in range(m + 1):
                table = self._cdf(WHOLE_FAMILY, provider, scale)
                cdf = table if scale < m else table[:K]
                try:
                    values = decoder.decode(cdf)
                except InvalidSourceStream:
                    raise
                except ValueError as error:
                    raise InvalidSourceStream(str(error)) from error
                canonical.encode(values, cdf)
                recovered.append(values.copy())
                if scale < m:
                    provider.advance(values.copy())
        if not np.array_equal(canonical.finish(), a):
            raise InvalidSourceStream('Noncanonical arithmetic termination or length')
        if decoder.position - len(a) != 30:
            raise InvalidSourceStream('Unexpected terminal lookahead')
        return dict(received_tokens=np.concatenate(recovered).astype(np.int64, copy=False),
                    source_status='ARITHMETIC_SOURCE_DECODED', canonical=True, zero_extension_reads=30,
                    transmitted_source_bits=len(a), source_family=family, m=m, K=K, order='raster',
                    receiver_truth_used=False, transmitter_probability_table_used=False,
                    same_scale_partial_conditioning=False)

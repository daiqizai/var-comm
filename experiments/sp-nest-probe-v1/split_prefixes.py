"""LE conditional coding, with an independently terminated (10,0) endpoint.

`codec` is the pinned LE/EP source codec. Its provider always starts from a fresh
VAR context. At RX it is advanced using recovered coarse tokens only. No logits,
source identifier or TX table is an argument of either receiver entry point.
"""
from contextlib import contextmanager
import numpy as np
from profiles import OFF, QUARTER, count, capacity

FAMILY = 'EC_VAR_WHOLE'
BASES = (4, 5, 6, 7)


class SourceReject(ValueError):
    """Malformed or noncanonical received stream, not a numerical/model exception."""


class SplitPrefixes:
    def __init__(self, codec):
        self.codec = codec

    @contextmanager
    def context(self):
        try:
            with self.codec._inference(FAMILY), self.codec._context(FAMILY) as p:
                yield p
        except SourceReject:
            raise
        except ValueError as error:
            # A failed model/CDF/context is an execution failure, never a gray frame.
            raise RuntimeError('source codec or probability provider failure') from error

    def encode(self, tokens):
        tokens = np.asarray(tokens)
        if tokens.shape != (OFF[-1],) or tokens.dtype.kind not in 'iu' or (tokens < 0).any() or (tokens >= 4096).any():
            raise ValueError('expected the registered 680 integral TX tokens')
        single, enc = self.codec.Encoder(), {}
        whole_streams, base, enhancement = {}, {}, {}

        def flush(end):
            whole_streams[end] = single.finish()
            if end[1] == 0 and end[0] in BASES:
                base[end[0]] = whole_streams[end]
            for j, coder in enc.items():
                if count(end) > OFF[j]:
                    enhancement[(j, *end)] = coder.finish()

        with self.context() as p:
            for s in range(10):
                vals = tokens[OFF[s]:OFF[s+1]]
                table = self.codec._cdf(FAMILY, p, s)
                lo = 0
                for hi in sorted({k for m,k in QUARTER if m == s and k} | {len(vals)}):
                    for coder in (single, *enc.values()):
                        coder.encode(vals[lo:hi], table[lo:hi])
                    end = (s, hi) if hi < len(vals) else (s + 1, 0)
                    if end in QUARTER:
                        flush(end)
                    lo = hi
                if s < 9:
                    p.advance(vals.copy())
                if s + 1 in BASES:
                    enc[s + 1] = self.codec.Encoder()
        if (len(whole_streams), len(base), len(enhancement)) != (25, 4, 72):
            raise RuntimeError('incomplete endpoint grid')
        return whole_streams, base, enhancement

    def _read(self, p, start, end, bits):
        b = np.asarray(bits)
        if b.ndim != 1 or not 2 <= len(b) <= 8191 or not np.isin(b, [0,1]).all():
            raise SourceReject('invalid paid bit vector')
        b = b.astype(np.uint8, copy=True)
        class Bounded(self.codec.Decoder):
            def read_bit(inner):
                if inner.position >= len(b) + 30:
                    raise SourceReject('terminal lookahead exceeded')
                return super().read_bit()
        decoder, canonical = Bounded(b), self.codec.Encoder()
        m,k = end
        parts = []
        for s in range(start, m + int(k > 0)):
            # Keep model errors outside the source rejection handler.
            cdf = self.codec._cdf(FAMILY, p, s)
            if s == m:
                cdf = cdf[:k]
            try:
                q = decoder.decode(cdf)
            except ValueError as error:
                raise SourceReject('arithmetic decoding failed') from error
            canonical.encode(q, cdf)
            parts.append(q)
            if s < m and s < 9:
                p.advance(q.copy())
        if decoder.position != len(b) + 30 or not np.array_equal(canonical.finish(), b):
            raise SourceReject('noncanonical independently terminated stream')
        return np.concatenate(parts).astype(np.int64, copy=False)

    def decode_base(self, j, bits):
        if j not in BASES:
            raise RuntimeError('invalid public base endpoint')
        with self.context() as p:
            return self._read(p, 0, (j, 0), bits)

    def decode_enhancement(self, j, end, recovered_base, bits):
        if j not in BASES or end not in QUARTER or count(end) <= OFF[j]:
            raise RuntimeError('invalid public enhancement endpoint')
        base = np.asarray(recovered_base)
        if base.shape != (OFF[j],) or base.dtype.kind not in 'iu' or (base < 0).any() or (base >= 4096).any():
            raise RuntimeError('invalid retained RX context')
        with self.context() as p:
            for s in range(j):
                # Some native providers prepare the next state in cdf(). Match
                # the historical RX transition, even when replaying verified B.
                self.codec._cdf(FAMILY, p, s)
                p.advance(base[OFF[s]:OFF[s+1]].copy())
            return self._read(p, j, end, bits)


def select_profile(base, enhancement, jcap, target, br, qe, er, ap, grid, inverse):
    """TX base-first fitting from actual bytes. None denotes a paid NULL frame.

    `inverse` maps the prepublished non-NULL profile tuples to their header ID.
    Missing stream keys raise: absence of an archive is not a channel failure.
    """
    if jcap not in (5,6,7) or grid not in ('QUARTER','WHOLE'):
        raise ValueError('unregistered policy')
    ends = [e for e in QUARTER if grid == 'QUARTER' or e[1] == 0]
    if target not in ends or target[0] < 6 or count(target) <= OFF[jcap]:
        raise ValueError('unregistered target endpoint')
    cb, ce = capacity(2,br), capacity(qe,er)
    for j in range(jcap,3,-1):
        if len(base[j]) > cb:
            continue
        eligible = [e for e in ends if OFF[j] < count(e) <= count(target)]
        for e in reversed(eligible):
            bits = enhancement[(j,*e)]
            if len(bits) <= ce:
                return inverse[(j,*e,br,qe,er,ap)], base[j], bits
        return inverse[(j,j,0,br,qe,er,ap)], base[j], np.empty(0,np.uint8)
    return None

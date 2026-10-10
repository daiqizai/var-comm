"""Source-only entropy helpers; no model import, training or PHY execution."""
from __future__ import annotations
import ast
import csv
import hashlib
import importlib.util
import json
from fractions import Fraction
from pathlib import Path
import numpy as np

SIZES = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
OFFSETS = np.cumsum((0,) + tuple(x*x for x in SIZES))
CODE_SHA = {
    'entropy.py': 'f3adf956fba05d60e687472fa153e0bf684119ef6c8885a75fe4ea21962b2b2b',
    'whole_entropy.py': '5d08d2f8c218f0bdf5b6d2a8ca7b3aa729baa75e274531afbfd8f31c95b24a5b',
}

def require(ok, message):
    if not ok:
        raise ValueError(message)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def write(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')

def csv_write(path, rows):
    require(rows, 'No rows to write')
    with Path(path).open('x', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

def verify(path, expected):
    require(sha(path) == expected, 'Changed SHA256: '+str(path))

def load_integer_codec(root):
    """Import the bound NumPy CDF module and only two exact historical classes.

    The original whole_entropy module imports torch and class-conditioned model
    wrappers. Extracting its unchanged integer classes avoids importing those
    unused paths. SHA binds the entire source before AST extraction.
    """
    source = Path(root)/'src/var_comm'
    for name, expected in CODE_SHA.items():
        verify(source/name, expected)
    spec = importlib.util.spec_from_file_location('_wcl_t1_bound_entropy', source/'entropy.py')
    entropy = importlib.util.module_from_spec(spec); spec.loader.exec_module(entropy)
    tree = ast.parse((source/'whole_entropy.py').read_text(encoding='utf-8'))
    selected = [n for n in tree.body if isinstance(n, ast.ClassDef)
                and n.name in ('ArithmeticEncoder', 'ArithmeticDecoder')]
    require(len(selected) == 2, 'Historical integer classes unavailable')
    ns = {k: getattr(entropy, k) for k in ('FULL','HALF','QUARTER','TOTAL','validate_cdf','validate_bits')}
    ns['np'] = np
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source/'whole_entropy.py'), 'exec'), ns)
    return entropy, ns['ArithmeticEncoder'], ns['ArithmeticDecoder']

def source_tokens(tokens):
    a = np.asarray(tokens)
    require(a.shape == (680,) and np.issubdtype(a.dtype, np.integer)
            and np.all((a >= 0) & (a < 4096)), 'Expected complete ten-scale source tokens')
    return a.astype(np.int64, copy=False)

def static_encode_decode(tokens, cdfs, encoder_type, decoder_type, modes=(5,6,7,8,9)):
    a = source_tokens(tokens)
    require(np.asarray(cdfs).shape == (10,4097), 'One shared full CDF per scale required')
    encoder = encoder_type(); streams = {}; records = []
    for i in range(max(modes)):
        values = a[OFFSETS[i]:OFFSETS[i+1]]
        cdf = np.broadcast_to(cdfs[i], (len(values),4097))
        encoder.encode(values, cdf)
        if i+1 not in modes:
            continue
        bits = np.asarray(encoder.finish(), dtype=np.uint8)
        decoder = decoder_type(bits.copy()); decoded = []
        canonical = encoder_type()
        for s in range(i+1):
            table = np.broadcast_to(cdfs[s], (SIZES[s]**2,4097))
            values_rx = decoder.decode(table)
            canonical.encode(values_rx, table); decoded.extend(values_rx.tolist())
        require(np.array_equal(a[:OFFSETS[i+1]], decoded), 'Static exact token roundtrip failed')
        require(np.array_equal(bits, canonical.finish()), 'Static canonical re-encode failed')
        require(decoder.position-len(bits) == 30, 'Historical terminal lookahead changed')
        streams[i+1] = bits
        records.append(dict(m=i+1,raw_bits=int(12*OFFSETS[i+1]),arithmetic_bits=len(bits),
            flush_bits=len(bits)-len(encoder.bits),zero_extension_reads=30))
    return streams, records

def candidates():
    rows = []
    for q in (2,4,6):
        for rate in ('1/2','2/3','3/4','5/6'):
            n = 956*q; r = Fraction(rate); k = n*r.numerator//r.denominator
            for target_m in (7,8,9):
                rows.append(dict(candidate_id=f'm{target_m}_q{q}_r{rate.replace("/","-")}',
                    N=1024,target_m=target_m,q=q,nominal_rate=rate,k=k,n=n,
                    header_symbols=68,body_symbols=956,frame_tail_symbols=0,
                    length_bits=13,crc_bits=16,source_capacity=k-29,
                    actual_ldpc_layout_status='NOT_QUALIFIED_BY_SOURCE_ONLY_CHECK'))
    return rows

def choose_arithmetic(lengths, candidate, minimum_m=5):
    """Pure entropy: no raw substitution; missing prefixes are not TX failure."""
    attempts = []
    for m in range(candidate['target_m'], minimum_m-1, -1):
        if m not in lengths:
            return dict(status='BLOCKED_MISSING_PREFIX',actual_m=None,arithmetic_bits=None,
                        missing_m=m,attempts=attempts)
        bits = int(lengths[m]); fits = bits <= candidate['source_capacity']
        attempts.append(dict(m=m,bits=bits,fits=fits))
        if fits:
            return dict(status='SOURCE_LENGTH_FITS_LAYOUT_PENDING',actual_m=m,
                        arithmetic_bits=bits,missing_m=None,attempts=attempts)
    return dict(status='TX_UNENCODABLE',actual_m=None,arithmetic_bits=None,
                missing_m=None,attempts=attempts)

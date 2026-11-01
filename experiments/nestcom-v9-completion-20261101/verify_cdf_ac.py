"""Bounded CPU audit of unchanged repository CDF/AC definitions; no image inference."""
from pathlib import Path
import ast
import hashlib
import importlib.util
import json
import numpy as np

import argparse
p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
SRC=args.repo/'src/var_comm'
OUT=args.output;OUT.mkdir(parents=True,exist_ok=False)
spec = importlib.util.spec_from_file_location('audited_entropy', SRC / 'entropy.py')
entropy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entropy)
tree = ast.parse((SRC / 'whole_entropy.py').read_text())
nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in ('ArithmeticEncoder', 'ArithmeticDecoder')]
assert len(nodes) == 2
ns = {k: getattr(entropy, k) for k in ('FULL', 'HALF', 'QUARTER', 'TOTAL', 'validate_cdf', 'validate_bits')}
ns['np'] = np
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SRC / 'whole_entropy.py'), 'exec'), ns)
Encoder, Decoder = ns['ArithmeticEncoder'], ns['ArithmeticDecoder']

def array_sha(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()

# No random source, no external model. Cases cover exact uniform, one dominant
# mass, tied maxima, tiny nonzero tails and ordinary nonuniform probability.
logp = np.zeros((6, 4096), dtype=np.float32)
logp[1] = -1000.; logp[1, 17] = 0.
logp[2] = -1000.; logp[2, :11] = 0.
logp[3] = np.linspace(-30., 0., 4096, dtype=np.float32)
logp[4] = -80.; logp[4, [0, 1, 2048, 4095]] = np.asarray([-1., -2., -3., -4.], np.float32)
logp[5] = np.arange(4096, dtype=np.float32) % 17 - 8
cdf = entropy.probability_cdf(logp)
freq = np.diff(cdf, axis=1)
assert cdf.dtype == np.int64 and np.all(freq >= 1) and np.all(freq.sum(axis=1) == 2**24)
assert np.all(freq[0] == 4096)
assert freq[1, 17] == 2**24 - 4095
assert np.all(np.delete(freq[1], 17) == 1)
base_tie = (2**24 - 4096) // 11 + 1
remainder = (2**24 - 4096) % 11
assert freq[2, 0] == base_tie + remainder
assert np.all(freq[2, 1:11] == base_tie)
assert np.all(freq[2, 11:] == 1)
tokens = np.asarray([4095, 17, 8, 4095, 2048, 1010], np.int64)

encoder = Encoder()
records = []
for stop in range(1, len(tokens) + 1):
    encoder.encode(tokens[stop-1:stop], cdf[stop-1:stop])
    state = (encoder.low, encoder.high, encoder.pending, tuple(encoder.bits))
    bits = encoder.finish()
    assert state == (encoder.low, encoder.high, encoder.pending, tuple(encoder.bits))
    independent = Encoder(); independent.encode(tokens[:stop], cdf[:stop])
    assert np.array_equal(bits, independent.finish())
    # Separate RX state. Only wire bits and the prescribed CDF enter decoder.
    decoder = Decoder(bits.copy())
    recovered = decoder.decode(cdf[:stop].copy())
    assert np.array_equal(recovered, tokens[:stop])
    assert decoder.position - len(bits) == 30
    canonical = Encoder(); canonical.encode(recovered, cdf[:stop])
    assert np.array_equal(bits, canonical.finish())
    extra = np.concatenate((bits, np.zeros(1, np.uint8)))
    bad_decoder = Decoder(extra); bad_tokens = bad_decoder.decode(cdf[:stop])
    bad_canonical = Encoder(); bad_canonical.encode(bad_tokens, cdf[:stop])
    assert not np.array_equal(extra, bad_canonical.finish())
    records.append(dict(tokens=stop, payload_bits=len(bits), bitstring=''.join(map(str, bits.tolist())), cdf_sha256=array_sha(cdf[:stop]), recovered_sha256=array_sha(recovered), terminal_zero_reads=decoder.position-len(bits)))

np.savez_compressed(OUT / 'cdf_ac_reference_vector.npz', log_probs=logp, integer_cdf=cdf, tokens=tokens)
result = dict(status='PASS_CPU_CDF_ARITHMETIC_REFERENCE_CHECK', scope='Synthetic integer coding only; no VAR inference, no channel or image-quality claim', source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (SRC/'entropy.py', SRC/'whole_entropy.py')}, numpy=np.__version__, source_cases=6, independent_terminations=len(records), records=records)
(OUT / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k:v for k,v in result.items() if k!='records'}))

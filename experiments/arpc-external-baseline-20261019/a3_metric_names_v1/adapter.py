"""Exact legacy BPG name normalization; scientific values and formulas unchanged."""
import ast
import hashlib

DATA_SHA = 'dd9ece6c4e2f8cfadbd0c9cf878e738cb93f147037642af5e394bed45527534d'
OLD = "m,s,i=r['method'],int(r['snr_db']),int(r['source_index'])"
NEW = "m,s,i=({'BPG_ADAPTIVE':'BPG_ADAPTIVE_DOWNSAMPLING_LDPC_N1024'}.get(r['method'],r['method'])),int(r['snr_db']),int(r['source_index'])"

def adapted(raw):
    assert hashlib.sha256(raw).hexdigest() == DATA_SHA
    text = raw.decode()
    assert text.count(OLD) == 2
    changed = text.replace(OLD, NEW, 1)
    assert changed.count(NEW) == 1
    assert ast.dump(ast.parse(changed.replace(NEW, OLD, 1)), include_attributes=False) == ast.dump(ast.parse(text), include_attributes=False)
    namespace = {'__name__': 'arpc_exact_legacy_names'}
    exec(compile(changed, '<ARPC exact legacy BPG main-table alias>', 'exec'), namespace)
    return namespace['baseline_rows']

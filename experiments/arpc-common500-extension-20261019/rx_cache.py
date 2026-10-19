"""Original cache key contract, enlarged state count only; math is untouched."""
import hashlib,pathlib
PE=pathlib.Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/arpc-external-baseline-20261019')
def load():
    p=PE/'a3_rx_v1/cache.py';raw=p.read_bytes()
    # The hash is checked against the closed original A3 source registration.
    import json
    reg=json.loads((PE.parents[1]/'results/arpc_external_baseline_20261019_v1/a3_rx_v1/registration.json').read_bytes())
    pin=next(d for d in reg['source_files']if d['path']==str(p))
    assert hashlib.sha256(raw).hexdigest()==pin['sha256'] and len(raw)==pin['bytes']
    s=raw.decode();assert s.count('len(new)<=700')==1
    ns={'__name__':'arpc_common500_rx_cache'};exec(compile(s.replace('len(new)<=700','len(new)<=3600'),str(p)+'::remaining400_cap','exec'),ns)
    return ns['match'],ns['extend_cache']

"""Resume the unchanged N3060 waiter with its exact pre-probe dependency manifest.
The original broad git enumeration gained unrelated RX probe files. All 470 original
bindings remain verified. Only added files under this isolated probe folder are
excluded from recomputing that historical enumeration; no original dependency is removed.
The actual GPU child still runs tools.replay_n3060_reference_timing --run unchanged.
"""
import run_preflight as env
import os,time,sys,json
from pathlib import Path
sys.path.insert(0,str(env.ROOT))
from latent_enhancement.runtime import digest,write_json
from tools import replay_n3060_reference_timing as legacy
ROOT=env.ROOT
HERE=env.OUT/'revision_v2'/'queue_restoration'/'n3060_manifest_adapter'
OLD=legacy.OUT/'worker_registration.json'
# Environment helper configures the RX supervisor; restore the original waiter stage root.
from token_efficiency import delivery_chain
delivery_chain.CHAIN=legacy.BASE/'delivery_chain_v1'
PREFIX=str(ROOT/'experiments/rx-posterior-step1-20260929')+'/'
def validate_delta(original,current,allowed_prefix):
    old=set(original);new=set(current)
    if old-new:raise RuntimeError('original registered dependency disappeared')
    added=new-old
    if any(not x.startswith(allowed_prefix) for x in added):
        raise RuntimeError('new dependency outside independent RX probe; requires review')
    return sorted(added)
def frozen_dependencies():
    registration=json.loads((HERE/'registration.json').read_text())
    assert digest(OLD)==registration['worker_registration_sha256']
    assert digest(__file__)==registration['adapter_sha256']
    record=json.loads(OLD.read_text())
    for p,h in record['bindings'].items():assert digest(p)==h,('original dependency changed',p)
    return [Path(x) for x in sorted(record['bindings'])]
def main():
    reg=json.loads(OLD.read_text());legacy.runtime_identity()
    assert len(reg['bindings'])==470
    current={str(p.resolve()) for p in legacy.dependency_files()}
    extras=validate_delta(reg['bindings'],current,PREFIX)
    frozen_dependencies()
    write_json(HERE/'runtime_receipt.json',dict(status='EXACT_ORIGINAL_DEPENDENCIES_VERIFIED',
        original_bindings=470,added_unrelated_RX_files=extras,runtime=legacy.runtime_identity(),
        time=time.time(),GPU_forward='NOT_RUN',new_holdout=False))
    legacy.dependency_files=frozen_dependencies
    legacy.wait()
if __name__=='__main__':main()

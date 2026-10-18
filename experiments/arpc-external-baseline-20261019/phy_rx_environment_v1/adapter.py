"""Exact environment-root and output-path revision; original PHY bytes unchanged."""
import ast,hashlib,pathlib
E=pathlib.Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/arpc-external-baseline-20261019')
ORIGINAL=E/'phy_v1/engineering.py'
SHA='359254256a22dfe79d012fc6943e534d4be6d29b16b3a03140f38b77fad02b18'
EDITS=[
 ("target=RUN/'rx';target.mkdir();", "target=RUN/'rx_environment_v1';target.mkdir();"),
 ("rtroot/'envs/frozen_ldpc_py311',RUN/'received_inputs',target]", "rtroot/'envs/frozen_ldpc_py311',rtroot/'envs/frozen_python_base_v1/lib/python3.11',rtroot/'assets/frozen_namespace_v1/VAR_COMM/outputs/PRIOR-AWARE-UEP-20261004-V1/ldpc_environment/lib/python3.11/site-packages',RUN/'received_inputs',target]"),
 ("RUN/'rx/completion.json'", "RUN/'rx_environment_v1/completion.json'"),
]
def adapt(data):
    assert hashlib.sha256(data).hexdigest()==SHA
    original=data.decode();text=original
    for a,b in EDITS:assert text.count(a)==1;text=text.replace(a,b)
    reverse=text
    for a,b in reversed(EDITS):assert reverse.count(b)==1;reverse=reverse.replace(b,a)
    assert reverse==original
    return text
def namespace():
    return {'__file__':str(ORIGINAL),'__name__':'_arpc_engineering_rx_environment'}

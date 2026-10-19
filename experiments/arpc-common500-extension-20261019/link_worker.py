"""Population/path-only adapter of the qualified original physical engine."""
import ast,hashlib,pathlib
PE=pathlib.Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/arpc-external-baseline-20261019')
SHA='248c92d388e5455a92e35301af1a2d24e8991900ee3e40818e483382d671634d'
EDITS=[
 ("RUN=L/'a2b_link_v1';OUT=O/'a2b_link_v1';REG=OUT/'registration.json'", "L=L.parent/'arpc_common500_extension_20261019_v1';O=O.parent/'arpc_common500_extension_20261019_v1'\nRUN=L/'link_v1';OUT=O/'link_v1';REG=OUT/'registration.json'"),
 ('18000','3600'),('len(sources)==1000','len(sources)==400'),
 ('CAL1000_PHYSICAL_FRAMES_CLOSED_AWAITING_ACTUAL_SOURCE_RX','COMMON500_EXTENSION_PHYSICAL_FRAMES_CLOSED_AWAITING_ACTUAL_SOURCE_RX'),
]
def adapt(raw):
    assert hashlib.sha256(raw).hexdigest()==SHA
    original=text=raw.decode()
    for a,b in EDITS:assert a in text and b not in text;text=text.replace(a,b)
    inverse=text
    for a,b in reversed(EDITS):inverse=inverse.replace(b,a)
    assert inverse==original
    calls=[]
    for s in (original,text):
        tree=ast.parse(s)
        calls.append([ast.dump(n,include_attributes=False)for n in ast.walk(tree)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr in ('transmit','receive','standard_noise')])
    assert calls[0]==calls[1]
    return text
if __name__=='__main__':
    p=PE/'a2b_link_v1/worker.py';exec(compile(adapt(p.read_bytes()),str(p)+'::common500_remaining400','exec'),{'__name__':'__main__','__file__':str(p)})

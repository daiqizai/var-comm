"""Population/path-only reuse of original, separately executed TX/RX PHY engine."""
import ast,hashlib,pathlib,sys
PE=pathlib.Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/arpc-external-baseline-20261019')
SHA='248c92d388e5455a92e35301af1a2d24e8991900ee3e40818e483382d671634d'
def adapt(raw):
    assert hashlib.sha256(raw).hexdigest()==SHA
    original=raw.decode();text=original
    edits=[("RUN=L/'a2b_link_v1';OUT=O/'a2b_link_v1';REG=OUT/'registration.json'", "L=L.parent/'kodak_current_20261026_v1/arpc_v1';O=O.parent/'kodak_current_20261026_v1/arpc_v1'\nRUN=L/'phy';OUT=O/'phy';REG=OUT/'registration.json'"),('from plan import events,digest','from kodak_plan import events,digest'),('18000','216'),('len(sources)==1000','len(sources)==24'),('CAL1000_PHYSICAL_FRAMES_CLOSED_AWAITING_ACTUAL_SOURCE_RX','KODAK24_PHYSICAL_FRAMES_CLOSED_AWAITING_ACTUAL_SOURCE_RX')]
    for a,b in edits:assert a in text;text=text.replace(a,b)
    inverse=text
    for a,b in reversed(edits):inverse=inverse.replace(b,a)
    assert inverse==original
    calls=lambda s:[ast.dump(n,include_attributes=False) for n in ast.walk(ast.parse(s)) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('transmit','receive','standard_noise')]
    assert calls(original)==calls(text)
    return text
if __name__=='__main__':
    sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
    p=PE/'a2b_link_v1/worker.py';exec(compile(adapt(p.read_bytes()),str(p)+'::Kodak24_population_v1','exec'),{'__name__':'__main__','__file__':str(p)})

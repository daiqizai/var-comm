"""Registered population/limit adapter; original actual physical operations unchanged."""
import ast,hashlib,pathlib
E=pathlib.Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/experiments/arpc-external-baseline-20261019')
ORIGINAL_SHA='248c92d388e5455a92e35301af1a2d24e8991900ee3e40818e483382d671634d'
EDITS=[('a2b_link_v1','a3_link_v1'),('18000','900'),('len(sources)==1000','len(sources)==100'),('CAL1000_PHYSICAL_FRAMES_CLOSED_AWAITING_ACTUAL_SOURCE_RX','EVAL100_PHYSICAL_FRAMES_CLOSED_AWAITING_ACTUAL_SOURCE_RX')]
def adapt(data):
    assert hashlib.sha256(data).hexdigest()==ORIGINAL_SHA
    text=original=data.decode()
    for a,b in EDITS:assert a in text;text=text.replace(a,b)
    back=text
    for a,b in reversed(EDITS):back=back.replace(b,a)
    assert back==original
    before=ast.parse(original);after=ast.parse(text)
    for name in ['cache_env','ledger_closed']:
        a=next(n for n in before.body if isinstance(n,ast.FunctionDef) and n.name==name)
        b=next(n for n in after.body if isinstance(n,ast.FunctionDef) and n.name==name)
        assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)
    for tree in [before,after]:
        calls=[ast.dump(n,include_attributes=False) for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ['transmit','receive','standard_noise']]
        if tree is before:original_calls=calls
        else:assert original_calls==calls
    return text
if __name__=='__main__':
    p=E/'a2b_link_v1/worker.py';exec(compile(adapt(p.read_bytes()),str(p)+'::A3_fixed900','exec'),{'__name__':'__main__','__file__':str(p)})

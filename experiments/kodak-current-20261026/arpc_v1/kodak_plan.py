"""Frozen selection and event rules with only Kodak population/noise labels changed."""
import hashlib, pathlib
R=pathlib.Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
def load(raw):
    before=raw.decode();assert before.count('[6201,6202,6203]')==1 and before.count('len(out)<=3600')==1
    text=before.replace('[6201,6202,6203]','[2001,2002,2003]').replace('len(out)<=3600','len(out)<=216')
    assert text.replace('[2001,2002,2003]','[6201,6202,6203]').replace('len(out)<=216','len(out)<=3600')==before
    ns={};exec(compile(text,'frozen_plan::Kodak24','exec'),ns);return ns
_ns=load((R/'experiments/arpc-common500-extension-20261019/plan.py').read_bytes()) if R.exists() else None
def events(*args):return _ns['events'](*args)
def digest(x):
    import json
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()

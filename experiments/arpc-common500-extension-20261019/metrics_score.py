"""Frozen seven-metric algorithm, expanded counters and separate output paths."""
import ast,hashlib,re
from common import *
def adapted(raw):
    p=PE/'a3_metrics_v1/score.py';reg=read(PO/'a3_metrics_v1/registration.json');pin=next(d for d in reg['source_files']if d['path']==str(p))
    assert hashlib.sha256(raw).hexdigest()==pin['sha256']and len(raw)==pin['bytes']
    original=s=raw.decode();edits=[('import control as c','import metrics_control as c'),('from data import METRICS','from metric_data import METRICS'),("c.E/'cuda_recovery_v1'","c.PE/'cuda_recovery_v1'"),('from metric_adapter import load','from metrics_math import load'),("/'metric_adapter.py'","/'metrics_math.py'"),('external_common100_v1','common500_remaining400')]
    for a,b in edits:assert a in s;s=s.replace(a,b)
    values={'100':'400','800':'3600','1600':'7200','900':'3600'}
    s=re.sub(r'\b(?:100|800|1600|900)\b',lambda m:values[m[0]],s)
    # Verify all model/mathematical calls preserve their original AST argument lists.
    def calls(t):
        names={'prepare','score','embed','preprocess_dreamsim','cosine_similarity','pair_cache_key'}
        return [ast.dump(n,include_attributes=False)for n in ast.walk(ast.parse(t))if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr in names]
    assert calls(original)==calls(s)
    return s
if __name__=='__main__':
    p=PE/'a3_metrics_v1/score.py';exec(compile(adapted(p.read_bytes()),str(p)+'::common500_remaining400','exec'),{'__name__':'__main__','__file__':str(E/'metrics_score.py')})

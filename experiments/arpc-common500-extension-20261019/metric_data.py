"""Original per-source grouping/means, exact original indices and 400-source cap."""
import hashlib,json
from common import PE,PO
def adapted(raw):
    p=PE/'a3_metrics_v1/data.py';r=json.loads((PO/'a3_metrics_v1/registration.json').read_bytes());d=next(x for x in r['source_files']if x['path']==str(p))
    assert len(raw)==d['bytes']and hashlib.sha256(raw).hexdigest()==d['sha256']
    edits=[('len(source)==100 and len(rows)==900','len(source)==400 and len(rows)==3600'),("sum(len(s['pairs']) for s in result)<=800","sum(len(s['pairs']) for s in result)<=3600"),('len(seen)==900 and len(result)==2100','len(seen)==3600 and len(result)==8400')]
    s=original=raw.decode()
    for a,b in edits:assert s.count(a)==1;s=s.replace(a,b,1)
    inverse=s
    for a,b in reversed(edits):inverse=inverse.replace(b,a,1)
    assert inverse==original
    return s
ns={'__name__':'_common500_metric_data'}
exec(compile(adapted((PE/'a3_metrics_v1/data.py').read_bytes()),'common500_metric_data','exec'),ns)
group=ns['group'];means=ns['means'];METRICS=ns['METRICS'];DIRECTIONS=ns['DIRECTIONS']

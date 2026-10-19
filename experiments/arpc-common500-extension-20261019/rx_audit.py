"""Qualified original post-exit pixel/status audit with population and paths only."""
import hashlib,json,sys
from common import PE,PO
import rx_control as c
def adapted(raw):
    p=PE/'a3_rx_v1/audit.py';reg=json.loads((PO/'a3_rx_v1/registration.json').read_bytes());pin=next(x for x in reg['source_files']if x['path']==str(p))
    assert len(raw)==pin['bytes']and hashlib.sha256(raw).hexdigest()==pin['sha256']
    edits=[('import control as c','import rx_control as c'),("c.O/'a3_link_v1/logical_frames.json'","c.O/'link_v1/logical_frames.json'"),('900','3600')]
    s=original=raw.decode()
    for a,b in edits:assert a in s;s=s.replace(a,b)
    inverse=s
    for a,b in reversed(edits):inverse=inverse.replace(b,a)
    assert inverse==original
    return s
if __name__=='__main__':
    p=PE/'a3_rx_v1/audit.py';c.success=c.check_wait
    exec(compile(adapted(p.read_bytes()),str(p)+'::remaining400','exec'),{'__name__':'__main__','__file__':str(p)})

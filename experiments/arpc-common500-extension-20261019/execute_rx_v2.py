"""Once-only RX preparation-v2 and original dispatch, with preserved v1 failure."""
import ast,hashlib
from common import *
OLD_SHA='aef9ca8629be68d3aeea5cc7c7bac62cbba004c961b6a5c89ee563d3c0216c4a'
def adapt(raw):
    assert hashlib.sha256(raw).hexdigest()==OLD_SHA
    s=original=raw.decode()
    edits=[("out=L/(a.stage+'_'+a.role+'_control_v1');assert not out.exists()","assert a.stage=='rx'\n    out=L/('rx_prepare_control_v2'if a.role=='prepare'else'rx_dispatch_control_v1');assert not out.exists()"),
           ("str(E/(a.stage+'_control.py'))","str(E/('rx_prepare_v2.py'if a.role=='prepare'else'rx_control.py'))"),
           ("(a.stage+'_prepare_control_v1')","(a.stage+'_prepare_control_v2')")]
    for a,b in edits:assert a in s;s=s.replace(a,b)
    inverse=s
    for a,b in reversed(edits):inverse=inverse.replace(b,a)
    assert inverse==original
    return s
if __name__=='__main__':
    exec(compile(adapt((E/'execute_stage_v1.py').read_bytes()),str(E/'execute_stage_v1.py')+'::rx_prepare_v2','exec'),{'__name__':'__main__','__file__':__file__})

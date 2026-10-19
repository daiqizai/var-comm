"""User-authorized host move; frozen source science is unchanged byte for byte."""
import ast, hashlib, os, pathlib, sys
from common import *
SOURCE_SHA='b18956cc008dc85ba025f59905fb88791288fe1d9d03afcc2aa2113332c493dc'
HOST='aibox-r59688baa77c-57c794f5c6-44lm7'
ADMISSION=O/'host3781_admission_v2/admission.json'
ADMISSION_SHA='13274d6f84b5873a0f8eacebff83f45ed5b3f92421fae750160c1f250a2d9901'
OLD_HOST_LINE="assert os.uname().nodename==read(PO/'a3_source_v1/registration.json')['host'],'Use the established native host/environment'"
EDITS=[
    (OLD_HOST_LINE,'require_admission()'),
    ("('common.py','source_control.py','source_native.py','execute.py')", "('common.py','source_control.py','source_native.py','execute.py','source_host_v1.py','execute_host_v1.py')"),
    ("desc(qual/'completion.json')],'source_files'", "desc(qual/'completion.json'),desc(ADMISSION)],'source_files'"),
]
def require_admission():
    assert os.uname().nodename==HOST and sys.executable=='/usr/bin/python3'
    assert desc(ADMISSION)['sha256']==ADMISSION_SHA
    d=read(ADMISSION);assert d['status']=='PASS' and d['host']==HOST
    assert d['runtime']['python_version']==list(sys.version_info[:3])==[3,12,3]
    assert d['runtime']['python']==sys.version
    assert d['new_scientific_calls']==0 and d['numeric_protocol_unchanged']
    for p in d['evidence']+d['source_files']:verify(p)
    q=L/'host3781_qualification_v2'
    check_wait(q/'imports.actual_wait.json')
    assert gone(read(q/'supervisor.json')['process'])
def adapt(data):
    assert hashlib.sha256(data).hexdigest()==SOURCE_SHA
    before=text=data.decode()
    for a,b in EDITS:assert text.count(a)==1;text=text.replace(a,b,1)
    inverse=text
    for a,b in reversed(EDITS):assert inverse.count(b)==1;inverse=inverse.replace(b,a,1)
    assert inverse==before
    trees=[ast.parse(s)for s in (before,text)]
    for name in ('run','audit','main'):
        nodes=[next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name==name)for t in trees]
        assert ast.dump(nodes[0],include_attributes=False)==ast.dump(nodes[1],include_attributes=False)
    return text
if __name__=='__main__':
    require_admission()
    p=E/'source_control.py'
    exec(compile(adapt(p.read_bytes()),str(p)+'::authorized_host3781','exec'),{**globals(),'__name__':'__main__','__file__':str(p)})

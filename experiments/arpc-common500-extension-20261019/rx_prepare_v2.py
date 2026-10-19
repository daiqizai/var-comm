"""Read the original clean-cache closure schema; native RX code is unchanged."""
import ast,hashlib
from common import *
OLD_SHA='c88a208117cb7bdf228e2a7ab7aba11ba6350399af0322419ced845300425581'

def normalized_clean_review(review,completion,completion_pin):
    assert review['status']=='PASS'and review['all_processes_gone']and not review['new_failure_files']
    assert completion['status']=='CLEAN_STATES_AUDITED_NO_METRICS_OR_WIRELESS'and completion['unresolved']==0 and completion['total_clean_states']==700
    assert review['registration_sha256']==completion['registration_sha256']
    assert completion_pin in review['receipts']
    assert review['process_checks']and all(x['observed_gone']for x in review['process_checks'])
    return dict(review,unresolved=completion['unresolved'],controls=review['receipts'],outputs=completion['outputs'])

def adapt(raw):
    assert hashlib.sha256(raw).hexdigest()==OLD_SHA
    original=raw.decode()
    a="rp=PO/name/'root_closure_review.json';r=read(rp);assert r['all_processes_gone'] and r['unresolved']==0"
    b="rp=PO/name/'root_closure_review.json';r=read(rp)\n        if name=='a2_clean_v1':r=normalized_clean_review(r,read(PO/name/'completion.json'),desc(PO/name/'completion.json'))\n        assert r['all_processes_gone'] and r['unresolved']==0"
    c="code=[desc(E/n)for n in ('common.py','stage_support.py','execute_stage_v1.py','rx_control.py','rx_native.py','rx_cache.py','rx_audit.py','source_host_v1.py')]"
    d=c+"\n    code+=[desc(E/n)for n in ('rx_prepare_v2.py','execute_rx_v2.py')]\n    inputs+=[desc(PO/'a2_clean_v1/completion.json'),desc(L/'rx_schema_qualification_v1/completion.json'),desc(L/'rx_prepare_control_v1/child.actual_wait.json'),desc(L/'rx_prepare_control_v1/failure.json')]"
    assert original.count(a)==original.count(c)==1
    text=original.replace(a,b).replace(c,d)
    assert text.replace(d,c).replace(b,a)==original
    old=next(x for x in ast.parse(original).body if isinstance(x,ast.FunctionDef)and x.name=='run')
    new=next(x for x in ast.parse(text).body if isinstance(x,ast.FunctionDef)and x.name=='run')
    assert ast.dump(old)==ast.dump(new)
    return text

def main():
    assert sys.argv[1:]==['prepare']
    assert not(L/'rx_v1').exists()and not(O/'rx_v1').exists()
    old=L/'rx_prepare_control_v1';w=read(old/'child.actual_wait.json')
    assert type(w['returncode'])is int and w['returncode']==1 and gone(w['process'])and gone(read(old/'supervisor.json')['process'])
    assert "KeyError: 'unresolved'"in(old/'child.stderr').read_text()
    q=read(L/'rx_schema_qualification_v1/completion.json');assert q['status']=='PASS'and q['scientific_calls']==0
    for p in q['sources']:verify(p)
    ns={'__name__':'_rx_schema_v2','__file__':str(E/'rx_control.py'),'normalized_clean_review':normalized_clean_review}
    exec(compile(adapt((E/'rx_control.py').read_bytes()),str(E/'rx_control.py')+'::clean_closure_schema_v2','exec'),ns)
    ns['prepare']()

if __name__=='__main__':main()

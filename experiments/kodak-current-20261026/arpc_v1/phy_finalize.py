"""Existing saved-packet finalization, population216; original PHY exit1 retained."""
import ast,collections,hashlib,importlib.util,json,os,pathlib,sys,time
E=pathlib.Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('kodak_original_control',E/'control.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
import phy as link_worker
OLD=c.R/'experiments/arpc-common500-extension-20261019/link_finalize_v1.py'
OLD_SHA='b2631ef791e8f836d512370aa187cde279f03a9514e4c548078de0253c958a7b'
OBS_SHA='c4b183cd52d2dec1dac739230f6dea46e803f33f312c8aba53caee0bdbdc7c77'
assert hashlib.sha256(OLD.read_bytes()).hexdigest()==OLD_SHA
# Reuse all three original pure functions exactly, including boundary proof and
# post-exit audit AST adaptation. They contain no receive/transmit/model calls.
_tree=ast.parse(OLD.read_bytes());_nodes=[n for n in _tree.body if isinstance(n,ast.FunctionDef)and n.name in ('ledger','boundary_proof','audit_source')]
assert len(_nodes)==3
exec(compile(ast.Module(body=_nodes,type_ignores=[]),str(OLD)+'::exact_pure_helpers','exec'),globals())
RUN=c.L/'phy';OUT=c.O/'phy';WORK=c.L/'phy_finalization_v1';REPORT=c.O/'phy_finalization_v1'

def failure():
    p=c.O/'phy_failure_observation_v1/observation.json';assert c.desc(p)['sha256']==OBS_SHA;d=c.read(p)
    for pin in d['waits']+d['packet_outputs']+d['evidence']:c.verify(pin)
    for name,rc in [('tx',0),('rx',1)]:
        w=c.read(RUN/(name+'.actual_wait.json'));assert type(w['returncode'])is int and w['returncode']==rc and c.gone(w['process'])
    w=c.read(c.L/'phy_control/worker.actual_wait.json');assert type(w['returncode'])is int and w['returncode']==1 and c.gone(w['process'])
    assert c.gone(c.read(c.L/'phy_control/supervisor.json')['process'])
    assert not(RUN/'rx/completion.json').exists()
    assert 'line 80, in rx'in(RUN/'rx.stderr').read_text()
    return d

def finalize_phy():
    assert not WORK.exists()and not REPORT.exists();d=failure()
    c.checkreg(c.desc(c.O/'registration.json')['sha256'])
    q=c.read(c.O/'phy_finalization_qualification.json');assert q['status']=='PASS'and q['science_calls']==0
    for p in q['sources']:c.verify(p)
    counts=ledger(RUN/'rx/calls.jsonl',{'header':216,'body':216});tx=c.read(RUN/'tx/completion.json');ledger(RUN/'tx/calls.jsonl',tx['counts'])
    boundary=boundary_proof((c.PE/'a1_native.py').read_text());jobs=d['received_jobs'];assert len(jobs)==72
    for j in jobs:c.verify(j['input']);assert j['path']==j['input']['path']
    WORK.mkdir();REPORT.mkdir();c.put(WORK/'launch.json',{'process':c.identity(os.getpid()),'parent_process':c.identity(os.getppid()),'argv':sys.argv,'science_calls':0})
    c.put(RUN/'rx/actual_source_jobs.json',jobs)
    receipt={'status':'ALL_SAVED_PACKETS_COMPLETE_ORIGINAL_RX_EXIT1','logical_frames':216,'counts':counts,'outputs':d['packet_outputs'],'receiver_boundary':boundary,'blocked_reads':None,'neural_calls':0,'actual_unique_payloads':72,'jobs':c.desc(RUN/'rx/actual_source_jobs.json'),'original_rx_wait':c.desc(RUN/'rx.actual_wait.json'),'replayed_decodes':0}
    c.put(WORK/'rx_receipt.json',receipt)
    text=audit_source((c.PE/'a2b_link_v1/worker.py').read_bytes())
    ns=dict(RUN=RUN,OUT=REPORT,REG=OUT/'registration.json',WORK=WORK,json=json,pathlib=pathlib,time=time,verify=c.verify,desc=c.desc,put=c.put)
    exec(compile(text,'original_phy_audit::saved_packets_only','exec'),ns);ns['audit'](c.read(OUT/'registration.json'))
    rows=c.read(REPORT/'logical_frames.json');a=c.read(REPORT/'completion.json');assert len(rows)==216
    c.put(OUT/'logical_frames.json',rows)
    a.update(status='COMPLETE_SAVED_PACKET_FINALIZATION_WITH_PRESERVED_RX_EXIT1',receiver_boundary=boundary,original_failed_rx_preserved=True,replayed_decodes=0,outputs=[c.desc(OUT/'logical_frames.json'),c.desc(RUN/'rx/actual_source_jobs.json'),c.desc(WORK/'rx_receipt.json')]);c.put(OUT/'completion.json',a)
    c.put(REPORT/'recovery_audit.json',{'status':'PASS_WITH_DISCLOSED_ORIGINAL_ATTEMPT_LOG_LOSS','scientific_calls':0,'replayed_decodes':0,'boundary':boundary,'original_observation':c.desc(c.O/'phy_failure_observation_v1/observation.json'),'original_waits':d['waits'],'output':c.desc(OUT/'completion.json'),'unresolved':0,'original_helper_source':c.desc(OLD)})
    for p in d['evidence']+d['packet_outputs']:c.verify(p)
    print(json.dumps(c.desc(REPORT/'recovery_audit.json')))

def finalize_delivery():
    import resource_continuation as resource
    resource.guard(c,True,False);failure()
    c.success(RUN/'audit.actual_wait.json');assert c.gone(c.read(RUN/'audit.actual_wait.json')['process'])
    for role in ('source','rx'):
        root=resource.control_dir(c,role);c.success(root/'worker.actual_wait.json');assert c.gone(c.read(root/'supervisor.json')['process'])
    assert c.read(REPORT/'recovery_audit.json')['replayed_decodes']==0
    source=(E/'control.py').read_text();fn=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)and n.name=='finalize');text=ast.get_source_segment(source,fn)
    old="for role in ('source','phy','rx'):\n        success(L/(role+'_control/worker.actual_wait.json'));assert read(L/(role+'_control/completion.json'))['status']=='COMPLETE'\n        assert not(L/role/'failure.json').exists()"
    new="assert read(O/'phy_finalization_v1/recovery_audit.json')['replayed_decodes']==0\n    success(L/'phy/audit.actual_wait.json')\n    for role in ('source','rx'):\n        root=resource.control_dir(original_control,role);success(root/'worker.actual_wait.json');assert read(root/'completion.json')['status']=='COMPLETE'\n        assert not(L/role/'failure.json').exists()"
    assert text.count(old)==1;text=text.replace(old,new)
    ns={**vars(c),'resource':resource,'original_control':c};exec(compile(text,'original_finalizer::explicit_preserved_failure_chain','exec'),ns);ns['finalize'](c.desc(c.O/'registration.json')['sha256'])
if __name__=='__main__':
    if sys.argv[1:]==['phy']:finalize_phy()
    elif sys.argv[1:]==['delivery']:finalize_delivery()
    else:raise ValueError('Specify phy or delivery')

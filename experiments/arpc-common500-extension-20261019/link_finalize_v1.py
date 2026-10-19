"""Finalize fixed packet outputs after the preserved final denied-read assertion.

No transmit/receive/model operation is available. This does not recover the lost
denied-path list or convert the original RX exit1 into exit0. The original Python
audit hook rejected every out-of-bound open before returning file data. Its
attempt-free completion condition failed; receiver data isolation stayed active.
"""
import ast,collections,fcntl,hashlib,time
from common import *
import link_worker

EXPECTED='4a36b8f0f0b73e803ce683ee3bd163a19e7cfb58f3e3448abedb89f075d7de01'
OBS_SHA='26d9c88b7acb2f98c4205f50da2105ef3769a39eec4212a751a660682f70d174'
RUN=L/'link_v1';OUT=O/'link_v1'
WORK=L/'link_finalization_v1';REPORT=O/'link_finalization_v1'

def ledger(path,expected):
    pending={};seen=set();counts=collections.Counter()
    for line in path.read_text().splitlines():
        a=json.loads(line);i=a['event']
        if a['state']=='RESERVED':
            assert i not in seen;seen.add(i);pending[i]=a;counts[a['kind']]+=1
        else:
            assert a['state']=='COMPLETE' and i in pending;pending.pop(i)
    assert not pending and dict(counts)==expected
    return dict(counts)

def old_exit(path,rc):
    a=read(path);assert type(a['returncode'])is int and a['returncode']==rc
    assert not any(a.get(k)for k in ('terminated','killed','timed_out','metadata_error'))
    assert gone(a['process']);return desc(path)

def boundary_proof(raw):
    tree=ast.parse(raw);f=next(x for x in tree.body if isinstance(x,ast.FunctionDef)and x.name=='install_rx_boundary')
    a=next(x for x in f.body if isinstance(x,ast.FunctionDef)and x.name=='audit')
    branch=next(x for x in ast.walk(a)if isinstance(x,ast.If)and ast.unparse(x.test).startswith('not any('))
    assert len(branch.body)==2 and ast.unparse(branch.body[0])=='blocked.append(str(p))'
    assert isinstance(branch.body[1],ast.Raise)and isinstance(branch.body[1].exc,ast.Call)
    assert ast.unparse(branch.body[1].exc.func)=='PermissionError'
    assert 'sys.addaudithook(audit)'in ast.unparse(f)
    return {'original_hook_ast_sha256':hashlib.sha256(ast.dump(f,include_attributes=False).encode()).hexdigest(),
            'out_of_allowlist_open_raises_before_returning_data':True,
            'original_blocked_attempts_nonzero':True,'original_blocked_paths':None,
            'original_attempt_log_recoverable':False,'attempt_free_gate_passed':False,
            'boundary_was_not_disabled_or_relaxed':True,
            'limitation':'Original denied-path list was memory-only and lost on process exit. Exact attempted paths are unknown; diagnostic runs do not reconstruct that historical list.'}

def audit_source(raw):
    original=link_worker.adapt(raw)
    tree=ast.parse(original)
    fns=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name in ('ledger_closed','audit')]
    assert len(fns)==2
    text=ast.unparse(ast.Module(body=fns,type_ignores=[]))
    old="json.loads((RUN / 'rx/completion.json').read_text())"
    new="json.loads((WORK / 'rx_receipt.json').read_text())"
    gate="and (not rx['blocked_reads'])"
    # ast.unparse on supported Python may omit the redundant parentheses.
    if gate not in text:gate="and (not rx['blocked_reads'])" if "and (not rx['blocked_reads'])"in text else "and not rx['blocked_reads']"
    replacement="and rx['receiver_boundary']['out_of_allowlist_open_raises_before_returning_data']"
    assert text.count(old)==1 and text.count(gate)==1
    changed=text.replace(old,new).replace(gate,replacement)
    assert changed.replace(new,old).replace(replacement,gate)==text
    calls=[n for n in ast.walk(ast.parse(changed))if isinstance(n,ast.Call)]
    assert not any(isinstance(n.func,ast.Attribute)and n.func.attr in ('receive','transmit','decode','forward','encode')for n in calls)
    return changed

def main():
    assert not WORK.exists()and not REPORT.exists()
    assert desc(OUT/'registration.json')['sha256']==EXPECTED
    obsfile=O/'link_failure_diagnosis_v1/observation.json';assert desc(obsfile)['sha256']==OBS_SHA
    obs=read(obsfile);reg=read(OUT/'registration.json')
    assert reg['host']==os.uname().nodename
    for d in reg['source_files']+reg['inputs']+obs['evidence']+obs['packet_outputs']:verify(d)
    q=read(L/'link_finalization_qualification_v1/completion.json');assert q['status']=='PASS'and q['scientific_calls']==0
    for d in q['sources']:verify(d)
    old_exit(L/'link_finalization_qualification_v1/tests.actual_wait.json',0)
    identities=[L/'link_dispatch_control_v1/supervisor.json',RUN/'owner.launch.json',RUN/'tx.launch.json',RUN/'rx.launch.json']
    for p in identities:assert gone(read(p)['process'])
    waits=[old_exit(RUN/'tx.actual_wait.json',0),old_exit(RUN/'rx.actual_wait.json',1),old_exit(L/'link_dispatch_control_v1/child.actual_wait.json',1)]
    assert (RUN/'failure.json').is_file()and(L/'link_dispatch_control_v1/failure.json').is_file()
    assert not(RUN/'rx/completion.json').exists()and not(OUT/'completion.json').exists()
    counts=ledger(RUN/'rx/calls.jsonl',{'header':3600,'body':3600})
    ledger(RUN/'tx/calls.jsonl',{'encode_waveform':1200})
    original_native=PE/'a1_native.py'
    # This exact boundary source was bound into the failed run, not newly substituted.
    nativepin=next(d for d in reg['source_files']if d['path']==str(original_native));verify(nativepin)
    boundary=boundary_proof(original_native.read_text())
    raw=(PE/'a2b_link_v1/worker.py').read_bytes();adapted=link_worker.adapt(raw)
    rx=next(n for n in ast.parse(adapted).body if isinstance(n,ast.FunctionDef)and n.name=='rx')
    assert ast.unparse(rx).count('install_rx_boundary(allowed)')==1
    assert 'source_truth_supplied=False'in ast.unparse(rx)
    assert not any(isinstance(n,ast.Name)and n.id=='reg'for n in ast.walk(rx))
    audit=audit_source(raw)
    lock=(PL/'arpc_science.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    WORK.mkdir();REPORT.mkdir()
    put(WORK/'launch.json',{'process':identity(os.getpid()),'parent_process':identity(os.getppid()),'argv':sys.argv,'started_unix':time.time(),'scientific_call_cap':0})
    jobs=obs['actual_jobs_from_saved_received_packets'];assert len(jobs)==1184
    for j in jobs:verify(j['input']);assert j['input']['path']==j['path']
    # Missing summary metadata only. Existing packet/payload/ledger bytes stay fixed.
    put(RUN/'rx/actual_source_jobs.json',jobs)
    receipt={'status':'SAVED_PACKETS_COMPLETE_ORIGINAL_RX_EXIT1','logical_frames':3600,'counts':counts,
             'outputs':obs['packet_outputs'],'receiver_boundary':boundary,'blocked_reads':None,
             'neural_calls':0,'actual_unique_payloads':1184,'jobs':desc(RUN/'rx/actual_source_jobs.json'),
             'original_rx_wait':waits[1],'replayed_decodes':0}
    put(WORK/'rx_receipt.json',receipt)
    # Original post-exit truth audit: only provenance/read locations and the explicit
    # preserved-failure condition differ. Packet decisions and payloads cannot change.
    ns=dict(RUN=RUN,OUT=REPORT,REG=OUT/'registration.json',WORK=WORK,json=json,pathlib=__import__('pathlib'),time=time,verify=verify,desc=desc,put=put)
    exec(compile(audit,str(PE/'a2b_link_v1/worker.py')+'::saved_packet_finalization','exec'),ns)
    ns['audit'](reg)
    a=read(REPORT/'completion.json');rows=read(REPORT/'logical_frames.json')
    assert len(rows)==3600 and collections.Counter(r['snr_db']for r in rows)=={4:1200,10:1200,19:1200}
    assert len({r['source_id']for r in rows})==400
    put(OUT/'logical_frames.json',rows)
    a.update(status='COMPLETE_SAVED_PACKET_FINALIZATION_WITH_PRESERVED_RX_EXIT1',receiver_boundary=boundary,
             original_failed_rx_preserved=True,replayed_decodes=0,
             outputs=[desc(OUT/'logical_frames.json'),desc(RUN/'rx/actual_source_jobs.json'),desc(WORK/'rx_receipt.json')])
    put(OUT/'completion.json',a)
    report={'status':'PASS_WITH_DISCLOSED_ORIGINAL_ATTEMPT_LOG_LOSS','scientific_calls':0,'replayed_decodes':0,
            'original_waits':waits,'boundary':boundary,'original_observation':desc(obsfile),
            'original_rx_ast_sha256':hashlib.sha256(ast.dump(rx,include_attributes=False).encode()).hexdigest(),
            'source_files':[desc(Path(__file__)),nativepin,desc(PE/'a2b_link_v1/worker.py')],
            'output':desc(OUT/'completion.json'),'unresolved':0,
            'reason':'All scientific calls and fixed packet outputs completed; only end-of-run no-denied-attempts assertion failed. The unchanged hook raised before outside file data could be obtained. Exact denied paths remain unknown. No data access is newly allowed and no packet output is recomputed.'}
    put(REPORT/'recovery_audit.json',report)
    for d in obs['packet_outputs']+obs['evidence']:verify(d)
    print(json.dumps({'recovery':desc(REPORT/'recovery_audit.json'),'completion':desc(OUT/'completion.json'),'logical_frames':3600,'received_states':1184,'scientific_calls':0}))

if __name__=='__main__':main()

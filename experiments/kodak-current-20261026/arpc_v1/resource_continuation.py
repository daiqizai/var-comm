"""One exact first launch after zero-native resource rejection; frozen math retained."""
import ast,hashlib,importlib.util,json,os,pathlib,subprocess,sys,time,traceback
E=pathlib.Path(__file__).resolve().parent
CONTROL_SHA='a685cdd92c6963f9b999ded73daafcd83c6a4fb7c865c08b409623efb490981d'
AUTH_SHA='d169aea0bc51a78c890db6980969cc918a8290a9d2f1dc69cd113d569511d4e4'

def adapt_worker(raw):
    assert hashlib.sha256(raw).hexdigest()==CONTROL_SHA
    tree=ast.parse(raw);node=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='worker')
    original=ast.get_source_segment(raw.decode(),node)
    edits=[("assert not(run/'worker.launch.json').exists()","assert (role=='source' and not(run/'native.launch.json').exists() and not(run/'tx').exists()) or (role=='rx' and not(run/'worker.launch.json').exists())"),
           ("put(run/'worker.launch.json'","put(run/('worker_resource.launch.json' if role=='source' else 'worker.launch.json')"),
           ("gpu['utilization']<=25","gpu['utilization']<=80")]
    text=original
    for a,b in edits:assert text.count(a)==1;text=text.replace(a,b,1)
    inverse=text
    for a,b in reversed(edits):inverse=inverse.replace(b,a,1)
    assert inverse==original
    return text

def load():
    raw=(E/'control.py').read_bytes();assert hashlib.sha256(raw).hexdigest()==CONTROL_SHA
    spec=importlib.util.spec_from_file_location('kodak_original_control',E/'control.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
    ns=dict(vars(c));exec(compile(adapt_worker(raw),str(E/'control.py')+'::shared_inference_resource80','exec'),ns)
    return c,ns['worker']

def control_dir(c,role):return c.L/('source_unlaunched_control_v1' if role=='source' else 'rx_resource_control_v1')

def guard(c,registered=True,source_unstarted=False):
    auth=c.O/'shared_inference_resource_authorization.json';assert c.desc(auth)['sha256']==AUTH_SHA
    a=c.read(auth)
    # The original failed process chain is evidence, not converted to success.
    w=c.read(c.L/'source_control/worker.actual_wait.json');assert type(w['returncode'])is int and w['returncode']==1 and not w['timed_out']
    assert c.gone(w['process']) and c.gone(c.read(c.L/'source_control/supervisor.json')['process'])
    err=(c.L/'source_control/worker.stderr').read_text()
    assert "gpu['utilization']<=25" in err and "'utilization': 36" in err
    if source_unstarted:assert not(c.L/'source/native.launch.json').exists() and not(c.L/'source/tx').exists()
    if registered:
        r=c.read(c.O/'resource_continuation_registration.json')
        for d in r['inputs']+r['sources']:c.verify(d)
        assert r['original_registration_sha256']==c.desc(c.O/'registration.json')['sha256']
    return a

def prepare():
    c,_=load();guard(c,False,True)
    q=c.read(c.O/'resource_qualification.json');assert q['status']=='PASS' and q['science_calls']==0
    for d in q['sources']:c.verify(d)
    pins=[c.desc(c.O/'shared_inference_resource_authorization.json'),c.desc(c.O/'registration.json'),c.desc(c.O/'resource_qualification.json')]
    pins += [c.desc(c.L/'source_control'/n) for n in ['worker.actual_wait.json','worker.stderr','failure.json','supervisor.json','worker.launch.json']]
    r={'schema':'ARPC_KODAK24_ZERO_NATIVE_RESOURCE_CONTINUATION_V1','original_registration_sha256':c.desc(c.O/'registration.json')['sha256'],'inputs':pins,'sources':[c.desc(E/'resource_continuation.py'),c.desc(E/'test_resource.py')],'resource_change':{'startup_utilization_max':80,'startup_free_bytes':64*1024**3,'allocator_max_bytes':70*1024**3,'running_free_bytes':8*1024**3},'scope':['source first native launch','rx first native launch'],'numeric_protocol_unchanged':True,'repeated_successful_science':0,'same_gpu_index':1}
    c.put(c.O/'resource_continuation_registration.json',r);print(json.dumps(c.desc(c.O/'resource_continuation_registration.json')))

def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('role',choices=['prepare','launch','supervisor','worker','finalize']);p.add_argument('--stage',choices=['source','rx']);a=p.parse_args()
    if a.role=='prepare':prepare();return
    c,worker=load();guard(c,True,a.stage=='source'and a.role in ['launch','worker'])
    expected=c.desc(c.O/'registration.json')['sha256'];c.checkreg(expected)
    if a.role=='finalize':
        # Only the exact known pre-Popen source failure is accepted. New source,
        # original PHY and resource-aware RX must have genuine wait0 evidence.
        for role in ('source','rx'):
            d=control_dir(c,role);c.success(d/'worker.actual_wait.json');assert c.read(d/'completion.json')['status']=='COMPLETE'
            assert c.gone(c.read(d/'supervisor.json')['process'])
        original=ast.get_source_segment((E/'control.py').read_text(),next(n for n in ast.parse((E/'control.py').read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='finalize'))
        old="success(L/(role+'_control/worker.actual_wait.json'));assert read(L/(role+'_control/completion.json'))['status']=='COMPLETE'"
        new="d=control_dir(original_control,role) if role in ('source','rx') else L/(role+'_control'); success(d/'worker.actual_wait.json');assert read(d/'completion.json')['status']=='COMPLETE'"
        assert original.count(old)==1;text=original.replace(old,new)
        ns={**vars(c),'control_dir':control_dir,'original_control':c};exec(compile(text,str(E/'control.py')+'::exact_source_resource_lineage','exec'),ns);ns['finalize'](expected);return
    d=control_dir(c,a.stage)
    if a.role=='launch':
        assert not d.exists();d.mkdir()
        argv=[sys.executable,'-B',str(E/'resource_continuation.py'),'supervisor','--stage',a.stage]
        with(d/'supervisor.stdout').open('xb')as out,(d/'supervisor.stderr').open('xb')as err:
            child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
        c.put(d/'dispatch.json',{'process':c.identity(child.pid),'parent_process':c.identity(os.getpid()),'argv':argv,'resource_registration':c.desc(c.O/'resource_continuation_registration.json')});print(json.dumps(c.desc(d/'dispatch.json')))
    elif a.role=='supervisor':
        c.put(d/'supervisor.json',{'process':c.identity(os.getpid()),'argv':sys.argv})
        try:
            c.stage_wait(d,'worker',[sys.executable,'-B',str(E/'resource_continuation.py'),'worker','--stage',a.stage],dict(os.environ),4200)
            c.put(d/'completion.json',{'status':'COMPLETE','actual_wait':c.desc(d/'worker.actual_wait.json')})
        except BaseException:c.put(d/'failure.json',{'error':traceback.format_exc(),'automatic_retry':False});raise
    else:worker(a.stage,expected)
if __name__=='__main__':main()

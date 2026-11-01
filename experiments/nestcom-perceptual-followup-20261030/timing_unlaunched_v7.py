"""First launches only after an exactly diagnosed unrelated BPG guard failure.

No old role is retried; completed groups and the BPG failure remain immutable.
"""
from followup_common import *
from launch import launch
import subprocess, traceback


def verify_failures(evidence):
    for role, item in evidence.items():
        c=L/(role+'_control'); w=cr(item['wait'])
        assert pin(c/'actual_wait.json')==item['wait'] and w['returncode']==1
        assert ident(w['child']['pid']) is None
        assert ident(read(c/'owner.json')['owner']['pid']) is None
        assert pin(L/role/'failure.json')==item['failure']
    b=cr(evidence['E5_dep5_g12_tx']['failure'])
    assert b['completed_rows']==0 and b['ledger']['unresolved']==1
    assert b['ledger']['counts']==[dict(kind='timing_trial',status='FAILED',count=1)]
    assert "88, '')" in b['traceback'] and 'bpgenc' in b['traceback']
    a=cr(evidence['E5_ARPC_IPC_queue_v2']['failure'])
    assert a['launched']==[] and "wait_external('E5_timing_recovery_v6')" in a['traceback']
    q=cr(evidence['E5_timing_recovery_v6']['failure'])
    assert q['launched'][-1]=='E5_dep5_g12_tx'
    assert all(not x.startswith(('E5_dep5_g09','E5_dep5_g13','E5_dep5_g14','E5_dep5_g15')) for x in q['launched'])


def main(request, expected):
    assert sha(request)==expected
    r=read(request); [checked(x) for x in r['code']]
    verify_failures(r['preserved_failures'])
    out=Path(r['output']); out.mkdir(exist_ok=False)
    os.sched_setaffinity(0,set(r['cpu_affinity']))
    beg=time.monotonic(); launched=[]
    def remaining():return min(r['seconds']-(time.monotonic()-beg),r['deadline_unix']-time.time())
    def check():
        if remaining()<=90 or (L/'STOP').exists():raise BudgetStop('Unchanged E5 deadline/stop boundary')
    def wait(role):
        c=L/(role+'_control')
        while not (c/'actual_wait.json').exists():check();time.sleep(3)
        w=read(c/'actual_wait.json'); assert ident(w['child']['pid']) is None
        owner=read(c/'owner.json')['owner']
        while True:
            check()
            live=ident(owner['pid'])
            if live is not None:assert live['start_ticks']==owner['start_ticks']
            try:got,status=os.waitpid(owner['pid'],os.WNOHANG)
            except ChildProcessError:
                assert live is None;break
            if got:
                save(out/('owner_wait_'+role+'.json'),dict(actual_wait=True,returncode=os.waitstatus_to_exitcode(status)))
                assert os.waitstatus_to_exitcode(status)==w['returncode'];break
            time.sleep(.1)
        assert w['returncode']==0, ('New stage failed, no retry',role)
        done=closed(role)
        assert done['rows']==160 and done['ledger']['unresolved']==0
        return done
    try:
        plan=Path(r['plan']); cfg=read(plan/'config.json'); groups=read(plan/'groups.json')
        old=cr(r['old_queue']); arpc=cr(r['old_ARPC_queue'])
        assert r['group_indices']==[13,14,15,9] and len(groups)==16
        proof=closed('E5_ARPC_IPC_preflight_v2')
        assert proof['engineering_TX']==proof['engineering_RX']==2 and proof['model_calls']==0
        assert pin(L/'E5_ARPC_IPC_preflight_v2/completion.json')==arpc['qualification']
        for gi in r['group_indices']:
            for direction in ['tx','rx']:
                role=f'E5_dep5_g{gi:02d}_{direction}'
                assert not (D/(role+'_registration.json')).exists()
                assert not (L/role).exists() and not (L/(role+'_control')).exists()
        for gi in r['group_indices']:
            group=groups[gi]; is_arpc=gi==9; base=arpc if is_arpc else old
            for direction in ['tx','rx']:
                check();role=f'E5_dep5_g{gi:02d}_{direction}'
                while True:
                    check()
                    gs=[x.split(', ') for x in subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()]
                    g=next(x for x in gs if x[0]==cfg['GPU_UUID'])
                    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
                    if int(g[1])>=60000 and int(g[2])<=50 and cfg['GPU_UUID'] not in apps:break
                    time.sleep(5)
                req=dict(code=base['worker_code'],config=pin(plan/'config.json'),group=group,
                    direction=direction,sources=pin(plan/'sources.json'),cpu_affinity=cfg['cpu_affinity'],
                    sionna_package=base['sionna_package'],device_mapping=base['device_mapping'],
                    first_launch_continuation=pin(request),training=0,scientific_quality_results=0,automatic_retry=False)
                if is_arpc:
                    req.update(qualification=base['qualification'],ARPC_scope=base['scope'],
                        model_route='original system Python/Torch2.10',PHY_route='original LDPC Python/Torch2.11',
                        transport='memory socketpair; all per-call transport inside measured boundaries')
                else:req.update(role_map=base['role_map'])
                if direction=='tx':req['expected_TX']=pin(plan/'expected_TX.json')
                else:req['received_symbols']=pin(L/f'E5_dep5_g{gi:02d}_tx/received_symbols.json')
                worker=E/('timing_end_to_end_arpc_ipc_v2.py' if is_arpc else 'timing_end_to_end_dependency_v5.py')
                python=Path('/usr/bin/python3') if is_arpc else RT/'envs/frozen_visual_py310/bin/python'
                launch(role,req,worker,python,min(43200,remaining()-60),gpu=True,required_uuid=cfg['GPU_UUID'],timing=True)
                launched.append(role);save(out/('launch_'+role+'.json'),dict(role=role,time=time.time()))
                wait(role)
                if is_arpc:
                    side=read(L/role/'phy_service_actual_wait.json')
                    assert side['actual_wait'] and side['returncode']==0 and side['final_identity'] is None
                    actual=side['report']['result'];assert actual['TX' if direction=='tx' else 'RX']==160
                    assert actual['RX' if direction=='tx' else 'TX']==0
        save(out/'completion.json',dict(status='COMPLETE',groups=r['group_indices'],launched=launched,
            full_E5_complete=False,BPG_preserved_failed=True,automatic_retry=False))
    except BaseException:
        save(out/'failure.json',dict(traceback=traceback.format_exc(),launched=launched,automatic_retry=False));raise


if __name__=='__main__':main(*sys.argv[1:])

"""Only the two unlaunched ARPC stages, after the existing timing owner closes."""
from followup_common import *
from launch import launch
import subprocess,traceback


def main(request,expected):
    assert sha(request)==expected
    r=read(request);[checked(v) for v in r['code']]
    out=Path(r['output']);out.mkdir(exist_ok=False)
    os.sched_setaffinity(0,set(r['cpu_affinity']))
    launched=[];beg=time.monotonic()
    def check():
        remaining=min(r['seconds']-(time.monotonic()-beg),r['deadline_unix']-time.time())
        if remaining<=90 or (L/'STOP').exists():raise BudgetStop('Original E5 deadline or stop boundary')
        return remaining
    def wait_external(role):
        c=L/(role+'_control')
        while not (c/'actual_wait.json').exists():
            check()
            if (c/'child.json').exists():
                child=read(c/'child.json')['child'];live=ident(child['pid'])
                if live is None:
                    time.sleep(1)
                    assert (c/'actual_wait.json').exists(), ('Vanished without actual wait',role)
                else: assert live['start_ticks']==child['start_ticks']
            time.sleep(10)
        done=closed(role)
        owner=read(c/'owner.json')['owner']
        while ident(owner['pid']) is not None:
            check();live=ident(owner['pid'])
            if live is not None:assert live['start_ticks']==owner['start_ticks']
            time.sleep(.2)
        return done
    def wait_owned(role):
        c=L/(role+'_control')
        while not (c/'actual_wait.json').exists():check();time.sleep(5)
        done=closed(role);owner=read(c/'owner.json')['owner']
        while True:
            check()
            try:got,status=os.waitpid(owner['pid'],os.WNOHANG)
            except ChildProcessError:
                assert ident(owner['pid']) is None;break
            if got:
                assert os.waitstatus_to_exitcode(status)==0;break
            time.sleep(.2)
        return done
    try:
        proof=closed('E5_ARPC_IPC_preflight_v2')
        assert proof['engineering_TX']==proof['engineering_RX']==2 and proof['model_calls']==0
        assert pin(L/'E5_ARPC_IPC_preflight_v2/completion.json')==r['qualification']
        # No GPU or ARPC stage registration while the other timing chain owns its slot.
        earlier=wait_external('E5_timing_recovery_v6')
        assert earlier['status']=='COMPLETE_AVAILABLE_GROUPS_ARPC_PENDING'
        plan=Path(r['plan']);cfg=read(plan/'config.json');group=read(plan/'groups.json')[9]
        assert group['name']=='ARPC_N1024_S10'
        for direction in ['tx','rx']:
            check();role='E5_dep5_g09_'+direction
            assert not (D/(role+'_registration.json')).exists() and not (L/role).exists()
            while True:
                check()
                gs=[x.split(', ') for x in subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()]
                g=next(x for x in gs if x[0]==cfg['GPU_UUID'])
                apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
                if int(g[1])>=60000 and int(g[2])<=50 and cfg['GPU_UUID'] not in apps:break
                time.sleep(10)
            req=dict(code=r['worker_code'],config=pin(plan/'config.json'),group=group,
                direction=direction,sources=pin(plan/'sources.json'),cpu_affinity=cfg['cpu_affinity'],
                sionna_package=r['sionna_package'],device_mapping=r['device_mapping'],
                qualification=r['qualification'],ARPC_scope=r['scope'],
                model_route='original system Python/Torch2.10',PHY_route='original LDPC Python/Torch2.11',
                transport='memory socketpair; all per-call transport inside measured boundaries',
                training=0,scientific_quality_results=0,automatic_retry=False)
            if direction=='tx':req['expected_TX']=pin(plan/'expected_TX.json')
            else:req['received_symbols']=pin(L/'E5_dep5_g09_tx/received_symbols.json')
            launch(role,req,E/'timing_end_to_end_arpc_ipc_v2.py',Path('/usr/bin/python3'),
                min(43200,check()-60),gpu=True,required_uuid=cfg['GPU_UUID'],timing=True)
            launched.append(role);save(out/('launch_'+role+'.json'),dict(role=role,time=time.time()))
            done=wait_owned(role);assert done['rows']==160 and done['ledger']['unresolved']==0
            side=read(L/role/'phy_service_actual_wait.json')
            assert side['actual_wait'] and side['returncode']==0 and side['final_identity'] is None
            actual=side['report']['result'];assert actual['TX' if direction=='tx' else 'RX']==160
            assert actual['RX' if direction=='tx' else 'TX']==0
        extra=sum(read(L/role/'phy_service_actual_wait.json')['report']['result']['CPU_process_wall_seconds'] for role in launched)
        save(out/'completion.json',dict(status='COMPLETE',launched=launched,rows=320,
            additional_PHY_service_CPU_process_wall_seconds=extra,
            shared_two_core_affinity=True,transport_included_in_timing=True,automatic_retry=False))
    except BaseException:
        save(out/'failure.json',dict(traceback=traceback.format_exc(),launched=launched,automatic_retry=False));raise


if __name__=='__main__':main(*sys.argv[1:])

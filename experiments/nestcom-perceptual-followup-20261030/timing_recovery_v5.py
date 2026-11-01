"""Diagnosed zero-trial initialization correction; finite first measurement roles.

The old failed roles, registrations, and elapsed windows remain immutable.
This independently registered corrective phase uses the original study's
remaining total resources. Unknown failures stop it without automatic retry.
"""
from followup_common import *
from launch import launch
import traceback, subprocess


def verify_old_failure(evidence):
    for role in ['E5_g00_tx', 'E5_timing_queue']:
        c = L / (role + '_control')
        o, ch, w = read(c / 'owner.json'), read(c / 'child.json'), read(c / 'actual_wait.json')
        assert w['returncode'] == 1
        assert ident(o['owner']['pid']) is None and ident(ch['child']['pid']) is None
        assert pin(L / role / 'failure.json') == evidence[role]
        assert not (L / role / 'completion.json').exists()
    f = cr(evidence['E5_g00_tx'])
    assert f['completed_rows'] == 0 and f['ledger']['counts'] == [] and f['ledger']['unresolved'] == 0
    assert "NameError: name 'sys' is not defined" in f['traceback']
    assert cr(evidence['E5_timing_queue'])['launched'] == ['E5_g00_tx']


def main(request, expected):
    assert sha(request) == expected
    r = read(request)
    [checked(p) for p in r['code']]
    verify_old_failure(r['old_failure_evidence'])
    for role in ['E5_ns2_g00_tx', 'E5_timing_recovery_v2']:
        c = L/(role+'_control'); w = read(c/'actual_wait.json')
        assert w['returncode'] == 1 and ident(w['child']['pid']) is None
        assert ident(read(c/'owner.json')['owner']['pid']) is None
        assert pin(L/role/'failure.json') == r['dependency_failure_evidence'][role]
    failed = cr(r['dependency_failure_evidence']['E5_ns2_g00_tx'])
    assert failed['completed_rows'] == 0 and failed['ledger']['counts'] == [] and failed['ledger']['unresolved'] == 0
    assert "ModuleNotFoundError: No module named 'sionna'" in failed['traceback']
    assert cr(r['dependency_failure_evidence']['E5_timing_recovery_v2'])['launched'] == ['E5_ns2_g00_tx']
    for role in ['E5_dep4_g00_tx', 'E5_timing_recovery_v4']:
        c = L/(role+'_control'); w = read(c/'actual_wait.json')
        assert w['returncode'] == 1 and ident(w['child']['pid']) is None
        assert ident(read(c/'owner.json')['owner']['pid']) is None
        assert pin(L/role/'failure.json') == r['device_failure_evidence'][role]
    f = cr(r['device_failure_evidence']['E5_dep4_g00_tx'])
    assert f['completed_rows'] == 0 and f['ledger']['counts'] == [] and f['ledger']['unresolved'] == 0
    assert "get_device_properties(0).uuid" in f['traceback']
    assert cr(r['device_failure_evidence']['E5_timing_recovery_v4'])['launched'] == ['E5_dep4_g00_tx']
    out = Path(r['output']); out.mkdir(exist_ok=False)
    os.sched_setaffinity(0, set(r['cpu_affinity']))
    start = time.monotonic(); launched = []

    def remaining():
        return min(r['seconds'] - (time.monotonic() - start), r['deadline_unix'] - time.time())

    def check():
        if remaining() <= 90 or (L / 'STOP').exists():
            raise BudgetStop('Corrective timing phase boundary; old deadlines remain unchanged')

    def wait(role):
        c = L / (role + '_control')
        while not (c / 'actual_wait.json').exists():
            check()
            if (c / 'child.json').exists():
                ch = read(c / 'child.json'); live = ident(ch['child']['pid'])
                if live is None:
                    time.sleep(1)
                    assert (c / 'actual_wait.json').exists(), ('Vanished without actual wait', role)
                else:
                    assert live['start_ticks'] == ch['child']['start_ticks']
            time.sleep(10)
        closed(role)
        o = read(c / 'owner.json')
        # The supervisor writes actual_wait immediately before its own exit.
        for _ in range(30):
            if ident(o['owner']['pid']) is None:
                break
            time.sleep(.1)
        assert ident(o['owner']['pid']) is None

    try:
        closed('completion_followup_queue')
        qualification = closed('E5_dependency_preflight_v4')
        assert qualification['runtime_constructors_qualified'] and len(qualification['routes']) == 2
        assert all(v['status'] == 'PASS' and v['qualified_N1024_profiles'] == 360 and v['matched_N512_layouts'] == 12 for v in qualification['routes'])
        assert qualification['model_calls'] == qualification['packet_calls'] == qualification['timing_calls'] == 0
        assert pin(L/'E5_dependency_preflight_v4/completion.json') == r['CPU_qualification']
        plan = Path(r['plan']); cfg = read(plan / 'config.json'); groups = read(plan / 'groups.json')
        mapping = cr(r['role_map']); assert len(groups) == 16 and len(mapping) == 32
        assert len({v['role'] for v in mapping}) == 32
        for p in L.glob('*_control/owner.json'):
            reg = cr(read(p)['registration'])
            if reg['gpu']:
                w = read(p.parent / 'actual_wait.json')
                assert ident(w['child']['pid']) is None
        for gi, group in enumerate(groups):
            if group['method'] == 'ARPC':
                save(out/'blocked_ARPC.json', dict(group_index=gi, group=group['name'], reason='Original ARPC system-Python numerical route requires its own qualified PHY integration', registrations=0, model_calls=0, timing_calls=0))
                continue
            for direction in ['tx', 'rx']:
                check()
                role = next(v['role'] for v in mapping if v['group_index'] == gi and v['direction'] == direction)
                req = dict(code=r['worker_code'], config=pin(plan / 'config.json'), group=group,
                           direction=direction, sources=pin(plan / 'sources.json'),
                           cpu_affinity=cfg['cpu_affinity'], training=0, scientific_quality_results=0,
                           corrective_phase=pin(request), role_map=r['role_map'], sionna_package=r['sionna_package'], device_mapping=r['device_mapping'])
                if direction == 'tx':
                    req['expected_TX'] = pin(plan / 'expected_TX.json')
                else:
                    tx_role = next(v['role'] for v in mapping if v['group_index'] == gi and v['direction'] == 'tx')
                    req['received_symbols'] = pin(L / tx_role / 'received_symbols.json')
                while True:
                    check()
                    gs = [v.split(', ') for v in subprocess.check_output(
                        ['nvidia-smi', '--query-gpu=uuid,memory.free,utilization.gpu', '--format=csv,noheader,nounits'], text=True).splitlines()]
                    g = next(v for v in gs if v[0] == cfg['GPU_UUID'])
                    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'], text=True)
                    if int(g[1]) >= 60000 and int(g[2]) <= 50 and cfg['GPU_UUID'] not in apps:
                        break
                    time.sleep(15)
                python = (RT / 'envs/frozen_ldpc_py311/bin/python' if group['method'] == 'BPG_LDPC'
                          else Path('/usr/bin/python3') if group['method'] == 'ARPC'
                          else RT / 'envs/frozen_visual_py310/bin/python')
                launch(role, req, E / 'timing_end_to_end_dependency_v5.py', python,
                       min(43200, remaining() - 60), gpu=True, required_uuid=cfg['GPU_UUID'], timing=True)
                launched.append(role)
                save(out / ('launch_' + role + '.json'), dict(role=role, time=time.time()))
                wait(role)
        save(out/'completion.json', dict(status='COMPLETE_AVAILABLE_GROUPS_ARPC_PENDING', launched=launched, groups=15,
             actual_wait0_for_launched=True, original_failures_preserved=True, seconds=time.monotonic()-start,
             full_E5_complete=False, blocked_group_indices=[9], new_training=0, automatic_retry=False))
    except BaseException:
        save(out / 'failure.json', dict(traceback=traceback.format_exc(), launched=launched, automatic_retry=False))
        raise


if __name__ == '__main__':
    main(*sys.argv[1:])

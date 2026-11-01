"""Bounded CPU construction of real qualified PHY in the existing numerical routes."""
from followup_common import *
import subprocess, traceback
from timing_dependency_v4 import activate


def probe(request, route):
    r = read(request); cfg = cr(r['config'])
    os.sched_setaffinity(0, set(r['cpu_affinity']))
    if route == 'ARPC':
        sys.path.insert(0, cfg['arpc_public']['environment'])
    activate(r['sionna_package'])
    import torch
    torch.set_num_threads(6); torch.set_num_interop_threads(2)
    sys.path[:0] = [str(R/'experiments/adatoken-response-20261026'),
                   str(R/'experiments/nestcom-n512-20261028')]
    from timing_phy_namespace_v2 import isolated_engine_with_explicit_sys
    import c_predclass_frame as cp
    phy = isolated_engine_with_explicit_sys(cp.isolated_engine, R)
    ns = phy.pack_body.__globals__
    ns['FAMILIES'] = (*ns['FAMILIES'][:-2], cp.FAMILY, cp.NULL)
    runtime = phy.create_runtime(R, str(checked(cfg['phy_qualification'])), configure_threads=False)
    assert runtime.qualified and len(runtime.profiles) == 360
    from n512_phy import engine
    n5, _ = engine(); nr = n5.Runtime(R, configure_threads=False)
    cat = cr(r['n512_catalogue'])
    layouts = {}
    for p in cat['profiles']:
        key = p['k'], p['n'], p['q']
        if key not in layouts:
            layouts[key] = nr.backend.plan(*key)
            assert layouts[key] == p['layout']
    print(json.dumps(dict(route=route, status='PASS', torch=torch.__version__, cuda=torch.version.cuda,
         sionna_version='2.2.0', qualified_N1024_profiles=360, matched_N512_layouts=len(layouts),
         backend_identity=runtime.backend.identity, model_calls=0, packet_calls=0, timing_calls=0)))


def main(request, expected):
    assert sha(request) == expected
    r = read(request); [checked(p) for p in r['code']]
    out = Path(r['output']); out.mkdir(exist_ok=False)
    os.sched_setaffinity(0, set(r['cpu_affinity']))
    began = time.monotonic(); reports = []
    try:
        for route, python in [('visual', RT/'envs/frozen_visual_py310/bin/python'),
                              ('LDPC', RT/'envs/frozen_ldpc_py311/bin/python')]:
            argv = [str(python), '-B', __file__, '--probe', request, route]
            q = subprocess.run(argv, text=True, capture_output=True, timeout=180)
            save(out/(route+'.json'), dict(returncode=q.returncode, stdout=q.stdout, stderr=q.stderr))
            assert q.returncode == 0, (route, q.stderr[-4000:])
            reports.append(json.loads(q.stdout.strip().splitlines()[-1]))
        save(out/'completion.json', dict(status='COMPLETE', routes=reports, model_calls=0,
             packet_calls=0, timing_calls=0, runtime_constructors_qualified=True,
             seconds=time.monotonic()-began))
    except BaseException:
        save(out/'failure.json', dict(traceback=traceback.format_exc(), completed_routes=reports,
             model_calls=0, packet_calls=0, timing_calls=0, automatic_retry=False))
        raise


if __name__ == '__main__':
    if sys.argv[1] == '--probe': probe(*sys.argv[2:])
    else: main(*sys.argv[1:])

"""Explicit single owner for frozen post-hoc common500 source preparation."""
import argparse,json,os,signal,time,traceback
from pathlib import Path
from t1_entropy_core import read,require,sha,verify
import t2_pilot as shared
from t1_holdout_sources import run_owned

def run(args):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='0','Explicit admittedGPU0 required')
    request=read(args.request);env=read(args.environment_request);root=Path(request['root']);out=Path(args.out).resolve()
    require(out.is_relative_to(root/'outputs/WCL-EVIDENCE-CLOSURE-20261009') and not out.exists(),'Fresh WCL source-owner output required')
    require(request['schema']=='T1_POSTHOC_COMMON500_SOURCE_REQUEST_V1','Frozen source request required')
    policy=read(request['freeze']['path']);verify(request['freeze']['path'],request['freeze']['sha256'])
    require(policy['status']=='T1_POLICIES_FROZEN_CALIBRATION_ONLY_V1','Calibrated policy freeze required')
    out.mkdir(parents=True);started=time.monotonic();native=None;event=0
    counts={k:dict(reserved=0,completed=0) for k in ('static_source_tx','static_source_rx','var_source_tx','var_source_rx')}
    caps=dict(static_source_tx=500,var_source_tx=500,static_source_rx=1500,var_source_rx=1500)
    conf=dict(deadline_unix=args.deadline_unix,max_seconds=args.max_seconds,stop_files=[str(root/'STOP'),str(out/'STOP')])
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,shared.stop)
    def boundary():
        shared.guard(conf,started)
        if native is not None:native.health_polling.next_check=0.;native.common.check()
    def account(kind,operation,**context):
        nonlocal event
        boundary();require(kind in caps and counts[kind]['reserved']<caps[kind],'Source call category/cap exceeded')
        event+=1;counts[kind]['reserved']+=1;p=out/'calls'/f'{event:05d}_{kind}'
        shared.save(p.with_suffix('.reserved.json'),dict(kind=kind,context=context,started_unix=time.time(),pid=os.getpid()))
        began=time.monotonic();value=operation();counts[kind]['completed']+=1
        shared.save(p.with_suffix('.complete.json'),dict(kind=kind,context=context,seconds=time.monotonic()-began))
        return value
    modules=['t1_holdout_source_owner.py','t1_holdout_sources.py','t1_holdout_metadata.py','t1_entropy_core.py','t1_codec_runtime.py','t1_source_asset_schema.py','t2_pilot.py']
    bindings={str(Path(__file__).with_name(n).resolve()):sha(Path(__file__).with_name(n)) for n in modules}
    shared.save(out/'owner_request.json',dict(request=shared.desc(args.request),environment=shared.desc(args.environment_request),
        caps=caps,source_bindings=bindings,freeze=request['freeze'],no_auto_retry=True,**conf))
    try:
        with shared.lock(env['visual_config']['visual_lock']),shared.lock(out/'gpu.lock'):
            boundary();shared.save(out/'owner_started.json',dict(pid=os.getpid(),time=time.time()))
            native,_,_=shared.gpu_build(env,out)
            result=run_owned(request=args.request,out=out/'sources500',native=native,boundary=boundary,account=account)
            require(all(c['reserved']==c['completed'] for c in counts.values()),'Unresolved source call')
            done=dict(status='T1_POSTHOC_SOURCE_OWNER_COMPLETE',source_completion=shared.desc(out/'sources500'/'completion.json'),
                counts=counts,seconds=time.monotonic()-started,new_packet_decodes=0,new_image_renders=0,new_metric_calls=0)
            shared.save(out/'completion.json',done);return done
    except BaseException:
        shared.save(out/'owner_failed.json',dict(counts=counts,traceback=traceback.format_exc(),no_auto_retry=True));raise
    finally:
        shared.save(out/'owner_exited.json',dict(pid=os.getpid(),counts=counts,complete=(out/'completion.json').exists(),time=time.time()))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('request','environment-request','out'):p.add_argument('--'+n,required=True)
    p.add_argument('--deadline-unix',type=float,required=True);p.add_argument('--max-seconds',type=float,default=86400)
    args=p.parse_args();require(0<args.max_seconds<=86400,'Finite source window required');print(json.dumps(run(args)))

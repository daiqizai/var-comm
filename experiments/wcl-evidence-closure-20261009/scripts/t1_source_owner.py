"""One explicitly launched, locked T1 source GPU window then CPU source sealing.

No SSH, subprocess scheduler, training, channel simulation, or quality selection.
Existing output directories are refused, preserving unresolved reservations.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import time
import traceback
from types import SimpleNamespace
from t1_entropy_core import read,require,sha,verify
import t2_pilot as shared
import t1_owned_source_gate as gate
import t1_owned_shortprefixes as short
import t1_merge_sources as merge
from t1_reference_cache import RawCalibrationReference


class CallLedger:
    """Immutable attempt reservations, written before every scientific call."""
    def __init__(self,out,caps,boundary):
        self.out=Path(out);self.caps=caps;self.boundary=boundary
        self.reserved={k:0 for k in caps};self.completed={k:0 for k in caps};self.events=0
    def __call__(self,kind,operation,**context):
        self.boundary();require(kind in self.caps,'Unregistered call category')
        require(self.reserved[kind]<self.caps[kind],'Registered call budget exhausted: '+kind)
        self.events+=1;self.reserved[kind]+=1
        path=self.out/f'{self.events:05d}_{kind}'
        started=time.monotonic()
        shared.save(path.with_suffix('.reserved.json'),dict(kind=kind,context=context,
            started_unix=time.time(),pid=os.getpid(),event=self.events))
        value=operation()
        self.completed[kind]+=1
        shared.save(path.with_suffix('.complete.json'),dict(kind=kind,context=context,event=self.events,
            elapsed_seconds=time.monotonic()-started,completed_unix=time.time()))
        return value
    def summary(self):
        return dict(caps=self.caps,reserved=self.reserved,completed=self.completed,events=self.events,
            unresolved=sum(self.reserved.values())-sum(self.completed.values()),
            new_VAR_prior_scale_evaluations_completed=5*self.completed['var_source_tx']+
                # The per-event m records retain the exact m4/m5 split on failure.
                sum(read(p)['context']['m'] for p in self.out.glob('*_var_source_rx.complete.json')),
            new_image_renders_completed=self.completed['image_render'])


def source_budget(source_completion):
    done=read(source_completion);base=Path(source_completion).parent;mp=base/'manifest.json'
    require(done['status']=='T1_CALIBRATION_SOURCE_CPU_PREPARATION_COMPLETE','CPU source preparation must finish first')
    verify(mp,done['outputs'][str(mp)]);manifest=read(mp)
    require(manifest['source_count']==1000 and len(manifest['records'])==1000,'Original full calibration1000 required')
    require([r['source_index'] for r in manifest['records']]==list(range(1000)),'Original source order required')
    needed=[r['source_index'] for r in manifest['records'] if int(r['families']['EC_VAR_WHOLE']['6'])>927]
    return dict(var_source_tx=len(needed),var_source_rx=2*len(needed),static_source_rx=96,image_render=192),needed


def run(args):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='0','Explicit admitted GPU0 environment required')
    root=Path(args.root).resolve();out=Path(args.out).resolve()
    require(out.is_relative_to(root/'outputs/WCL-EVIDENCE-CLOSURE-20261009'),'Separate WCL output namespace required')
    require(not out.exists(),'Existing source owner output must be inspected, never relaunched automatically')
    environment=read(args.environment_request)
    require(environment['root']==str(root),'Environment belongs to a different project')
    caps,needed=source_budget(args.source_population_completion)
    begun=time.monotonic();config=dict(deadline_unix=args.deadline_unix,max_seconds=args.max_seconds,
        stop_files=[str(root/'STOP'),str(out/'STOP')])
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,shared.stop)
    def guard():shared.guard(config,begun)
    guard();out.mkdir(parents=True)
    files=['t1_source_owner.py','t1_owned_source_gate.py','t1_owned_shortprefixes.py','t1_reference_cache.py',
        't1_merge_sources.py','t1_entropy_core.py','t1_codec_runtime.py','t1_source_asset_schema.py','t1_source_check.py','t2_pilot.py']
    bindings={str(Path(__file__).with_name(n).resolve()):sha(Path(__file__).with_name(n)) for n in files}
    inputs={str(Path(getattr(args,k)).resolve()):sha(getattr(args,k)) for k in (
        'environment_request','calibration','h_completion','static_completion','source32_completion','source_population_completion')}
    request=dict(status='REGISTERED_SOURCE_OWNER_WINDOW',pid=os.getpid(),root=str(root),out=str(out),
        input_bindings=inputs,source_bindings=bindings,caps=caps,short_prefix_source_indices=needed,
        required_VAR_short_prefix_sources=len(needed),no_source_quality_selection=True,
        prior_scale_evaluation_upper_bound=14*len(needed),image_render_upper_bound=192,
        new_packet_decodes=0,training_updates=0,holdout_used=False,**config)
    shared.save(out/'request.json',request)
    native=None;ledger=None;stage='waiting_for_owner_locks'
    try:
        with shared.lock(environment['visual_config']['visual_lock']),shared.lock(out/'gpu.lock'):
            guard();stage='model_loading'
            for p,h in {**inputs,**bindings}.items():verify(p,h)
            shared.save(out/'owner_started.json',dict(pid=os.getpid(),started_unix=time.time(),request_sha256=sha(out/'request.json')))
            native,_,_=shared.gpu_build(environment,out)
            def boundary():
                guard();native.health_polling.next_check=0.;native.common.check()
            ledger=CallLedger(out/'calls',caps,boundary)
            reference=RawCalibrationReference(args.environment_request,native)
            stage='source32_gate';boundary()
            gate_done=gate.run_owned(root=root,calibration=args.calibration,h_completion=args.h_completion,
                static_completion=args.static_completion,source32_completion=args.source32_completion,
                out=out/'source32_gate',native=native,boundary=boundary,reference_lookup=reference,account=ledger)
            stage='calibration_short_prefixes';boundary()
            short_done=short.run_owned(root=root,source_population_completion=args.source_population_completion,
                static_completion=args.static_completion,out=out/'calibration_short_prefixes',native=native,
                boundary=boundary,existing_fallback_manifests=[out/'source32_gate'/'fallback_manifest.json'],account=ledger)
            stage='cpu_source_merge';boundary()
            merged=merge.run(SimpleNamespace(source_completion=args.source_population_completion,
                fallback_completion=[str(out/'calibration_short_prefixes'/'completion.json')],out=str(out/'source1000_sealed')))
            native.frozen();guard()
            counts=ledger.summary()
            require(counts['unresolved']==0 and counts['completed']['var_source_tx']==len(needed)
                and counts['completed']['var_source_rx']==2*len(needed) and counts['completed']['static_source_rx']==96,
                'Actual source call accounting differs from registered needed population')
            result=dict(status='T1_SOURCE_OWNER_COMPLETE',pid=os.getpid(),counts=counts,
                gate=shared.desc(out/'source32_gate'/'completion.json'),
                short_prefixes=shared.desc(out/'calibration_short_prefixes'/'completion.json'),
                sealed_sources=shared.desc(out/'source1000_sealed'/'completion.json'),
                source_visual_gate_status=gate_done['status'],elapsed_seconds=time.monotonic()-begun,
                new_packet_decodes=0,training_updates=0,holdout_used=False)
            shared.save(out/'completion.json',result);return result
    except BaseException as error:
        shared.save(out/'owner_failed.json',dict(status='T1_SOURCE_OWNER_STOPPED_OR_FAILED',pid=os.getpid(),stage=stage,
            exception_type=type(error).__name__,message=str(error),traceback=traceback.format_exc(),
            counts=None if ledger is None else ledger.summary(),elapsed_seconds=time.monotonic()-begun,
            rerun_forbidden_without_inspecting_reserved_calls=True))
        raise
    finally:
        shared.save(out/'owner_exited.json',dict(pid=os.getpid(),exited_unix=time.time(),last_stage=stage,
            completion_present=(out/'completion.json').exists(),counts=None if ledger is None else ledger.summary()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','environment-request','calibration','h-completion','static-completion',
                 'source32-completion','source-population-completion','out'):p.add_argument('--'+name,required=True)
    p.add_argument('--deadline-unix',type=float,required=True);p.add_argument('--max-seconds',type=float,default=86400)
    args=p.parse_args();require(0<args.max_seconds<=86400,'Explicit finite at-most24h source window')
    print(json.dumps(run(args),sort_keys=True))

if __name__=='__main__':main()

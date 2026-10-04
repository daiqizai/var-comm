"""Original M1 batch-one scientific runner over an exclusive source assignment.

Benchmark recomputes complete 2070-row sources in isolated scratch and requires
exact row identity with the original run. Production keeps its original source
checkpoint schema/binding and never writes aggregate policies or development.
"""
from __future__ import annotations
import argparse
import importlib
import os
from pathlib import Path
import signal
import sys
import time
import traceback
import cohort_runtime as c

STOP=False


def stopped(*_):
    global STOP
    STOP=True


def validate_config(config):
    required=('root','original_out','original_code','protocol','cohort_manifest','worker_id','source_indices',
        'stage','scratch_out','output_dir','stop_file','source_bindings','expected_calibration_registration_sha256')
    c.require(all(key in config for key in required),'Incomplete source-worker registration')
    c.require(config['stage'] in ('benchmark','production'),'Only whole-source benchmark/production is authorized')
    indices=config['source_indices'];c.require(isinstance(indices,list) and indices and len(set(indices))==len(indices)
        and all(type(i)is int and 0<=i<1000 for i in indices),'Invalid exclusive source assignment')
    old=Path(config['original_out']).resolve();scratch=Path(config['scratch_out']).resolve()
    c.require(old!=scratch and old not in scratch.parents,'Scratch must be separate from original scientific output')
    reg=old/'calibrate_registration.json'
    c.require(c.sha(reg)==config['expected_calibration_registration_sha256'],'Original calibration registration changed')
    c.verify_bindings(config['source_bindings'])
    return config


def configure_runner(config,cohort):
    """Import the original modules after cohort admission; patch only this process."""
    root=Path(config['root']).resolve();code=Path(config['original_code']).resolve()
    sys.path.insert(0,str(code));base=importlib.import_module('m1_run');u=importlib.import_module('m1_common')
    c.require(Path(base.__file__).resolve()==code/'m1_run.py' and Path(u.__file__).resolve()==code/'m1_common.py',
        'Original M1 module alias points elsewhere')

    class SourceRunner(base.Runner):
        def __init__(self):
            super().__init__(root,config['protocol'],config['original_out'])
            self.stage='calibrate';self.worker_output=Path(config['output_dir']).resolve()
        def guard(self):
            if STOP or self.stop or Path(config['stop_file']).exists():
                raise base.PauseRequested('Whole-source cohort pause requested')
            # Keep the original run's explicit stop file and immutable checks.
            if (Path(config['original_out'])/'STOP').exists():
                raise base.PauseRequested('Original M1 stop file is present')
        def status(self,status,**fields):
            c.write(self.worker_output/'status.json',dict(status=status,worker_id=config['worker_id'],
                stage=config['stage'],pid=os.getpid(),time=time.time(),**fields))
        def load(self):
            self.guard()
            if self.native is None:
                self.status('LOADING_FROZEN_MODELS')
                old=root/'experiments/rx-posterior-step1-20260929';sys.path.insert(0,str(old))
                environment=importlib.import_module('run_preflight')
                c.require(Path(environment.__file__).resolve()==old/'run_preflight.py','Original preflight module differs')
                original=importlib.import_module('rx_v3_common');probe=original.b
                runtime=importlib.import_module('latent_enhancement.runtime')
                cohort.install(probe,runtime)
                native_module=importlib.import_module('m1_native');performance=importlib.import_module('m1_performance')
                c.require(Path(native_module.__file__).resolve()==code/'m1_native.py'
                    and Path(performance.__file__).resolve()==code/'m1_performance.py','Original native/performance module differs')
                self.native=native_module.Native(root,self.stopped)
                c.require(self.native.assets.old.b is probe and probe.SAFETY is not None,
                    'Native did not use the admitted original safety instance')
                probe.SAFETY=performance.HealthyPolling(probe.SAFETY,interval=5.)
                self.native.health_polling=probe.SAFETY;self.native.timings['health_query']=probe.SAFETY.timing
            return self.native
    return SourceRunner(),u,base


def row_digest(saved,u):
    c.require(saved.get('synthetic') is False and saved.get('development_read') is False and len(saved['rows'])==2070,
        'Only complete original real calibration sources are admissible')
    return u.identity(saved['rows'])


def source_once(runner,u,config,binding,data,actions,index):
    """Lock one complete source; no partial frame or alternate numerical path."""
    original=Path(config['original_out']).resolve();scratch=Path(config['scratch_out']).resolve()
    lock_root=(original/'calibrate/source_locks' if config['stage']=='production' else scratch/'source_locks')
    reference=original/'calibrate/source_checkpoints'/('%04d.json'%index)
    with c.source_lock(lock_root,index):
        runner.guard();c.verify_bindings(config['source_bindings'])
        c.require(c.sha(original/'calibrate_registration.json')==config['expected_calibration_registration_sha256'],
            'Original calibration registration changed during worker execution')
        if config['stage']=='benchmark':
            c.require(reference.exists(),'Benchmark requires a completed original reference source')
            expected=u.checkpoint(reference,binding,index);reference_sha=c.sha(reference)
            prior_out=runner.out
            try:
                runner.out=scratch
                # A fresh scratch target is mandatory; never time an existing cache.
                path=scratch/'cohort_benchmark/source_checkpoints'/('%04d.json'%index)
                c.require(not path.exists(),'Benchmark scratch source already exists; do not time cache reuse')
                saved,path=runner.calibration_source('cohort_benchmark',binding,data,index,actions)
            finally:runner.out=prior_out
            c.require(c.sha(reference)==reference_sha and row_digest(saved,u)==row_digest(expected,u),
                'Complete 2070-row benchmark differs from the original source')
        else:
            saved,path=runner.calibration_source('calibrate',binding,data,index,actions)
        runner.validate_calibration_rows(saved['rows'],data,index,actions)
        return saved,path,row_digest(saved,u)


def run(config):
    config=validate_config(config);output=Path(config['output_dir']);output.mkdir(parents=True,exist_ok=True)
    c.require(not (output/'completion.json').exists(),'Worker output already completed; owner must verify rather than rerun')
    cohort=c.await_cohort(config,stopped=lambda:STOP)
    runner,u,base=configure_runner(config,cohort)
    signal.signal(signal.SIGTERM,runner.stopped);signal.signal(signal.SIGINT,runner.stopped)
    native=runner.load();data=native.data('calibration');actions=native.actions()
    c.require(len(actions)==138 and len(data['records'])==1000,'Original full calibration/action grid changed')
    binding,registration=runner.stage_registration('calibrate',data,
        dict(action_grid=[u.action_dict(a) for a in actions],noise_seeds=u.CAL_SEEDS,
             source_count=1000,full_grid=True,shortlist_used=False))
    c.require(u.sha(registration)==config['expected_calibration_registration_sha256'],
        'Cohort model/runtime/population differ from original calibration registration')
    outputs={};digests={};seconds={};details=[]
    native.torch.cuda.synchronize();compute_started_unix=time.time();began=time.monotonic()
    for index in config['source_indices']:
        runner.guard();native.health_polling.next_check=0.;native.common.check()
        original_path=Path(config['original_out'])/'calibrate/source_checkpoints'/('%04d.json'%index)
        reused=config['stage']=='production' and original_path.exists()
        saved,path,digest=source_once(runner,u,config,binding,data,actions,index)
        outputs[str(path)]=u.sha(path);digests[str(index)]=digest;seconds[str(index)]=float(saved['seconds'])
        lineage=output/'source_lineage'/('%04d.json'%index)
        c.write(lineage,dict(status='M1_SOURCE_EXECUTOR_LINEAGE',execution_revision='independent-source-gpu-cohort-v1',
            worker_id=config['worker_id'],pid=os.getpid(),source_index=index,stage=config['stage'],
            generated_by_this_worker=not reused,checkpoint=str(path),checkpoint_sha256=u.sha(path),
            scientific_rows_sha256=digest,calibration_binding=binding,
            calibration_registration_sha256=config['expected_calibration_registration_sha256'],
            original_registration_sha256=runner.registration_sha256,worker_config_sha256=c.identity(config),
            cohort_manifest_sha256=cohort.manifest_sha256,source_bindings=config['source_bindings'],
            scientific_checkpoint_format_changed=False,numerical_runtime=native.flags,time=time.time()))
        outputs[str(lineage)]=c.sha(lineage)
        details.append({k:saved[k] for k in ('source_index','seconds','unique_receiver_events','unique_quality_images',
            'unique_tx_waveforms','unique_tx_prefix_logits','timing')})
        runner.status('RUNNING',completed_sources=len(details),assigned_sources=len(config['source_indices']),
            source_index=index,source_seconds=saved['seconds'],rows_sha256=digest)
    native.torch.cuda.synchronize();steady_seconds=time.monotonic()-began;compute_finished_unix=time.time()
    native.frozen();c.verify_bindings(config['source_bindings']);cohort.require_available()
    value=dict(status='M1_SOURCE_WORKER_COMPLETE',worker_id=config['worker_id'],stage=config['stage'],pid=os.getpid(),
        source_indices=config['source_indices'],physical_frames=2070*len(details),rows_sha256_by_source=digests,
        per_source_seconds=seconds,per_source=details,steady_seconds=steady_seconds,
        compute_started_unix=compute_started_unix,compute_finished_unix=compute_finished_unix,
        peak_cuda_reserved_bytes=int(native.torch.cuda.max_memory_reserved()),
        peak_cuda_allocated_bytes=int(native.torch.cuda.max_memory_allocated()),
        registration_sha256=u.sha(registration),calibration_binding=binding,original_registration_sha256=runner.registration_sha256,
        cohort_manifest_sha256=cohort.manifest_sha256,source_bindings=config['source_bindings'],outputs=outputs,
        complete_row_parity_verified=config['stage']=='benchmark',numerical_runtime=native.flags,
        batch_size=1,training_updates=0,development_read=False,synthetic=False)
    c.write(output/'completion.json',value);runner.status('COMPLETE',sources=len(details),steady_seconds=steady_seconds)
    return value


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);args=parser.parse_args()
    config=c.read(args.config);signal.signal(signal.SIGTERM,stopped);signal.signal(signal.SIGINT,stopped)
    try:run(config)
    except BaseException as error:
        paused=STOP or error.__class__.__name__ in ('PauseRequested','ResourceBusy')
        folder=Path(config['output_dir']);c.write(folder/('pause.json' if paused else 'failure.json'),
            dict(status='PAUSED' if paused else 'FAILED',error=repr(error),traceback=traceback.format_exc(),
                worker_id=config.get('worker_id'),pid=os.getpid(),time=time.time(),automatic_retry=False))
        if paused:raise SystemExit(75)
        raise


if __name__=='__main__':main()

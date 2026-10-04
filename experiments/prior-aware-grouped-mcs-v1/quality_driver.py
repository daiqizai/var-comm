"""Frozen calibration source-Q driver; launched explicitly by the study owner.

No model training, development access, policy selection, or background launch.
The original GPU admission/thermal policy is retained by the native factory.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import gc
import importlib.util
import math
import os
from pathlib import Path
import signal
import sys
import time
import numpy as np
import source_quality as q

STOP = False


def stop(*_):
    global STOP
    STOP = True


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


@contextmanager
def single_writer(output):
    """Kernel lock releases on abnormal process exit; no stale-PID override."""
    import fcntl
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    with (output/"quality.lock").open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another source-quality writer owns this output") from None
        handle.seek(0); handle.truncate(); handle.write(str(os.getpid())+"\n"); handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def verify_repeated_interop(torch):
    """The two original loaders both set interop threads; only same-value reuse."""
    original = torch.set_num_interop_threads
    def same(value):
        q.require(type(value) is int and value == torch.get_num_interop_threads(),
                  "The metric loader requests different interop threads")
    torch.set_num_interop_threads = same
    try:
        yield
    finally:
        torch.set_num_interop_threads = original


def build_native(root, runtime, signal_handler=stop):
    runtime = Path(runtime).resolve()
    sys.path.insert(0, str(runtime))
    common = module("m1_common", runtime/"m1_common.py")
    implementation = module("_uep_source_quality_native", runtime/"m1_native.py")
    native = implementation.Native(root, signal_handler)
    performance = module("_uep_source_quality_polling", runtime/"m1_performance.py")
    probe = native.assets.old.b
    q.require(probe.SAFETY is not None, "Original safety monitor is absent")
    probe.SAFETY = performance.HealthyPolling(probe.SAFETY)
    native.health_polling = probe.SAFETY
    native.driver_bindings = {str(p): q.sha(p) for p in runtime.glob("m1_*.py")}
    native.driver_bindings[str(Path(__file__).resolve())] = q.sha(__file__)
    native.driver_bindings[str(Path(q.__file__).resolve())] = q.sha(q.__file__)
    native.driver_bindings.update(native.common.source_bindings())
    native.common.verify_bindings(native.driver_bindings)
    return native


def boundary(native):
    # A real check is forced at each source; tight model calls retain the
    # independently qualified five-second healthy-device polling semantics.
    native.health_polling.next_check = 0.
    native.common.check()
    return STOP


def batch8_qualification(evaluator, metric_module, torch, device, manifest, output, context):
    """Check fixed B8 and every permitted tail size against B1, without search."""
    import batch_speed
    device = torch.device(device)
    q.require(device.type == "cuda" and torch.cuda.is_available(), "Native CUDA metric qualification required")
    device_index = torch.cuda.current_device() if device.index is None else device.index
    properties = torch.cuda.get_device_properties(device_index)
    binding = dict(evaluator=evaluator.identity(), modelmanifest_sha256=q.sha(manifest), context=context,
        driver_sha256=q.sha(__file__), original_batch_helper_sha256=q.sha(batch_speed.__file__),
        metric_source_sha256=q.sha(metric_module.__file__), hardware=dict(name=properties.name,
            total_memory_bytes=properties.total_memory, compute_capability=[properties.major, properties.minor]))
    path = Path(output)/"metric_batch8_qualification.json"
    if path.exists():
        receipt = q.read(path)
        q.require(receipt["binding"] == binding and receipt["status"] == "FIXED_BATCH8_QUALIFICATION_PASS"
                  and receipt["payload_sha256"] == q.identity({k:v for k,v in receipt.items() if k!='payload_sha256'}),
                  "Fixed batch8 metric qualification is stale")
        return receipt
    checks=[]
    with torch.random.fork_rng(devices=[device_index]), metric_module.no_network(), torch.no_grad():
        source, images = batch_speed.synthetic_fixtures(torch)
        prepared = evaluator.prepare_reference(source)
        truth = int(prepared["resnet50"][0])
        scalar=[]
        for index in range(len(images)):
            scalar.extend(evaluator.score(source, images[index:index+1], torch.tensor([truth]),
                          prepared=prepared, label_conditioned=False))
        batch_speed.compare_rows(scalar, scalar)
        for batch in (1,2,4,8):
            torch.cuda.synchronize(device); began=time.monotonic(); values=[]
            for offset in range(0,len(images),batch):
                reference,cached=evaluator.expand_reference(source,prepared,batch)
                values.extend(evaluator.score(reference,images[offset:offset+batch],torch.tensor([truth]*batch),
                              prepared=cached,label_conditioned=False))
            torch.cuda.synchronize(device)
            check=batch_speed.compare_rows(values,scalar)
            q.require(check["passed"] and check["classification_exact"], "Fixed batch8 or tail differs from scalar metrics")
            checks.append(dict(batch_size=batch,comparison=check,total_seconds=time.monotonic()-began,images=len(images)))
    receipt=dict(status="FIXED_BATCH8_QUALIFICATION_PASS",binding=binding,checks=checks,
        chosen_batch_size=8,qualified_batch_sizes=[1,2,4,8],selection="fixed before timing; no batch search",
        synthetic_images=True,source_or_development_images_used=False,real_model_weights=True,
        rng_state_preserved=True,training_updates=0,scientific_result=False)
    receipt['payload_sha256']=q.identity(receipt);q.seal(path,receipt)
    return receipt


class MetricScorer:
    """Source-specific metric cache; no source identity or truth enters VAR."""
    def __init__(self,native,root,output):
        root=Path(root);self.native=native;self.torch=native.torch
        sys.path.insert(0,str(root/'outputs/EXTERNAL-COMPARISON-20261004/runtime'))
        sys.path.insert(0,str(root/'experiments/unified-metrics-20261002'))
        from step0_reference_metrics import load_suite,numeric_flags
        import runner, replay
        before=numeric_flags(self.torch)
        with verify_repeated_interop(self.torch):
            evaluator,metric,quality,lpips,dino,bindings,flags=load_suite(root,'cuda:0')
        q.require(numeric_flags(self.torch)==before==flags,"Metric construction changed frozen numerical settings")
        # These are the same original LPIPS and DINO-S14 weights. Verify before
        # replacing duplicate loader references with the live frozen originals.
        state_sha=native.assets.old.b.state_sha256
        q.require(state_sha(lpips)==native.loaded['identity']['models']['lpips']
                  and state_sha(dino)==native.loaded['identity']['models']['dino'],"Legacy metric weights differ")
        del lpips,dino;gc.collect()
        self.evaluator,self.metric,self.quality,self.runner,self.replay=evaluator,metric,quality,runner,replay
        self.lpips,self.dino=native.loaded['lpips'],native.loaded['dino']
        self.device=native.loaded['device'];self.prepared=None
        manifest=root/'outputs/UNIFIED-METRICS-20261002/modelmanifest.json'
        self.batch_receipt=batch8_qualification(evaluator,metric,self.torch,self.device,manifest,output,
                                               dict(visual_identity=native.loaded['identity'],flags=flags))
        self.identity=dict(modelmanifest_sha256=q.sha(manifest),metric_evaluator_identity=evaluator.identity(),
            model_metadata=evaluator.metadata,source_bindings=bindings,numerical_runtime=flags,
            batch_qualification_sha256=q.sha(Path(output)/'metric_batch8_qualification.json'),
            driver_sha256=q.sha(__file__),batch_size=8,qualified_batch_sizes=[1,2,4,8],
            primary_objective=q.PRIMARY_METRIC,ConvNeXt="not loaded; development only",
            mismatch_specificity="actual development only; no calibration mismatch population invented")

    def prepare_source(self,record,target,source_index):
        boundary(self.native)
        q.rgb_sha(target)
        self.target=np.asarray(target,dtype=np.float32)
        self.reference=self.torch.from_numpy(self.target[None])
        with self.torch.no_grad(), self.metric.no_network():
            self.prepared=self.evaluator.prepare_reference(self.reference)
        self.record_identity=(record['image_id'],record['preprocessing_id'],source_index)

    def __call__(self,record,images):
        q.require(self.prepared is not None and self.record_identity==
            (record['image_id'],record['preprocessing_id'],record['source_index']),"Metric reference source differs")
        with self.torch.no_grad(), self.metric.no_network():
            legacy,_,_=self.quality.quality_metrics(self.target,images,self.lpips,self.dino,self.device)
            scorer=self.runner.PendingMetricScorer(self.evaluator,self.torch,self.reference,self.prepared,
                        int(record['class_index']),8,[1,2,4,8])
            rows=[]
            for image,old in zip(images,legacy):
                row=dict(old); rows.append(row)
                scorer.add(self.replay.metric_cache_key(image,self.target,self.evaluator.identity()),image,row)
            scorer.flush()
        for row in rows:
            value=float(row['psnr_db'])
            q.require(math.isfinite(value) or value==math.inf,"Invalid PSNR")
            row['psnr_db']=value if math.isfinite(value) else None
            row['psnr_infinite']=value==math.inf
            row['semantic_error']=1-int(row['resnet50_top1_source_prediction'])
        return rows


def pilot(native,states,scorer,output):
    """Two original calibration sources, all states; only cost/equivalence use."""
    data=native.data('calibration');timings=[]
    for index in (0,1):
        boundary(native); native.torch.cuda.synchronize(); began=time.monotonic()
        renderer=q.SourceRenderer(native.loaded,native.receiver,data['T'][index].cpu().numpy(),states,use_snapshots=True)
        try:
            images={}
            for state in states:
                for receiver in ('VAR','direct'):
                    image=renderer.render(state,receiver)['image'];images[q.rgb_sha(image)]=image
            image=np.full((3,256,256),.5,np.float32);images[q.rgb_sha(image)]=image
            native.torch.cuda.synchronize();render_seconds=time.monotonic()-began
            record=dict(data['records'][index],source_index=index)
            scorer.prepare_source(record,record['pixels'].astype(np.float32)/np.float32(255),index)
            began_metrics=time.monotonic();rows=scorer(record,list(images.values()))
            native.torch.cuda.synchronize();metrics_seconds=time.monotonic()-began_metrics
            archive=Path(output)/'pilot_reconstructions'/('%04d.npz'%index)
            archive.parent.mkdir(parents=True,exist_ok=True)
            began_archive=time.monotonic()
            target=record['pixels'].astype(np.float32)/np.float32(255)
            if archive.exists():
                with np.load(archive,allow_pickle=False) as existing:
                    q.require(np.array_equal(existing['images'],np.stack(list(images.values())))
                              and np.array_equal(existing['source_rgb'],target),'Engineering pilot archive differs')
            else:
                temp=archive.with_name(archive.name+'.tmp')
                with temp.open('wb') as handle:
                    np.savez_compressed(handle,images=np.stack(list(images.values())),source_rgb=target)
                os.replace(temp,archive)
            archive_sha=q.sha(archive)
            timings.append(dict(source_index=index,source_id=record['image_id'],states=len(states),unique_images=len(images),
                prior_forward_calls=renderer.prior_forward_calls,render_total_seconds=render_seconds,
                metrics_total_seconds=metrics_seconds,archive_total_seconds=time.monotonic()-began_archive,
                archive_path=str(archive),archive_sha256=archive_sha,archive_bytes=archive.stat().st_size,
                source_total_seconds=time.monotonic()-began,
                metric_rows=len(rows)))
        finally:
            renderer.close()
    return dict(status='SOURCE_QUALITY_COST_PILOT_COMPLETE',population='calibration',source_indices=[0,1],
        scientific_result=False,development_read=False,training_updates=0,selection_updates=0,rows=timings,
        estimated_full1000_source_Q_hours=sum(r['source_total_seconds'] for r in timings)/2*1000/3600,
        estimate_scope='source Q only; excludes BLER, actual development, reporting and final1000 confirmation',
        stage_rule='Study owner combines all stage costs before invoking the frozen >48h fallback; no automatic shortlist')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True,type=Path)
    parser.add_argument('--native-runtime',type=Path)
    parser.add_argument('--states',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--stage',choices=('qualify','benchmark','run'),required=True)
    parser.add_argument('--source-limit',type=int,choices=(300,1000),default=1000)
    parser.add_argument('--reuse-from',type=Path)
    args=parser.parse_args();root=args.root.resolve();output=args.output.resolve()
    runtime=args.native_runtime or root/'experiments/m1-n2048-full-grid-20261004'
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    with single_writer(output):
        native=build_native(root,runtime)
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
        states_doc=q.read(args.states)
        q.require(states_doc['registered_before_quality'] is True and states_doc.get('development_read',False) is False,
                  'Quality states were not frozen before quality selection')
        states=q.validate_states(states_doc['states'])
        q.seal(output/'driver_registration.json',dict(source_bindings=native.driver_bindings,
            states_registration_path=str(args.states.resolve()),states_registration_sha256=q.sha(args.states),
            population='calibration',generation=q.GENERATION,independent_ConvNeXt='development only',
            visual_identity=native.loaded['identity'],numeric_flags=native.flags,health_polling_seconds=5.))
        qualification_path=output/'qualification.json'
        if qualification_path.exists():
            qualification=q.read(qualification_path)
            q.require(qualification['status']=='REAL_SOURCE_QUALITY_PARITY_PASS'
                      and qualification['frozen_identity']==native.loaded['identity']
                      and qualification['source_sha256']==q.sha(q.__file__),'Source-Q qualification changed')
        else:
            qualification=q.qualify(native,states);q.seal(qualification_path,qualification)
        if args.stage=='qualify':
            native.frozen();return
        scorer=MetricScorer(native,root,output)
        if args.stage=='benchmark':
            receipt=pilot(native,states,scorer,output)
            receipt.update(quality_qualification_sha256=q.sha(qualification_path),score_identity=scorer.identity,
                           driver_registration_sha256=q.sha(output/'driver_registration.json'))
            q.seal(output/'benchmark.json',receipt)
        else:
            q.run_quality(native,args.states,output,scorer,scorer.identity,source_limit=args.source_limit,
                          reuse_from=args.reuse_from,qualification=qualification,
                          stop_requested=lambda:boundary(native))
        native.frozen()


if __name__=='__main__':main()

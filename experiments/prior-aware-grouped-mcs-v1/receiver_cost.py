"""Fixed16 uncached TX entropy and actual-payload receiver timing, after freeze."""
from __future__ import annotations
import argparse
from collections import defaultdict
from pathlib import Path
import signal
import statistics
import time
import numpy as np
import source_quality as q
import quality_driver as driver
from score_actual import actual_state,event_completion,frame_rows,write_csv
from report_builder import FIXED

WARMUP=1
REPEATS=3
METHODS=('B0','B1','B2','B3','B4')


def timing_samples(call,synchronize,guard,fingerprint_of,clock=time.perf_counter):
    warm=[];measured=[];fingerprint=None
    for phase,count in (('warmup',WARMUP),('measured',REPEATS)):
        for _ in range(count):
            guard();synchronize();began=clock();result=call();synchronize();elapsed=clock()-began
            q.require(elapsed>=0 and np.isfinite(elapsed),'Invalid synchronized timing')
            digest=fingerprint_of(result)
            if fingerprint is None:fingerprint=digest
            q.require(digest==fingerprint,'Repeated uncached computation changed its numeric output')
            (warm if phase=='warmup' else measured).append(elapsed)
    return dict(status='MEASURED_UNCACHED',warmup_repetitions=WARMUP,warmup_total_seconds=sum(warm),
        measured_repetitions=REPEATS,seconds=measured,mean_seconds=statistics.mean(measured),
        median_seconds=statistics.median(measured),total_seconds=sum(measured),output_fingerprint=fingerprint)


def fresh_entropy(native,scales,m):
    """No source snapshot, logits, final-order or image memo is accepted."""
    with native.torch.no_grad():
        prior=native.receiver._Prior(native.loaded['vae'],native.loaded['var'],native.loaded['device'])
        try:
            for k in range(m):prior.logits(k);prior.advance(scales[k],k)
            logits=prior.logits(m)[0].float().cpu().numpy()
            order=q.entropy_order(native.receiver,logits)
        finally:prior.close()
    return order


def admitted(policies_path,events):
    from evaluate import selected_schedule
    from profiles import freeze_codebook
    policies=q.read(policies_path);labels,profiles=selected_schedule(policies)
    codebook=freeze_codebook(profiles);events=Path(events).resolve();reg=q.read(events/'registration.json')
    scope=dict(policies=policies,codebook=codebook,labels_by_snr=labels,
        selected_policies_sha256=q.sha(policies_path),visual_identity=reg['visual_identity'])
    # Rebuild only the finite already-selected public codebook, without search.
    for values in labels.values():
        for value in values:
            if value['status']=='FROZEN':value['profile_id']=codebook['candidate_to_profile'][value['stable_id']]
    event_reg,event_done=event_completion(events,scope)
    return scope,event_reg,event_done


def measure(native,scope,event_reg,event_done,events,output):
    events,output=Path(events).resolve(),Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    q.require(native.loaded['identity']==scope['visual_identity'],'Frozen timing models differ from evaluated models')
    data=native.data('development');records=data['records']
    q.require([r['image_id'] for r in records]==event_reg['source_ids'],'Timing uses different development sources')
    inputs=dict(event_reg['input_bindings'],**{str(events/'registration.json'):q.sha(events/'registration.json'),
                                            str(events/'completion.json'):q.sha(events/'completion.json')})
    registration=dict(status='UEP_FIXED16_RECEIVER_COST_REGISTERED',source_indices=list(FIXED),noise_seed=2001,
        snrs_db=[4,7,10,13],methods=list(METHODS),warmup_repetitions=WARMUP,measured_repetitions=REPEATS,
        model_identity=native.loaded['identity'],numeric_flags=native.flags,source_bindings=dict(native.driver_bindings,**{str(Path(__file__).resolve()):q.sha(__file__)}),
        input_bindings=inputs,training_updates=0,policy_selection_updates=0,uncached=True,
        TX_scope='known coarse token prefix through original VAR and full entropy ordering; excludes encoder/VQ and packet/FEC encoding',
        RX_scope='actual CRC-accepted payload through fresh VAR and Dc to CPU float32 RGB; no final-image/KV cache, metric scoring excluded',
        synchronization='CUDA synchronized before and after each measured callable',
        guard_timing='original safety check before each repetition, excluded from timed callable',
        failure_handling='gray measured as gray allocation; no phantom neural decoder cost on erasure',
        same_wire_aliases='one measured actual frame shared by explicit method labels',
        PHY_scope='PHY CPU/GPU decoding times absent in old event ledger remain explicitly unmeasured; never added to image-receiver time')
    q.seal(output/'registration.json',registration);binding=q.identity(registration)
    receiver=q.ReceivedRenderer(native.loaded,native.receiver);profiles={int(p['profile_id']):p for p in scope['codebook']['entries']}
    allrows=[];outputs={}
    for index in FIXED:
        if driver.boundary(native) or (output/'STOP').exists():
            q.write(output/'status.json',dict(status='PAUSED_AT_SOURCE_BOUNDARY',completed_sources=len(outputs)));return None
        source_path=events/'source_checkpoints'/('%04d.json'%index);cp=q.read(source_path)
        q.require(event_done['outputs'][str(source_path)]==q.sha(source_path)
                  and cp['binding']==q.identity(event_reg) and cp['payload_sha256']==q.identity({k:v for k,v in cp.items() if k!='payload_sha256'}),'Timing actual-event checkpoint changed')
        frame_rows(cp,index,records[index]['image_id'],scope['codebook'],scope['labels_by_snr'])
        output_cp=output/'source_checkpoints'/('%04d.json'%index)
        if output_cp.exists():
            saved=q.read(output_cp)
            q.require(saved['binding']==binding and saved['input_checkpoint_sha256']==q.sha(source_path)
                      and saved['payload_sha256']==q.identity({k:v for k,v in saved.items() if k!='payload_sha256'}),'Timing resume receipt changed')
        else:
            rows=[];tx_measurements={};scales=q.split_tokens(data['T'][index].cpu().numpy())
            for frame in cp['frames']:
                spec=frame['spec'];methods=[m for m in frame['methods'] if m['family'] in METHODS]
                if spec['noise_seed']!=2001 or not methods:continue
                profile=profiles[int(spec['profile_id'])];state=actual_state(frame['receiver_event'],scope['codebook'])
                m=profile['m']
                if profile['K'] and m not in tx_measurements:
                    tx_measurements[m]=timing_samples(lambda:fresh_entropy(native,scales,m),native.torch.cuda.synchronize,
                                                       lambda:driver.boundary(native),lambda value:q.identity(value.tolist()))
                tx=tx_measurements[m] if profile['K'] else dict(status='NOT_REQUIRED_K0',mean_seconds=0.,median_seconds=0.,measured_repetitions=0)
                def receive():
                    if state['kind']=='gray':image=np.full((3,256,256),.5,np.float32)
                    else:
                        value=receiver.render_received(state['prefix'],state['partial_values'],state['m'],state['K'],'VAR',cache=None)
                        image=value['image']
                    return image
                measured=timing_samples(receive,native.torch.cuda.synchronize,lambda:driver.boundary(native),q.rgb_sha)
                row=dict(source_index=index,source_id=records[index]['image_id'],N=1024,snr_db=spec['snr_db'],noise_seed=2001,
                    physical_frame_id=frame['physical_frame_id'],profile_id=spec['profile_id'],methods=methods,
                    receiver_state=state['kind'],accepted_m=state['m'],accepted_K=state['K'],tx_m=profile['m'],tx_K=profile['K'],
                    TX_entropy=tx,RX_image=measured,neural_receiver_invoked=state['kind']!='gray',
                    actual_frame_energy=frame['ledger']['E'],PHY_timing_status='NOT_RECORDED_IN_ACTUAL_LEDGER',
                    PHY_timing_seconds=None,end_to_end_receiver_seconds=None,
                    original_ledger_timing=frame['ledger'].get('timing'),
                    same_wire_measurement_reused=len(methods)>1,final_image_cache_enabled=False,synthetic=False)
                rows.append(row)
            saved=dict(binding=binding,source_index=index,input_checkpoint_sha256=q.sha(source_path),rows=rows)
            saved['payload_sha256']=q.identity(saved);q.seal(output_cp,saved)
        allrows.extend(saved['rows']);outputs[str(output_cp)]=q.sha(output_cp)
        q.write(output/'status.json',dict(status='RUNNING',completed_sources=len(outputs),total_sources=16))
    groups=defaultdict(list)
    for row in allrows:
        for method in row['methods']:groups[row['snr_db'],method['family']].append(row)
    summary=[]
    for (snr,method),rows in sorted(groups.items()):
        q.require({r['source_index'] for r in rows}==set(FIXED) and len(rows)==16,'Incomplete fixed16 cost population')
        for name,field in (('TX_entropy','TX_entropy'),('RX_image','RX_image')):
            values=np.array([r[field]['mean_seconds']*1000 for r in rows])
            summary.append(dict(N=1024,snr_db=snr,method=method,component=name,mean_ms=float(values.mean()),
                median_ms=float(np.median(values)),p95_ms=float(np.quantile(values,.95)),sources=16,noise_seed=2001,
                repetition_mean_before_population_summary=True,uncached=True,
                gray_frames=sum(r['receiver_state']=='gray' for r in rows),PHY_included=False,metrics_included=False))
    summary_path=output/'timing_summary.csv';write_csv(summary_path,summary);outputs[str(summary_path)]=q.sha(summary_path)
    native.frozen()
    for path,digest in registration['source_bindings'].items():q.require(q.sha(path)==digest,'Timing implementation changed')
    receipt=dict(status='UEP_UNCACHED_RECEIVER_COST_COMPLETE',registration_sha256=q.sha(output/'registration.json'),
        source_indices=list(FIXED),rows=allrows,summary=summary,outputs=outputs,input_bindings=inputs,
        uncached_image_receiver=True,uncached_TX_entropy=True,metric_scoring_executed=False,extra_metric_suite_loaded=False,
        resident_original_quality_models=['lpips','dino'],standalone_receiver_memory_measured=False,
        PHY_latency_measured=False,end_to_end_latency_claimed=False,training_updates=0,policy_selection_updates=0)
    q.seal(output/'timing.json',receipt);q.write(output/'status.json',dict(status='COMPLETE',sources=16,physical_frames=len(allrows)))
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','policies','events','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--native-runtime',type=Path);args=parser.parse_args()
    scope,reg,done=admitted(args.policies,args.events)
    with driver.single_writer(args.output):
        native=driver.build_native(args.root,args.native_runtime or args.root/'experiments/m1-n2048-full-grid-20261004')
        signal.signal(signal.SIGTERM,driver.stop);signal.signal(signal.SIGINT,driver.stop)
        measure(native,scope,reg,done,args.events,args.output)


if __name__=='__main__':main()

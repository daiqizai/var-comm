"""Closed-run metadata review. No model, channel, scoring or bootstrap imports."""
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'bpg_threads_recovery_v1'))
import resume as a


def main():
    r, rec = a.bind('f513248d68146729ced3ef9dc7b591acf1a697c3931acab0d9345ad06fe16dbb')
    out = a.O/'execution_review_v1'
    assert not out.exists()
    for v in r['original_bindings']:
        a.checked(v)
    delivery = a.O/'delivery'
    c = a.read(delivery/'completion.json')
    assert c['full_calls']==1440 and c['formal_calls']==720
    a.checked(c['manifest'])
    for v in a.read(delivery/'manifest.json')['files']:
        a.checked(v)
    with (delivery/'per_call.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    assert len(rows)==1440
    outcome = []
    for method in a.METHODS:
        for snr in a.SNRS:
            part = [v for v in rows if v['method']==method and int(v['snr_db'])==snr]
            formal = [v for v in part if v['phase']=='measure']
            assert len(formal)==48 and len(part)==96
            assert {int(v['source_index']) for v in formal}==set(a.INDICES)
            for v in part:
                assert math.isclose(float(v['tx_ms'])+float(v['rx_ms']),float(v['total_ms']),rel_tol=1e-12)
            varying = []
            for index in a.INDICES:
                source = [v for v in part if int(v['source_index'])==index]
                assert len(source)==6 and len({v['counter'] for v in source})==1
                if len({(v['status'],v['output_sha256']) for v in source})!=1:
                    varying.append(index)
            outcome.append(dict(method=method,snr_db=snr,formal_calls=48,
                receiver_statuses=dict(Counter(v['status'] for v in formal)),
                repeated_output_variation_source_indices=varying))
    resources = []
    stages = []
    for method in a.METHODS:
        roles = ('tx','rx') if method in ('P100000','Swin200000') else ('tx','phy','rx')
        for role in roles:
            d = a.stage_dir(method,role)
            root = (a.T if (method,role) in [('NEST','tx'),('NEST','phy')]
                    else a.CONTROL if method in ('BPG','ARPC') else a.previous.CONTROL)
            waitfile = root/'control'/(method+'_'+role)/'actual_wait.json'
            wait = a.read(waitfile)
            assert type(wait['returncode']) is int and wait['returncode']==0
            assert not wait['timed_out'] and a.gone(wait['child'])
            comp = a.read(d/'completion.json')
            assert comp['calls']==288 and comp['formal']==144 and not comp['blocked_reads']
            assert not (d/'failure.json').exists()
            readings = [a.read(d/'resource_before.json'),comp['resource_after']]
            for row in r['grid']:
                p = d/'cases'/row['key']
                assert (p/'complete.json').is_file()
                readings.append(a.read(p/'reservation.json')['resource'])
            assert all(v['cpu_affinity']==list(range(6)) for v in readings)
            assert {v['uuid'] for v in readings}=={r['resource']['gpu_uuid']}
            resources.append(dict(method=method,role=role,
                GPU_free_MiB_min=min(v['free_MiB'] for v in readings),
                GPU_utilization_min=min(v['utilization'] for v in readings),
                GPU_utilization_max=max(v['utilization'] for v in readings),
                system_load1_min=min(v['load_average'][0] for v in readings),
                system_load1_max=max(v['load_average'][0] for v in readings),
                samples=len(readings),CPU_affinity=list(range(6)),
                setup=a.read(d/'setup.json'),
                native_identity=a.read(d/'native_identity.json') if (d/'native_identity.json').exists() else None))
            stages.append(dict(method=method,role=role,wait=a.pin(waitfile),
                launch=a.pin(root/'control'/(method+'_'+role)/'launch.json'),
                completion=a.pin(d/'completion.json'),
                cumulative_last_model_calls=comp['results'][-1]['model_calls'],
                cumulative_last_packet_calls=comp['results'][-1]['evidence'].get('packet_calls')))
    out.mkdir()
    a.save(out/'review.json',dict(status='PASS_COMPLETENESS_AND_MAPPING',
        reviewer='root assistant',registration=a.pin(a.O/'registration.json'),
        recovery_registration=a.pin(a.REG),delivery_completion=a.pin(delivery/'completion.json'),
        scope=dict(methods=5,sources=16,SNRs=list(a.SNRS),formal_calls=720,warmups=720,batch=1),
        outcome_by_method_SNR=outcome,resource_ranges=resources,stages=stages,
        failed_initial_receiver_cases=0,incomplete_failed_BPG_encoder_attempts=1,
        replayed_successful_scientific_calls=0,
        BPG_resource_qualification=a.pin(a.T/'bpg_threads_qualification_v1/completion.json'),
        quality_metrics=0,bootstrap=0,training=0,
        fairness_notes=[
            'All timing calls share the same physical host/GPU and CPU0..5; BPG and channel decoders are CPU work.',
            'Original software stacks and original per-method numerical precision remain disclosed, not claimed identical.',
            'Three warmups plus three measured fresh calls per source/SNR; repetitions use one fixed noise realization.',
            'Substage CUDA synchronization instrumentation is included in endpoint totals.',
            'BPG includes the frozen multi-resolution/QP search and candidate decoding, not only one final bpgenc call.',
            'Shared load samples are observed, not proof of exclusive machine use.',
            'All outcomes including rejects/gray are included; status counts accompany latencies.']))
    print(json.dumps(a.pin(out/'review.json')))


if __name__ == '__main__':
    main()

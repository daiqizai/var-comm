"""Metadata-only publication export after a named stage is actually closed.

This is not a launcher and never scores, renders, or resamples. It is separate
from the immutable running continuations. Its first execution is still pending.
"""
from followup_common import *
from package_closed_evidence import seal_closed,copy_exact
from collections import defaultdict,Counter

def main(stage):
    assert stage in ['e2','e5','e6']
    if stage=='e2':
        roles=['E2_nest',*[f'E2_ada_{i:02d}' for i in range(4)],'E2_ada_unlaunched_queue_v2','E2_three','E2_clip','E2_dists','E2_statistics']
    elif stage=='e6':roles=['E6_phy','E6_dreamsim','E6_dists','E6_statistics']
    else:roles=[*[f'E5_g{i:02d}_{d}' for i in range(16) for d in ['tx','rx']],'E5_timing_queue']
    receipts={r:seal_closed(r) for r in roles}
    out=D/'remaining_evidence_v1'/stage;out.mkdir(parents=True,exist_ok=False)
    src=D/stage;manifest=dict(stage=stage,status='ACTUAL_CLOSED',closed_roles=receipts,new_model_calls=0,new_PHY=0,bootstrap_reruns=0,files=[])
    names={'e2':['frame_metrics.csv','source_means.csv','summary.csv','paired_deltas.csv','efficiency.csv','selected_prefixes.csv','statistics_identity.json','actual_compute_and_reuse.json','REPORT_zh.md'],
           'e5':['call_timings.csv','source_means.csv','summary.csv','identity.json'],
           'e6':['frame_metrics.csv','source_means.csv','summary.csv','statistics_identity.json','actual_compute_and_reuse.json','REPORT_zh.md']}[stage]
    for name in names:manifest['files'].append(copy_exact(src/name,out/name))
    parent=read(D/'protocol.json');design_key={'e2':'E2_information_budget','e5':'E5_timing','e6':'E6_threshold'}[stage]
    save(out/'protocol.json',dict(parent=pin(D/'protocol.json'),design=parent['design'][design_key],source_sets=pin(D/'source_sets.json')))
    statrole={'e2':'E2_statistics','e6':'E6_statistics','e5':'E5_timing_queue'}[stage]
    save(out/'execution_closure.json',dict(role=statrole,actual_wait_returncode=0,owner_gone=True,child_gone=True,failure_present=False,summary_sha256=sha(src/'summary.csv'),evidence=receipts[statrole]))
    if stage=='e2':
        frames=rows(src/'frame_metrics.csv');assert len(frames)==15000
        source_groups=defaultdict(list)
        for v in frames:source_groups[v['method'],int(v['B_bits']),v['source_id']].append(v)
        assert len(source_groups)==9000
        cells=defaultdict(list);endpoints_count=Counter()
        for (method,budget,sid),vs in source_groups.items():
            assert len(vs)==(3 if method=='Ada' else 1)
            for k in ['endpoint','source_bits','total_overhead','padding_bits','length_checks','receiver_status']:
                assert len({v[k] for v in vs})==1,(method,budget,sid,k)
            v=vs[0];endpoint=ast.literal_eval(v['endpoint']);full=endpoint==([256,0] if method=='Ada' else [10,0])
            null=v['receiver_status']=='FIXED_GRAY'
            assert v['receiver_status'] in ['SOURCE_INPUT','FIXED_GRAY']
            cells[method,budget].append(dict(full=full,null=null,source_bits=int(v['source_bits']),padding_bits=int(v['padding_bits']),length_checks=int(v['length_checks'])))
            endpoints_count[method,budget,tuple(endpoint),v['receiver_status']]+=1
        data=[]
        for (m,b),vs in sorted(cells.items()):
            assert len(vs)==500
            data.append(dict(method=m,B_bits=b,n_sources=500,full_representation_sources=sum(v['full'] for v in vs),full_representation_fraction=sum(v['full'] for v in vs)/500,no_feasible_prefix_sources=sum(v['null'] for v in vs),no_feasible_prefix_fraction=sum(v['null'] for v in vs)/500,mean_actual_stream_bits=float(np.mean([v['source_bits'] for v in vs])),mean_padding_bits=float(np.mean([v['padding_bits'] for v in vs])),mean_length_checks=float(np.mean([v['length_checks'] for v in vs])),NULL_stream_note='If NULL occurs, two protocol placeholder bits are not a decoded source representation.'))
        csvout(out/'representation_and_capacity.csv',data)
        csvout(out/'endpoint_counts.csv',[dict(method=m,B_bits=b,endpoint=list(ep),receiver_status=status,n_sources=n) for (m,b,ep,status),n in sorted(endpoints_count.items())])
        manifest['descriptive_derivation']=dict(input=pin(src/'frame_metrics.csv'),source_units=9000,statistical_resampling=0,source_once_before_averaging=True,all_gen_repeats_same_endpoint_asserted=True)
    if stage=='e5':
        config=read(L/'E5_plan_assets_v3/config.json')
        save(out/'boundary_and_resource_protocol.json',dict(config=pin(L/'E5_plan_assets_v3/config.json'),groups=pin(L/'E5_plan_assets_v3/groups.json'),boundaries=config['boundaries'],excluded=config['excluded'],GPU_UUID=config['GPU_UUID'],cpu_affinity=config['cpu_affinity'],cpu_cores=2,torch_threads=6,torch_interop_threads=2,batch=1,warmups=2,measurements=3,BPG_CPU_only=True,GPU_reservation_accounting='Timing launch reserves the fixed GPU for exclusive conditions, including the CPU-only BPG slots; charged reservation time is not a claim of GPU kernel usage.'))
    save(out/'delivery_manifest.json',manifest)
    print(json.dumps(dict(stage=stage,output=str(out),manifest=pin(out/'delivery_manifest.json'),new_scientific_calls=0)))
if __name__=='__main__':main(*sys.argv[1:])

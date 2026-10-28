"""Read cached streams, freeze equal ceilings and exact physical-event reuse."""
import pathlib,sys,json,collections,subprocess
R=pathlib.Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
sys.path.insert(0,str(R/'experiments/shared-codeword-fit-20261028'))
from fit_common import *
from fit_protocol import endpoints,P,OFFSETS
E2=R/'experiments/two-route-confirmation-20261028';O2=R/'results/two_route_confirmation_20261028_v1';L2=B/'two_route_confirmation_20261028_v1'

def main():
    O2.mkdir(exist_ok=False);L2.mkdir(exist_ok=False)
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=R).decode().strip()=='6bff6eab3157244f11fd4f705d204f4adedcb87a'
    al=L;bl=B/'nested_residual_rd_20261028_v1';bo=R/'results/nested_residual_rd_20261028_v1'
    for role in ['train_scalar_to5000','train_progressive_to5000']:
        w=read(bl/(role+'_control')/'actual_wait.json');assert w['returncode']==0 and gone(w['owner'])and gone(w['process'])
    pause=read(bo/'user_pause_20261028_verified.json');assert pin(bl/'STOP')==pause['STOP']
    quarter=[(4,0),(4,6),(4,12),(4,18),(5,0),(5,9),(5,18),(5,27)]
    fine=[p for p in endpoints()if OFFSETS[p[0]]+p[1]<=OFFSETS[5]+27]
    assert len(quarter)==8 and len(fine)==53 and set(quarter)<=set(fine)
    source=read(al/'source32_numeric_v1/sources.json');assert len(source)==32
    cat=read(al/'geometry/profile_catalog.json');oldjobs=read(al/'actual_frames.json')
    oldfill={(j['source_id'],j['snr_db'],j['replicate']):j for j in oldjobs if j['arm']=='S_FILL'};assert len(oldfill)==288
    oldevents={}
    for role in ['actual_frames_00','actual_frames_01']:
        w=read(al/(role+'_control')/'actual_wait.json');assert w['returncode']==0 and gone(w['owner'])and gone(w['process'])
        oldevents.update({e['event_id']:e for e in read(al/role/'events.json')if e['arm']=='S_FILL'})
    bundle=read(B/'nextscale_selective_blocks_20261028_v1/p3_replay_v2/receiver_cache.json');decode=dict(bundle['cache']);decode.update(read(al/'rx_clean/novel_decode_cache.json'))
    jobs=[];logical=[];accounts=[];newkeys={};old_reused=set();misses=[]
    def identity(j):return digest({k:j[k]for k in ['source_id','source_index','snr_db','replicate','profile_id','information','payloads','counter']})
    for s in source:
        with np.load(checked(s['streams']),allow_pickle=False)as z:
            for arm,ee in [('QUARTER',quarter),('TOKEN',fine)]:
                feasible=[e for e in ee if s['lengths'][f'm{e[0]}_K{e[1]}']<=608];assert feasible
                m,K=max(feasible,key=lambda e:OFFSETS[e[0]]+e[1]);bits=z[f'm{m}_K{K}'].tolist()
                pp=[p for p in cat.values()if(p['mode'],p['m'],p['K'],p['k'],p['null'])==('S_FILL',m,K,637,False)];assert len(pp)==1;p=pp[0]
                projection=dict(status='SOURCE_INPUT',codec='NEST',m=m,K=K,family='EC_VAR_PARTIAL'if K else'EC_VAR_WHOLE',bits=bits)
                ck=digest(projection)
                if ck not in decode:misses.append(projection)
                else:assert decode[ck]['projection']==projection
                accounts.append(dict(arm=arm,source_id=s['source_id'],source_index=s['source_index'],m=m,K=K,source_bits=len(bits),fields=29,padding=608-len(bits),source_capacity=608,utilization=len(bits)/608,k=637,n=1912,N=1024,Es=2,energy=2048,ceiling_m=5,ceiling_K=27))
                for snr in [0,2,4]:
                    for rep in [1,2,3]:
                        old=oldfill[s['source_id'],snr,rep]
                        j=dict(old,profile_id=p['profile_id'],information=P.pack_group(bits,637).tolist(),payloads=[bits],source_bits=len(bits),fields=29,padding=608-len(bits))
                        key=identity(j)
                        if key==identity(old):physical=old['event_id'];origin='EXACT_PRIOR_S_FILL_EVENT';old_reused.add(physical)
                        else:
                            if key not in newkeys:
                                physical=digest(dict(study='TWO_ROUTE_CONFIRMATION_V1',physical_identity=key));newkeys[key]=physical
                                jobs.append(dict(j,event_id=physical,arm='MATCHED_SINGLE_STREAM'))
                            physical=newkeys[key];origin='NEW_PHYSICAL_EVENT'
                        logical.append(dict(arm=arm,source_id=s['source_id'],source_index=s['source_index'],snr_db=snr,replicate=rep,physical_event_id=physical,origin=origin,transmitted_profile_id=p['profile_id'],m=m,K=K))
    save(L2/'cache_miss_preflight.json',misses);assert not misses,'Existing independent RX cache incomplete; no automatic model inference'
    assert len(logical)==576 and len(jobs)<=288
    save(L2/'a_jobs.json',jobs);save(L2/'a_logical_events.json',logical);save(L2/'a_prior_events.json',[oldevents[k]for k in sorted(old_reused)])
    csvsave(O2/'a_source_accounting.csv',accounts)
    plan=pin(bo/'diagnostic16_plan_v1.json');breg=read(bo/'train_progressive_to5000_registration.json');checkpoint=pin('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-checkpoint/liulu/var_comm/nested_residual_rd_20261028_v1/train_progressive_to5000/checkpoint_03000.pt')
    scope=dict(base_commit='6bff6eab3157244f11fd4f705d204f4adedcb87a',user_direction='Prioritize finite same-ceiling S_FILL confirmation and fixed16 transmitted-versus-entropy-predicted residual diagnostic; keep shared extension and training paused',A=dict(sources=32,quarter_endpoints=quarter,token_endpoints=fine,ceiling=[5,27],MCS=dict(q=2,k=637,n=1912),N=1024,header_symbols=68,body_symbols=956,paid_body_fields=29,source_capacity=608,SNR=[0,2,4],noise_replicates=[1,2,3],noise_namespace='SHARED_FIT_AWGN_V1',counter_namespace='SHARED_FIT_FRAME_V1',physical_session='NESTCOM_SHARED_FIT_20261028_V1',logical_events=len(logical),new_physical_frames=len(jobs),reused_distinct_physical_events=len(old_reused),jobs=pin(L2/'a_jobs.json'),logical=pin(L2/'a_logical_events.json'),old_events=pin(L2/'a_prior_events.json'),catalog=pin(al/'geometry/profile_catalog.json'),source_cache=pin(al/'source32_numeric_v1/sources.json'),new_source_model_calls=0,new_RX_model_calls=0,new_metric_calls=0),B=dict(plan=plan,sources=read(checked(plan))['sources'],order='progressive',checkpoint=checkpoint,step=3000,j=256,conditions=['independently_decoded_transmitted','conditional_per_scalar_argmax16'],precision='FP32_TF32_OFF',same_original_GPU=5,old_summary=pin(bl/'train_progressive_to5000/eval_03000/summary.json'),training_stop=pin(bl/'STOP'),training_updates=0),caps=dict(A_physical_frames=288,B_source_materializations=16,B_independent_decodes=16,B_image_renders=32,B_new_lpips_pairs=16,worker_seconds_each=1800,GPU_device_seconds=1800,total_worker_wall_seconds=3600),statistics='Fixed development subsets; descriptive source-paired differences, no new bootstrap or formal/generalization claims',no_automatic_expansion=True,created_unix=time.time())
    save(O2/'scope.json',scope);save(L2/'preparation_completion.json',dict(status='PREPARED_NO_SCIENTIFIC_LAUNCH',scope=pin(O2/'scope.json')))
    print(json.dumps(dict(scope=pin(O2/'scope.json'),A_new_frames=len(jobs),A_reused_physical=len(old_reused),A_logical=len(logical),B_sources=16,B_checkpoint=checkpoint)))
if __name__=='__main__':main()

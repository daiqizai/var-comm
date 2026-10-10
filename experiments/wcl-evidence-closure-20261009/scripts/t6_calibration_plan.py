"""Finite N2048 calibration registration; no channel, model or image reads.

Raw pilot covers every actual legal wire once per first100 original source/SNR.
Full raw recheck is a deterministic finite union, never a new proxy optimum.
Entropy family is fixed exclusively by the completed N1024 calibration decision.
"""
import argparse,csv,json,math
from fractions import Fraction
from pathlib import Path
import t2_pilot as h
from t6_plan import write_csv

SNRS=[4,10,19]


def entropy_candidates():
    rows=[]
    for q in(2,4,6):
        for nominal in('1/2','2/3','3/4','5/6'):
            rate=Fraction(nominal);n=1980*q;k=n*rate.numerator//rate.denominator
            for m in(7,8,9,10):
                rows.append(dict(candidate_id=f'N2048_m{m}_q{q}_r{nominal.replace("/","-")}',N=2048,target_m=m,
                    q=q,nominal_rate=nominal,k=k,n=n,header_symbols=68,body_symbols=1980,frame_tail_symbols=0,
                    length_bits=13,crc_bits=16,source_capacity=min(k-29,8191),minimum_m=4,
                    capacity_rule='min(k-13-16,2**13-1); existing paid uint13 L, no free side information',
                    actual_ldpc_layout_status='REQUIRES_NEW_N2048_ENTROPY_ACTUAL_CONSTRUCTOR_AND_PHY_QUALIFICATION'))
    return rows


def raw_schedule(cat,phase,pilot_rows=None):
    profiles={p['candidate_id']:p for p in cat['profiles']};allids=set(profiles)
    h.require(len(profiles)==521 and sum(p['K']==0 for p in profiles.values())==131,'Bound actual N2048 catalogue has521 raw/131 whole actions')
    full={c for c,p in profiles.items()if p['K']==0 and'full_budget'in p['allocation_modes']}
    m10={c for c,p in profiles.items()if p['m']==10};h.require(len(full)==9 and len(m10)==3,'All9 full-budget whole and3 ten-scale actions required')
    schedule=[]
    for snr in SNRS:
        if phase=='pilot':selected=allids;reasons={c:['ALL_ACTUAL_LEGAL_WIRES']for c in selected}
        else:
            rows=[r for r in pilot_rows if int(r['snr_db'])==snr]
            h.require(len(rows)==521 and {r['candidate_id']for r in rows}==allids,'Full raw stage requires complete all521 pilot quality rows')
            h.require(all(int(r['source_count'])==100 and int(r['noise_count'])==1 and math.isfinite(float(r['dinov2_vitl14_cosine']))for r in rows),'Complete fixed100 pilot required')
            whole=sorted([r for r in rows if profiles[r['candidate_id']]['K']==0],key=lambda r:(-float(r['dinov2_vitl14_cosine']),r['candidate_id']))[:5]
            partial=sorted(rows,key=lambda r:(-float(r['dinov2_vitl14_cosine']),r['candidate_id']))[:5]
            wids={r['candidate_id']for r in whole};pids={r['candidate_id']for r in partial};selected=wids|pids|full|m10
            reasons={c:([x for enabled,x in[(c in wids,'WHOLE_PILOT_TOP5'),(c in pids,'PARTIAL_PILOT_TOP5'),
                (c in full,'ALL_FULL_BUDGET_WHOLE'),(c in m10,'ALL_M10')]if enabled])for c in selected}
        for c in sorted(selected):
            p=profiles[c];schedule.append(dict(snr_db=snr,candidate_id=c,profile_id=p['profile_id'],m=p['m'],K=p['K'],
                eligible_WHOLE=p['K']==0,eligible_PARTIAL=True,reasons=reasons[c],profile=p))
    return schedule


def preview(a):
    cat=h.read(a.candidate_catalogue);done=h.read(a.metadata_completion)
    sealed=[v for p,v in done['outputs'].items()if Path(p).name=='candidate_catalogue.json']
    h.require(done['status']=='T6_RESOURCE_METADATA_COMPLETE_NOT_PHY_QUALIFIED'and sealed==[h.sha(a.candidate_catalogue)],'Actual N2048 catalogue SHA differs')
    confirmation=h.read(a.confirmation_registration)
    h.require(confirmation['source_count']==100 and confirmation['noise_seeds']==[9201,9202,9203]
        and confirmation['pixel_reads']==0 and len(confirmation['records'])==100,'Pre-pixel fixed100 confirmation registration required')
    schedule=raw_schedule(cat,'pilot');ec=entropy_candidates();out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True)
    write_csv(out/'raw_pilot_action_schedule.csv',[{k:(';'.join(v)if k=='reasons'else v)for k,v in x.items()if k!='profile'}for x in schedule])
    write_csv(out/'entropy_candidate_resource_queries.csv',ec)
    value=dict(status='T6_FINITE_CALIBRATION_SCOPE_PREVIEW_NOT_EXECUTION_REGISTRATION',N=2048,snrs=SNRS,
        actual_raw_actions=521,whole_actions=131,full_budget_whole_actions=9,m10_actions=3,
        raw_pilot=dict(source_count=100,source_order='first100 original calibration1000',noise_seeds=[4101],action_SNR_points=len(schedule),
            logical_frames=len(schedule)*100,packet_cap=len(schedule)*200),
        raw_full_rule='perSNR union WHOLE top5, PARTIAL top5, all9 whole full-budget and all3 m10; exact count registered after pilot',
        raw_full_upper_bound=dict(action_SNR_points=66,logical_frames=198000,packet_cap=396000),
        entropy_pilot=dict(families=1,family='PENDING_SEALED_N1024_CALIBRATION_CHOICE',candidate_queries=len(ec),source_count=100,
            noise_seeds=[4101],logical_frames=14400,packet_cap=28800),
        entropy_full_rule='one frozen family; perSNR top3 by pilot DINOv2-L; original1000cal x3noise',
        entropy_full=dict(logical_frames=27000,packet_cap=54000),
        formal_confirmation=dict(registration=h.desc(a.confirmation_registration),source_count=100,noise_seeds=[9201,9202,9203],
            methods=3,logical_method_frame_conditions=2700,maximum_physical_packet_calls=5400,
            content_duplicate_check=confirmation['content_duplicate_check'],scope=confirmation['scope']),
        new_actual_constructor_calls=0,new_packet_decodes=0,new_model_calls=0,source_pixels_opened=False,
        source_inputs=dict(candidate_catalogue=h.desc(a.candidate_catalogue),metadata_completion=h.desc(a.metadata_completion)),
        outputs={str(out/n):h.sha(out/n)for n in('raw_pilot_action_schedule.csv','entropy_candidate_resource_queries.csv')})
    h.save(out/'completion.json',value);return value


def prepare_raw(a):
    qual=h.read(a.phy_qualification);qr=h.checked(qual['request']);cat=h.checked(qr['candidate_catalogue'])
    h.require(qual['schema']=='WCL_T6_N2048_RAW_REAL_PHY_QUALIFICATION_V1'and qual['status']=='PASS'
        and qual['packet_decode_count']==qr['packet_cap']and qual['ledger']==dict(total=qr['packet_cap'],unresolved=0,cap=qr['packet_cap']),
        'Actual independent N2048 PHY qualification first')
    parent=h.read(a.original_calibration_request)
    h.require(parent['schema']=='WCL_T2_FULL1000_RECHECK_V1'and parent['source_count']==1000
        and [r['source_index']for r in parent['records']]==list(range(1000))
        and len({r['source_id']for r in parent['records']})==1000,'Exact original calibration1000 source order required')
    family=h.read(a.entropy_family_selection)
    h.require(family['status']=='T1_SINGLE_ENTROPY_FAMILY_FROZEN_CALIBRATION_ONLY'
        and not family['holdout_used_for_selection']and not family['holdout_files_read']and'T6_additional_budget'in family['uses'],
        'N1024-only common entropy family choice must be frozen before N2048 science')
    h.checked(family['original_policy_freeze']);pilot=None;ranking=None
    if a.phase=='full':
        h.require(a.pilot_completion is not None,'Actual pilot closure required');pilot=h.read(a.pilot_completion)
        h.require(pilot['status']=='T6_RAW_PILOT_CALIBRATION_COMPLETE'and pilot['source_count']==100 and pilot['noise_count']==1,'Actual all521 pilot required')
        ranking=Path(pilot['rankings_path']);h.require(h.sha(ranking)==pilot['outputs'][str(ranking)],'Sealed pilot rankings required')
        with ranking.open(newline='',encoding='utf-8')as f:rankings=list(csv.DictReader(f))
    else:rankings=None
    schedule=raw_schedule(cat,a.phase,rankings);count=100 if a.phase=='pilot'else 1000;seeds=[4101]if a.phase=='pilot'else[4101,4102,4103]
    frames=len(schedule)*count*len(seeds);h.require(a.packet_cap==2*frames,'Explicit exact finite packet cap required')
    out=Path(a.out).resolve();root=Path(parent['root']);h.require(out.is_relative_to(root/'outputs/WCL-EVIDENCE-CLOSURE-20261009'),'Independent WCL output')
    records=parent['records'][:count]
    request=dict(schema='WCL_T6_N2048_RAW_CALIBRATION_V1',phase=a.phase,N=2048,root=str(root),out=str(out),
        source_count=count,records=records,source_ids=[r['source_id']for r in records],snrs=SNRS,noise_seeds=seeds,
        schedule=schedule,frame_count=frames,packet_cap=2*frames,workers=a.workers,
        phy_qualification=h.desc(a.phy_qualification),candidate_catalogue=qr['candidate_catalogue'],entropy_family_selection=h.desc(a.entropy_family_selection),
        original_calibration_request=h.desc(a.original_calibration_request),environment_request=h.desc(a.environment_request),
        pilot_completion=None if pilot is None else h.desc(a.pilot_completion),
        policy_selection='source mean of noise, then source mean DINOv2-L; lexical candidate tie-break; WHOLE K0 only; PARTIAL all same grid',
        source_rule='original calibration manifest first100 pilot / all1000 full, no quality selection',
        noise_rule='t6_raw_phy.standard_noise(source_id,registered_seed), full2048, same noise for both raw families',
        counter_rule='(index_in_[4,10,19]*1000+source_index)*3+(noise_seed-4101)',
        receiver_rule='original actual raw hard bits KEEP even CRC reject; accepted RX header alone selects profile',
        old_RX_reuse=False,old_exact_state_RGB_metric_reuse=True,
        GPU_upper_bound_new_renders=frames,GPU_execution_requires_post_CPU_exact_state_plan=True,
        confirmation_images_opened=False,holdout_used_for_selection=False,training_updates=0,
        deadline_unix=a.deadline_unix,max_seconds=172800,stop_files=[str(root/'STOP'),str(out/'STOP')],
        source_bindings={str(Path(__file__).resolve()):h.sha(__file__)},automatic_successor=False)
    h.require(1<=a.workers<=16,'Finite workers');h.save(out/'request.json',request)
    return dict(status='T6_RAW_CALIBRATION_METADATA_REGISTERED_NOT_RUN',request=h.desc(out/'request.json'),frame_count=frames,packet_cap=frames*2,
        points_per_SNR={s:sum(x['snr_db']==s for x in schedule)for s in SNRS})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True);q=s.add_parser('preview')
    for n in('candidate-catalogue','metadata-completion','confirmation-registration','out'):q.add_argument('--'+n,required=True)
    q=s.add_parser('prepare-raw')
    for n in('phy-qualification','original-calibration-request','entropy-family-selection','environment-request','out'):q.add_argument('--'+n,required=True)
    q.add_argument('--phase',choices=['pilot','full'],required=True);q.add_argument('--pilot-completion');q.add_argument('--packet-cap',type=int,required=True)
    q.add_argument('--workers',type=int,default=8);q.add_argument('--deadline-unix',type=float,required=True)
    a=p.parse_args();print(json.dumps(preview(a)if a.command=='preview'else prepare_raw(a),indent=2))

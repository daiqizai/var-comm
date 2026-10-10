"""Read-only T2 publication export from sealed pilot and full calibration CSVs.

No model, source image, PHY, bootstrap or policy-selection operation is run.
Local copies are verified against original remote SHA bindings without rewriting
the original logical paths embedded in those scientific artifacts.
"""
import argparse,csv,io,json
from pathlib import Path
import t2_pilot as h

METRICS=['psnr_db','lpips_alex','dinov2_vitl14_cosine']
FULL_SCHEMA='WCL_T2_FULL1000_RECHECK_V1'


def rows(path):
    with Path(path).open(newline='',encoding='utf-8-sig')as f:return list(csv.DictReader(f))


def truth(value):
    h.require(str(value)in('True','False','true','false'),'Boolean value required');return str(value).lower()=='true'


def sealed_copy(path,done,name):
    found=[(p,v)for p,v in done['outputs'].items()if Path(p).name==name]
    h.require(len(found)==1,'Unique sealed output required: '+name);remote,digest=found[0]
    h.require(h.sha(path)==digest,'Downloaded output hash differs: '+name)
    return dict(logical_remote_path=remote,sha256=digest,local_copy=str(Path(path).resolve()))


def write_csv(path,values):
    buf=io.StringIO(newline='');w=csv.DictWriter(buf,list(values[0]),lineterminator='\n');w.writeheader();w.writerows(values)
    write_text(path,buf.getvalue())


def write_text(path,text):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);data=text.encode()
    if p.exists():h.require(p.read_bytes()==data,'Previously exported report differs')
    else:p.write_bytes(data)


def fmt(value,digits=5):return format(float(value),'.'+str(digits)+'f')


def run(a):
    r=h.read(a.full_request);done=h.read(a.full_completion);pilot=h.read(a.pilot_completion);policy=h.read(a.expanded_policy)
    h.require(done['status']=='T2_FULL1000_CALIBRATION_COMPLETE_NOT_HOLDOUT' and done['source_count']==1000 and done['noise_count']==3,
        'Actual closed full1000 calibration required')
    h.require(done['request_sha256']==h.sha(a.full_request) and r['schema']==FULL_SCHEMA and r['noise_seeds']==[4101,4102,4103]
        and r['snrs']==[10,19] and r['source_count']==1000,'Frozen full calibration identity differs')
    h.require(done['frame_count']==r['frame_count']==len(r['schedule'])*3000 and done['packet_cap']==r['packet_cap']
        and done['new_packet_calls']<=done['packet_cap'] and not done['original_ledgers_mutated'] and not done['original_policy_overwritten'],
        'Actual finite completion/accounting differs')
    h.require(pilot['status']=='T2_PILOT_COMPLETE_ONLY' and pilot['frame_count']==21200 and pilot['source_count']==100 and pilot['noise_count']==1
        and pilot['request_sha256']==r['parent_pilot_request']['sha256'] and h.sha(a.pilot_completion)==r['parent_pilot_completion']['sha256'],
        'Completed parent pilot differs')
    bindings=[sealed_copy(a.full_rankings,done,'calibration_rankings.csv'),sealed_copy(a.expanded_policy,done,'expanded_whole_policy.json'),
        sealed_copy(a.pilot_rankings,pilot,'calibration_rankings.csv')]
    full=rows(a.full_rankings);pr=rows(a.pilot_rankings);coverage=rows(a.original_coverage)
    h.require(len(pr)==212 and len(full)==len(r['schedule']) and len(coverage)==433*6,'Complete finite candidate tables required')
    pb={(int(x['snr_db']),x['candidate_id']):x for x in pr};fb={(int(x['snr_db']),x['candidate_id']):x for x in full}
    cb={(int(x['snr_db']),x['candidate_id']):x for x in coverage}
    h.require(len(pb)==212 and len(fb)==len(full) and len(cb)==2598,'Duplicate candidate rows')
    schedule={(x['snr_db'],x['candidate_id']):x for x in r['schedule']};h.require(set(schedule)==set(fb),'Full measurement schedule differs')
    source_ids=[x['source_id']for x in r['records']]
    h.require(len(set(source_ids))==1000 and [x['source_index']for x in r['records']]==list(range(1000)),'Original ordered1000 required')
    observations=[]
    for snr in(10,19):
        group=[x for x in pr if int(x['snr_db'])==snr]
        h.require(sorted(int(x['pilot_rank'])for x in group)==list(range(1,107))
            and all(int(x['K'])==0 and int(x['source_count'])==100 and int(x['noise_count'])==1 for x in group),'All106 pilot actions required')
        fg=[x for x in full if int(x['snr_db'])==snr];whole=[x for x in fg if truth(x['eligible_for_expanded_whole'])]
        h.require(sorted(int(x['whole_rank'])for x in whole)==list(range(1,len(whole)+1)),'Full whole rankings incomplete')
        h.require(sum('full_budget'in x['allocation_modes'].split(';')for x in whole)==7,'All7 full-budget whole controls required')
        pilot_top=sorted(group,key=lambda x:int(x['pilot_rank']))[:5]
        h.require(all((snr,x['candidate_id'])in fb for x in pilot_top),'Pilot top5 missing from full recheck')
        win=next(x for x in whole if x['whole_rank']=='1');pwin=next(x for x in policy['winners']if x['snr_db']==snr)
        h.require(win['candidate_id']==pwin['candidate_id'] and float(win['dinov2_vitl14_cosine'])==pwin['mean_DINO_L'],
            'Sealed winner and full mean differ')
        outside=[x for x in pilot_top if not truth(cb[(snr,x['candidate_id'])]['original_whole_shortlist'])]
        observations.append(dict(snr_db=snr,pilot_whole_actions=106,full_whole_actions=len(whole),full_partial_reference_actions=len(fg)-len(whole),
            full_budget_whole_actions=7,pilot_top5_outside_original_shortlist=len(outside),winner_profile_id=int(win['profile_id']),
            winner_candidate_id=win['candidate_id'],winner_changed=pwin['changed'],winner_original_proxy_rank=int(win['original_proxy_rank_whole']),
            winner_pilot_rank=int(win['pilot_rank']),**{m:win[m]for m in METRICS},
            body_crc_rejected_fraction=win['body_crc_rejected_fraction'],header_rejected_fraction=win['header_rejected_fraction']))
    for key,x in pb.items():
        old=cb[key]
        h.require(float(x['original_proxy_score'])==float(old['original_proxy_score'])
            and int(x['original_proxy_rank_whole'])==int(old['original_proxy_rank_whole'])
            and x['profile_id']==old['profile_id'] and x['m']==old['m'],'Pilot and original candidate identity/proxy differs')
    for key,x in fb.items():
        old=cb[key];p=schedule[key]['profile']
        h.require(int(x['source_count'])==1000 and int(x['noise_count'])==3 and int(x['K'])==p['K']
            and int(x['profile_id'])==p['profile_id'] and x['candidate_id']==p['candidate_id']
            and float(x['original_proxy_score'])==float(old['original_proxy_score']),'Full candidate/source/score identity differs')
    exported=[]
    for old in coverage:
        key=(int(old['snr_db']),old['candidate_id']);p=pb.get(key);f=fb.get(key)
        # Drop only empty T0 placeholders; historical metadata values are copied.
        item={k:v for k,v in old.items()if not k.startswith('new_actual_')and k not in('new_crc_failure_fraction','new_measurement_status')}
        item.update(T2_pilot_completed=p is not None,T2_full1000_completed=f is not None,
            T2_scope='PILOT100_ONE_NOISE_AND_FULL1000_THREE_NOISE'if f else('PILOT100_ONE_NOISE_ONLY'if p else'NOT_ADDITIONALLY_MEASURED_BY_T2'),
            pilot_source_count=100 if p else'',pilot_noise_count=1 if p else'',pilot_rank=p['pilot_rank']if p else'',
            full_source_count=1000 if f else'',full_noise_count=3 if f else'',full_whole_rank=f['whole_rank']if f else'',
            full_schedule_reasons=f['reasons']if f else'')
        for name in METRICS+['body_crc_rejected_fraction','header_rejected_fraction']:
            item['pilot_'+name]=p[name]if p else'';item['full_'+name]=f[name]if f else''
        exported.append(item)
    out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True)
    write_csv(out/'candidate_coverage.csv',exported);write_csv(out/'pilot_rankings.csv',pr)
    write_csv(out/'full_calibration_rankings.csv',full);write_csv(out/'winner_summary.csv',observations)
    h.save(out/'expanded_whole_policy.json',policy)
    total_calls=pilot['new_packet_calls']+done['new_packet_calls'];unchanged=all(not x['winner_changed']for x in observations)
    text=['# T2 finite whole-action calibration audit','',
        ('The expanded finite calibration check retained both original whole-scale winners at 10 and 19 dB.'if unchanged else'The expanded finite calibration check changed at least one whole-scale winner.'),
        'This is calibration evidence, not a new holdout result or a global-optimality claim.','',
        '## Actual completed scope','',
        '- Original catalogue: 433 unique wire actions, including 106 whole-scale actions and 7 whole-scale full-budget protection actions.',
        '- Pilot: every one of the 106 whole actions, the first 100 original calibration sources, one original noise realization (4101), at 10 and 19 dB: 21,200 logical frames.',
        '- Full recheck: the pre-registered union of pilot top5, both original winners and all seven full-budget whole actions. Each SNR has '+str(observations[0]['full_whole_actions'])+' eligible whole actions and one fixed partial reference; 1,000 original calibration sources and three noises (4101/4102/4103).',
        '- Actual full closure: '+str(done['frame_count'])+' logical frames and '+str(done['new_packet_calls'])+' newly charged packet decodes, within the '+str(done['packet_cap'])+' independent cap.',
        '- Pilot actual packet decodes: '+str(pilot['new_packet_calls'])+'; pilot plus full total: '+str(total_calls)+'. Exact received-state/image/score reuse did not invoke new packet decoding.',
        '- Two GPU source workers passed the registered exact RGB and metric parity checks before full production; their qualification receipts remain in the scientific closure.',
        '- Full rankings average the three noises within each source before averaging sources. Only DINOv2-L ranks eligible K=0 actions; candidate ID is the deterministic tie break.','',
        '## Retained winners','',
        '| SNR (dB) | Profile | Original proxy rank | Pilot rank | Full PSNR (dB) | Full LPIPS | Full DINOv2-L | Body CRC rejection | Header rejection | Changed |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|']
    for x in observations:
        text.append('| '+str(x['snr_db'])+' | '+str(x['winner_profile_id'])+' | '+str(x['winner_original_proxy_rank'])+' | '+str(x['winner_pilot_rank'])
            +' | '+fmt(x['psnr_db'])+' | '+fmt(x['lpips_alex'])+' | '+fmt(x['dinov2_vitl14_cosine'])+' | '+fmt(100*float(x['body_crc_rejected_fraction']),3)+'% | '
            +fmt(100*float(x['header_rejected_fraction']),3)+'% | '+str(x['winner_changed'])+' |')
    text+=['','CSV exports retain the input numeric strings at original precision; the table rounds for reading. No confidence intervals or bootstrap were newly computed.','',
        '## Interpretation and limits','',
        'The complete whole-action pilot makes the original proxy pre-screen visible. `candidate_coverage.csv` reports the original proxy rank, original shortlist membership, pilot rank, full eligibility and every actually measured quality/failure rate separately. A candidate missing full1000 measurements is left blank, never imputed from its pilot mean.',
        'The 100-source, one-noise pilot and the 1,000-source, three-noise recheck have different populations and precision. Pilot quality levels should not be compared directly with full means as a performance change.',
        'All seven full-budget whole controls were included in the full recheck. Padding in the retained 19 dB winner therefore describes the selected configuration; it does not establish that the whole family was denied full-budget protection.',
        'The partial arm was a fixed reference and was never eligible to win the K=0 ranking. CRC rejection rates describe receiver events; under raw KEEP, a CRC-rejected packet may still supply a reconstruction and does not automatically mean a gray image.',
        'This audit covers only 10 and 19 dB. It does not establish full1000 optimality across all106 whole actions, extend the check to the other four SNRs, or establish a global optimum over modulation, coding, packetization or source representations.',
        ('Both winners remained unchanged, so the registered policy requires no new500 holdout rerun. Published original holdout results and the original PARTIAL policy remain unchanged.'if unchanged else'Changed winners require the separately registered same-source post-hoc holdout specified by the frozen protocol; this exporter starts nothing.'),
        '', '## Provenance','',
        '- Full request SHA256: `'+done['request_sha256']+'`.',
        '- Full completion SHA256: `'+h.sha(a.full_completion)+'`.',
        '- Pilot request SHA256: `'+pilot['request_sha256']+'`.',
        '- Full execution commit: `'+r['execution_commit']+'`.',
        '- Original result files, policies and ledgers are unchanged. This export reads only JSON/CSV, writes a new directory and starts no scientific successor.','']
    write_text(out/'REPORT_T2.md','\n'.join(text))
    inputs={str(Path(getattr(a,n)).resolve()):h.sha(getattr(a,n))for n in('full_request','full_completion','full_rankings','expanded_policy','pilot_completion','pilot_rankings','original_coverage')}
    h.save(out/'provenance.json',dict(input_local_copies=inputs,sealed_remote_outputs=bindings,script=h.desc(__file__),
        source_population_sha256=h.digest(source_ids),new_scientific_calls=0))
    names=['candidate_coverage.csv','pilot_rankings.csv','full_calibration_rankings.csv','winner_summary.csv','expanded_whole_policy.json','REPORT_T2.md','provenance.json']
    result=dict(status='T2_SEALED_CALIBRATION_REPORT_EXPORTED',source_count=1000,noise_count=3,full_logical_frames=done['frame_count'],
        pilot_new_packet_calls=pilot['new_packet_calls'],full_new_packet_calls=done['new_packet_calls'],combined_new_packet_calls=total_calls,
        full_candidate_points=len(full),candidate_coverage_rows=len(exported),all_original_whole_winners_unchanged=unchanged,
        new_holdout_required_at=policy['same_source_posthoc_holdout_required_at'],new_scientific_calls=0,
        outputs={str(out/n):h.sha(out/n)for n in names})
    h.save(out/'completion.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in('full-request','full-completion','full-rankings','expanded-policy','pilot-completion','pilot-rankings','original-coverage','out'):p.add_argument('--'+name,required=True)
    print(json.dumps(run(p.parse_args()),indent=2))

"""Frozen finite A2a proxy nomination; no channel, model or bootstrap call."""
import csv,hashlib,json,math,pathlib,time
ROOT=pathlib.Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM/results/arpc_external_baseline_20261019_v1')
OUT=ROOT/'a2_screen_v1'
RATES=['1/2','2/3','3/4','5/6']
def bounds(gray,clean,p,low,high):
    gain=clean-gray;ends=[gray+low*gain,gray+high*gain]
    return gray+p*gain,min(ends),max(ends)
def nominate(rows):
    # All endpoints within one MCS share the same empirical reliability proxy.
    # Larger mean actual-fallback DINO dominates smaller means in this proxy
    # for every p>=0. This is not a payload-independent reliability theorem.
    leaders=[]
    for q in [2,4,6]:
        for rate in RATES:
            group=[r for r in rows if r['q']==q and r['nominal_rate']==rate]
            assert len(group)==7
            group.sort(key=lambda x:(-x['clean_dino_mean'],-x['target_endpoint']))
            leaders.append(group[0])
    def order(r):return (-r['expected_dino'],r['q'],RATES.index(r['nominal_rate']),-r['target_endpoint'])
    best=sorted(leaders,key=order)[0]
    competitor=sorted([r for r in leaders if r is not best],key=lambda r:(-r['upper_dino'],*order(r)))[0]
    lower=max(r['lower_dino'] for r in leaders)
    possible=[r for r in leaders if r['upper_dino']>=lower]
    # Never conceal a third plausible MCS tradeoff to satisfy the two-item cap.
    selected=[best,competitor];covered={r['strategy_id'] for r in selected}
    missing=[r['strategy_id'] for r in possible if r['strategy_id'] not in covered]
    return {'status':'CANDIDATES_WITHIN_ORIGINAL_CAP' if not missing else 'BLOCKED_RELIABILITY_UNCERTAINTY_EXCEEDS_TWO',
      'best_estimate':best['strategy_id'],'competitor':competitor['strategy_id'],'proposed_candidates':[r['strategy_id'] for r in selected],
      'possible_best_within_simultaneous_bounds':[r['strategy_id'] for r in possible],
      'uncovered_possible_candidates':missing,'within_MCS_leaders':[r['strategy_id'] for r in leaders],
      'candidate_cap':2,'new_channel_authorized_by_this_output':not missing}
def pin(p):
    p=pathlib.Path(p);b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def verify_completion(root,expected):
    a=json.loads((root/'completion.json').read_text());assert a['status']==expected
    for d in a['outputs']:assert pin(d['path'])==d
    return a
def write_csv(p,rows):
    with p.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def put(p,a):
    with p.open('x') as f:json.dump(a,f,indent=2)
def main():
    assert not OUT.exists();clean=ROOT/'a2_clean_v1';metric=ROOT/'a2_metrics_v1';phy=ROOT/'a2_reliability_v1'
    verify_completion(clean,'CLEAN_STATES_AUDITED_NO_METRICS_OR_WIRELESS')
    verify_completion(metric,'COMPLETE_CAL100_THREE_METRICS');verify_completion(phy,'MATCHED_BACKEND_HISTORICAL_PROXY_ONLY')
    lengths=list(csv.DictReader((clean/'source_lengths.csv').open()));quality=list(csv.DictReader((metric/'clean_quality.csv').open()));cells=list(csv.DictReader((phy/'reliability_cells.csv').open()))
    assert len(lengths)==700 and len(quality)==800 and len(cells)==36
    lengths={(int(r['rank']),int(r['endpoint'])):r for r in lengths};quality={(int(r['rank']),int(r['endpoint'])):r for r in quality}
    assert len(lengths)==700 and len(quality)==800
    ids={i:quality[i,0]['source_id'] for i in range(100)};assert len(set(ids.values()))==100
    for (i,e),r in lengths.items():
        assert r['source_id']==ids[i]==quality[i,e]['source_id'] and int(r['outer_length_bits'])+int(r['outer_crc_bits'])==29
    gray=sum(float(quality[i,0]['dinov2_vitl14_cosine']) for i in range(100))/100
    details=[];summaries=[]
    for q in [2,4,6]:
        for rate in RATES:
            num,den=map(int,rate.split('/'));n=956*q;k=n*num//den;capacity=k-29
            for target in range(1,8):
                sid=f'e{target}_q{q}_r{rate.replace("/","-")}';actual=[];values=[]
                for i in range(100):
                    fits=[e for e in range(1,target+1) if int(lengths[i,e]['payload_bits'])<=capacity];end=max(fits,default=0)
                    row=quality[i,end];actual.append(end);values.append(float(row['dinov2_vitl14_cosine']))
                    details.append({'strategy_id':sid,'rank':i,'source_id':ids[i],'q':q,'nominal_rate':rate,'n':n,'k':k,'capacity_bits':capacity,
                      'target_endpoint':target,'actual_endpoint':end,'source_no_fit':end==0,'payload_bits':0 if end==0 else int(lengths[i,end]['payload_bits']),
                      'actual_prefix_dino':float(row['dinov2_vitl14_cosine']),'actual_prefix_psnr':float(row['psnr_db']),
                      'actual_prefix_lpips':float(row['lpips_alex']),'gray_dino':float(quality[i,0]['dinov2_vitl14_cosine'])})
                mean=sum(values)/100
                for snr in [4,10,19]:
                    matches=[r for r in cells if int(r['snr_db'])==snr and int(r['q'])==q and r['nominal_rate']==rate];assert len(matches)==1
                    cell=matches[0];assert (int(cell['k']),int(cell['n']),int(cell['capacity_bits']))==(k,n,capacity)
                    expected,lo,hi=bounds(gray,mean,float(cell['success_rate']),float(cell['simultaneous95_bonferroni_low']),float(cell['simultaneous95_bonferroni_high']))
                    summaries.append({'strategy_id':sid,'snr_db':snr,'q':q,'nominal_rate':rate,'target_endpoint':target,'capacity_bits':capacity,
                      'clean_dino_mean':mean,'gray_dino_mean':gray,'expected_dino':expected,'lower_dino':lo,'upper_dino':hi,
                      'historical_joint_success_rate':float(cell['success_rate']),'historical_trials':int(cell['trials']),
                      'actual_endpoint_vector_sha256':hashlib.sha256(json.dumps(actual).encode()).hexdigest(),'source_no_fit_count':actual.count(0)})
    assert len(details)==8400 and len(summaries)==252
    decisions={str(snr):nominate([r for r in summaries if r['snr_db']==snr]) for snr in [4,10,19]}
    passed=all(d['status']=='CANDIDATES_WITHIN_ORIGINAL_CAP' for d in decisions.values())
    OUT.mkdir();write_csv(OUT/'actual_fallback_quality.csv',details);write_csv(OUT/'strategy_proxy.csv',summaries)
    report={'status':'A2A_NOMINATION_READY_FOR_REGISTERED_LINK_QUALIFICATION' if passed else 'BLOCKED_RELIABILITY_UNCERTAINTY_EXCEEDS_TWO',
      'decisions':decisions,'A2b_started':False,'A3_started':False,'new_models':0,'new_packets':0,'bootstrap_calls':0,
      'objective':'DINOv2-L only; actual fallback quality averaged over all100 sources; uniform per-MCS historical joint-success proxy',
      'uncertainty':'Clopper-Pearson per36cell Bonferroni95 bounds; source binomial model; no arbitrary-payload transfer guarantee',
      'tie_rule':'Higher target endpoint within equal-clean MCS; across equal point scores lowerq then lower nominalrate, then higher target.',
      'dominance':'Within-MCS proxy only because common p>=0; all84 original strategies remain in tables.',
      'no_fit':'Endpoint0 always gray regardless of hypothetical p; actual future null-profile still pays full frame.',
      'limitations':'Historical entropy payloads/catalogue provide a pre-screen proxy only. ARPC actual packets/CRC/parser failures must be evaluated without TX correction.',
      'inputs':[pin(root/n) for root,n in [(clean,'completion.json'),(clean,'source_lengths.csv'),(metric,'completion.json'),(metric,'clean_quality.csv'),(phy,'completion.json'),(phy,'reliability_cells.csv')]],
      'source':pin(__file__),'ended_unix':time.time()}
    put(OUT/'nomination.json',report);print(json.dumps(report,indent=2))
if __name__=='__main__':main()

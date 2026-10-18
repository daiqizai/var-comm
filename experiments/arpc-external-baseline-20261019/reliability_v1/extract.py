"""Read-only historical PHY proxy: one fixed m7-whole event per source/MCS/SNR.

No packet constructor, receiver, model, image or quality score is executed.
The sample selector is fixed before packet outcomes are opened. Correlated
candidate duplicates never increase the trial count. This is an empirical
matched-backend proxy, not an ARPC wireless result or payload-independence proof.
"""
import csv,hashlib,json,math,pathlib,time
from scipy.stats import beta
R=pathlib.Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
RT=R.parent.parent/'var_comm_runtime_20261010';H=RT/'qualification/h800_ep_pilot_phy_v1_attempt1'
O=R/'results/arpc_external_baseline_20261019_v1/a2_reliability_v1'
PINS={'request.json':'34a3538a57650dcc54c00abcf9ca45b3cde2dfc11e02d125b96e42555accd478',
 'run/worker_completion.json':'e635d27a7d355f2260c20390d8366e33a7722f2edb77676287f9d3a59414d880'}
RATES=['1/2','2/3','3/4','5/6']
def sha(b):return hashlib.sha256(b).hexdigest()
def read_pin(d):
    p=pathlib.Path(d['path']);assert p.is_file() and not p.is_symlink();b=p.read_bytes();assert sha(b)==d['sha256'],p
    if 'bytes' in d:assert len(b)==d['bytes']
    return json.loads(b)
def desc(p):
    p=pathlib.Path(p);b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':sha(b)}
def put(p,a):
    with p.open('x') as f:json.dump(a,f,indent=2)
def physical_key(packet,phy_identity):
    r=packet['logical_event']
    d={'phy_identity':phy_identity,'snr_db':r['snr_db'],'source_id':r['source_id'],'noise_seed':r['noise_seed'],
       'public_frame_counter':r['public_frame_counter'],'transmission':packet['transmission'],
       'payload_sha256':packet['payload_sha256'],'noise_sha256':packet['noise_sha256'],'observation_sha256':packet['observation_sha256']}
    return sha(json.dumps(d,sort_keys=True,separators=(',',':'),allow_nan=False).encode())
def exact_event(packet):
    tx=packet['transmission'];rx=packet['actual_RX'];head=rx['header'];body=rx.get('body') or {}
    header_exact=bool(head['header_ok']) and head['profile_id']==tx['profile_id']
    body_exact=body.get('hard_bits_sha256')==tx['information_sha256']
    accepted=rx['status']=='PAYLOAD_PARSED' and body.get('crc_accepted',False) and body.get('parser_accepted',False)
    return {'header_exact':header_exact,'body_information_exact':body_exact,'joint_exact_success':header_exact and body_exact and accepted,
      'accepted_but_information_or_profile_wrong':accepted and not(header_exact and body_exact),'rx_status':rx['status']}
def interval(success,n,alpha):
    assert type(success) is int and type(n) is int and 0<=success<=n and n>0
    return (0.0 if success==0 else float(beta.ppf(alpha/2,success,n-success+1)),
            1.0 if success==n else float(beta.ppf(1-alpha/2,success+1,n-success)))
def write_csv(p,rows):
    with p.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    assert not O.exists(),'Historical extraction already attempted; inspect instead of repeating'
    req=read_pin({'path':str(H/'registered/request.json'),'sha256':PINS['request.json']})
    done=read_pin({'path':str(H/'registered/run/worker_completion.json'),'sha256':PINS['run/worker_completion.json']})
    wait=json.loads((H/'run_owner_actual_wait.json').read_text());assert type(wait['returncode']) is int and wait['returncode']==0 and wait['actual_wait'] is True
    assert done['request_sha256']==PINS['request.json'] and done['status']=='PASS_H800_ORIGINAL100_PILOT_PHY_ONLY'
    assert done['neural_model_calls']==done['quality_scores']==0 and done['results']['packet_ledger']['unresolved']==0
    backend=req['phy_identity']['backend_identity'];decoder=backend['decoder']
    assert backend['implementation']=='Sionna_5G_LDPC_WCL_T1_ADAPTER_V1' and backend['device']=='cpu'
    assert decoder['num_iter']==20 and decoder['cn_update']=='boxplus-phi' and decoder['llr_max']==20.0 and not decoder['early_stop']
    assert backend['original_backend_identity']['version']=='2.2.0'
    selected=[(i,r) for i,r in enumerate(req['frames']) if r['family']=='EC_VAR_WHOLE' and r['target_m']==7 and r['target_K']==0]
    expected={(s,snr,q,rate) for s in range(100) for snr in [4,10,19] for q in [2,4,6] for rate in RATES}
    assert len(selected)==3600 and {(r['source_index'],r['snr_db'],r['q'],r['nominal_rate']) for _,r in selected}==expected
    # The selected index set is now complete; no success/failure has influenced it.
    O.mkdir();put(O/'selection.json',{'rule':'original cal indices0..99; WHOLE targetm7 K0; all12 MCS; SNR4/10/19; originalseed4101; one event/source/cell',
       'request':desc(H/'registered/request.json'),'selected_frame_indices':[i for i,_ in selected],'outcomes_read_before_selection':0})
    rows=[];pins={};sources=set()
    for index,r in selected:
        assert r['noise_seed']==4101 and r['N']==1024 and r['header_symbols']==68 and r['body_symbols']==956
        assert r['length_bits']==13 and r['crc_bits']==16 and r['n']==956*r['q']
        num,den=map(int,r['nominal_rate'].split('/'));assert r['k']==r['n']*num//den
        lp=done['results']['logical_frames'][index];logical=read_pin(lp);assert logical['logical_event']==r and logical['frame_index']==index
        pin=logical['physical_frame'];packet=read_pin(pin);tx=packet['transmission'];pr=packet['logical_event']
        assert tx['n']==r['n'] and tx['k']==r['k'] and tx['q']==r['q'] and tx['total_symbols']==1024
        assert pr['source_id']==r['source_id'] and pr['noise_seed']==4101 and pr['snr_db']==r['snr_db']
        key=physical_key(packet,req['phy_identity'])
        assert key==logical['physical_key'] and packet.get('physical_key',key)==key
        assert packet['full_public_receive_catalogue'] and not packet['source_truth_supplied_to_RX']
        sources.add(r['source_id']);pins[lp['path']]=lp;pins[pin['path']]=pin
        rows.append({'source_id':r['source_id'],'source_index':r['source_index'],'snr_db':r['snr_db'],'q':r['q'],'nominal_rate':r['nominal_rate'],
          'n':r['n'],'k':r['k'],'noise_seed':4101,'physical_key':key,'logical_index':index,
          'historical_payload_bits':tx['arithmetic_bits'],'actual_profile_id':tx['profile_id'],**exact_event(packet)})
    assert len(sources)==100 and len({r['physical_key'] for r in rows})==3600
    cells=[]
    for snr in [4,10,19]:
        for q in [2,4,6]:
            for rate in RATES:
                group=[r for r in rows if (r['snr_db'],r['q'],r['nominal_rate'])==(snr,q,rate)];assert len(group)==100
                success=sum(int(r['joint_exact_success']) for r in group);lo,hi=interval(success,100,.05);slo,shi=interval(success,100,.05/36)
                cells.append({'snr_db':snr,'q':q,'nominal_rate':rate,'n':group[0]['n'],'k':group[0]['k'],'capacity_bits':group[0]['k']-29,
                  'trials':100,'successes':success,'success_rate':success/100,'pointwise95_low':lo,'pointwise95_high':hi,
                  'simultaneous95_bonferroni_low':slo,'simultaneous95_bonferroni_high':shi,
                  'header_errors':sum(not r['header_exact'] for r in group),'false_accepts':sum(r['accepted_but_information_or_profile_wrong'] for r in group)})
    write_csv(O/'historical_events.csv',rows);write_csv(O/'reliability_cells.csv',cells)
    put(O/'input_manifest.json',{'request':desc(H/'registered/request.json'),'worker_completion':desc(H/'registered/run/worker_completion.json'),
      'owner_actual_wait':desc(H/'run_owner_actual_wait.json'),'packet_and_logical_pins':list(pins.values()),'phy_identity':req['phy_identity']})
    put(O/'completion.json',{'status':'MATCHED_BACKEND_HISTORICAL_PROXY_ONLY','ended_unix':time.time(),'cells':36,'independent_sources_per_cell':100,
      'new_packets':0,'new_models':0,'new_quality_scores':0,'science_outcomes_unchanged':True,
      'limitations':['Historical NeST whole payloads and its360-profile catalogue, not ARPC waveforms or an ARPC error-rate claim.',
       'Original common standard-noise seed4101; independence across sources, correlation across MCS/SNR retained.',
       'Intervals condition on binomial source/noise trial model; do not establish arbitrary-payload equivalence.',
       'Exact information equality is post-RX reliability audit only; no truth correction in future ARPC RX.',
       'Zero observed errors retain nonzero error upper bounds. Header and body errors are included jointly.'],
      'outputs':[desc(O/n) for n in ['selection.json','historical_events.csv','reliability_cells.csv','input_manifest.json']]})
    print(json.dumps(cells,indent=2))
if __name__=='__main__':main()

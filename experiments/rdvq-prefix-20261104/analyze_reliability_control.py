"""Native source-acceptance audit and conditional expected-distortion reference.

No new radio, model, metric or selector-training calls. The source-informed
reference uses successful candidate reconstructions at TX, never future noise.
Its source CI conditions on the finite reliability table, whose Wilson limits
are reported separately. This is not P4's measured test communication result.
"""
from pathlib import Path
import argparse,csv,hashlib,json,math
from collections import defaultdict
import numpy as np
METRICS=['dists','dreamsim','psnr_db','lpips_alex']

def csv_write(p,rows):
    with Path(p).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def wilson(k,n):
    z=1.959963984540054;p=k/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den;radius=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return max(0.,center-radius),min(1.,center+radius)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--request',required=True);a=ap.parse_args();r=json.loads(Path(a.request).read_text());O=Path(r['output'])
    assert json.loads(Path(r['source_gate']).read_text())['passed']
    wait=json.loads(Path(r['source_outer_wait']).read_text());assert wait['returncode']==0 and wait['child_gone']
    sources=json.loads(Path(r['manifest']).read_text())['sources'];by_source={v['source_id']:v for v in sources}
    score_rows=list(csv.DictReader(Path(r['scores']).open()));scores={(v['source'],int(v['slices'])):{m:float(v[m]) for m in METRICS} for v in score_rows}
    assert len(scores)==1628*29
    lengths={}
    for v in sources:lengths[v['source_id']]={0:72,**{int(x['slices']):int(x['packet_bits']) for x in json.loads(Path(v['lengths_path']).read_text())}}
    groups=defaultdict(list);events=[];unknown=[];wrong=[];cache={}
    for directory in r['reliability_roots']:
        d=Path(directory);w=json.loads((d/'actual_wait.json').read_text());assert w['returncode']==0 and w['child_gone']
        for line in (d/'events.jsonl').read_text().splitlines():
            event=json.loads(line);sid=event['source_id'];key=(event['N'],event['snr_db'],event['mcs'],event['target']);accepted=event['rx_packet_sha256'] is not None
            assert not event['receiver_given_tx_mcs'] and not event['receiver_given_tx_prefix']
            if sid not in cache:
                row=by_source[sid];known={}
                for t in range(29):
                    wire=(Path(row['public_directory'])/f's{t:02d}.bin').read_bytes();receipt=json.loads((Path(row['rx_directory'])/f's{t:02d}_complete.json').read_text())
                    h=hashlib.sha256(wire).hexdigest();assert receipt['packet_sha256']==h
                    known[h]=dict(slices=t,packet_sha256=h,receipt=str(Path(row['rx_directory'])/f's{t:02d}_complete.json'))
                cache[sid]=known
            if accepted:
                wire=(Path(event['event_directory'])/'received_packet.bin').read_bytes();h=hashlib.sha256(wire).hexdigest();assert h==event['rx_packet_sha256']
                if h not in cache[sid]:unknown.append(dict(event_id=event['event_id'],source_id=sid,path=str(Path(event['event_directory'])/'received_packet.bin'),sha256=h))
                elif h!=event['tx_packet_sha256']:wrong.append(event['event_id'])
                else:
                    native=cache[sid][h];assert native['slices']==event['received']['received_slices']
            packet_bits=event['transmission']['packet_bits'];capacity=event['transmission']['k'];assert packet_bits<=capacity and event['transmission']['N']==event['N']
            row=dict(event_id=event['event_id'],source_id=sid,N=key[0],snr_db=key[1],mcs=key[2],target=key[3],accepted=int(accepted),status=event['received']['status'],received_slices=event['received'].get('received_slices',-1),tx_slices=event['transmitted_slices'],packet_bits=packet_bits,capacity_bits=capacity,rho=event['transmission']['rho'],rx_packet_sha256=event['rx_packet_sha256'],source_cache_verified=int(accepted and event['rx_packet_sha256'] in cache[sid]))
            events.append(row);groups[key].append(row)
    assert len(events)==len({v['event_id'] for v in events})==14400 and len(groups)==144
    (O/'unknown_received_packets.json').write_text(json.dumps(unknown,indent=2));(O/'accepted_wrong_packets.json').write_text(json.dumps(wrong,indent=2));csv_write(O/'frame_audit.csv',events)
    if unknown or wrong:raise RuntimeError('Accepted packet lacks exact source RX lineage, or is undetected wrong content. Independent RX / error model review required; no automatic success or retry.')
    rel=[];prob={};capacities={}
    for key,rows in sorted(groups.items()):
        assert len(rows)==len({v['source_id'] for v in rows})==100
        count=sum(v['accepted'] for v in rows);lo,hi=wilson(100-count,100);prob[key]=count/100
        assert len({v['capacity_bits'] for v in rows})==1;capacities[(key[0],key[2])]=rows[0]['capacity_bits']
        rel.append(dict(N=key[0],snr_db=key[1],mcs=key[2],target=key[3],frames=100,accepted=count,rejected=100-count,fer=(100-count)/100,fer_wilson_low=lo,fer_wilson_high=hi,capacity_bits=rows[0]['capacity_bits'],mean_rho=float(np.mean([v['rho'] for v in rows])),min_rho=min(v['rho'] for v in rows),max_rho=max(v['rho'] for v in rows)))
    csv_write(O/'reliability.csv',rel)
    cal=[v['source_id'] for v in sources if v['split']=='calibration'];test=[v['source_id'] for v in sources if v['split']=='test'];assert len(cal)==1000 and len(test)==500
    identity=json.loads(Path(r['statistics_identity']).read_text());assert identity['sources']==test
    bank=np.load(r['source_bank']);assert bank.shape==(10000,500) and hashlib.sha256(bank.tobytes()).hexdigest()==identity['array_sha256']
    actions=[(m,t) for m in range(12) for t in [14,28]];policies=[];selected=[];candidate_scores=[]
    def values(s,N,snr,m,target):
        capacity=capacities[(N,m)];t=max(t for t,b in lengths[s].items() if t<=target and b<=capacity);p=prob[(N,snr,m,target)]
        return dict(mcs=m,target=target,slices=t,packet_bits=lengths[s][t],capacity_bits=capacity,probability=p,**{metric:p*scores[(s,t)][metric]+(1-p)*scores[(s,0)][metric] for metric in METRICS})
    for N in [512,1024]:
        for snr in [4,10,19]:
            candidates=[]
            for m,t in actions:
                loss=float(np.mean([values(s,N,snr,m,t)['dists'] for s in cal]));candidates.append((loss,m,t));candidate_scores.append(dict(N=N,snr_db=snr,mcs=m,target=t,calibration_expected_dists=loss,calibration_sources=1000))
            loss,m,t=min(candidates);policies.append(dict(N=N,snr_db=snr,mcs=m,target=t,calibration_expected_dists=loss))
            for s in test:
                fixed=values(s,N,snr,m,t);options=[values(s,N,snr,mi,ti) for mi,ti in actions];informed=min(options,key=lambda v:(v['dists'],v['mcs'],v['target']))
                for name,v in [('C0_calibrated_SNR_only',fixed),('C1_source_informed_reference',informed)]:selected.append(dict(source=s,N=N,snr_db=snr,policy=name,**v))
    csv_write(O/'calibration_candidates.csv',candidate_scores);csv_write(O/'frozen_reference_policies.csv',policies);csv_write(O/'test_expected_source_metrics.csv',selected)
    summary=[];paired=[];lookup={(v['source'],v['N'],v['snr_db'],v['policy']):v for v in selected}
    def stats(x):
        q=np.quantile(x[bank].mean(1),[.025,.975]);return dict(mean=float(x.mean()),ci_low=float(q[0]),ci_high=float(q[1]),sources=500,scope='source uncertainty conditional on measured reliability')
    for N in [512,1024]:
        for snr in [4,10,19]:
            for metric in METRICS:
                vv=[]
                for name in ['C0_calibrated_SNR_only','C1_source_informed_reference']:
                    x=np.array([lookup[(s,N,snr,name)][metric] for s in test]);vv.append(x);summary.append(dict(N=N,snr_db=snr,policy=name,metric=metric,**stats(x)))
                paired.append(dict(N=N,snr_db=snr,contrast='C1 minus C0',metric=metric,**stats(vv[1]-vv[0])))
    csv_write(O/'expected_summary.csv',summary);csv_write(O/'expected_paired_deltas.csv',paired)
    (O/'statistics_identity.json').write_text(json.dumps(dict(source_bootstrap=identity,reliability_uncertainty='pointwise Wilson95 on100 independent-source channel events per action; not pooled into source CI',test_scope='expected metrics conditional on fixed empirical reliability table; no new test PHY',tie_rule='lower DISTS then MCS ID then target'),indent=2))
    (O/'complete.json').write_text(json.dumps(dict(passed=True,actual_frames=14400,accepted_native_cache=sum(v['accepted'] for v in events),rejected_gray=sum(1-v['accepted'] for v in events),unknown_packets=0,accepted_wrong=0,new_model_calls=0,new_metric_calls=0,new_test_PHY=0,expected_test_source_records=len(selected),mean_CI=len(summary),paired_CI=len(paired),source_informed_reference_is_expensive=True,learned_controller='NOT_RUN'),indent=2))
if __name__=='__main__':main()

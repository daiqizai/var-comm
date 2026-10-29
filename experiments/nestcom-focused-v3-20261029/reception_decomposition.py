"""Read existing scores only; preserve the original all-event estimand."""
import argparse, csv, hashlib, json, math
from collections import Counter, defaultdict
from pathlib import Path

METRICS = ('psnr_db', 'lpips_alex', 'dreamsim', 'clip_image_cosine')

def pin(p):
    p=Path(p); return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())

def mean(v):
    return math.fsum(v)/len(v)

def writecsv(p,rows):
    with p.open('x',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main(root,out):
    src=root/'results/nestcom_n512_20261028_v1/delivery/frame_scores.csv'
    summary=root/'results/nestcom_n512_20261028_v1/delivery/summary.csv'
    failures=root/'results/nestcom_focused_review_20261029_v2/failure_summary.csv'
    rows=list(csv.DictReader(src.open(newline='')));assert len(rows)==13500
    original={(r['method'],int(r['snr_db']),r['metric']):float(r['mean']) for r in csv.DictReader(summary.open(newline=''))}
    by=defaultdict(list); unique={};identity={};statuses=Counter()
    for r in rows:
        assert r['N']=='512';s=int(r['snr_db']);i=int(r['source_index']);seed=int(r['noise_seed'])
        assert seed in (6201,6202,6203)
        k=(r['source_id'],s,seed,r['method']);assert k not in unique;unique[k]=r
        ident=(i,r['reference_sha256']);assert identity.setdefault(r['source_id'],ident)==ident
        r['ok']=r['accepted'].lower()=='true'
        assert r['fixed_gray'].lower()==str(not r['ok']).lower()
        assert r['null_received'].lower()=='false' and r['TX_NO_FIT'].lower()=='false'
        for metric in METRICS:
            r[metric]=float(r[metric]);assert math.isfinite(r[metric])
        by[r['method'],s].append(r);statuses[r['method'],s,r['receiver_status']]+=1
    assert len(identity)==500 and len(by)==9
    decomposed=[];joint=[];per_source=[]
    for (method,s),rr in sorted(by.items()):
        assert len(rr)==1500 and Counter(r['source_id'] for r in rr)==Counter({sid:3 for sid in identity})
        good=[r for r in rr if r['ok']];bad=[r for r in rr if not r['ok']];p=len(good)/1500
        for metric in METRICS:
            allq=mean([r[metric] for r in rr]);qs=mean([r[metric] for r in good]);qf=mean([r[metric] for r in bad]) if bad else None
            failed_term=(1-p)*qf if bad else 0.;residual=allq-(p*qs+failed_term)
            assert abs(residual)<1e-12 and abs(allq-original[method,s,metric])<1e-12
            decomposed.append(dict(N=512,method=method,snr_db=s,metric=metric,frames=1500,sources=500,success_frames=len(good),failed_frames=len(bad),p_success=p,all_frame_mean=allq,successful_frame_mean=qs,failed_frame_mean=qf,weighted_success=p*qs,weighted_failure=failed_term,mixture_residual=residual,original_summary_mean=original[method,s,metric]))
    for other in ('WHOLE','ADA'):
        for snr in (4,10,19):
            per=defaultdict(list)
            for sid in identity:
                for seed in (6201,6202,6203):
                    a=unique[sid,snr,seed,'NEST'];b=unique[sid,snr,seed,other]
                    assert a['source_index']==b['source_index'] and a['reference_sha256']==b['reference_sha256']
                    if a['ok'] and b['ok']:per[sid].append((a,b))
            counts=Counter(len(v) for v in per.values())
            for metric in METRICS:
                deltas=[]
                for sid,events in per.items():
                    delta=mean([a[metric]-b[metric] for a,b in events]);deltas.append(delta)
                    per_source.append(dict(comparison='NEST-'+other,snr_db=snr,source_id=sid,source_index=identity[sid][0],metric=metric,joint_success_repeats=len(events),mean_paired_delta=delta))
                joint.append(dict(N=512,comparison='NEST-'+other,snr_db=snr,metric=metric,joint_success_frames=sum(map(len,per.values())),retained_sources=len(per),sources_with_0_repeats=500-len(per),sources_with_1_repeat=counts[1],sources_with_2_repeats=counts[2],sources_with_3_repeats=counts[3],source_equal_paired_delta=mean(deltas),confidence_interval='not computed; descriptive conditional comparison',primary_all_event_delta=original['NEST',snr,metric]-original[other,snr,metric]))
    out.mkdir(parents=True,exist_ok=False)
    for name,data in [('success_failure_quality.csv',decomposed),('joint_success_paired_quality.csv',joint),('joint_success_source_deltas.csv',per_source)]:writecsv(out/name,data)
    record=dict(schema='N512_EXISTING_SCORE_DECOMPOSITION_V1',inputs=[pin(src),pin(summary),pin(failures)],source_order=list(identity),source_order_sha256=hashlib.sha256(json.dumps(list(identity),separators=(',',':')).encode()).hexdigest(),join=['full source_id','original source_index verified','reference_sha256 verified','snr_db','noise_seed'],mixture='Frame-weighted success/failure means; all-event mean matches fixed-three-repeat equal-source mean.',joint='Intersect actual successful events, average differences within each source over retained repeats, then equal-weight retained sources.',failed_mean_without_failure=None,metrics=METRICS,logical_frames=13500,new_model_calls=0,new_metric_calls=0,new_channel_calls=0,new_bootstrap=0,outputs=[pin(p) for p in sorted(out.glob('*.csv'))],status_counts=[dict(method=k[0],snr_db=k[1],status=k[2],count=v) for k,v in sorted(statuses.items())])
    (out/'reception_decomposition_identity.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(status='COMPLETE',rows=len(decomposed),joint_rows=len(joint),max_mixture_residual=max(abs(x['mixture_residual']) for x in decomposed))))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.root,a.out)

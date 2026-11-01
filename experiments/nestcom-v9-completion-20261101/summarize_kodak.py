"""Closed-input source-paired Kodak summary; no inference or new bootstrap draws."""
from pathlib import Path
import sys,csv,itertools,time
import numpy as np
R=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(R/'experiments/nestcom-opt-20261030'))
from opt_common import read,save,pin,checked,ident
B=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm');L=B/'nestcom_v9_completion_20261101_v1';D=R/'results/nestcom_v9_completion_20261101_v1'
METRICS=['psnr_db','dists','lpips_alex','dreamsim','clip_image_cosine']
def csvout(path,rows):
    fields=list(dict.fromkeys(k for v in rows for k in v))
    with path.open('x',newline='')as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
def main():
    started=time.time();receipts={}
    for role in ['token_source','token_phy','token_rx','token_three','token_clip','kodak_dists']:
        wait=read(L/(role+'_control')/'actual_wait.json');assert wait['returncode']==0 and ident(wait['child']['pid'])is None
        done=read(L/role/'completion.json');assert done['ledger']['unresolved']==0
        receipts[role]=dict(wait=wait,completion=done)
    bankmeta=read(R/'results/kodak_current_20261026_v1/statistics_v1/bootstrap_bank.json')
    bank=np.load(checked(bankmeta['bank']),allow_pickle=False);assert bank.shape==(10000,24)and bank.dtype==np.int64
    sources=read(L/'sources.json');assert [s['source_id']for s in sources]==['kodak/kodim%02d'%i for i in range(1,25)]
    assert [s['source_index']for s in sources]==bankmeta['source_order']
    out=D/'kodak';out.mkdir(exist_ok=False)
    save(out/'statistics_identity.json',dict(bank=bankmeta,source_order=[s['source_id']for s in sources],aggregation='mean of three events within each source, then equal sources; paired differences before bootstrap',percentile=[.025,.975],quantile_method='linear',new_random_draws=0,inputs=[pin(L/'metric_plan/frames.json')]+[pin(L/r/'scores.json')for r in ['token_three','token_clip','kodak_dists']],code=pin(Path(__file__))))
    frames=read(L/'metric_plan/frames.json');three=read(L/'token_three/scores.json');clip=read(L/'token_clip/scores.json');dists=read(L/'kodak_dists/scores.json')
    for f in frames:
        k=f['pair_key']
        if f['four_origin']=='NEW_TOKEN_PAIR':
            for m in ['psnr_db','lpips_alex','dreamsim']:f[m]=three[k][m]
            f['clip_image_cosine']=clip[k]['clip_image_cosine']
        f['dists']=dists[k]['dists'];assert all(np.isfinite(f[m])for m in METRICS)
        f['method']='Token'if f['method']=='NeST-Token'else f['method']
    methods=['WHOLE','Quarter','Token','Ada'];means=[];summary=[];vectors={};paired=[];dist=[]
    def ci(v):
        samples=v[bank].mean(1)
        return dict(mean=float(v.mean()),ci_low=float(np.quantile(samples,.025)),ci_high=float(np.quantile(samples,.975)))
    for method,snr in itertools.product(methods,[4,10,19]):
        ff=[f for f in frames if f['method']==method and f['snr_db']==snr];assert len(ff)==72
        for s in sources:
            fs=[f for f in ff if f['source_id']==s['source_id']];assert len(fs)==3 and len({f['noise_seed']for f in fs})==3
            means.append(dict(method=method,N=1024,snr_db=snr,source_id=s['source_id'],source_index=s['source_index'],events=3,**{m:float(np.mean([f[m]for f in fs]))for m in METRICS}))
        for m in METRICS:
            v=np.array([s[m]for s in means if s['method']==method and s['snr_db']==snr]);vectors[method,snr,m]=v
            summary.append(dict(method=method,N=1024,snr_db=snr,metric=m,sources=24,events=72,**ci(v)))
    for a,b in [('Token','WHOLE'),('Token','Quarter'),('Token','Ada'),('Quarter','WHOLE'),('Quarter','Ada'),('WHOLE','Ada')]:
        for snr,m in itertools.product([4,10,19],METRICS):
            v=vectors[a,snr,m]-vectors[b,snr,m];gain=v if m in ['psnr_db','clip_image_cosine']else -v
            paired.append(dict(comparison=a+' minus '+b,N=1024,snr_db=snr,metric=m,sources=24,events_per_arm=72,**ci(v)))
            dist.append(dict(comparison=a+' vs '+b,N=1024,snr_db=snr,metric=m,sources=24,better=int((gain>1e-10).sum()),equal=int((abs(gain)<=1e-10).sum()),worse=int((gain< -1e-10).sum()),median_gain=float(np.median(gain)),q25_gain=float(np.quantile(gain,.25)),q75_gain=float(np.quantile(gain,.75))))
    for name,rows in [('frame_metrics',frames),('source_means',means),('summary',summary),('paired_deltas',paired),('per_source_improvement',dist)]:csvout(out/(name+'.csv'),rows)
    csvout(out/'capacity.csv',[{k:f[k]for k in ['source_id','snr_db','noise_seed','m','K','payload_bits','capacity_bits','utilization']if k in f}for f in frames if f['method']=='Token'])
    save(out/'source_manifest.json',sources);save(out/'actual_compute_and_reuse.json',dict(receipts=receipts,formal_logical_new_arm=216,physical_new=213,physical_reused=3,formal_unique_rx=72,formal_new_rx=71,formal_reused_rx=1,reference_extra_rx=1,new_four_metric_pairs=71,new_dists_pairs=269,baseline_four_values_reused=648,token_four_logical_reused=3,device_seconds=sum(v['wait']['device_seconds']for v in receipts.values()),new_training=0,new_tokenizer=0))
    text='# Kodak24 V9 results\n\nN=1024; 24 fixed center-cropped sources, SNR 4/10/19, three original repetitions. Existing policies remain fixed. WHOLE, Quarter and Ada native outputs and four metrics were reused; DISTS was computed from native arrays for all methods. Ada is the predicted-class fixed-frame adaptation.\n\nAll 864 logical results are retained. Confidence intervals use the exact historical 10000 x 24 bank after within-source averaging. No new wireless experiment was performed for the reference-vector diagnostic.\n\n'
    for p in paired:
        if p['comparison'].startswith('Token'):text+=f"{p['comparison']}, {p['snr_db']} dB, {p['metric']}: {p['mean']:+.6f} [{p['ci_low']:+.6f}, {p['ci_high']:+.6f}]\n\n"
    (out/'REPORT_zh.md').write_text(text)
    save(out/'completion.json',dict(status='COMPLETE',frames=864,source_rows=len(means),mean_intervals=len(summary),paired_intervals=len(paired),statuses=sorted(set(f['status']for f in frames)),all_denominators_preserved=True,new_scientific_calls=0,seconds=time.time()-started))
    print(read(out/'completion.json'))
    for p in paired:
        if p['comparison'].startswith('Token')and p['metric']in ['psnr_db','dists','dreamsim']:print(p)
if __name__=='__main__':main()

"""CPU-only paired calibration summary; a single registered shared index bank."""
from common import *
import csv,collections,itertools
import numpy as np

def csvread(p):
    with pathlib.Path(p).open(newline='')as f:return list(csv.DictReader(f))
def csvsave(p,rows):
    with pathlib.Path(p).open('x',newline='')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    reg=read(O/'decoder_pilot_registration.json');regsha=sha(O/'decoder_pilot_registration.json')
    initial=read(L/'decoder_initial_control_v1/launch.json')['process'];assert process(initial['pid'])is None
    out=O/'decoder_delivery_v1';assert not out.exists()
    index=read(checked(reg['seed_index']));ids=[x['source_id']for x in index['sources']if x['split']=='calibration'];assert len(ids)==100
    baseline=csvread(checked(read(checked(reg['seed_closure']))['rows']));records=[]
    for x in baseline:
        if x['split']=='calibration':
            row=dict(x,arm='current_dc');row['utility']=float(row['mse'])+.1*float(row['lpips_alex']);records.append(row)
    closures=[];initialshas=[];exposure=None
    for arm in reg['arms']:
        d=L/'decoder_dispatch_v1'/arm;p=L/'decoder_pilot_v1'/arm
        w=read(d/'actual_wait.json');assert w['returncode']==0 and not w['timed_out'] and process(w['process']['pid'])is None and not(p/'failure.json').exists()
        c=read(p/'completion.json');assert c['registration_sha256']==regsha and c['training_updates']==5000 and c['training_samples']==20000 and c['calibration_endpoint_rows']==800
        for k in ['checkpoint','calibration','ledger','updates','initial_checkpoint']:checked(c[k])
        events=[json.loads(x)for x in checked(c['ledger']).read_text().splitlines()];start=collections.Counter((x['kind'],x['key'])for x in events if x['status']=='STARTED');end=collections.Counter((x['kind'],x['key'])for x in events if x['status']=='COMPLETE')
        assert start==end and all(v==1 for v in start.values()) and all(x['status']in['STARTED','COMPLETE']for x in events)
        identity=read(p/'identity.json');initialshas.append(identity['initial_model_sha256'])
        current=[x['pair_indices']for x in read(checked(c['updates']))]
        if exposure is None:exposure=current
        else:assert current==exposure
        r=csvread(checked(c['calibration']));assert len(r)==800;records.extend(r)
        closures.append(dict(arm=arm,completion=pin(p/'completion.json'),actual_wait=pin(d/'actual_wait.json'),checkpoint=c['checkpoint'],identity=identity,counts=dict(collections.Counter(k[0]for k in end))))
    assert len(set(initialshas))==1 and len(records)==3200
    out.mkdir();b=L/'decoder_statistics_v1';b.mkdir(exist_ok=False)
    bank=np.random.default_rng(reg['bootstrap']['seed']).integers(0,100,(10000,100),dtype=np.int64)
    with(b/'calibration_bootstrap_indices.npy').open('xb')as f:np.save(f,bank,allow_pickle=False)
    metrics=['mse','psnr_db','lpips_alex','ms_ssim','utility'];arms=['current_dc']+list(reg['arms']);lookup={(x['arm'],x['source_id'],int(x['m']),int(x['K'])):x for x in records};means=[];paired=[]
    for m,k in reg['endpoints']:
        for metric in metrics:
            values={arm:np.asarray([float(lookup[arm,sid,m,k][metric])for sid in ids])for arm in arms}
            for arm,v in values.items():
                lo,hi=np.quantile(v[bank].mean(1),[.025,.975]);means.append(dict(arm=arm,m=m,K=k,metric=metric,mean=float(v.mean()),ci_low=float(lo),ci_high=float(hi),sources=100,subset='calibration',unit='source'))
            for a,bref in [('multiprefix_unconditional','current_dc'),('old_recipe_control','current_dc'),('endpoint_conditioned','current_dc'),('multiprefix_unconditional','old_recipe_control'),('endpoint_conditioned','multiprefix_unconditional')]:
                delta=values[a]-values[bref];lo,hi=np.quantile(delta[bank].mean(1),[.025,.975]);paired.append(dict(comparison=a+'_minus_'+bref,m=m,K=k,metric=metric,mean=float(delta.mean()),ci_low=float(lo),ci_high=float(hi),sources=100,subset='calibration',unit='source-paired'))
    keys=['source_id','source_index','split','arm','m','K','terminated_bits']+metrics
    csvsave(out/'source_metrics.csv',[{k:x[k]for k in keys}for x in records]);csvsave(out/'summary.csv',means);csvsave(out/'paired_differences.csv',paired)
    save(out/'closure_audit.json',dict(status='PASS',registration=pin(O/'decoder_pilot_registration.json'),arms=closures,shared_initial_state=True,identical_pair_order=True,three_arms_updates=15000,three_arms_training_samples=60000,new_calibration_rows=2400,reused_current_dc_calibration_rows=800,bootstrap=pin(b/'calibration_bootstrap_indices.npy'),bootstrap_seed=reg['bootstrap']['seed'],scientific_subset='Calibration100 only; no heldout or wireless claims',encoder_or_VAR_updates=0))
    print(json.dumps(dict(summary=pin(out/'summary.csv'),paired=pin(out/'paired_differences.csv'),closure=pin(out/'closure_audit.json'))))

if __name__=='__main__':main()

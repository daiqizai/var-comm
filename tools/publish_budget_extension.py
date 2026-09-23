"""Publish an immutable completed budget extension, without launching GPU work."""
import argparse
import csv
import itertools
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess
from tools.publish_budget_milestone import ROOT, digest, read, check


def calibration(path, N, step, expected_ids=None):
    receipt=read(path.with_suffix('.json'))
    check(receipt['step']==step and receipt['sha256']==digest(path),'calibration receipt/hash')
    with path.open() as f: rows=list(csv.DictReader(f))
    keys=[(r['image_id'],int(r['snr_db']),int(r['seed'])) for r in rows]
    ids={r['image_id']:int(r['source_index']) for r in rows}
    check(len(rows)==receipt['rows']==15000 and len(ids)==1000,'complete calibration size')
    check(set(ids.values())==set(range(1000)) and (expected_ids is None or ids==expected_ids),'source identity')
    check(len(set(keys))==len(keys) and set(keys)==set(itertools.product(ids,(1,4,7,13,19),(4101,4102,4103))),'source/SNR/noise grid')
    for r in rows:
        check(r['method']==f'P{N}' and int(r['source_index'])==ids[r['image_id']],'arm/source')
        check(all(math.isfinite(float(r[k])) for k in ('mse','lpips_alex','normalized_latent','utility','U_image')),'nonfinite metric')
        check(abs(float(r['U_image'])-float(r['mse'])-.1*float(r['lpips_alex']))<1e-7,'U_image')
    mean=statistics.fmean(float(r['utility']) for r in rows)
    check(abs(mean-receipt['summary'][f'P{N}'])<1e-12,'selection mean')
    return ids,{'step':step,'selection_utility':mean,**{k:statistics.fmean(float(r[k]) for r in rows) for k in ('U_image','mse','lpips_alex','normalized_latent')}}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--N',type=int,choices=(2048,3060),required=True);p.add_argument('--until',type=int,required=True);a=p.parse_args()
    check(a.until>=30000 and a.until%10000==0,'completed 10k extension boundary')
    run=f'P{a.N}_seed2026092304';source=ROOT/'outputs/TOKEN-CHANNEL-EFFICIENCY-20260923/training'/run
    extension=source/'extensions'/f'until_{a.until}';done=read(extension/'completion.json');plan=read(extension/'plan.json');selected=done['selected'][f'P{a.N}']
    check(done['synthetic'] is False and done['state']['step']==a.until and plan['until']==a.until,'actual extension completion')
    check(plan['parent']['step']==a.until-10000 and plan['N']==a.N,'extension parent scope')
    check(digest(plan['parent']['path'])==plan['parent']['sha256'],'actual parent checkpoint')
    check(digest(selected['checkpoint'])==selected['checkpoint_sha256'],'actual selected checkpoint')
    check(digest(source/'registration.json')==selected['registration_sha256']==done['registration_sha256'],'historical registration')
    files=[extension/n for n in ('completion.json','plan.json','registration.json')]+[source/'decisions'/f'at_{a.until:05d}.json']
    decision=read(files[-1]);check(decision['step']==a.until and decision['selected']==selected,'completed decision/selected identity')
    check(digest(plan['decision'])==plan['decision_sha256'],'parent decision binding');files.append(Path(plan['decision']))
    bindings={}
    for reg in [source/'registration.json',source/'execution/microbatch8-v1/registration.json',*sorted((source/'extensions').glob('*/registration.json'))]:
        if not reg.exists():continue
        record=read(reg)
        if record.get('spec',{}).get('until',0)>a.until:continue
        for name,h in record['bindings'].items():check(digest(name)==h,'runtime source/config binding: '+name)
        bindings.update(record['bindings']);files.append(reg)
    e=selected.get('execution_identity');check(e is not None and digest(e['path'])==e['sha256'],'selected execution identity')
    qualification=read(source/'qualification.json');check(qualification['synthetic'] is False and qualification['status'].endswith('_PASS'),'real original qualification')
    curves=[];ids=None;cal_refs={}
    for step in range(0,a.until+1,2500):
        path=source/'calibration'/f'full_{step:05d}.csv';ids,row=calibration(path,a.N,step,ids);curves.append(row)
        if step>plan['parent']['step']:files.extend([path,path.with_suffix('.json')])
        else:cal_refs[str(path.relative_to(ROOT))]=digest(path)
    best=min(curves,key=lambda r:(r['selection_utility'],r['step']))
    check(selected['step']==best['step'] and abs(selected['utility']-best['selection_utility'])<1e-12,'full-history selected minimum')
    values=[r['selection_utility'] for r in curves[-3:]];improvements=[(x-y)/x for x,y in zip(values,values[1:])]
    check(decision['utilities']==values and decision['relative_improvements']==improvements,'decision calibration evidence')
    check(decision['extend']==all(x>=.002 for x in improvements),'registered extension criterion')
    for name,h in decision['evidence_bindings'].items():check(digest(name)==h,'decision evidence binding')
    final=source/'finalization.json'
    if not decision['extend']:
        f=read(final);check(f['selected']==selected and f['completed_step']==a.until and f['decision_sha256']==digest(source/'decisions'/f'at_{a.until:05d}.json'),'finalization identity');files.append(final)
    dest=ROOT/'results/token_channel_efficiency_20260923/budget_extensions'/f'{run}_until{a.until}'
    check(not dest.exists(),'immutable publication exists');dest.mkdir(parents=True)
    published={}
    for path in sorted(set(files)):
        rel=path.relative_to(source);target=dest/rel;target.parent.mkdir(exist_ok=True,parents=True);shutil.copyfile(path,target);published[str(rel)]={'source':str(path.relative_to(ROOT)),'sha256':digest(target),'bytes':target.stat().st_size}
    (dest/f'selected_P{a.N}.json').write_text(json.dumps(selected,indent=2)+'\n')
    with (dest/'calibration_curve.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(curves[0]));w.writeheader();w.writerows(curves)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,4));ax.plot([r['step'] for r in curves],[r['selection_utility'] for r in curves],'o-',label='Selection utility');ax.plot([r['step'] for r in curves],[r['U_image'] for r in curves],'s--',label='MSE + 0.1 LPIPS');ax.axvline(plan['parent']['step'],color='gray',linestyle=':',label='Extension parent');ax.set(xlabel='Total training updates',ylabel='1000-source calibration mean',title=f'P{a.N} through {a.until} updates');ax.grid(alpha=.25);ax.legend();fig.tight_layout();svg=dest/'calibration_curve.svg';fig.savefig(svg);plt.close(fig);svg.write_text('\n'.join(x.rstrip() for x in svg.read_text().splitlines())+'\n')
    audit={'status':'REAL_BUDGET_EXTENSION_AND_DECISION_VERIFIED_NOT_QUALITY_EVALUATION','N':a.N,'E':2*a.N,'completed_step':a.until,'selected':selected,'calibration_rows_checked':len(curves)*15000,'new_calibration_rows_published':60000,'source_count':1000,'relative_improvements':improvements,'extend_again':decision['extend'],'synthetic':False,'development_used':False,'new_holdout_used':False,'training_launched':False,'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'publisher_sha256':digest(__file__),'verified_runtime_bindings':bindings,'prior_calibration_references':cal_refs,'files':published,'pending':['selected development and shared online timing','B1/B2/C and historical compatibility','final paired publication and remote acceptance']}
    (dest/'audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps({k:audit[k] for k in ('status','completed_step','calibration_rows_checked','relative_improvements','extend_again')},indent=2))

if __name__=='__main__':main()

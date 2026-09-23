"""Calibration-only paired extension and preregistered short-prefix choices."""
import csv,math
from pathlib import Path
from latent_enhancement.runtime import digest,verify_snapshot
from latent_followup.run_identity import checked_checkpoint
from .common import ROOT,read,register
from .budget_lifecycle import RULE,should_extend

SHORT=ROOT/'outputs/SHORT-PREFIX-20260923'

def next_group_action(folder):
    folder=Path(folder);done=read(folder/'completion.json');step=done['state']['step'];latest=read(folder/'latest.json');checked_checkpoint(latest,ROOT)
    for p in sorted((folder/'delivery_decisions').glob('at_*.json')):
        old=read(p)
        if old['extend'] and step<old['until']:return {'kind':'extend','until':old['until']}
    if step<20000 or step%10000:raise RuntimeError('completed20k/10k C decision boundary')
    arms=sorted(done['state']['updates']);evidence={};values={n:[] for n in arms}
    for s in (step-5000,step-2500,step):
        p=folder/'calibration'/f'full_{s:05d}.json';r=read(p);csvfile=p.with_suffix('.csv')
        if r['step']!=s or r['rows']!=15000*len(arms) or r['sha256']!=digest(csvfile):raise RuntimeError('C full calibration grid/receipt')
        for n in arms:values[n].append(r['summary'][n])
        evidence[str(p)]=digest(p);evidence[str(csvfile)]=digest(csvfile)
    arm_decisions={n:{'extend':should_extend(v)[0],'relative_improvements':should_extend(v)[1]} for n,v in values.items()};extend=any(r['extend'] for r in arm_decisions.values())
    record={'rule':RULE,'step':step,'arms':arm_decisions,'extend':extend,'until':step+10000 if extend else step,'paired_arms_extend_together':True,'values':values,'evidence_bindings':evidence,'development_used':False}
    p=folder/'delivery_decisions'/f'at_{step:05d}.json';register(p,record)
    if extend:return {'kind':'extend','until':step+10000}
    return {'kind':'final','step':step,'decision':str(p),'sha256':digest(p)}

def selected_score(folder,arm):
    folder=Path(folder);rec=read(folder/f'selected_{arm}.json');checked_checkpoint(rec,ROOT);p=folder/'calibration'/f"full_{rec['step']:05d}.json";meta=read(p);csvfile=p.with_suffix('.csv')
    if digest(csvfile)!=meta['sha256']:raise RuntimeError('selected calibration hash')
    with csvfile.open() as f:rows=[r for r in csv.DictReader(f) if r['method']==arm and float(r['snr_db']) in (1,4,7)]
    if len(rows)!=9000 or len({(r['image_id'],float(r['snr_db']),int(r['seed'])) for r in rows})!=9000 or len({r['image_id'] for r in rows})!=1000:raise RuntimeError('selected primary calibration coverage')
    values=[float(r['U_image']) for r in rows]
    if not all(math.isfinite(x) for x in values):raise RuntimeError('finite primary utility')
    return {'primary_U_image':sum(values)/len(values),'selected':rec,'selected_sha256':digest(folder/f'selected_{arm}.json'),'calibration_csv':str(csvfile),'calibration_csv_sha256':digest(csvfile)}

def choose_candidate(destination):
    candidates={str(m):selected_score(SHORT/'training'/f'm{m}_seed2026092304',f'H{m}-V') for m in (6,7)}
    m=min((6,7),key=lambda n:(candidates[str(n)]['primary_U_image'],n))
    record={'status':'FROZEN_CALIBRATION_ONLY_SHORT_PREFIX_CHOICE','m':m,'rule':'minimum selected-model complete primary1/4/7 calibration U_image; ties lower m','candidates':candidates,'N3060_group':f'm{m}','repeat_controls':[f'H{m}-P','H8-V','P4084'],'additional_seeds':[2026092404,2026092504],'development_used':False,'new_holdout':False}
    register(Path(destination),record);return record

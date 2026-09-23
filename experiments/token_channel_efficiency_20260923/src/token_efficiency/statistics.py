"""Strict full-grid source-level statistics; no development-selected policies.

The caller supplies the frozen population and expected eligible method roster.
Unencodable cells belong in a separate candidate ledger, never partial frame grids.
"""
import hashlib,json
import numpy as np

METRICS=('mse','psnr_db','lpips_alex','dino_cosine')
CONTEXT=('run_id','context_sha256','family','N','mcs','energy_constraint','noise_namespace')

def canonical_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

class FrameTable:
    def __init__(self,rows,population,sources,snrs,seeds,methods):
        self.population=population;self.sources=dict(sources);self.ids=tuple(self.sources);self.snrs=tuple(snrs);self.seeds=tuple(seeds);self.methods=tuple(methods)
        if population not in ('calibration','development') or not self.ids or not self.methods or not self.snrs or not self.seeds or len(set(self.methods))!=len(self.methods) or len(set(self.snrs))!=len(self.snrs) or len(set(self.seeds))!=len(self.seeds):raise ValueError('explicit unique population/grid/method roster required')
        if len(rows)!=len(self.ids)*len(self.snrs)*len(self.seeds)*len(self.methods):raise ValueError('incomplete frame grid')
        self.context={};self.frames={};positions={v:i for i,v in enumerate(self.ids)};snrpos={v:i for i,v in enumerate(self.snrs)};seedpos={v:i for i,v in enumerate(self.seeds)}
        self.values={m:np.empty((len(self.ids),len(self.snrs),len(self.seeds),len(METRICS))) for m in self.methods}
        self.energies={m:np.empty((len(self.ids),len(self.snrs),len(self.seeds))) for m in self.methods}
        for row in rows:
            method=row['method'];source=row['source_id'];snr=row['snr_db'];seed=row['noise_seed']
            if method not in self.methods or source not in positions or snr not in snrpos or seed not in seedpos or row['population']!=population or row['preprocessing_id']!=self.sources[source]:raise ValueError('row escaped registered population/preprocessing/grid')
            key=(method,source,snr,seed)
            if key in self.frames:raise ValueError('duplicate frame key')
            context={k:row[k] for k in CONTEXT}
            if any(row[k] in ('',None) for k in CONTEXT):raise ValueError('missing execution context')
            if method in self.context and self.context[method]!=context:raise ValueError('mixed run/model/protocol/budget/noise scope within method')
            self.context[method]=context
            numbers=np.asarray([row[k] for k in METRICS],dtype=float);energy=float(row['E'])
            if not np.isfinite(numbers).all() or not np.isfinite(energy) or energy<=0 or numbers[0]<0 or numbers[2]<0:raise ValueError('invalid/nonfinite frame metric or energy')
            N=row['N'];segments=[row[k] for k in ('N_header','N_data','N_continuous')]
            if any(isinstance(x,bool) or not isinstance(x,(int,np.integer)) or x<0 for x in [N,*segments]) or sum(segments)!=N or N not in (2048,3060,4084):raise ValueError('actual integer resource ledger')
            if row['energy_constraint']=='per_frame_2N' and not np.isclose(energy,2*N,atol=.02,rtol=1e-5):raise ValueError('per-frame energy violation')
            if row['energy_constraint'] not in ('per_frame_2N','fixed_constellation_average_2_per_symbol'):raise ValueError('unregistered energy constraint')
            if row['family']=='continuous' and (row['energy_constraint']!='per_frame_2N' or row['mcs']!='continuous'):raise ValueError('continuous protocol constraint')
            if row['mcs']=='16QAM' and row['energy_constraint']!='fixed_constellation_average_2_per_symbol':raise ValueError('16QAM cannot claim per-frame constant energy')
            # Failure events must be retained, including a legitimate not_applicable.
            for field in ('header_ok','body_crc_ok','source_overflow_erasure'):
                if row[field] not in (True,False,0,1,'not_applicable'):raise ValueError('failure status missing')
            pos=(positions[source],snrpos[snr],seedpos[seed]);self.values[method][pos]=numbers;self.energies[method][pos]=energy;self.frames[key]=dict(row)
        self.sha256=canonical_sha([self.frames[k] for k in sorted(self.frames)])
    def source_values(self,method,snr):return self.values[method][:,self.snrs.index(snr)].mean(axis=1)
    def summary(self):
        result=[]
        for method in self.methods:
            for snr in self.snrs:
                source=self.source_values(method,snr);e=self.energies[method][:,self.snrs.index(snr)].ravel()
                result.append({'method':method,'population':self.population,'snr_db':snr,'sources':len(self.ids),'noise_seeds':len(self.seeds),**self.context[method],**dict(zip(METRICS,map(float,source.mean(axis=0)))),'E_mean':float(e.mean()),'E_min':float(e.min()),'E_p05':float(np.quantile(e,.05)),'E_p95':float(np.quantile(e,.95)),'E_max':float(e.max()),'table_sha256':self.sha256})
        return result

def freeze_digital_policy(table):
    if table.population!='calibration':raise ValueError('policy selection requires calibration')
    groups={}
    for method,ctx in table.context.items():
        if ctx['family'] not in ('raw','arithmetic'):continue
        # Keep constant-energy QPSK and average-energy16QAM separate. Protocol
        # versions may compete only as entire pre-shared SNR/cell configurations.
        key=(ctx['family'],ctx['N'],ctx['mcs'],ctx['energy_constraint']);groups.setdefault(key,[]).append(method)
    if not groups:raise ValueError('no digital candidates')
    choices=[]
    for key,methods in sorted(groups.items()):
        for snr in table.snrs:
            scores={m:float(np.mean(table.source_values(m,snr)[:,0]+.1*table.source_values(m,snr)[:,2])) for m in methods}
            selected=min(methods,key=lambda m:(scores[m],m))
            choices.append({'family':key[0],'N':key[1],'mcs':key[2],'energy_constraint':key[3],'snr_db':snr,'method':selected,'context_sha256':table.context[selected]['context_sha256'],'calibration_utility':scores[selected],'candidate_scores':scores})
    return {'status':'FROZEN_CALIBRATION_POLICY','selection':'source mean of MSE+0.1LPIPS; deterministic method-name tie break; DINO report-only','calibration_table_sha256':table.sha256,'choices':choices,'noise_pairing':'same source and registered seed index; namespace and waveform length may differ'}

def validate_policy_on_development(table,policy):
    if table.population!='development' or policy.get('status')!='FROZEN_CALIBRATION_POLICY' or not policy.get('calibration_table_sha256'):raise ValueError('development with frozen calibration policy required')
    expected={(c['family'],c['N'],c['mcs'],c['energy_constraint'],s) for c in table.context.values() if c['family'] in ('raw','arithmetic') for s in table.snrs}
    found=set()
    for choice in policy['choices']:
        key=tuple(choice[k] for k in ('family','N','mcs','energy_constraint','snr_db'))
        if key in found:raise ValueError('duplicate policy cell')
        found.add(key);ctx=table.context.get(choice['method'])
        if not ctx or ctx['context_sha256']!=choice['context_sha256'] or any(ctx[k]!=choice[k] for k in ('family','N','mcs','energy_constraint')):raise ValueError('selected method execution context changed')
    if found!=expected:raise ValueError('policy/development cell coverage mismatch')

def paired(table,left,right,snr,*,repeats=10000,seed=20260923):
    a=table.source_values(left,snr);b=table.source_values(right,snr);difference=a-b
    if not np.allclose(difference.mean(0),a.mean(0)-b.mean(0),atol=1e-12,rtol=1e-12):raise RuntimeError('inconsistent paired estimand')
    if repeats<1:raise ValueError('bootstrap repetitions')
    rng=np.random.default_rng(seed);means=[]
    for start in range(0,repeats,250):
        ids=rng.integers(0,len(a),(min(250,repeats-start),len(a)));means.append(difference[ids].mean(axis=1))
    intervals=np.quantile(np.concatenate(means),[.025,.975],axis=0)
    return {'left':left,'right':right,'snr_db':snr,'sources':len(a),'paired_unit':'source image; average registered noise seeds first','same_noise_observations_claimed':False,'left_context':table.context[left],'right_context':table.context[right],'bootstrap_repeats':repeats,'training_seed_variation_included':False,'table_sha256':table.sha256,'metrics':{m:{'mean_left':float(a[:,i].mean()),'mean_right':float(b[:,i].mean()),'difference':float(difference[:,i].mean()),'ci95':intervals[:,i].tolist()} for i,m in enumerate(METRICS)}}

def target_rates(table,method,snr,target):
    frames=table.values[method][:,table.snrs.index(snr)];source=frames.mean(axis=1);mean=source.mean(axis=0)
    return {'mean_target_met':bool(mean[1]>=target['psnr_min'] and mean[2]<=target['lpips_max']),'frame_joint_success_rate':float(((frames[:,:,1]>=target['psnr_min'])&(frames[:,:,2]<=target['lpips_max'])).mean()),'source_noise_mean_success_rate':float(((source[:,1]>=target['psnr_min'])&(source[:,2]<=target['lpips_max'])).mean()),'mean_psnr_db':float(mean[1]),'mean_lpips_alex':float(mean[2])}

def minimum_tested_uses(table,policy,targets,continuous_methods):
    validate_policy_on_development(table,policy)
    if targets.get('status')!='FROZEN_FROM_CALIBRATION_BEFORE_NEW_DEVELOPMENT' or not targets.get('source_csv_sha256'):raise ValueError('frozen quality target identity required')
    for target in targets['targets'].values():
        if not np.isfinite([target['psnr_min'],target['lpips_max']]).all() or target['lpips_max']<0:raise ValueError('finite valid quality targets required')
    if not continuous_methods or any(table.context[m]['family']!='continuous' for m in continuous_methods):raise ValueError('explicit continuous reference roster required')
    if len({table.context[m]['N'] for m in continuous_methods})!=len(continuous_methods):raise ValueError('one calibration-selected continuous checkpoint per budget, no development oracle')
    rows=[]
    for snr in table.snrs:
        for level,target in targets['targets'].items():
            candidates=[(m,target_rates(table,m,snr,target)) for m in continuous_methods]
            eligible=[m for m,r in candidates if r['mean_target_met']]
            continuous_N=min((table.context[m]['N'] for m in eligible),default=None)
            families=sorted({(c['family'],c['mcs'],c['energy_constraint']) for c in policy['choices']})
            for family,mcs,constraint in families:
                methods=[c['method'] for c in policy['choices'] if c['snr_db']==snr and (c['family'],c['mcs'],c['energy_constraint'])==(family,mcs,constraint)]
                evidence={m:target_rates(table,m,snr,target) for m in methods};valid=[m for m,r in evidence.items() if r['mean_target_met']]
                digital_N=min((table.context[m]['N'] for m in valid),default=None)
                rows.append({'snr_db':snr,'target':level,'family':family,'mcs':mcs,'energy_constraint':constraint,'continuous_N_required':continuous_N,'digital_N_required':digital_N,'saving':None if continuous_N is None or digital_N is None else 1-digital_N/continuous_N,'status':'MEASURED_GRID_TARGET_MET' if continuous_N is not None and digital_N is not None else 'NOT_REACHED_IN_MEASURED_GRID','strict_per_frame_equal_energy_protocol':constraint=='per_frame_2N','digital_evidence':evidence,'continuous_evidence':dict(candidates),'interpolation_used':False,'table_sha256':table.sha256})
    return rows

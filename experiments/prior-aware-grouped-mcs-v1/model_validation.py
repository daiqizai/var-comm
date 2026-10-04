"""Post-freeze diagnosis of the expected-quality model on matched source images.

No policy selection, neural inference, additional samples or N2048 execution.
Insufficient statistical resolution is explicit UNCERTAIN, never automatic PASS.
"""
from __future__ import annotations
import argparse
import csv
import gzip
import itertools
from pathlib import Path
from statistics import NormalDist
import numpy as np
import optimize as o
from uep_common import read,require,sha,identity,seal

HERE=Path(__file__).resolve().parent
RULES=HERE/'model_validation_rules.json'
METHODS=('B3','strong_baseline','validation_candidate')
STATES=('gray','prefix','full','other')


def wilson_fraction(mean,n,z):
    den=1+z*z/n;center=(mean+z*z/(2*n))/den
    radius=z*np.sqrt(mean*(1-mean)/n+z*z/(4*n*n))/den
    return max(0.,center-radius),min(1.,center+radius)


def interval(values,draws,mass=.975):
    values=np.asarray(values,dtype=np.float64)
    require(values.ndim==1 and np.isfinite(values).all(),'Finite matched-source vector required')
    lower,upper=np.quantile(values[draws].mean(1),[(1-mass)/2,1-(1-mass)/2])
    return dict(mean=float(values.mean()),ci_low=float(lower),ci_high=float(upper),n_sources=len(values))


def weight_vectors(probabilities):
    header,*groups=probabilities;first=header*groups[0]
    full=first if len(groups)==1 else first*groups[1]
    return dict(gray=1-first,prefix=np.zeros_like(first) if len(groups)==1 else first-full,
                full=full,other=np.zeros_like(first))


def prediction(profile,snr,ids,quality,probabilities,z):
    keys=[profile['header_phy_key'],*[g['phy_key'] for g in profile['groups']]]
    means=[];bounds=[]
    for key in keys:
        values,_,rows=probabilities.vector(key,snr);means.append(values);lo=[];hi=[]
        for row in rows:
            n=row.get('n_blocks',0);correct=row.get('n_correct',-1)
            require(n>=256 and (n-correct>=100 or n>=20000),'A validation physical point has not finished refinement')
            a,b=wilson_fraction(correct/n,n,z);lo.append(a);hi.append(b)
        bounds.append((np.asarray(lo),np.asarray(hi)))
    gray=np.asarray([quality[source,'gray','gray'] for source in ids])
    states=([profile['prefix_state_id']] if profile['G']==2 else [])+[profile['full_state_id']]
    qs=[np.asarray([quality[source,state,'VAR'] for source in ids]) for state in states]
    estimate=o.expected_quality(gray,qs,means[0],means[1:])
    qlower,qupper=o.expected_quality_bounds(gray,qs,bounds[0],bounds[1:])
    weights=weight_vectors(means);corners=[weight_vectors(values) for values in itertools.product(*bounds)]
    state_bounds={key:(np.min([x[key] for x in corners],axis=0),np.max([x[key] for x in corners],axis=0)) for key in STATES}
    return dict(quality=estimate,lower=qlower,upper=qupper,weights=weights,state_bounds=state_bounds)


def observed_state(frame,profile):
    state=frame['receiver_event']['state'];truth=frame['offline_truth']
    if state['kind']=='gray':return 'gray'
    if not truth.get('header_correct',False):return 'other'
    groups=truth.get('groups',[]);kept=int(state['accepted_groups'])
    if len(groups)<kept or any(not g.get('truth_correct_after_receiver',False) for g in groups[:kept]):return 'other'
    if (state['m'],state['K'])==(profile['m'],profile['K']):return 'full'
    if profile['G']==2 and (state['m'],state['K'])==(profile['j'],0):return 'prefix'
    return 'other'


def envelope(actual,predicted,pred_low,pred_high,draws,mass):
    residual=interval(actual-predicted,draws,mass)
    residual['envelope_low']=residual['ci_low']-float(np.mean(pred_high-predicted))
    residual['envelope_high']=residual['ci_high']+float(np.mean(predicted-pred_low))
    residual['absolute_error_upper']=max(abs(residual['envelope_low']),abs(residual['envelope_high']))
    return residual


def validate(policies,bler,source_ids,quality_rows,scored_rows,frames,*,synthetic=False,bootstrap_replicates=None):
    rules=read(RULES);count=len(source_ids)
    require(count==len(set(source_ids)) and (synthetic or count==100),'Exactly registered100 development sources required')
    require(set(source_ids).isdisjoint(policies['source_ids']),'Calibration/development source overlap')
    require(synthetic or (policies['status']=='CALIBRATION_POLICIES_SELECTED' and policies['synthetic'] is False
        and policies['source_count']==1000 and policies['development_read'] is False),'Final frozen1000cal policy required')
    require(bler['status']=='REFINEMENT_COMPLETE' and bler['synthetic'] is False,'Actual completed refined probabilities required')
    cells={c['snr_db']:c for c in policies['cells'] if c['N']==1024 and c['snr_db'] in o.ACTUAL_SNRS}
    require(set(cells)==set(o.ACTUAL_SNRS) and all(c['ready_for_actual_link'] for c in cells.values()),'Incomplete four-point frozen policy')
    profiles={};choices={}
    for snr,cell in cells.items():
        choices[snr]={'B3':cell['independent_optima']['B3']['selected'],'strong_baseline':cell['strong_baseline'],
            'validation_candidate':cell['frozen_distinct_validation_candidate']['selected']}
        require(cell['frozen_distinct_validation_candidate']['status']=='FROZEN','Validation neighbour not frozen on calibration')
        for method,choice in choices[snr].items():profiles[snr,method]=choice['profile']
        require(len({profiles[snr,m]['wire_key'] for m in METHODS})>=2 and
            all(profiles[snr,'validation_candidate']['wire_key']!=profiles[snr,m]['wire_key'] for m in ('B3','strong_baseline')),
            'Validation neighbour must be a distinct frozen physical resource configuration')
    expected={(source,snr,seed,method) for source in source_ids for snr in o.ACTUAL_SNRS for seed in o.SEEDS for method in METHODS}
    scored={}
    for row in scored_rows:
        method=row.get('method',row.get('family'));snr=int(row['snr_db'])
        if method not in METHODS or snr not in o.ACTUAL_SNRS:continue
        key=(row['source_id'],snr,int(row['noise_seed']),method)
        require(key in expected and key not in scored and int(row['N'])==1024,'Duplicate/unexpected validation metric row')
        frame=frames[key];require(frame['synthetic'] is synthetic and frame['actual_bit_chain_executed'] is (not synthetic),'Nonphysical receive frame')
        require(row['stable_id']==profiles[snr,method]['stable_id'] and row['physical_frame_id']==frame['physical_frame_id'],'Metric/actual frame/frozen policy mismatch')
        scored[key]=o.finite(row[o.OBJECTIVE],o.OBJECTIVE)
    require(set(scored)==expected and expected<=set(frames),'Incomplete fixed-source three-noise actual comparison')
    quality={}
    for row in quality_rows:
        require(row['source_id'] in source_ids and row['receiver'] in ('VAR','gray'),'No matched development source-Q row')
        require(row.get('diagnostic_only') is True and row.get('development_Q_used_for_selection') is False
            and row.get('policy_reselection_allowed') is False,'Development Q must be diagnostic-only')
        key=(row['source_id'],row['state_id'],row['receiver']);require(key not in quality,'Duplicate development Q state')
        quality[key]=o.finite(row[o.OBJECTIVE],o.OBJECTIVE)
    require(all((source,'gray','gray') in quality for source in source_ids),'Missing same-source gray Q')
    probability=o.ProbabilityTable(bler,source_ids)
    for (snr,method),profile in profiles.items():
        for key in [profile['header_phy_key'],*[g['phy_key'] for g in profile['groups']]]:probability.vector(key,snr)
    # This correction includes shared probability keys once, across all points.
    z=NormalDist().inv_cdf(1-rules['physical_probability_joint_alpha']/(2*len(probability.used)))
    repeats=bootstrap_replicates or rules['bootstrap_replicates']
    require(synthetic or repeats==10000,'Production bootstrap count cannot be reduced')
    draws=np.random.default_rng(rules['bootstrap_seed']).integers(0,count,(repeats,count))
    mass=rules['source_bootstrap_interval_mass'];z_source=NormalDist().inv_cdf((1+mass)/2)
    results=[]
    for snr in o.ACTUAL_SNRS:
        estimates={};actuals={};per_method={};state_pass=True;state_fail=False
        for method in METHODS:
            profile=profiles[snr,method];pred=prediction(profile,snr,source_ids,quality,probability,z);estimates[method]=pred
            actual=np.asarray([np.mean([scored[source,snr,seed,method] for seed in o.SEEDS]) for source in source_ids]);actuals[method]=actual
            frequency=[]
            for state_name in STATES:
                values=np.asarray([np.mean([observed_state(frames[source,snr,seed,method],profile)==state_name for seed in o.SEEDS]) for source in source_ids])
                lo,hi=pred['state_bounds'][state_name];check=envelope(values,pred['weights'][state_name],lo,hi,draws,mass)
                # Source-level bootstrap collapses at zero observed errors.
                # A source-effective Wilson envelope preserves finite-sample uncertainty.
                wl,wh=wilson_fraction(float(values.mean()),count,z_source)
                check['envelope_low']=min(check['envelope_low'],wl-float(hi.mean()))
                check['envelope_high']=max(check['envelope_high'],wh-float(lo.mean()))
                margin=rules['state_frequency_equivalence_margin'];passed=check['envelope_low']>-margin and check['envelope_high']<margin
                failed=check['envelope_low']>margin or check['envelope_high']<-margin
                state_pass=state_pass and passed;state_fail=state_fail or failed
                frequency.append(dict(state=state_name,actual_mean=float(values.mean()),predicted_mean=float(pred['weights'][state_name].mean()),
                    equivalence_margin=margin,qualified=passed,status='PASS' if passed else 'FAIL' if failed else 'UNCERTAIN',**check))
            mismatch=[]
            for source in source_ids:
                for seed in o.SEEDS:
                    state=observed_state(frames[source,snr,seed,method],profile)
                    sid='gray' if state=='gray' else profile['full_state_id'] if state=='full' else profile.get('prefix_state_id')
                    if state!='other':mismatch.append(abs(scored[source,snr,seed,method]-quality[source,sid,'gray' if state=='gray' else 'VAR']))
            require(not mismatch or max(mismatch)<=rules['clean_actual_quality_tolerance'],'Correct receive state does not match its frozen noiseless source quality')
            per_method[method]=dict(stable_id=profile['stable_id'],wire_key=profile['wire_key'],state_frequencies=frequency,
                calibration_expected_mean=choices[snr][method]['predicted_VAR_mean'],matched_development_expected=interval(pred['quality'],draws,mass),
                actual_quality=interval(actual,draws,mass),absolute_model_error=envelope(actual,pred['quality'],pred['lower'],pred['upper'],draws,mass),
                matched_Q_changes_policy=False)
        contrasts=[];ranking=True;resolved=0;error_qualified=True;rank_fail=False;validated={m:False for m in METHODS}
        for a,b in itertools.combinations(METHODS,2):
            pa,pb=profiles[snr,a],profiles[snr,b];preda,predb=estimates[a],estimates[b]
            if pa['wire_key']==pb['wire_key']:
                require(np.array_equal(actuals[a],actuals[b]),'Identical actual wire aliases produced different metrics')
                contrasts.append(dict(method_A=a,method_B=b,status='STRUCTURAL_IDENTITY',independent_validation_evidence=False));continue
            delta=preda['quality']-predb['quality'];lower=preda['lower']-predb['upper'];upper=preda['upper']-predb['lower']
            observed=actuals[a]-actuals[b];pi=interval(delta,draws,mass);ai=interval(observed,draws,mass)
            pi['envelope_low']=pi['ci_low']-float(np.mean(delta-lower));pi['envelope_high']=pi['ci_high']+float(np.mean(upper-delta))
            sign_pred=1 if pi['envelope_low']>0 else -1 if pi['envelope_high']<0 else 0
            sign_actual=1 if ai['ci_low']>0 else -1 if ai['ci_high']<0 else 0
            same=sign_pred!=0 and sign_pred==sign_actual;reversed_order=sign_pred*sign_actual==-1
            tie=rules['quality_practical_tie_margin'];equivalent=max(abs(pi['envelope_low']),abs(pi['envelope_high']),abs(ai['ci_low']),abs(ai['ci_high']))<tie
            error=envelope(observed,delta,lower,upper,draws,mass)
            gain=min(abs(pi['envelope_low']),abs(pi['envelope_high']),abs(ai['ci_low']),abs(ai['ci_high'])) if same else 0.
            error_pass=same and error['absolute_error_upper']<rules['model_error_to_resolved_gain_max_ratio']*gain
            ranking=ranking and (same or equivalent);rank_fail=rank_fail or reversed_order;resolved+=int(same)
            error_qualified=error_qualified and (error_pass if same else equivalent)
            if same and error_pass:validated[a]=True;validated[b]=True
            contrasts.append(dict(method_A=a,method_B=b,status='PASS' if (same and error_pass) or equivalent else 'FAIL' if reversed_order else 'UNCERTAIN',
                actual_delta=ai,matched_source_predicted_delta=pi,actual_minus_predicted_delta=error,
                resolved_matching_direction=same,practical_equivalence=equivalent,conservative_gain_lower_bound=gain,
                error_less_than_half_gain=error_pass,independent_validation_evidence=True))
        ranking=ranking and resolved>0;error_qualified=error_qualified and resolved>0
        # A wire-identical strong baseline inherits B3's independently validated neighbour contrast.
        if profiles[snr,'strong_baseline']['wire_key']==profiles[snr,'B3']['wire_key']:
            validated['strong_baseline']=validated['B3']
        passed=state_pass and ranking and error_qualified and validated['strong_baseline'] and validated['validation_candidate']
        results.append(dict(N=1024,snr_db=snr,status='PASS' if passed else 'FAIL' if state_fail or rank_fail else 'UNCERTAIN',
            synthetic=synthetic,state_frequency_qualified=state_pass,ranking_qualified=ranking,model_error_vs_gain_qualified=error_qualified,
            strong_baseline_validated=validated['strong_baseline'],distinct_resource_candidate_validated=validated['validation_candidate'],
            methods=per_method,contrasts=contrasts,source_count=count,noise_seeds=list(o.SEEDS),
            approximate_probability_intervals=True,uncertain_blocks_N2048=True,policy_reselection_allowed=False))
    return dict(status='SYNTHETIC_MODEL_VALIDATION_FIXTURE' if synthetic else 'MODEL_VALIDATION_COMPLETE',synthetic=synthetic,
        rules_sha256=sha(RULES),rules=rules,source_ids=source_ids,model_validation=results,
        all_four_points_qualified=all(r['status']=='PASS' for r in results),development_Q_used_for_selection=False,
        additional_samples_allowed=False,independent_ConvNeXt_used=False)


def rows_csv(path):
    opener=gzip.open if str(path).endswith('.gz') else open
    with opener(path,'rt',newline='',encoding='utf-8-sig') as stream:
        rows=list(csv.DictReader(stream))
    for row in rows:
        for key,value in row.items():
            if value in ('True','False','true','false'):row[key]=value.lower()=='true'
    return rows


def run(policies_path,bler_path,events,scores,preregistration):
    # Admission precedes any development file read.
    prior=read(preregistration);sources=prior.get('source_bindings',{})
    for path in (Path(__file__).resolve(),RULES):require(sources.get(str(path))==sha(path),'Model validation code/rules were not preregistered')
    policies_path,bler_path,events,scores=map(lambda x:Path(x).resolve(),(policies_path,bler_path,events,scores))
    policies,bler=read(policies_path),read(bler_path)
    require(identity(bler)==policies['bler_input_sha256'],'Frozen refined probability input changed')
    er,ec,sr,sc=[read(p) for p in (events/'registration.json',events/'completion.json',scores/'registration.json',scores/'completion.json')]
    require(ec['status']=='UEP_ACTUAL_EVENTS_COMPLETE' and sc['status']=='UEP_ACTUAL_SCORING_COMPLETE'
        and ec['sources']==sc['sources']==100 and ec['synthetic'] is False and sc['synthetic'] is False,'Complete actual100source events and scoring required')
    require(ec['registration_sha256']==sha(events/'registration.json') and sc['registration_sha256']==sha(scores/'registration.json'),'Changed event/scoring registration')
    require(er['selected_policies_sha256']==sr['selected_policies_sha256']==sha(policies_path)
        and er['codebook_sha256']==sr['codebook_sha256'],'Selected policy/codebook differs between actual and metrics')
    require(sr['development_source_Q']['diagnostic_only'] is True and sr['development_source_Q']['development_Q_used_for_selection'] is False
        and sr['development_source_Q']['policy_reselection_allowed'] is False,'No frozen diagnostic-only source-Q registration')
    bindings={str(Path(preregistration).resolve()):sha(preregistration),str(policies_path):sha(policies_path),str(bler_path):sha(bler_path)}
    for folder,reg,done in ((events,er,ec),(scores,sr,sc)):
        for path,digest in reg.get('input_bindings',{}).items():require(sha(path)==digest,'Bound scientific input changed');bindings[path]=digest
        for name in ('registration.json','completion.json'):bindings[str(folder/name)]=sha(folder/name)
    frames={};ids=er['source_ids'];require(len(ids)==len(set(ids))==100,'Original100development identities required')
    for i,source in enumerate(ids):
        path=events/'source_checkpoints'/f'{i:04d}.json';require(ec['outputs'].get(str(path))==sha(path),'Unbound actual source checkpoint')
        cp=read(path);require(cp['binding']==identity(er) and cp['complete'] is True and cp['source_id']==source and cp['source_index']==i
            and cp['payload_sha256']==identity({k:v for k,v in cp.items() if k!='payload_sha256'}),'Changed actual source receipt')
        bindings[str(path)]=sha(path)
        for frame in cp['frames']:
            spec=frame['spec'];require(spec['source_id']==source and spec['source_index']==i and spec['N']==1024,'Actual event source identity differs')
            for method in frame['methods']:
                key=(source,int(spec['snr_db']),int(spec['noise_seed']),method['family']);require(key not in frames,'Duplicate actual method frame')
                frames[key]=frame
    paths=[scores/'development_source_quality.csv',scores/'metrics_per_frame.csv.gz']
    for path in paths:require(sc['outputs'].get(str(path))==sha(path),'Unbound scored/Q table');bindings[str(path)]=sha(path)
    result=validate(policies,bler,ids,rows_csv(paths[0]),rows_csv(paths[1]),frames)
    result['input_bindings']=bindings;return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('policies','bler','events','scores','preregistration','output'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args();result=run(args.policies,args.bler,args.events,args.scores,args.preregistration)
    seal(args.output,result);print(result['status'],[(r['snr_db'],r['status']) for r in result['model_validation']])


if __name__=='__main__':main()

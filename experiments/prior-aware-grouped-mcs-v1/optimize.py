"""CPU-only, calibration-only finite-profile search for prior-aware UEP.

No models, channel simulator, calibration writer, or development metric enters
selection. Correct probabilities are combined with quality for EACH source.
The separate development gate only decides whether the frozen N2048 extension
is allowed; it never changes a selected policy or adds observations.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import numpy as np

OBJECTIVE='dinov2_vitl14_cosine'
INDEPENDENT='convnext_source_prediction_agreement'
FAMILIES=('B0','B1','B2','B3','B4')
LOOKUP_SNRS=(1,4,7,10,13,19)
ACTUAL_SNRS=(4,7,10,13)
SCALE_SIZES=(1,2,3,4,5,6,8,10,13,16)
SEEDS=(2001,2002,2003)
BOOTSTRAP_SEED=20261002
BOOTSTRAP_REPLICATES=10000
VERSION='PRIOR-AWARE-UEP-CALIBRATION-OPTIMIZER-V1'


def require(condition,message):
    if not condition:raise ValueError(message)


def identity(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def finite(value,name):
    require(not isinstance(value,bool),name+' must be a number')
    value=float(value);require(math.isfinite(value),'Nonfinite '+name);return value


def expected_quality(gray,states,p_header,p_groups):
    """Contiguous trusted-prefix model; undetected-error effects are approximate.

    `states` and `p_groups` follow wire order. A later accepted group cannot
    restore an earlier failed group. Quality need not be monotonic.
    """
    require(len(states)==len(p_groups) and len(states) in (1,2),'Only G1/G2 are registered')
    q0=np.asarray(gray,dtype=np.float64)
    ph=np.asarray(p_header,dtype=np.float64)
    require(np.isfinite(q0).all() and np.isfinite(ph).all() and ((ph>=0)&(ph<=1)).all(),'Invalid gray/header values')
    result=q0.copy();previous=q0;survival=np.ones_like(q0)
    for quality,probability in zip(states,p_groups):
        quality=np.asarray(quality,dtype=np.float64);probability=np.asarray(probability,dtype=np.float64)
        require(quality.shape==q0.shape and probability.shape==q0.shape and ph.shape==q0.shape,'Per-source vector shapes differ')
        require(np.isfinite(quality).all() and np.isfinite(probability).all() and ((probability>=0)&(probability<=1)).all(),'Invalid quality/probability')
        survival=survival*probability
        result=result+ph*(quality-previous)*survival
        previous=quality
    return result


def expected_quality_bounds(gray,states,header_interval,group_intervals):
    """Multilinear corner extrema, valid even when later quality is worse.

    These are conservative plug-in probability-box bounds, NOT joint 95% CIs.
    """
    choices=[header_interval,*group_intervals]
    require(len(group_intervals)==len(states),'Probability interval dimension differs')
    candidates=[expected_quality(gray,states,corner[0],corner[1:]) for corner in itertools.product(*choices)]
    return np.min(candidates,axis=0),np.max(candidates,axis=0)


def memberships(profile):
    groups=profile['groups'];g=profile['G'];answer=[]
    if g==1:answer.append('B0')
    if g==2:
        mcs=[item.get('mcs_id',item['modulation']+':r'+str(item['nominal_rate'])) for item in groups]
        if mcs[0]==mcs[1]:answer.append('B1')
        if groups[0]['modulation']==groups[1]['modulation']:answer.append('B2')
    return answer+['B3','B4']


def stable_id(profile):return str(profile['stable_id'])


def profile_wire_key(profile):
    if profile.get('wire_key'):return str(profile['wire_key'])
    return identity({k:profile[k] for k in ('N','G','m','K','j')}|{'phy_keys':[g['phy_key'] for g in profile['groups']]})


def validate_profiles(bundle):
    profiles=bundle['profiles'] if isinstance(bundle,dict) else bundle
    require(isinstance(profiles,list) and profiles,'No frozen candidate profiles')
    legal=[];seen=set();excluded=[]
    for profile in profiles:
        sid=stable_id(profile);require(sid and sid not in seen,'Duplicate/missing stable profile ID');seen.add(sid)
        if profile.get('status')=='NOT_FEASIBLE' or profile.get('feasible') is False:
            excluded.append(sid);continue
        require(profile['N'] in (1024,2048) and type(profile['G']) is int and profile['G'] in (1,2),'Budget/group scope differs')
        m,k,j=profile['m'],profile['K'],profile['j']
        require(type(m)is int and 4<=m<=10 and type(k)is int and k>=0,'Source state differs')
        require(k==0 if m==10 else k<SCALE_SIZES[m]**2,'Full next scale must be normalized into the next complete prefix')
        require(len(profile['groups'])==profile['G'] and profile['header_phy_key'],'Physical layout missing')
        require(profile['full_state_id']==f'm{m}_K{k}','Unnormalized full-state identity')
        if profile['G']==2:
            require(type(j)is int and j in (4,5,6,7) and j<=m and (j<m or k>0),'Empty/illegal second group')
            require(profile.get('prefix_state_id')==f'm{j}_K0','First-group state differs')
        else:require(j in (None,'',0) and profile.get('prefix_state_id') in (None,''),'Single group cannot have an intermediate prefix')
        for group in profile['groups']:
            require(group['modulation'] in ('QPSK','16QAM') and group['phy_key'] and group['nominal_rate'],'Missing modulation/code identity')
            require(str(group['nominal_rate']) in ('1/3','1/2','2/3','3/4','5/6'),'Unregistered nominal rate')
            if 'mcs_id' in group:require(group['mcs_id']==group['modulation']+':r'+str(group['nominal_rate']),'MCS label disagrees with actual nominal configuration')
        if 'family_memberships' in profile:
            require(set(profile['family_memberships'])==set(memberships(profile)),'Resource/optimizer family definitions disagree')
        legal.append(profile)
    require(legal,'No feasible profiles');return legal,excluded


class QualityTable:
    def __init__(self,bundle,*,stage,synthetic):
        require(bundle.get('population')=='calibration','Only calibration quality may select strategies')
        require(bundle.get('development_read',False) is False,'Development information entered selection')
        self.ids=bundle['source_ids'];require(len(self.ids)==len(set(self.ids)) and self.ids,'Duplicate/empty calibration population')
        require(stage in ('screen','final'),'Unknown selection stage')
        if not synthetic:
            require(len(self.ids)==(300 if stage=='screen' else 1000),'Selection needs registered 300-screen/1000-final calibration sources')
            if stage=='screen':
                all_ids=bundle['calibration_source_ids'];require(len(all_ids)==1000 and self.ids==all_ids[:300],'Screen must use first 300 registered calibration sources')
        require(bundle.get('metric',OBJECTIVE)==OBJECTIVE,'Optimization target is frozen DINOv2-L/14')
        by_key={}
        for row in bundle['rows']:
            require(row['source_id'] in self.ids and row.get('population','calibration')=='calibration','Unexpected quality source/population')
            receiver=row['receiver'];state=row['state_id']
            require(receiver in ('VAR','direct','gray') and (receiver!='gray' or state=='gray'),'Quality receiver/state differs')
            key=(row['source_id'],state,receiver);require(key not in by_key,'Duplicate per-source quality state')
            value=finite(row[OBJECTIVE],OBJECTIVE);require(-1.000001<=value<=1.000001,'DINO cosine is out of range');by_key[key]=value
        self.rows=by_key;self.cache={}

    def vector(self,state,receiver):
        key=(state,receiver)
        if key not in self.cache:
            keys=[(source,state,receiver) for source in self.ids]
            require(all(k in self.rows for k in keys),'Incomplete calibration quality: '+str(key))
            self.cache[key]=np.asarray([self.rows[k] for k in keys],dtype=np.float64)
        return self.cache[key]


class ProbabilityTable:
    def __init__(self,bundle,source_ids):
        require(bundle.get('conditional_group_independence') is True,'Explicit conditional independent-AWGN assumption required')
        self.ids=source_ids;self.rows={};self.cache={};self.used={}
        for row in bundle['rows']:
            sid=row.get('source_id','*');key=(row['phy_key'],int(row['snr_db']),sid)
            require(sid=='*' or sid in self.ids,'Probability row has an unknown source')
            require(key not in self.rows,'Duplicate physical probability point')
            if sid=='*':require(row.get('applicability') in ('source_independent_verified','iid_scrambled_payload_approximation'),
                'Wildcard probability needs explicit payload applicability/approximation')
            probabilities=[finite(row[k],k) for k in ('p_correct','p_reject','p_undetected')]
            require(all(0<=v<=1 for v in probabilities) and abs(math.fsum(probabilities)-1.)<1e-9,'Correct/rejected/undetected probabilities must sum to one')
            interval=row.get('p_correct_ci')
            if interval is not None:
                require(len(interval)==2 and 0<=interval[0]<=probabilities[0]<=interval[1]<=1,'Invalid correct-probability interval')
            count_names=('n_blocks','n_correct','n_reject','n_undetected')
            if any(k in row for k in count_names):
                require(all(type(row.get(k))is int and row[k]>=0 for k in count_names),'Incomplete block counts')
                require(row['n_blocks']>0 and row['n_correct']+row['n_reject']+row['n_undetected']==row['n_blocks'],'Block counts differ')
                require(all(abs(row[k]/row['n_blocks']-p)<1e-9 for k,p in zip(count_names[1:],probabilities)),'Probabilities differ from actual block counts')
            self.rows[key]=row

    def vector(self,phy_key,snr):
        key=(phy_key,snr)
        if key not in self.cache:
            rows=[]
            for source in self.ids:
                exact=(phy_key,snr,source);fallback=(phy_key,snr,'*')
                require(exact in self.rows or fallback in self.rows,'Unmeasured physical point: '+str(exact))
                used=exact if exact in self.rows else fallback;row=self.rows[used];rows.append(row);self.used[used]=row
            interval=None
            if all('p_correct_ci' in r for r in rows):interval=tuple(np.asarray([r['p_correct_ci'][i] for r in rows]) for i in (0,1))
            self.cache[key]=(np.asarray([r['p_correct'] for r in rows]),interval,rows)
        return self.cache[key]


def predict(profile,snr,quality,probability,receiver):
    gray=quality.vector('gray','gray')
    states=([quality.vector(profile['prefix_state_id'],receiver)] if profile['G']==2 else [])+[quality.vector(profile['full_state_id'],receiver)]
    header,hi,hr=probability.vector(profile['header_phy_key'],snr)
    vectors=[probability.vector(group['phy_key'],snr) for group in profile['groups']]
    groups=[value[0] for value in vectors];values=expected_quality(gray,states,header,groups)
    mean=math.fsum(values.tolist())/len(values)
    lower=upper=None
    if hi is not None and all(value[1] is not None for value in vectors):
        lo,up=expected_quality_bounds(gray,states,hi,[value[1] for value in vectors])
        lower=math.fsum(lo.tolist())/len(lo);upper=math.fsum(up.tolist())/len(up)
    survive=header.copy();weights={'gray':1-header*groups[0]}
    for i,p in enumerate(groups):
        survive=survive*p
        weights['full' if i==len(groups)-1 else 'prefix']=survive if i==len(groups)-1 else survive*(1-groups[i+1])
    approximate=any(r.get('applicability')=='iid_scrambled_payload_approximation' for r in hr+sum([v[2] for v in vectors],[]))
    return dict(mean=mean,probability_box_lower=lower,probability_box_upper=upper,
        probability_box_is_joint_confidence_interval=False,per_source=values.tolist(),
        state_frequencies={name:math.fsum(value.tolist())/len(value) for name,value in weights.items()},
        wildcard_iid_payload_approximation_used=approximate,
        undetected_errors_ignored_in_expected_quality=True)


def ranked(profiles,predictions):
    return sorted(profiles,key=lambda p:(-predictions[stable_id(p)]['mean'],p['G'],stable_id(p)))


def result_profile(profile,var,direct,*,objective_receiver='VAR'):
    sid=stable_id(profile)
    return dict(stable_id=sid,wire_key=profile_wire_key(profile),N=profile['N'],G=profile['G'],m=profile['m'],K=profile['K'],j=profile['j'],
        objective_receiver=objective_receiver,objective=OBJECTIVE,selection_score=(direct if objective_receiver=='direct' else var)[sid]['mean'],
        predicted_VAR_mean=var[sid]['mean'],predicted_direct_mean=direct[sid]['mean'],
        predicted_VAR_probability_box=[var[sid]['probability_box_lower'],var[sid]['probability_box_upper']],
        predicted_VAR_state_frequencies=var[sid]['state_frequencies'],
        physical_keys=[profile['header_phy_key'],*[g['phy_key'] for g in profile['groups']]],profile=profile)


def optimize(profile_bundle,quality_bundle,bler_bundle,*,snrs=LOOKUP_SNRS,stage='final',synthetic=False):
    if not synthetic:require(bler_bundle.get('synthetic') is False,'Actual physical BLER table required; synthetic fixtures cannot select real policies')
    profiles,excluded=validate_profiles(profile_bundle);q=QualityTable(quality_bundle,stage=stage,synthetic=synthetic)
    probabilities=ProbabilityTable(bler_bundle,q.ids)
    require(set(snrs)<=set(LOOKUP_SNRS) and len(snrs)==len(set(snrs)),'Unregistered/duplicate optimization SNR')
    cells=[];refinement=defaultdict(set);per_source=[]
    for n in sorted({p['N'] for p in profiles}):
        group=[p for p in profiles if p['N']==n]
        require(any('B0' in memberships(p) for p in group),'Strong single-packet baseline cannot be omitted')
        for snr in snrs:
            var={stable_id(p):predict(p,snr,q,probabilities,'VAR') for p in group}
            direct={stable_id(p):predict(p,snr,q,probabilities,'direct') for p in group}
            families={};selected={}
            for family in FAMILIES:
                candidates=[p for p in group if family in memberships(p)]
                if not candidates:
                    families[family]=dict(status='NOT_FEASIBLE',candidate_count=0);continue
                ranking=ranked(candidates,direct if family=='B4' else var);selected[family]=ranking[0]
                top=[result_profile(p,var,direct,objective_receiver='direct' if family=='B4' else 'VAR') for p in ranking[:3]]
                families[family]=dict(status='CALIBRATION_SELECTED',candidate_count=len(ranking),selected=top[0],top3=top)
                for p in ranking[:3]:
                    for key in [p['header_phy_key'],*[g['phy_key'] for g in p['groups']]]:refinement[key,snr].add(f'{n}/{family}/top3')
            winner=selected['B3'];matched={}
            for family in ('B1','B2','B3'):
                if winner['G']==1:
                    matched[family]=dict(status='NOT_APPLICABLE',reason='B3 selected the permitted single-packet degeneration');continue
                candidates=[p for p in group if family in memberships(p) and p['G']==2 and
                    (p['m'],p['K'],p['j'])==(winner['m'],winner['K'],winner['j'])]
                if not candidates:matched[family]=dict(status='NOT_FEASIBLE',reason='No legal same-(m,K,j) family profile');continue
                chosen=ranked(candidates,var)[0];matched[family]=dict(status='CALIBRATION_SELECTED',selected=result_profile(chosen,var,direct))
                for key in [chosen['header_phy_key'],*[g['phy_key'] for g in chosen['groups']]]:refinement[key,snr].add(f'{n}/{family}/matched')
            strongest=ranked([selected[f] for f in ('B0','B1','B2') if f in selected],var)[0]
            used_wires={profile_wire_key(winner),profile_wire_key(strongest)}
            distinct=[p for p in ranked(group,var) if profile_wire_key(p) not in used_wires]
            validation=(dict(status='FROZEN',selected=result_profile(distinct[0],var,direct),
                selection_rule='highest predicted VAR score with actual wire layout distinct from B3 and strong baseline; calibration only') if distinct else
                dict(status='NOT_AVAILABLE',reason='Finite candidate set has no additional distinct wire layout beyond B3 and strong baseline; model-ranking validation cannot pass'))
            for tag,p in [('strong_baseline',strongest)]+([('distinct_validation',distinct[0])] if distinct else []):
                for key in [p['header_phy_key'],*[g['phy_key'] for g in p['groups']]]:refinement[key,snr].add(f'{n}/{tag}')
            a,b=stable_id(winner),stable_id(selected['B4'])
            for source_index,source_id in enumerate(q.ids):
                for family,p in selected.items():per_source.append(dict(N=n,snr_db=snr,source_id=source_id,family=family,stable_id=stable_id(p),
                    predicted_VAR=var[stable_id(p)]['per_source'][source_index],predicted_direct=direct[stable_id(p)]['per_source'][source_index]))
            baseline=var[stable_id(selected['B0'])]['mean']
            cell=dict(N=n,snr_db=snr,independent_optima=families,matched_to_B3=matched,
                strong_baseline=result_profile(strongest,var,direct),frozen_distinct_validation_candidate=validation,
                B3_predicted_delta_vs_B0=var[a]['mean']-baseline,
                prior_aware_delta_same_VAR_receiver=var[a]['mean']-var[b]['mean'],
                prior_aware_prediction_is_not_an_experimental_finding=True,
                B3_degenerated_to_single_group=winner['G']==1,
                B3_uses_common_MCS='B1' in memberships(winner),B3_uses_common_modulation=winner['G']==2 and 'B2' in memberships(winner),
                probability_box_may_change_VAR_winner=None)
            if var[a]['probability_box_lower'] is not None and all(v['probability_box_upper'] is not None for v in var.values()):
                cell['probability_box_may_change_VAR_winner']=any(v['probability_box_upper']>=var[a]['probability_box_lower'] for sid,v in var.items() if sid!=a)
            cells.append(cell)
    points=[]
    for (phy,snr),reasons in sorted(refinement.items()):
        rows=probabilities.vector(phy,snr)[2];observed=[]
        unique={identity(row):row for row in rows}
        for row in unique.values():
            errors=None if 'n_blocks' not in row else row['n_blocks']-row['n_correct']
            observed.append(dict(source_id=row.get('source_id','*'),n_blocks=row.get('n_blocks'),block_errors=errors,
                stopping_rule_reached=errors is not None and (errors>=100 or row['n_blocks']>=20000),
                probability_interval=row.get('p_correct_ci')))
        points.append(dict(phy_key=phy,snr_db=snr,reasons=sorted(reasons),coarse_min_blocks=256,
            refine_until_block_errors=100,refine_max_blocks=20000,zero_observed_errors_is_not_zero_true_BLER=True,
            measured_strata=observed,automatic_measurement_started=False))
    for cell in cells:
        needed=[point for point in points if point['snr_db']==cell['snr_db'] and
            any(reason.startswith(str(cell['N'])+'/') for reason in point['reasons'])]
        cell['probability_refinement_complete']=bool(needed) and all(
            item['stopping_rule_reached'] for point in needed for item in point['measured_strata'])
        cell['ready_for_actual_link']=stage=='final' and not synthetic and cell['probability_refinement_complete'] and \
            cell['frozen_distinct_validation_candidate']['status']=='FROZEN'
    return dict(version=VERSION,status='SYNTHETIC_CPU_FIXTURE' if synthetic else 'SCREEN_ONLY_NOT_DEPLOYABLE' if stage=='screen' else 'CALIBRATION_POLICIES_SELECTED',
        stage=stage,synthetic=synthetic,source_count=len(q.ids),source_ids=q.ids,objective=OBJECTIVE,
        development_read=False,selection_is_per_image_adaptive=False,neural_training_updates=0,independent_ConvNeXt_used_for_selection=False,
        profile_input_sha256=identity(profile_bundle),quality_input_sha256=identity(quality_bundle),bler_input_sha256=identity(bler_bundle),
        tie_rule='exact equal score: fewer groups, then stable_id lexical order',
        assumptions=dict(independent_AWGN_given_SNR_source_and_layout=True,undetected_errors_ignored_only_in_expectation=True,
            source_probabilities_combined_with_source_quality_before_averaging=True,power_protocol='fixed_constellation_average_complex_symbol_energy_2'),
        not_feasible_profiles=excluded,cells=cells,refinement_requirements=points,per_source_predictions=per_source,
        independent_metric=dict(name=INDEPENDENT,model='torchvision ConvNeXt-Tiny IMAGENET1K_V1',used_for_candidate_refinement=False))


def development_extension_gate(rows,development_source_ids,selected_policies,model_validation):
    """Fixed-size post-evaluation gate, with paired source-level resampling.

    It returns a stop/extension receipt. It never requests more samples, changes
    policies, pools SNRs, or retries with another metric or bootstrap seed.
    """
    require(len(development_source_ids)==100 and len(set(development_source_ids))==100,'Exactly 100 registered development sources required')
    require(selected_policies.get('status')=='CALIBRATION_POLICIES_SELECTED' and selected_policies.get('synthetic') is False
        and selected_policies.get('source_count')==1000 and selected_policies.get('development_read') is False,'Final1000 calibration freeze required')
    require(set(development_source_ids).isdisjoint(selected_policies['source_ids']),'Calibration/development overlap')
    selected={cell['snr_db']:cell for cell in selected_policies['cells'] if cell['N']==1024 and cell['snr_db'] in ACTUAL_SNRS}
    require(set(selected)==set(ACTUAL_SNRS),'Missing frozen N1024 policy working point')
    require(all(cell.get('ready_for_actual_link') is True for cell in selected.values()),'Actual policies were not frozen after full Q/probability refinement')
    expected={(source,snr,seed,family) for source in development_source_ids for snr in ACTUAL_SNRS for seed in SEEDS for family in ('B0','B3')}
    observed={}
    for row in rows:
        if int(row['N'])!=1024 or row['family'] not in ('B0','B3') or int(row['snr_db']) not in ACTUAL_SNRS:continue
        key=(row['source_id'],int(row['snr_db']),int(row['noise_seed']),row['family'])
        require(key in expected and key not in observed,'Unexpected/duplicate development pair')
        require(row.get('synthetic') is False and row.get('actual_bit_chain_executed') is True,'Actual decoded physical frames required')
        profile=selected[key[1]]['independent_optima'][key[3]]['selected']
        require(str(row['stable_id'])==profile['stable_id'],'Development output is not the frozen policy')
        require(row.get('convnext_model_id')=='torchvision_ConvNeXt_Tiny_IMAGENET1K_V1','Independent classifier identity differs')
        values={metric:finite(row[metric],metric) for metric in (OBJECTIVE,INDEPENDENT)}
        require(values[INDEPENDENT] in (0.,1.),'Per-frame ConvNeXt agreement must be an indicator')
        observed[key]=values
    require(set(observed)==expected,'Incomplete fixed100 × three-noise paired development grid')
    validations={int(v['snr_db']):v for v in model_validation if int(v['N'])==1024}
    require(len(validations)==len([v for v in model_validation if int(v['N'])==1024]),'Duplicate model validation point')
    required_validation=('state_frequency_qualified','ranking_qualified','model_error_vs_gain_qualified',
        'strong_baseline_validated','distinct_resource_candidate_validated')
    qualified={snr:snr in validations and validations[snr].get('status')=='PASS' and validations[snr].get('synthetic') is False
        and all(validations[snr].get(name) is True for name in required_validation) for snr in ACTUAL_SNRS}
    rng=np.random.default_rng(BOOTSTRAP_SEED);draws=rng.integers(0,100,size=(BOOTSTRAP_REPLICATES,100));results=[]
    for snr in ACTUAL_SNRS:
        metrics={}
        for metric in (OBJECTIVE,INDEPENDENT):
            differences=np.asarray([math.fsum(observed[source,snr,seed,'B3'][metric]-observed[source,snr,seed,'B0'][metric] for seed in SEEDS)/3
                for source in development_source_ids],dtype=np.float64)
            replicates=differences[draws].mean(axis=1);lo,hi=np.percentile(replicates,[2.5,97.5])
            metrics[metric]=dict(mean=math.fsum(differences.tolist())/100,ci_low=float(lo),ci_high=float(hi),
                strictly_positive_95pct_lower_bound=bool(lo>0),n_sources=100,n_frames_per_method=300,
                source_noise_averaging_before_bootstrap=True)
        nondegenerate=selected[snr]['independent_optima']['B3']['selected']['G']==2
        dual=all(metric['strictly_positive_95pct_lower_bound'] for metric in metrics.values())
        results.append(dict(snr_db=snr,paired_B3_minus_B0=metrics,B3_not_single_group=nondegenerate,
            model_validation_qualified=qualified[snr],joint_positive=dual,qualifying_working_point=dual and nondegenerate and qualified[snr]))
    passing=[r['snr_db'] for r in results if r['qualifying_working_point']]
    allow=len(passing)>=2 and all(qualified.values())
    reasons=[]
    if len(passing)<2:reasons.append('Fewer than two registered SNRs pass both paired metrics with non-single-group B3')
    if not all(qualified.values()):reasons.append('Expected-quality state-frequency/ranking validation is not qualified at all four working points')
    return dict(status='N2048_EXTENSION_ALLOWED' if allow else 'DELIVER_AND_STOP_NO_EXTENSION',N2048_extension_allowed=allow,
        N2048_actual_snrs=[4,7,13] if allow else [],qualifying_snrs=passing,working_points=results,reasons=reasons,
        new_samples_to_seek_significance_allowed=False,policy_reselection_allowed=False,full_results_must_be_delivered=True,
        bootstrap_unit='original source after averaging registered three noise realizations',bootstrap_seed=BOOTSTRAP_SEED,
        bootstrap_replicates=BOOTSTRAP_REPLICATES,source_count=100,exploratory_reused_development=True,
        selection_metric=OBJECTIVE,independent_metric=INDEPENDENT,gate_is_not_holdout_confirmation=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--profiles',required=True);parser.add_argument('--quality',required=True)
    parser.add_argument('--bler',required=True);parser.add_argument('--output',required=True);parser.add_argument('--stage',choices=('screen','final'),default='final')
    args=parser.parse_args()
    read=lambda path:json.loads(Path(path).read_text(encoding='utf-8-sig'))
    result=optimize(read(args.profiles),read(args.quality),read(args.bler),stage=args.stage)
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():require(read(path)==result,'Frozen policy output differs; use an explicit new refinement revision')
    else:path.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=result['status'],cells=len(result['cells']),refinement_points=len(result['refinement_requirements']))))


if __name__=='__main__':main()

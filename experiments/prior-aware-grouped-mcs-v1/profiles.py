"""Finite, calibration-blind resource enumeration for prior-aware UEP V1.

No LDPC algorithm is implemented here. An injected mature encoder planning
backend supplies actual lifting, filler and rate-matching layout. Candidate
IDs are not transmitted; the final selected finite codebook receives 12-bit IDs.
"""
from __future__ import annotations
import argparse
import csv
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import sys

SIZES=(1,2,3,4,5,6,8,10,13,16)
PREFIX_M=tuple(range(4,10))
BOUNDARIES=(4,5,6,7)
MODULATIONS={'QPSK':2,'16QAM':4}
RATES=('1/3','1/2','2/3','3/4','5/6')
HEADER_SYMBOLS=68
CRC_BITS=16
HEADER_PHY_KEY='uep-v1-profile12-crc16-tail6-conv-ratematch136-qpsk-Es2'


def require(ok,message):
    if not ok:raise ValueError(message)


def identity(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


class UnsupportedConfiguration(ValueError):
    """Only this explicit rejection is treated as a non-feasible coding choice."""


@dataclass(frozen=True)
class MCS:
    modulation:str
    nominal_rate:str

    def __post_init__(self):
        require(self.modulation in MODULATIONS and self.nominal_rate in RATES,'Unregistered MCS')

    @property
    def width(self):return MODULATIONS[self.modulation]

    @property
    def name(self):return self.modulation+':r'+self.nominal_rate


MCS_GRID=tuple(MCS(mod,rate) for mod in MODULATIONS for rate in RATES)


def normalize_state(m,K):
    require(type(m) is int and type(K) is int and 4<=m<=10,'Invalid source state')
    require(K>=0 and (K==0 if m==10 else K<=SIZES[m]**2),'Invalid partial count')
    return (m+1,0) if m<10 and K==SIZES[m]**2 else (m,K)


def state_id(m,K):
    m,K=normalize_state(m,K)
    return f'm{m}_K{K}'


def tokens(m,K=0):
    m,K=normalize_state(m,K)
    return sum(s*s for s in SIZES[:m])+K


def payloads(m,K=0,j=None):
    m,K=normalize_state(m,K)
    if j is None:return (12*tokens(m,K),)
    require(type(j) is int and j in BOUNDARIES and j<=m,'Invalid complete-scale split')
    first=12*tokens(j)
    second=12*tokens(m,K)-first
    require(second>0,'The second group must not be empty')
    return first,second


def initial_symbols(source_bits,mcs):
    """Nominal rate generates a starting length; actual coding is backend-owned."""
    r=Fraction(mcs.nominal_rate)
    numerator=(source_bits+CRC_BITS)*r.denominator
    coded=(numerator+r.numerator-1)//r.numerator
    return (coded+mcs.width-1)//mcs.width


def fill_symbols(start,budget):
    require(all(type(n) is int and n>0 for n in start) and sum(start)<=budget,'Invalid starting budget')
    spare=budget-sum(start);denominator=sum(start)
    added=[spare*n//denominator for n in start]
    # Fixed rule: residual integer symbols go in increasing group order.
    for i in range(spare-sum(added)):added[i]+=1
    return tuple(n+a for n,a in zip(start,added))


def family_memberships(groups):
    if len(groups)==1:return ['B0','B3','B4']
    same_mod=groups[0]['modulation']==groups[1]['modulation']
    same_mcs=same_mod and groups[0]['nominal_rate']==groups[1]['nominal_rate']
    return (['B1'] if same_mcs else [])+(['B2'] if same_mod else [])+['B3','B4']


class Planner:
    """Backend API: identity dict; qualified bool; plan(k,n,num_bits_per_symbol).

    k includes exactly this protocol's CRC16. plan returns a JSON dictionary:
      k,n,k_ldpc,k_filler,n_cb,bg,z,code_blocks,
      mother_bits,puncturing_bits,shortening_bits,repetition_bits,
      layout_id,decoder_config,bit_mapping,scrambling,power_protocol.
    Lengths are bits. Unsupported lengths raise UnsupportedConfiguration.
    Other failures remain fatal. The real backend must verify these accounting
    fields against the actual encoder output and its rate-matching indices.
    """
    def __init__(self,backend):
        self.backend=backend
        require(isinstance(backend.identity,dict),'Backend needs an immutable JSON identity')
        self.backend_id=identity(backend.identity)
        self.cache={};self.calls=0;self.rejections=Counter()

    def group(self,source_bits,mcs,symbols):
        require(source_bits>0 and source_bits%12==0,'Raw 12-bit token payload required')
        k=source_bits+CRC_BITS;n=symbols*mcs.width
        key=(k,n,mcs.width)
        if key not in self.cache:
            self.calls+=1
            try:
                layout=self.backend.plan(k=k,n=n,num_bits_per_symbol=mcs.width)
            except UnsupportedConfiguration as error:
                self.cache[key]=None;self.rejections[str(error)]+=1
            else:
                require(isinstance(layout,dict),'Backend plan must be a JSON dictionary')
                require(layout.get('k')==k and layout.get('n')==n and layout.get('code_blocks')==1,
                        'Backend changed message/transmitted length or introduced segmentation')
                require(layout.get('k_ldpc',-1)-layout.get('k_filler',-1)==k,
                        'Real encoder message/filler accounting differs')
                for name in ('k_ldpc','k_filler','n_cb','z','mother_bits','puncturing_bits','shortening_bits','repetition_bits'):
                    require(type(layout.get(name)) is int and layout[name]>=0,'Missing exact backend layout: '+name)
                require(layout['bg'] in (1,2,'bg1','bg2') and layout['z']>0,'Registered single-block 5G base graph required')
                for name in ('layout_id','decoder_config','bit_mapping','scrambling','power_protocol'):
                    require(layout.get(name) not in (None,'',{}),'Missing physical identity: '+name)
                require(layout['power_protocol']=='fixed_constellation_average_Es2','Unexpected power protocol')
                self.cache[key]=layout
        layout=self.cache[key]
        if layout is None:return None
        physical=dict(backend=self.backend.identity,source_bits=source_bits,CRC=CRC_BITS,layout=layout,
                      modulation=mcs.modulation,modulation_padding_bits=0)
        return dict(modulation=mcs.modulation,nominal_rate=mcs.nominal_rate,mcs_id=mcs.name,
            phy_key=identity(physical),source_bits=source_bits,crc_bits=CRC_BITS,tail_bits=0,
            information_bits=k,ldpc_k=layout['k_ldpc'],filler_bits=layout['k_filler'],
            mother_bits=layout['mother_bits'],transmitted_bits=n,symbols=symbols,
            modulation_padding_bits=0,effective_information_rate=k/n,
            source_rate=source_bits/n,layout=layout)

    def candidate(self,N,m,K,j,mcs,mode):
        m,K=normalize_state(m,K)
        try:source=payloads(m,K,j)
        except ValueError:return None
        require(len(source)==len(mcs),'MCS/group count mismatch')
        start=tuple(initial_symbols(b,c) for b,c in zip(source,mcs));body=N-HEADER_SYMBOLS
        if sum(start)>body:return None
        if mode=='nominal':allocation=start
        elif mode=='full_budget':allocation=fill_symbols(start,body)
        else:raise ValueError('Unknown allocation rule')
        groups=[self.group(b,c,n) for b,c,n in zip(source,mcs,allocation)]
        if any(g is None for g in groups):return None
        wire=dict(N=N,G=len(groups),m=m,K=K,j=j,group_phy_keys=[g['phy_key'] for g in groups],
                  header_phy_key=HEADER_PHY_KEY,order='entropy',receiver_rule='consecutive_crc_accepted_prefix_v1',
                  idle_symbols=body-sum(allocation),idle_rule='public_known_QPSK_Es2_no_source_information')
        candidate_id=identity(dict(wire=wire,nominal_mcs=[c.name for c in mcs]))
        return dict(stable_id=candidate_id,wire_key=identity(wire),N=N,G=len(groups),m=m,K=K,j=j,
            groups=groups,header_phy_key=HEADER_PHY_KEY,header_symbols=HEADER_SYMBOLS,
            header_information_bits=12,header_crc_bits=16,header_tail_bits=6,header_transmitted_bits=136,
            body_symbols=body,used_body_symbols=sum(allocation),idle_symbols=body-sum(allocation),
            idle_rule=wire['idle_rule'],full_state_id=state_id(m,K),
            prefix_state_id=state_id(j,0) if j is not None else None,gray_state_id='gray',
            allocation_modes=[mode],family_memberships=family_memberships(groups),
            order='entropy',receiver_rule=wire['receiver_rule'],backend_id=self.backend_id,
            encoder_qualified=bool(self.backend.qualified))

    def max_K(self,N,m,j,mcs,mode='nominal'):
        """Search this configuration only, before Q exists; never cross-expand K.

        A budget-only exact bound removes impossible K before querying the
        backend. The remaining descending scan does not assume LDPC feasibility
        is monotone at lifting or base-graph boundaries.
        """
        body=N-HEADER_SYMBOLS;length=SIZES[m]**2
        prefix=12*tokens(m)
        if j is None:
            b2=prefix;c=mcs[0];available=body
        else:
            b1=12*tokens(j);available=body-initial_symbols(b1,mcs[0]);b2=prefix-b1;c=mcs[1]
        if available<=0:return None
        r=Fraction(c.nominal_rate)
        max_information=(available*c.width*r.numerator)//r.denominator
        upper=min(length,(max_information-CRC_BITS-b2)//12)
        for K in range(upper,-1,-1):
            if self.candidate(N,m,K,j,mcs,mode) is not None:return K
        return None


def enumerate_profiles(backend,budgets=(1024,2048)):
    require(tuple(budgets) and len(set(budgets))==len(budgets) and all(n in (1024,2048) for n in budgets),'Registered budgets required')
    planner=Planner(backend);candidates={};generation=[];counts=Counter()
    for N in budgets:
        for m in PREFIX_M:
            settings=[(None,(c,)) for c in MCS_GRID]
            settings += [(j,pair) for j in BOUNDARIES if j<=m for pair in itertools.product(MCS_GRID,repeat=2)]
            for j,mcs in settings:
                counts[N]+=1
                length=SIZES[m]**2
                for mode in ('nominal','full_budget'):
                    maximum=planner.max_K(N,m,j,mcs,mode)
                    Kset={0,length//4,length//2,3*length//4}
                    if maximum is not None:Kset.add(maximum)
                    generation.append(dict(N=N,m=m,j=j,mcs=[c.name for c in mcs],allocation_mode=mode,max_legal_K=maximum,K_candidates=sorted(Kset)))
                    for K in sorted(Kset):
                        row=planner.candidate(N,m,K,j,mcs,mode)
                        if row is None:continue
                        key=row['stable_id']
                        if key in candidates:
                            candidates[key]['allocation_modes']=sorted(set(candidates[key]['allocation_modes']+row['allocation_modes']))
                        else:candidates[key]=row
    rows=sorted(candidates.values(),key=lambda r:(r['N'],r['G'],r['m'],r['K'],-1 if r['j'] is None else r['j'],r['stable_id']))
    states={};physical={}
    for row in rows:
        for receiver_state in (row['full_state_id'],row['prefix_state_id']):
            if receiver_state is not None:states.setdefault(row['N'],set()).add(receiver_state)
        for group in row['groups']:physical[group['phy_key']]=group
    all_states=set().union(*states.values()) if states else set()
    main_states=states.get(1024,set())
    main_phy={g['phy_key'] for r in rows if r['N']==1024 for g in r['groups']}
    report=dict(status='ACTUAL_BACKEND_ENUMERATED_REQUIRES_ROUNDTRIP_QUALIFICATION' if not backend.qualified else 'QUALIFIED_BACKEND_RESOURCE_ENUMERATION',
        backend_identity=backend.identity,encoder_qualified=bool(backend.qualified),calibration_read=False,development_read=False,
        selection_used=False,header_symbols=68,codebook_assigned=False,
        candidate_count=len(rows),distinct_wire_profiles=len({r['wire_key'] for r in rows}),
        distinct_body_phy_configurations=len(physical),header_phy_configurations=1,
        backend_planning_calls=planner.calls,backend_rejections=dict(planner.rejections),
        per_budget=[dict(N=N,starting_configurations=counts[N],candidates=sum(r['N']==N for r in rows),
            distinct_wire_profiles=len({r['wire_key'] for r in rows if r['N']==N}),
            distinct_full_endpoints=len({r['full_state_id'] for r in rows if r['N']==N}),
            distinct_Q_states=len(states.get(N,set())),Q_states=sorted(states.get(N,set())),
            endpoint_K_by_m={str(m):sorted({r['K'] for r in rows if r['N']==N and r['m']==m}) for m in range(4,11)},
            distinct_body_phy_configurations=len({g['phy_key'] for r in rows if r['N']==N for g in r['groups']}),
            Q_images_at_300_sources=300*(2*len(states.get(N,set()))+1),
            Q_images_at_1000_sources=1000*(2*len(states.get(N,set()))+1)) for N in budgets],
        union_Q_states_resource_only_not_scheduled=sorted(all_states),
        main_budget=1024,auxiliary_budget_2048_scope='resource enumeration and already measured matching PHY lookup only; no Q/BLER/real-link run',
        main_Q_states=[dict(state_id=name,m=int(name.split('_')[0][1:]),K=int(name.split('_K')[1])) for name in sorted(main_states)],
        Q_cost=dict(scope='N1024 only',unique_non_gray_states=len(main_states),
            unique_receivers=2,receiver_image_evaluations_per_calibration_source=2*len(main_states)+1,
            VAR_completions_per_calibration_source=len(main_states),direct_decodes_per_calibration_source=len(main_states),
            gray_evaluations_per_calibration_source=1,
            at_300_sources=dict(VAR_completions=300*len(main_states),receiver_images=300*(2*len(main_states)+1)),
            at_1000_sources=dict(VAR_completions=1000*len(main_states),receiver_images=1000*(2*len(main_states)+1))),
        BLER_coarse_cost=dict(scope='N1024 only',body_phy_configurations=len(main_phy),snr_points=6,
                             blocks_per_point=256,body_blocks=len(main_phy)*6*256,header_blocks=6*256),
        maximum_K_rule='per N,m,j,nominal MCS tuple,allocation mode; inclusive K=L normalized; no cross-product of maxima',
        deterministic_quality_state_reused_across_N=True)
    return rows,report,generation


def matched_controls(rows,selected):
    """Finite matched-source/split pools; no quality or development access."""
    base=[r for r in rows if r['N']==selected['N'] and (r['m'],r['K'])==(selected['m'],selected['K'])]
    same_split=[r for r in base if r['G']==selected['G'] and r['j']==selected['j']]
    return {family:[r['stable_id'] for r in same_split if family in r['family_memberships']]
            for family in ('B1','B2','B3')}


def freeze_codebook(selected_rows):
    """Called only on the final calibration-selected finite set and controls."""
    require(selected_rows,'Cannot freeze an empty profile codebook')
    physical={}
    for row in selected_rows:
        require(row.get('encoder_qualified') is True,'A real encoder qualification is required before deployment')
        physical.setdefault(row['wire_key'],[]).append(row)
    require(len(physical)<=4096,'Final unique profiles exceed the paid 12-bit header')
    entries=[];candidate_to_profile={}
    for profile_id,(wire_key,aliases) in enumerate(sorted(physical.items())):
        canonical=min(aliases,key=lambda r:r['stable_id'])
        value={k:v for k,v in canonical.items() if k not in ('allocation_modes','family_memberships')}
        ids=sorted({r['stable_id'] for r in aliases})
        entries.append(dict(value,profile_id=profile_id,candidate_aliases=ids))
        for key in ids:candidate_to_profile[key]=profile_id
    return dict(schema='uep-v1-final-codebook',profile_bits=12,entries=entries,
                candidate_to_profile=candidate_to_profile,codebook_sha256=identity(entries))


def main():
    p=argparse.ArgumentParser();p.add_argument('--backend-module',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path);p.add_argument('--budgets',type=int,nargs='+',default=[1024,2048]);a=p.parse_args()
    spec=importlib.util.spec_from_file_location('uep_actual_ldpc_backend',a.backend_module)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rows,report,generation=enumerate_profiles(module.make_backend(),tuple(a.budgets))
    a.output.mkdir(parents=True,exist_ok=True)
    artifacts=[('candidate_profiles_all_resources.json',rows),('resource_enumeration.json',report),('K_generation.json',generation),
               ('quality_states_N1024.json',report['main_Q_states'])]
    artifacts += [(f'candidate_profiles_N{n}.json',[r for r in rows if r['N']==n]) for n in a.budgets]
    for name,value in artifacts:
        (a.output/name).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    flat=[{k:json.dumps(v,sort_keys=True,separators=(',',':')) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows]
    with (a.output/'candidate_profiles.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]) if flat else ['stable_id']);writer.writeheader();writer.writerows(flat)
    ledger={}
    for r in rows:
        for g in r['groups']:
            key=g['phy_key'];ledger.setdefault(key,dict(g,budgets=set()))['budgets'].add(r['N'])
    flat=[{k:json.dumps(sorted(v) if isinstance(v,set) else v,sort_keys=True,separators=(',',':')) if isinstance(v,(dict,list,set)) else v for k,v in r.items()} for _,r in sorted(ledger.items())]
    with (a.output/'resource_ledger.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]) if flat else ['phy_key']);writer.writeheader();writer.writerows(flat)
    print(json.dumps(report,ensure_ascii=False,allow_nan=False))


if __name__=='__main__':
    # Backends import this exception by the stable module name even when the
    # enumerator is launched as a script; keep one exception class identity.
    sys.modules['profiles']=sys.modules[__name__]
    main()

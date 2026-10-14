"""Source-resolved finite strategy nomination; no unmeasured BLER assignment."""
from common import *
from fractions import Fraction
import numpy as np,itertools,statistics,sys,importlib.util
MCS=tuple(itertools.product((2,4,6),RATES))
def proportion(b,e):return max(2,min(954,2*round(956*(b+29)/(b+e+58)/2)))
def load():
 lengths=list(csv.DictReader((O1/'source_lengths.csv').open()));clean=list(csv.DictReader((O1/'clean_quality.csv').open()))
 assert len(lengths)==3400 and len(clean)==1200
 d={(int(r['pilot_slot']),int(r['j']),int(r['m']),int(r['K'])):r for r in lengths}
 Q={(int(r['pilot_slot']),int(r['m']),int(r['K'])):float(r['dinov2_vitl14_cosine']) for r in clean}
 Q0=np.array([float(r['dinov2_vitl14_cosine']) for r in sorted(csv.DictReader((O/'gray_quality.csv').open()),key=lambda r:int(r['pilot_slot']))]);assert Q0.shape==(50,)
 return d,Q,Q0
def source_states(d,Q,Q0,j,end,CB,CE=None):
 states=[];ceiling=OFFSETS[end[0]]+end[1]
 for slot in range(50):
  if j is None:
   ordered=sorted((x for x in ENDS if OFFSETS[x[0]]+x[1]<=ceiling),key=lambda x:OFFSETS[x[0]]+x[1],reverse=True)
   slen=lambda x:int(d[slot,4,9,192]['b_base']) if x==(4,0) else int(d[slot,4,*x]['b_single'])
   chosen=next((x for x in ordered if slen(x)<=min(8191,CB)),None)
   q=Q[slot,*chosen] if chosen else Q0[slot]
   states.append(dict(pilot_slot=slot,status='SINGLE' if chosen else 'NULL',actual_j=None,actual_m=chosen[0] if chosen else None,actual_K=chosen[1] if chosen else None,Q0=float(Q0[slot]),QB=q,QF=q,L_B=slen(chosen) if chosen else 0,L_E=0));continue
  base=next((v for v in range(j,3,-1) if int(d[slot,v,9,192]['b_base'])<=min(8191,CB)),None)
  if base is None:states.append(dict(pilot_slot=slot,status='NULL',actual_j=None,actual_m=None,actual_K=None,Q0=float(Q0[slot]),QB=float(Q0[slot]),QF=float(Q0[slot]),L_B=0,L_E=0));continue
  candidates=sorted((x for x in ENDS if OFFSETS[base]<OFFSETS[x[0]]+x[1]<=ceiling),key=lambda x:OFFSETS[x[0]]+x[1],reverse=True)
  final=next((x for x in candidates if int(d[slot,base,*x]['b_enhancement'])<=min(8191,CE)),(base,0))
  states.append(dict(pilot_slot=slot,status='EMPTY_ENHANCEMENT' if final==(base,0) else 'TWO_GROUPS',actual_j=base,actual_m=final[0],actual_K=final[1],Q0=float(Q0[slot]),QB=Q[slot,base,0],QF=Q[slot,*final],L_B=int(d[slot,base,9,192]['b_base']),L_E=0 if final==(base,0) else int(d[slot,base,*final]['b_enhancement'])))
 return states
def coefficients(states):return float(np.mean([s['QB']-s['Q0'] for s in states])),float(np.mean([s['QF']-s['QB'] for s in states]))
def key(row):return digest({k:row[k] for k in ('family','target_j','target_m','target_K','N_B','q_B','rate_B','q_E','rate_E')})
def make_layouts(backend):
 spec=importlib.util.spec_from_file_location('_LE2_constructor_design',E1/'group_layout_design.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 d=list(csv.DictReader((O1/'source_lengths.csv').open()));index={(int(r['pilot_slot']),int(r['j']),int(r['m']),int(r['K'])):r for r in d}
 targets=[(j,*end) for j in (6,7) for end in ENDS if OFFSETS[end[0]]+end[1]>OFFSETS[j]]
 props={tuple(t):proportion(statistics.median(int(index[s,*t]['b_base']) for s in range(50)),statistics.median(int(index[s,*t]['b_enhancement']) for s in range(50))) for t in targets}
 old=read(T1/'feasibility/group_layouts.json');assert old['backend_identity']==backend.identity
 rows={(r['N'],r['q'],r['nominal_rate']):r for r in old['layouts']}
 planner=mod.GroupLayoutPlanner(backend.Encoder,backend.identity)
 for N in sorted({956}|{n for t in targets for nb in (*range(256,769,32),props[tuple(t)]) for n in (nb,956-nb)}):
  for q,r in MCS:
   if (N,q,r) not in rows:rows[N,q,r]=planner.plan(N,q,r)
 save(T/'layouts.json',dict(layouts=list(rows.values()),constructor_only=True,new_constructor_calls=planner.constructor_calls,old_constructor_evidence=pin(T1/'feasibility/group_layouts.json'),backend_identity=backend.identity))
 csvsave(O/'proportional_allocation_correction.csv',[dict(j=j,m=m,K=K,old_N_B=2*round(956*statistics.median(int(index[s,j,m,K]['b_base']) for s in range(50))/(statistics.median(int(index[s,j,m,K]['b_base']) for s in range(50))+statistics.median(int(index[s,j,m,K]['b_enhancement']) for s in range(50)))/2),new_N_B=n,new_N_E=956-n,base_field_bits=29,enhancement_field_bits=29) for (j,m,K),n in props.items()])
 return rows,props
def singles(layouts,data):
 d,Q,Q0=data;out=[]
 for q,r in MCS:
  g=layouts[956,q,r]
  assert g['layout_status'].startswith('ACTUAL')
  for m,K in ENDS:
   states=source_states(d,Q,Q0,None,(m,K),g['k']-29);a,_=coefficients(states)
   row=dict(family='SINGLE',target_j=None,target_m=m,target_K=K,N_B=956,q_B=q,rate_B=r,q_E=None,rate_E=None,B=g['geometry_id'],E=None,A=a,Bcoef=0.,null_sources=sum(s['status']=='NULL' for s in states));row['policy_id']=key(row);out.append(row)
 return out
def candidates(layouts,props,data):
 d,Q,Q0=data;out=[];cache={}
 for (j,m,K),nbp in props.items():
  for nb in sorted(set(range(256,769,32))|{nbp}):
   for (qb,rb),(qe,re) in itertools.product(MCS,repeat=2):
    b=layouts[nb,qb,rb];e=layouts[956-nb,qe,re]
    if not(b['layout_status'].startswith('ACTUAL') and e['layout_status'].startswith('ACTUAL')):continue
    ck=j,m,K,b['k']-29,e['k']-29
    if ck not in cache:
     st=source_states(d,Q,Q0,j,(m,K),ck[-2],ck[-1]);a,be=coefficients(st)
     cache[ck]=(a,be,sum(s['status']=='NULL' for s in st),sum(s['status']=='EMPTY_ENHANCEMENT' for s in st))
    a,be,nul,emp=cache[ck]
    if all(abs(x)<1e-15 for x in (a,be)):continue
    row=dict(family='EEP' if (qb,rb)==(qe,re) else 'U',target_j=j,target_m=m,target_K=K,N_B=nb,q_B=qb,rate_B=rb,q_E=qe,rate_E=re,B=b['geometry_id'],E=e['geometry_id'],A=a,Bcoef=be,null_sources=nul,empty_enhancement_sources=emp,proportional=(nb==nbp));row['policy_id']=key(row);out.append(row)
 return out
def proxy(row,table,q0,lower=False):
 result=[]
 for snr in SNRS:
  def p(g):v=table[g,snr];return v['ci'][0] if lower else v['p_correct']
  ph=p('header');pb=p(row['B']);pe=p(row['E']) if row['E'] else 0.
  result.append(q0+ph*pb*(row['A']+pe*row['Bcoef']))
 return dict(Q7=result[0],Q10=result[1],J=.25*result[0]+.75*result[1])
def nominate(pool,single,layouts,table,q0):
 # The borrowed single956 rate is explicitly screen-only and never inserted
 # as a measured group probability. All new body cells remain unknown.
 bymcs={(r['q_B'],r['rate_B']):r['B'] for r in single}
 for r in pool:
  transfer=dict(table)
  for role in ('B','E'):
   mc=(r['q_'+role],r['rate_'+role]);gid=bymcs[mc]
   for s in SNRS:transfer[r[role],s]=table[gid,s]
  r['nomination_transfer_J']=proxy(r,transfer,q0)['J'];r['nomination_transfer_low']=proxy(r,transfer,q0,True)['J']
 order=lambda r:(-r['nomination_transfer_low'],-r['nomination_transfer_J'],float(Fraction(r['rate_B'])+Fraction(r['rate_E'])),not r['proportional'],r['policy_id'])
 selected=[];physical=set()
 def choose(rows,reason):
  row=next((r for r in sorted(rows,key=order) if (r['B'],r['E']) not in physical),None)
  if row:selected.append(dict(row,nomination_reason=reason));physical.add((row['B'],row['E']))
 for family in ('EEP','U'):
  for j in (6,7):
   for q in (2,4,6):choose([r for r in pool if (family=='U' or r['family']=='EEP') and r['target_j']==j and r['q_B']==q],f'{family}_actual_fallback_quality_j{j}_baseq{q}')
 for i in range(4):
  used={(r['q_E'],r['rate_E']) for r in selected if r['family']=='U'}
  options=[r for r in pool if r['family']=='U' and (r['q_E'],r['rate_E']) not in used]
  choose(options or [r for r in pool if r['family']=='U'],f'U_open_{i}_distinct_enhancement_MCS')
 assert len(selected)<=16
 return selected
def interval(row,table,q0):
 bounds=[]
 for s in SNRS:
  b=table[row['B'],s]['ci'];e=table[row['E'],s]['ci'] if row['E'] else (0,0);h=table['header',s]['ci'];v=[]
  for pb,pe,ph in itertools.product(b,e,h):v.append(q0+ph*pb*(row['A']+pe*row['Bcoef']))
  # If B=E, quality is a quadratic in the SAME probability; add its stationary
  # point and do not give two fictitious independent estimates to the alias.
  if row['E']==row['B']:
   ps=list(b)
   if row['Bcoef'] and b[0]<=-row['A']/(2*row['Bcoef'])<=b[1]:ps.append(-row['A']/(2*row['Bcoef']))
   v=[q0+ph*(p*row['A']+p*p*row['Bcoef']) for p,ph in itertools.product(ps,h)]
  bounds.append((min(v),max(v)))
 return .25*bounds[0][0]+.75*bounds[1][0],.25*bounds[0][1]+.75*bounds[1][1]
def delta_bounds(row,base,table):
 import itertools
 bounds=[]
 for s in SNRS:
  gids=list(dict.fromkeys([row['B'],row['E'],base['B']]));gids=[g for g in gids if g]
  values=[]
  for pvalues in itertools.product(*(table[g,s]['ci'] for g in gids)):
   p=dict(zip(gids,pvalues));pb=p[row['B']];pe=p[row['E']] if row['E'] else 0.;ps=p[base['B']]
   coefficient=pb*(row['A']+pe*row['Bcoef'])-ps*base['A']
   values.extend(h*coefficient for h in table['header',s]['ci'])
  if row['B']==row['E'] and row['Bcoef']:
   other=[g for g in gids if g!=row['B']]
   for vs in itertools.product(*(table[g,s]['ci'] for g in other)):
    psame=base['B']==row['B'];a=row['A']-(base['A'] if psame else 0);critical=-a/(2*row['Bcoef'])
    lo,hi=table[row['B'],s]['ci']
    if lo<=critical<=hi:
     p=dict(zip(other,vs));coefficient=critical*(row['A']+critical*row['Bcoef'])-(critical if psame else p[base['B']])*base['A']
     values.extend(h*coefficient for h in table['header',s]['ci'])
  bounds.append((min(values),max(values)))
 return .25*bounds[0][0]+.75*bounds[1][0],.25*bounds[0][1]+.75*bounds[1][1]

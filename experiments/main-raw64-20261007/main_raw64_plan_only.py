"""Prepared raw12/raster/KEEP resource extension. No packet or visual calls.

queries creates the calibration-blind integer-K planning superset. plan may
construct pinned CPU encoder metadata for missing tuples, never forward/decode.
assemble replays the original finite-K generator and proposes appended public
IDs. Its output is not a PHY qualification, ranking or execution registration.
"""
from dataclasses import dataclass
from fractions import Fraction
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback

SIZES=(1,2,3,4,5,6,8,10,13,16)
PREFIX=tuple(sum(s*s for s in SIZES[:m]) for m in range(11))
RATES=('1/3','1/2','2/3','3/4','5/6')
MODULATIONS={'QPSK':2,'16QAM':4,'64QAM':6}
BODY=956
MAPPING64='sequential_I_then_Q_3bits;Gray_inverse_binary_PAM;axis[-7,-5,-1,-3,7,5,1,3]/sqrt21'
OLD_PLAN_SHA='ce68bb1bfcb638feda1edeecdc6cf8c4087c95c1d1382e1c14337cd5f36d133c'
OLD_PROFILES_SHA='f2717e697e62325a4d3d252f95bc770bbbd451d12e3d490d51cda15848425b81'
OLD_ALIASES_SHA='d455e0d04b6da68ab4c441bdaf5c0a7bbc79722b56232ba3809061eaa46493bb'
OLD_QUALIFICATION_SHA='d23d9dc2c9c08855f2b539db90f51b83d034b61d97a3e0bf19093db95a232dd7'
FROZEN_SOURCES={
    'action_core':'7ca5809c00f8fd5ac0040d0abd664e1673cee71287bbe383e07fd1039a87e28d',
    'original_planner':'f9ea41595efa976a93c921f7be5241ad83d30777f4cc7735f099b6db7fcc86f0',
    'original_backend':'b085a300899bfd996f3a068955fa9694ea0fd911fd2de248022ddf9523211a3f',
    'original_common':'65080a95fd227f7f7001127356f81ab691899a9a540dc8ffba9fbe0ad72eb758'}

def require(v,m):
    if not v:raise ValueError(m)
def identity(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,v):
    with Path(p).open('x',encoding='utf-8',newline='\n') as f:json.dump(v,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
def load(p,name):
    if name in sys.modules:
        require(Path(sys.modules[name].__file__).absolute()==Path(p).absolute(),'Unexpected imported module: '+name)
        return sys.modules[name]
    s=importlib.util.spec_from_file_location(name,p);v=importlib.util.module_from_spec(s);sys.modules[name]=v;s.loader.exec_module(v);return v

@dataclass(frozen=True)
class MCS:
    modulation:str
    nominal_rate:str
    def __post_init__(self):require(self.modulation in MODULATIONS and self.nominal_rate in RATES,'Fixed MAIN MCS only')
    @property
    def width(self):return MODULATIONS[self.modulation]
    @property
    def name(self):return self.modulation+':r'+self.nominal_rate

def symbols(T,rate,q):
    r=Fraction(rate);k=12*T+16
    return ((k*r.denominator+r.numerator-1)//r.numerator+q-1)//q

def query_superset(modulations=tuple(MODULATIONS)):
    rows={}
    for mod in modulations:
        q=MODULATIONS[mod]
        for rate in RATES:
            for T in range(1,681):
                first=symbols(T,rate,q)
                if first>BODY:continue
                for mode,uses in [('nominal',first),('full_budget',BODY)]:
                    key=(12*T+16,uses*q,q)
                    row=rows.setdefault(key,dict(k=key[0],n=key[1],num_bits_per_symbol=q,source_token_count=T,query_id=identity(list(key)),labels=[]))
                    row['labels'].append(dict(modulation=mod,nominal_rate=rate,allocation_mode=mode))
    return [rows[k] for k in sorted(rows)]

def index_plans(rows):
    result={}
    for row in rows:
        key=(row['k'],row['n'],row['num_bits_per_symbol'])
        require(all(type(x) is int for x in key) and key[2] in (2,4,6) and key[0]>=28
            and (key[0]-16)%12==0 and key[1]>=key[0] and key[1]%key[2]==0,'Invalid raw query dimensions')
        require(key not in result and row['query_id']==identity(list(key)),'Duplicate/changed query identity')
        require(row['disposition'] in ('PLANNED','UNSUPPORTED'),'No silent resource failures')
        if row['disposition']=='PLANNED':
            layout=row['layout'];require(layout['k']==key[0] and layout['n']==key[1],'Plan dimensions changed')
            require(layout['layout_id']==identity({k:v for k,v in layout.items() if k!='layout_id'}),'Layout hash changed')
        else:require(row.get('exception_type')=='UnsupportedConfiguration' and row.get('reason'),'Explicit backend rejection required')
        result[key]=row
    return result

def split_queries(queries,old):
    require(old['status']=='MAIN_BACKEND_PLAN_ONLY_COMPLETE' and old['synthetic'] is False
        and old['encoder_forward_calls']==old['decoder_calls']==0,'Original completed metadata-only evidence required')
    indexed=index_plans(old['plans']);missing=[];reused=[]
    require(set(indexed)=={(r['k'],r['n'],r['num_bits_per_symbol']) for r in query_superset(('QPSK','16QAM'))}
        and old['query_count']==1848,'Original full1848 response grid required')
    for row in queries:
        key=(row['k'],row['n'],row['num_bits_per_symbol'])
        if key in indexed:
            require(key[2] in (2,4),'Old MAIN evidence cannot impersonate64QAM');reused.append(copy.deepcopy(indexed[key]))
        else:missing.append(row)
    return reused,missing

def new_identity(old_identity,adapter_sha):
    value=copy.deepcopy(old_identity)
    value.update(implementation='MAIN_RAW64_RESOURCE_PLAN_ADAPTER_V1',bit_mapping=MAPPING64,
        original_backend_identity=copy.deepcopy(old_identity),resource_adapter_sha256=adapter_sha)
    return value

class UnsupportedConfiguration(ValueError):pass

def encoder_layout(Encoder,k,n,q,implementation):
    """Only actual encoder-constructor fields; k28 m1 is deliberately legal."""
    require(q==6 and k>=28 and (k-16)%12==0 and n%q==0 and n>=k,'Raw MAIN64 dimensions invalid')
    try:e=Encoder(k,n,num_bits_per_symbol=q,device='cpu',precision='single')
    except (ValueError,AssertionError) as error:raise UnsupportedConfiguration(type(error).__name__+': '+str(error)) from error
    def forbidden(*a,**kw):raise RuntimeError('Encoder forward forbidden in plan-only stage')
    e.forward=forbidden
    require(e.k==k and e.n==n and e.k_ldpc-e.k_filler==k,'Encoder k/filler differs')
    if n>e.n_cb_comp:raise UnsupportedConfiguration('Repetition outside pinned RV0 unsupported')
    a=e.out_int.cpu().numpy();require(sorted(a.tolist())==list(range(n)),'Interleaver is not exact permutation')
    layout=dict(k=k,n=n,k_ldpc=int(e.k_ldpc),k_filler=int(e.k_filler),n_cb=int(e.n_cb),n_cb_comp=int(e.n_cb_comp),
        bg=str(e._bg),z=int(e.z),code_blocks=1,mother_bits=int(e.n_ldpc),puncturing_bits=int(e.n_ldpc-e.k_filler-n),
        puncturing_first_2Z=int(2*e.z),puncturing_remaining=int(e.n_cb_comp-n),shortening_bits=int(e.k_filler),
        repetition_bits=0,modulation_padding_bits=0,interleaver_sha256=hashlib.sha256(a.tobytes()).hexdigest(),
        decoder_config=implementation['decoder'],bit_mapping=MAPPING64,scrambling=implementation['scrambling'],
        power_protocol='fixed_constellation_average_Es2',actual_effective_rate=k/n,implementation=implementation)
    layout['layout_id']=identity(layout);return layout

def enumerate64(core,original,plans,backend_identity):
    cache=index_plans(plans)
    class Backend:
        identity=backend_identity
        qualified=False
        def plan(self,k,n,num_bits_per_symbol):
            key=(k,n,num_bits_per_symbol);require(key in cache,'Missing actual plan response; cannot prune source range')
            r=cache[key]
            if r['disposition']=='UNSUPPORTED':raise original.UnsupportedConfiguration(r['reason'])
            require(r['layout']['implementation']==backend_identity,'Wrong64 layout implementation')
            return r['layout']
    planner=core.MainPlanner(original.Planner(Backend()));rows={};generation=[]
    def add(row):
        if row is None:return
        cid=row['candidate_id']
        if cid in rows:rows[cid]['allocation_modes']=sorted(set(rows[cid]['allocation_modes']+row['allocation_modes']))
        else:rows[cid]=copy.deepcopy(row)
    for rate in RATES:
        mcs=MCS('64QAM',rate)
        for mode in ('nominal','full_budget'):
            for m in range(1,11):add(planner.candidate(m,0,mcs,mode))
            for m in range(1,10):
                L=SIZES[m]**2;maximum=planner.max_K(m,mcs,mode);grid={0,L//4,L//2,3*L//4}
                if maximum is not None:grid.add(maximum)
                generation.append(dict(m=m,nominal_rate=rate,allocation_mode=mode,max_legal_K=maximum,K_candidates=sorted(grid)))
                for K in sorted(grid):add(planner.candidate(m,K,mcs,mode))
    return dict(profiles=sorted(rows.values(),key=lambda r:r['candidate_id']),generation=generation,
        attempts=[{k:v for k,v in a.items() if k!='candidate'} for _,a in sorted(planner.attempts.items())],
        all_integer_K_is_only_a_resource_query_superset=True,finite_K_scientific_candidate_rule_unchanged=True)

def append_catalogue(old_profiles,old_aliases,new_candidates):
    require(len(old_profiles)==270 and len(old_aliases)==329 and [p['profile_id'] for p in old_profiles]==list(range(270)),'Original full270/329 catalogue required')
    profiles=copy.deepcopy(old_profiles);aliases=copy.deepcopy(old_aliases);oldw={p['wire_key'] for p in profiles};grouped={}
    for row in new_candidates:
        require(row['groups'][0]['modulation']=='64QAM' and row['wire_key'] not in oldw,'Only newly admitted64 wires may be appended')
        grouped.setdefault(row['wire_key'],[]).append(row)
    require(len(profiles)+len(grouped)<=4096,'Paid12bit header capacity exceeded')
    for wire,rows in sorted(grouped.items()):
        rows=sorted(rows,key=lambda r:r['candidate_id']);require(len({r['candidate_id'] for r in rows})==len(rows),'Duplicate alias')
        pid=len(profiles);representative=copy.deepcopy(rows[0]);representative.update(profile_id=pid,alias_candidate_ids=[r['candidate_id'] for r in rows]);profiles.append(representative)
        for r in rows:aliases.append(dict(candidate_id=r['candidate_id'],representative_candidate_id=representative['candidate_id'],wire_key=wire,
            profile_id=pid,nominal_rate=r['groups'][0]['nominal_rate'],modulation='64QAM',allocation_modes=r['allocation_modes']))
    require(profiles[:270]==old_profiles and aliases[:329]==old_aliases,'Prior transmitted IDs/aliases must be byte-structure preserved')
    return dict(status='PROPOSED_EXTENDED_PUBLIC_CATALOGUE_REQUIRES_REGISTRATION',profiles=profiles,aliases=aliases,
        legacy_profile_count=270,new_wire_count=len(grouped),catalogue_digest=identity(profiles),
        original_RX_results_automatically_reusable=False,header_known_ID_set_changed=bool(grouped))

def validate_response(nr,old,missing,adapter_sha):
    require(nr['status']=='MAIN_RAW64_MISSING_RESOURCE_PLANS_COMPLETE'
        and nr['encoder_forward_calls']==nr['decoder_calls']==0 and nr['GPU_used'] is False,'Completed metadata-only plans required')
    require(nr['old_backend_identity']==old['backend_identity']
        and nr['backend64_identity']==new_identity(old['backend_identity'],adapter_sha),'Plan backend identity changed')
    require(set(index_plans(nr['plans']))=={(r['k'],r['n'],r['num_bits_per_symbol']) for r in missing},'Only exact missing query responses allowed')

def context(path):
    c=read(path);require(c['schema']=='MAIN_RAW64_PLAN_ONLY_CONFIG_V1','Explicit new plan-only config required')
    require(c['scope']=='CPU_RESOURCE_METADATA_ONLY' and c['candidate_K_rule']=='ORIGINAL_FINITE_QUARTILES_AND_BACKEND_MAX'
        and c['new_packet_decodes']==0 and c['GPU_jobs']==0,'Prepared resource-only scope required')
    for p,h in c['source_bindings'].items():
        require(p not in c['input_bindings'] or c['input_bindings'][p]==h,'Conflicting input/source pin')
    for p,h in {**c['source_bindings'],**c['input_bindings']}.items():require(sha(p)==h,'Bound file changed: '+p)
    require(c['source_bindings'].get(str(Path(__file__).absolute()))==sha(__file__),'Plan entry unbound')
    for k,h in FROZEN_SOURCES.items():
        require(c['source_bindings'].get(c[k])==h==sha(c[k]),'Frozen original source changed: '+k)
    for k,h in [('old_plan',OLD_PLAN_SHA),('old_profiles',OLD_PROFILES_SHA),('old_aliases',OLD_ALIASES_SHA)]:
        require(c['input_bindings'].get(c[k])==h==sha(c[k]),'Original exact receipt changed: '+k)
    require(c['input_bindings'].get(c['old_qualification'])==OLD_QUALIFICATION_SHA==sha(c['old_qualification']),'Backend qualification not bound')
    old=read(c['old_plan']);q=read(c['old_qualification']);require(old['backend_identity']==q['backend_identity'] and q['status']=='PASS','Original qualified backend identity mismatch')
    return c,old

def run(stage,config,out):
    c,old=context(config);out=Path(out);require(not out.exists(),'Fresh stage output only; no retry');out.mkdir(parents=True)
    try:
        queries=query_superset();reused,missing=split_queries(queries,old)
        if stage=='queries':
            result=dict(status='MAIN_RAW64_RESOURCE_QUERY_PREPARED',queries=queries,reused_count=len(reused),missing_queries=missing,
                all_scales=list(range(1,11)),new_packet_decodes=0,GPU_used=False,quality_read=False)
        elif stage=='plan':
            require(sys.platform.startswith('linux') and os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU only actual plan invocation')
            require(str(Path(sys.executable).absolute())==c['python'],'Original LDPC interpreter path required')
            require(c['threads']==2 and c['nice']==15 and len(set(c['cpu_affinity']))==2 and 0<c['max_seconds']<=7200,'Bounded two-thread metadata plan only')
            os.sched_setaffinity(0,set(c['cpu_affinity']));os.setpriority(os.PRIO_PROCESS,0,15)
            for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='2'
            load(c['original_common'],'uep_common');original=load(c['original_planner'],'profiles');bm=load(c['original_backend'],'main64_original_backend')
            backend=bm.SionnaBackend('cpu',c['old_qualification']);require(backend.identity==old['backend_identity'],'Actual LDPC runtime differs')
            backend.torch.set_num_threads(2);backend.torch.set_num_interop_threads(2)
            def forbidden(*a,**kw):raise RuntimeError('Packet encoder/decoder forbidden')
            Encoder=backend.Encoder
            def guarded(*a,**kw):
                e=Encoder(*a,**kw);e.forward=forbidden;return e
            backend.Encoder=guarded;backend.Decoder=forbidden;backend.codecs=backend.encode=backend.decode=forbidden
            ident=new_identity(backend.identity,sha(__file__));rows=[];start=time.monotonic()
            for query in missing:
                require(time.time()<c['deadline_unix'] and time.monotonic()-start<c['max_seconds'] and not Path(c['stop_file']).exists(),'Plan STOP/deadline')
                k,n,q=(query[t] for t in ('k','n','num_bits_per_symbol'));row={t:query[t] for t in ('query_id','k','n','num_bits_per_symbol')}
                try:layout=encoder_layout(guarded,k,n,q,ident) if q==6 else backend.plan(k,n,q)
                except (UnsupportedConfiguration,original.UnsupportedConfiguration) as e:row.update(disposition='UNSUPPORTED',exception_type='UnsupportedConfiguration',reason=str(e))
                else:row.update(disposition='PLANNED',layout=layout)
                save(out/(row['query_id']+'.json'),row);rows.append(row)
            result=dict(status='MAIN_RAW64_MISSING_RESOURCE_PLANS_COMPLETE',plans=rows,backend64_identity=ident,
                old_backend_identity=backend.identity,reused_count=len(reused),encoder_forward_calls=0,decoder_calls=0,GPU_used=False)
        elif stage=='assemble':
            require(c['input_bindings'].get(c['new_plans'])==sha(c['new_plans']),'New metadata plans must be explicitly bound')
            nr=read(c['new_plans']);validate_response(nr,old,missing,sha(__file__))
            indexed=index_plans(reused+nr['plans']);require(set(indexed)=={(r['k'],r['n'],r['num_bits_per_symbol']) for r in queries},'All legal-query responses required')
            original=load(c['original_planner'],'profiles');core=load(c['action_core'],'original_main_finite_action_core')
            enum=enumerate64(core,original,nr['plans'],nr['backend64_identity'])
            result=append_catalogue(read(c['old_profiles']),read(c['old_aliases']),enum['profiles']);result['enumeration64']=enum
        else:raise ValueError('Unknown explicit stage')
        result.update(source_bindings=c['source_bindings'],input_bindings=c['input_bindings'],config_sha256=sha(config),
            qualified_new_PHY=False,ledger_created=False,quality_ranking=False,development_read=False,automatic_successor=False)
        if stage=='plan':require(time.time()<c['deadline_unix'] and time.monotonic()-start<c['max_seconds'] and not Path(c['stop_file']).exists(),'Plan STOP/deadline at completion')
        context(config)
        save(out/'completion.json',result);return result
    except BaseException:save(out/'failure.json',dict(status='FAILED_PRESERVE_NO_RETRY',traceback=traceback.format_exc()));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['queries','plan','assemble'],required=True);p.add_argument('--config',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    print(run(a.stage,a.config,a.out)['status'])

"""N2048 raw action resource planning only; no PHY, neural inference or selection.

The unchanged finite action generator and actual encoder constructors are used
in a private module instance with only N/body/modulation constants extended.
"""
from __future__ import annotations
import argparse,csv,io,json,os,signal,sys,time
from pathlib import Path
from fractions import Fraction
import t2_pilot as h

SCHEMA='WCL_T6_N2048_METADATA_PLAN_V1'
N=2048;HEADER=68;BODY=1980
RATES=('1/3','1/2','2/3','3/4','5/6')
MODULATIONS={'QPSK':2,'16QAM':4,'64QAM':6}
CORE_SHA='7ca5809c00f8fd5ac0040d0abd664e1673cee71287bbe383e07fd1039a87e28d'
FIXED={'original_common':'65080a95fd227f7f7001127356f81ab691899a9a540dc8ffba9fbe0ad72eb758',
       'original_planner':'f9ea41595efa976a93c921f7be5241ad83d30777f4cc7735f099b6db7fcc86f0',
       'original_backend':'b085a300899bfd996f3a068955fa9694ea0fd911fd2de248022ddf9523211a3f'}
QUAL_SHA='d23d9dc2c9c08855f2b539db90f51b83d034b61d97a3e0bf19093db95a232dd7'

def queries():
    rows={}
    for modulation,q in MODULATIONS.items():
        for rate in RATES:
            r=Fraction(rate)
            for tokens in range(1,681):
                k=12*tokens+16;symbols=((k*r.denominator+r.numerator-1)//r.numerator+q-1)//q
                if symbols>BODY:continue
                for mode,uses in [('nominal',symbols),('full_budget',BODY)]:
                    key=(k,uses*q,q);value=rows.setdefault(key,dict(k=k,n=key[1],num_bits_per_symbol=q,source_token_count=tokens,
                        query_id=h.digest(list(key)),labels=[]))
                    value['labels'].append(dict(modulation=modulation,nominal_rate=rate,allocation_mode=mode))
    return [rows[k]for k in sorted(rows)]

def write_csv(path,rows):
    buffer=io.StringIO(newline='');w=csv.DictWriter(buffer,list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);data=buffer.getvalue().encode()
    if p.exists():h.require(p.read_bytes()==data,'Completed metadata table changed')
    else:p.write_bytes(data)

def emit_queries(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);q=queries()
    h.save(out/'query_superset.json',dict(status='METADATA_QUERIES_NOT_LEGAL_ACTIONS',N=N,header_symbols=HEADER,body_symbols=BODY,
        queries=q,all_integer_K_only_resource_query_superset=True,candidate_K_rule='ORIGINAL_FINITE_QUARTILES_AND_BACKEND_MAX',
        actual_constructor_calls=0,encoder_forward_calls=0,packet_decodes=0,GPU_used=False))
    write_csv(out/'query_superset.csv',[{k:v for k,v in x.items()if k!='labels'}for x in q])
    result=dict(status='QUERY_SUPERSET_WRITTEN_NOT_ACTUAL_LAYOUTS',query_count=len(q),N=N,m10_raw_bits=8160,m10_information_bits=8176,
        formal_test_design=dict(SNRs=[4,10,19],source_count=100,noise_count=3,method_count=3,logical_frames=2700),
        entropy_family='PENDING_N1024_CONCLUSIONS',confirmation_sources='NOT_SELECTED',new_scientific_calls=0)
    h.save(out/'completion.json',result);return result

def prepare(a):
    env=h.read(a.environment_request);cc=env['cpu_config'];root=Path(a.root).resolve();out=Path(a.out).resolve()
    h.require(Path(env['root']).resolve()==root,'Metadata environment repository differs')
    h.require(out.is_relative_to(root/'outputs/WCL-EVIDENCE-CLOSURE-20261009'),'New WCL output namespace')
    fields={k:cc['adapter_config'][k]for k in FIXED}
    fields['raw64_planner']=cc['adapter_config']['plan_module']
    fields['action_core']=str(root/'outputs/CONTENT-REAL-64QAM-20261006/prepared_v1/main_plan_only_runtime/main_action_space.py')
    qualification=cc['adapter_config']['legacy_qualification'];h.require(h.sha(qualification)==QUAL_SHA,'Original actual backend qualification changed')
    bindings={str(Path(__file__).resolve()):h.sha(__file__),str(Path(h.__file__).resolve()):h.sha(h.__file__),
        str(Path(h.__file__).with_name('t2_ledger.py').resolve()):h.sha(Path(h.__file__).with_name('t2_ledger.py'))}
    for name,path in fields.items():
        actual=h.sha(path);expected=CORE_SHA if name=='action_core'else FIXED.get(name,env['source_bindings'].get(path))
        h.require(actual==expected,'Frozen source differs: '+name);bindings[path]=actual
    h.require(Path(cc['python']).is_file(),'Original CPU interpreter unavailable')
    request=dict(schema=SCHEMA,N=N,header_symbols=HEADER,body_symbols=BODY,root=str(root),out=str(out),python=cc['python'],
        **fields,source_bindings=bindings,qualification=h.desc(qualification),environment_request=h.desc(a.environment_request),
        threads=2,nice=15,max_seconds=a.max_seconds,deadline_unix=a.deadline_unix,stop_files=[str(root/'STOP'),str(out/'STOP')],
        rate_grid=list(RATES),modulation_grid=MODULATIONS,all_scales=list(range(1,11)),source_format='raw12_raster_one_body_group_CRC16_KEEP',
        finite_K_rule='per m/MCS/mode:0,L//4,L//2,3L//4,descending_actual_backend_max; canonical full next scale',
        partial_family_includes_whole=True,whole_has_full_budget_protection=True,packet_cap=0,GPU_cap=0,
        entropy_family=None,confirmation_sources=None,quality_selection_allowed=False,formal_test_allowed=False,
        formal_test_design=dict(SNRs=[4,10,19],source_count=100,noise_count=3,method_count=3,logical_frames=2700))
    h.save(out/'request.json',request);return dict(status='T6_METADATA_PLAN_REGISTERED_NOT_RUN',request=h.desc(out/'request.json'),packet_cap=0,GPU_cap=0)

def configure_core(core):
    h.require(core.N==1024 and core.BODY==956 and core.HEADER==68,'Original main generator constants changed')
    core.N=N;core.BODY=BODY;core.MODULATIONS=dict(MODULATIONS)
    return core

def catalogue(enumeration):
    groups={}
    for row in enumeration['profiles']:groups.setdefault(row['wire_key'],[]).append(row)
    h.require(len(groups)<=4096,'Paid twelve-bit profile ID space exhausted')
    profiles=[];aliases=[]
    for pid,(wire,rows)in enumerate(sorted(groups.items())):
        rows=sorted(rows,key=lambda x:x['candidate_id']);representative=dict(rows[0]);representative.update(profile_id=pid,
            alias_candidate_ids=[x['candidate_id']for x in rows],allocation_modes=sorted({m for x in rows for m in x['allocation_modes']}))
        profiles.append(representative)
        aliases.extend(dict(candidate_id=x['candidate_id'],profile_id=pid,wire_key=wire,allocation_modes=x['allocation_modes'],
            nominal_rate=x['groups'][0]['nominal_rate'])for x in rows)
    h.require(all(p['N']==N and p['header_symbols']==HEADER and p['body_symbols']==BODY and p['used_body_symbols']+p['idle_symbols']==BODY for p in profiles),'Exact N2048 accounting')
    h.require(all(('WHOLE'in p['family_memberships'])==(p['K']==0) and'PARTIAL_WITH_WHOLE_FALLBACK'in p['family_memberships']for p in profiles),'Equal family permissions')
    return dict(status='T6_ACTUAL_CONSTRUCTOR_ACTION_CATALOGUE_NOT_PHY_QUALIFIED',profiles=profiles,aliases=aliases,
        catalogue_sha256=h.digest(profiles),profile_namespace='WCL_T6_N2048_RAW_V1',original_N1024_ids_reused=False)

def plan(path):
    r=h.read(path);rh=h.sha(path);out=Path(r['out']);started=time.monotonic()
    h.require(r['schema']==SCHEMA and r['N']==N and r['body_symbols']==BODY and r['packet_cap']==r['GPU_cap']==0,'Metadata-only frozen scope')
    h.require(sys.platform.startswith('linux') and os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU-only Linux metadata constructor stage')
    h.require(str(Path(sys.executable).absolute())==r['python'],'Use registered original LDPC interpreter')
    for p,v in r['source_bindings'].items():h.require(h.sha(p)==v,'Bound planning source changed')
    h.require(h.sha(r['qualification']['path'])==r['qualification']['sha256']==QUAL_SHA,'Old qualification changed')
    h.require(h.sha(r['environment_request']['path'])==r['environment_request']['sha256'],'Bound environment request changed')
    if(out/'completion.json').exists():
        done=h.read(out/'completion.json');h.require(done['request_sha256']==rh,'Completed planning request differs')
        for p,v in done['outputs'].items():h.require(h.sha(p)==v,'Completed planning output changed')
        return done
    h.require(0<r['max_seconds']<=7200 and r['threads']==2 and r['nice']==15,'Bounded CPU metadata scope')
    for sig in(signal.SIGINT,signal.SIGTERM):signal.signal(sig,h.stop)
    for key in('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='2'
    os.setpriority(os.PRIO_PROCESS,0,15)
    with h.lock(out/'metadata_plan.lock'):
        h.load(r['original_common'],'uep_common',r['source_bindings']);original=h.load(r['original_planner'],'profiles',r['source_bindings'])
        bm=h.load(r['original_backend'],'_t6_original_backend',r['source_bindings']);raw=h.load(r['raw64_planner'],'_t6_raw64_planner',r['source_bindings'])
        core=configure_core(h.load(r['action_core'],'_t6_original_finite_generator',r['source_bindings']))
        base=bm.SionnaBackend('cpu',r['qualification']['path']);base.torch.set_num_threads(2);base.torch.set_num_interop_threads(2)
        identity=raw.new_identity(base.identity,r['source_bindings'][r['raw64_planner']]);Encoder=base.Encoder
        def forbidden(*a,**kw):raise RuntimeError('Packet encoding/decoding forbidden during T6 resource planning')
        def guarded(*a,**kw):
            instance=Encoder(*a,**kw);instance.forward=forbidden;return instance
        base.Encoder=guarded;base.Decoder=forbidden;base.codecs=base.encode=base.decode=forbidden
        new=0;reused=0;proofs={}
        class Backend:
            qualified=False
            def __init__(self):self.identity=identity
            def plan(self,k,n,num_bits_per_symbol):
                nonlocal new,reused
                h.guard(r,started);q=num_bits_per_symbol;key=h.digest([k,n,q]);p=out/'plans'/(key+'.json')
                if p.exists():
                    row=h.read(p);h.require(row['request_sha256']==rh and row['dimensions']==[k,n,q],'Resource checkpoint identity differs');reused+=1
                else:
                    try:layout=raw.encoder_layout(guarded,k,n,q,identity)if q==6 else base.plan(k,n,q)
                    except(raw.UnsupportedConfiguration,original.UnsupportedConfiguration)as exc:
                        row=dict(request_sha256=rh,dimensions=[k,n,q],disposition='UNSUPPORTED',exception_type='UnsupportedConfiguration',reason=str(exc))
                    else:row=dict(request_sha256=rh,dimensions=[k,n,q],disposition='PLANNED',layout=layout)
                    h.save(p,row);new+=1
                proofs[str(p)]=h.sha(p)
                if row['disposition']=='UNSUPPORTED':raise original.UnsupportedConfiguration(row['reason'])
                h.require(row['layout']['layout_id']==h.digest({k:v for k,v in row['layout'].items()if k!='layout_id'}),'Actual layout hash differs')
                return row['layout']
        enumeration=core.enumerate_actions(original.Planner(Backend()));cat=catalogue(enumeration)
        h.save(out/'finite_enumeration.json',enumeration);h.save(out/'candidate_catalogue.json',cat)
        table=[]
        for p in cat['profiles']:
            g=p['groups'][0];table.append(dict(profile_id=p['profile_id'],candidate_id=p['candidate_id'],wire_key=p['wire_key'],N=N,m=p['m'],K=p['K'],
                token_count=p['token_count'],modulation=g['modulation'],nominal_rate=g['nominal_rate'],source_bits=g['source_bits'],information_bits=g['information_bits'],
                transmitted_bits=g['transmitted_bits'],effective_rate=g['effective_information_rate'],header_symbols=HEADER,body_symbols=p['used_body_symbols'],
                padding_symbols=p['idle_symbols'],padding_energy_rule='known_QPSK_Es2',allocation_modes=';'.join(p['allocation_modes']),
                families=';'.join(p['family_memberships']),layout_id=g['layout']['layout_id'],new_packet_decodes=0,PHY_roundtrip_qualified=False))
        write_csv(out/'legal_actions_metadata.csv',table)
        for name in('finite_enumeration.json','candidate_catalogue.json','legal_actions_metadata.csv'):proofs[str(out/name)]=h.sha(out/name)
        h.guard(r,started)
        done=dict(status='T6_RESOURCE_METADATA_COMPLETE_NOT_PHY_QUALIFIED',request_sha256=rh,request_path=str(Path(path).absolute()),N=N,
            alias_candidates=len(cat['aliases']),unique_wire_actions=len(cat['profiles']),whole_actions=sum(p['K']==0 for p in cat['profiles']),
            full_budget_whole_actions=sum(p['K']==0 and'full_budget'in p['allocation_modes']for p in cat['profiles']),m10_actions=sum(p['m']==10 for p in cat['profiles']),
            resource_query_count=len(proofs)-3,new_constructor_query_checkpoints=new,reused_constructor_query_checkpoints=reused,
            encoder_forward_calls=0,packet_decodes=0,GPU_used=False,quality_read=False,calibration_read=False,outputs=proofs,
            entropy_family=None,test_sources=None,formal_test_started=False,automatic_successor=False)
        h.save(out/'completion.json',done);return done

def main():
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='command',required=True)
    a=s.add_parser('queries');a.add_argument('--out',required=True)
    a=s.add_parser('prepare')
    for k in('root','environment-request','out'):a.add_argument('--'+k,required=True)
    a.add_argument('--deadline-unix',type=float,required=True);a.add_argument('--max-seconds',type=int,default=7200)
    a=s.add_parser('plan');a.add_argument('--request',required=True);a=p.parse_args()
    result=emit_queries(a.out)if a.command=='queries'else(prepare(a)if a.command=='prepare'else plan(a.request))
    print(h.canonical({k:v for k,v in result.items()if k!='outputs'}))
if __name__=='__main__':main()

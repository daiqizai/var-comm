"""LE1 source feasibility only; layouts and proposed missing cells, no packets."""
from common import *
import sys,statistics,itertools,importlib.util
from collections import Counter
from group_layout_design import GroupLayoutPlanner

def run():
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
 reg=read(O/'registration.json');assert reg['authorized_stages']==['LE0','LE1']
 for p in reg['source_bindings']:checked(p)
 assert sha(R/'experiments/entropy-partial-mismatch-20261010/scripts/ep_phy.py')=='12cad3e495fc91d8f6819b895674df08e312da3833291019b8966a2211ec8c75'
 out=O/'feasibility';out.mkdir();data=T/'feasibility';data.mkdir()
 lengths=list(csv.DictReader((O/'source_lengths.csv').open()));assert len(lengths)==3400
 for r in lengths:
  for k in ('pilot_slot','j','m','K','b_single','b_base','b_enhancement'):r[k]=int(r[k])
 keyed={(r['pilot_slot'],r['j'],r['m'],r['K']):r for r in lengths}
 base={(r['pilot_slot'],r['j']):r['b_base'] for r in lengths}
 targets=[(j,m,K) for j in (6,7) for m,K in ENDS if OFFSETS[m]+K>OFFSETS[j]];assert len(targets)==26
 # Proportional length allocation is allowed for U as well as EEP, even when
 # an eventual U and S budget choice coincide. No performance forced by labels.
 proportional={}
 for j,m,K in targets:
  b=statistics.median(keyed[s,j,m,K]['b_base'] for s in range(50));e=statistics.median(keyed[s,j,m,K]['b_enhancement'] for s in range(50))
  proportional[j,m,K]=max(2,min(954,2*round((956*b/(b+e))/2)))
 allocations=sorted(set(range(256,769,32))|set(proportional.values()))
 sys.path.insert(0,str(R/'experiments/prior-aware-grouped-mcs-v1'))
 spec=importlib.util.spec_from_file_location('_LE1_original_ldpc',R/'experiments/prior-aware-grouped-mcs-v1/ldpc_backend.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 backend=mod.SionnaBackend('cpu',R/'outputs/PRIOR-AWARE-UEP-20261004-V1/ldpc_qualification.json')
 t=backend.torch;t.set_num_threads(1);t.set_num_interop_threads(1);assert not t.cuda.is_initialized()
 # Block all scientific LDPC tensor forwards. Constructor metadata only.
 forward_calls=[0]
 def forbidden(*args,**kwargs):forward_calls[0]+=1;raise AssertionError('LE1 prohibits packet/LDPC forward')
 backend.Encoder.forward=forbidden;backend.Decoder.forward=forbidden
 planner=GroupLayoutPlanner(backend.Encoder,backend.identity);rates=('1/3','1/2','2/3','3/4','5/6');mcs=list(itertools.product((2,4,6),rates))
 groups={}
 for N in sorted(set(allocations)|{956-N for N in allocations}):
  for q,r in mcs:groups[N,q,r]=planner.plan(N,q,r)
 save(data/'group_layouts.json',dict(status='CONSTRUCTOR_ONLY_NO_FORWARD_NO_PACKET_QUALIFICATION',source=pin(E/'group_layout_design.py'),original_ep_phy=pin(R/'experiments/entropy-partial-mismatch-20261010/scripts/ep_phy.py'),backend_identity=backend.identity,layouts=list(planner.cache.values()),encoder_decoder_forward_calls=forward_calls[0],wireless_frames=0,packet_trials=0))
 usage=Counter();fullpath=data/'capacity_grid.csv';summary={};rows_written=0;feasible_settings=0
 with fullpath.open('x',encoding='utf8',newline='') as f:
  fieldnames=['target_j','target_m','target_K','N_B','N_E','q_B','rate_B','q_E','rate_E','k_B','k_E','C_B','C_E','status','allocation_modes','full_target_sources','fallback_enhanced_sources','base_only_sources','null_sources','actual_base_depths','minimum_base_margin','median_base_margin','minimum_enhancement_margin','median_enhancement_margin']
  writer=csv.DictWriter(f,fieldnames=fieldnames);writer.writeheader()
  for j,m,K in targets:
   best=None
   for N in allocations:
    for (qb,rb),(qe,re) in itertools.product(mcs,repeat=2):
     Bp=groups[N,qb,rb];Ep=groups[956-N,qe,re];supported=Bp['layout_status'].startswith('ACTUAL') and Ep['layout_status'].startswith('ACTUAL')
     modes=['U_UNRESTRICTED']
     if qb==qe and rb==re:modes.append('EEP_EQUAL_MCS')
     if N==proportional[j,m,K]:modes.append('PROPORTIONAL_MEDIAN_LENGTH_AVAILABLE_TO_U_AND_EEP')
     if N in range(256,769,32):modes.append('COARSE_GRID')
     row=dict(target_j=j,target_m=m,target_K=K,N_B=N,N_E=956-N,q_B=qb,rate_B=rb,q_E=qe,rate_E=re,k_B=Bp['k'],k_E=Ep['k'],C_B=Bp['source_capacity_bits'],C_E=Ep['source_capacity_bits'],status='CAPACITY_ONLY_NOT_WIRELESS' if supported else 'UNSUPPORTED_CONSTRUCTOR',allocation_modes=';'.join(modes))
     null=baseonly=enhanced=full=0;depths=Counter();bmargin=[];emargin=[]
     if supported:
      for slot in range(50):
       actual_j=next((d for d in range(j,3,-1) if base[slot,d]<=min(8191,Bp['source_capacity_bits'])),None)
       if actual_j is None:null+=1;continue
       depths[actual_j]+=1;bmargin.append(Bp['source_capacity_bits']-base[slot,actual_j])
       ends=sorted([(a,b) for a,b in ENDS if OFFSETS[actual_j]<OFFSETS[a]+b<=OFFSETS[m]+K],key=lambda z:OFFSETS[z[0]]+z[1],reverse=True)
       actual_end=next(((a,b) for a,b in ends if keyed[slot,actual_j,a,b]['b_enhancement']<=min(8191,Ep['source_capacity_bits'])),None)
       if actual_end is None:baseonly+=1;continue
       enhanced+=1;emargin.append(Ep['source_capacity_bits']-keyed[slot,actual_j,*actual_end]['b_enhancement'])
       full+=actual_j==j and actual_end==(m,K)
      assert null+baseonly+enhanced==50
      usage[Bp['geometry_id']]+=50-null;usage[Ep['geometry_id']]+=enhanced+baseonly
      if enhanced>0:feasible_settings+=1
     row.update(full_target_sources=full,fallback_enhanced_sources=enhanced-full,base_only_sources=baseonly,null_sources=null if supported else 50,actual_base_depths=canonical(depths),minimum_base_margin=min(bmargin) if bmargin else '',median_base_margin=statistics.median(bmargin) if bmargin else '',minimum_enhancement_margin=min(emargin) if emargin else '',median_enhancement_margin=statistics.median(emargin) if emargin else '')
     writer.writerow(row);rows_written+=1
     if supported and (best is None or (full,enhanced,-null,-(qb+qe),-N)>(best['full_target_sources'],best['full_target_sources']+best['fallback_enhanced_sources'],-best['null_sources'],-(best['q_B']+best['q_E']),-best['N_B'])):best=dict(row)
   summary[j,m,K]=best
   print('CAPACITY_TARGET_COMPLETE',j,m,K,flush=True)
  f.flush();os.fsync(f.fileno())
 csvsave(out/'capacity_envelope.csv',[dict(v,selection_purpose='Descriptive capacity envelope only, never quality/policy/checkpoint selection') for v in summary.values() if v])
 # Exact geometry lookup retains provenance and iid payload limitations.
 master=read(OLD/'r1_execution_v1/registered/registration.json');catalogue=read(checked(master['pins']['grouped_catalogue']))
 oldlayouts={g['layout']['layout_id']:dict(g['layout'],q={'QPSK':2,'16QAM':4,'64QAM':6}.get(g['modulation'])) for p in catalogue['profiles'] for g in p['groups']}
 oldpoints=list(csv.DictReader((OLD/'r0_bler_delivery_v1/body_points.csv').open()))
 geometry={}
 for lid,v in oldlayouts.items():
  if v['q'] is None:continue
  key=(v['k'],v['n'],v['q'],v['bg'],v['z'],v['k_ldpc'],v['k_filler'],v['mother_bits'],v['puncturing_first_2Z'],v['puncturing_remaining'],v['interleaver_sha256'])
  if v['implementation']['encoder_source_sha256']==backend.identity['encoder_source_sha256'] and v['decoder_config']==backend.identity['decoder']:geometry.setdefault(key,[]).append(lid)
 gaprows=[];unique={v['geometry_id']:v for v in planner.cache.values() if v['layout_status'].startswith('ACTUAL')}
 for gid,v in sorted(unique.items()):
  key=(v['k'],v['n'],v['q'],v['bg'],v['z'],v['k_ldpc'],v['k_filler'],v['mother_bits'],v['puncturing_first_2Z'],v['puncturing_remaining'],v['interleaver_raw_bytes_sha256']);matched=geometry.get(key,[])
  for snr in (7,10):
   points=[p for p in oldpoints if p['layout_id'] in matched and float(p['snr_db'])==snr]
   gaprows.append(dict(kind='body',geometry_id=gid,N=v['N'],k=v['k'],n=v['n'],q=v['q'],nominal_rate=v['nominal_rate'],snr_db=snr,bg=v['bg'],z=v['z'],interleaver_raw_sha256=v['interleaver_raw_bytes_sha256'],constructor_layout_source=str(data/'group_layouts.json'),legacy_geometry_points=canonical([dict(layout_id=p['layout_id'],trials=int(p['n_blocks']),namespace=p['random_namespace'],payload_applicability=p['applicability']) for p in points]),status='LE2_MISSING_NEW_LENGTH_CRC_PROFILE_PROTOCOL',legacy_geometry_match=bool(points),capacity_usage=usage[gid],reason='Old iid raw-payload geometry evidence is not qualification of the new 13-bit length/canonical parser and paid profile domain'))
 # Two headers count in the proposed initial96 budget. Correct header coding is
 # unchanged, but false accepts must use the new received public profile domain.
 for snr in (7,10):gaprows.append(dict(kind='header',geometry_id='profile12_crc16_tail6_conv136_qpsk',N=68,k=28,n=136,q=2,nominal_rate='header',snr_db=snr,bg='',z='',interleaver_raw_sha256='',constructor_layout_source='Original paid68 header; new public catalogue required',legacy_geometry_points='[]',status='LE2_MISSING_NEW_RECEIVED_PROFILE_DOMAIN',legacy_geometry_match=False,capacity_usage=0,reason='Header correct/invalid/error-accepted outcomes must retain new public received profile domain'))
 csvsave(out/'LE2_exact_missing_cells.csv',gaprows)
 headers=[x for x in gaprows if x['kind']=='header'];bodies=sorted((x for x in gaprows if x['kind']=='body'),key=lambda x:(-x['capacity_usage'],x['geometry_id'],x['snr_db']))
 proposal=headers+bodies[:94];csvsave(out/'LE2_next96_proposal.csv',[dict(x,proposed_trials=128,execution='NOT_AUTHORIZED_NOT_RUN') for x in proposal])
 save(out/'summary.json',dict(status='LE1_CONSTRUCTOR_AND_CAPACITY_ONLY_COMPLETE',source_lengths=pin(O/'source_lengths.csv'),source=pin(E/'candidate_plan.py'),independent_group_design=pin(E/'group_layout_design.py'),original_ep_phy_unchanged=True,nominal_Es=2,header_symbols=68,body_symbols=956,frame_symbols=1024,group_overheads=dict(base_length=13,base_CRC=16,enhancement_length=13,enhancement_CRC=16),allocation_counts=len(allocations),proportional_allocations=[dict(j=j,m=m,K=K,N_B=N,available_to=['EEP','U']) for (j,m,K),N in proportional.items()],capacity_grid_rows=rows_written,enhancement_feasible_grid_settings=feasible_settings,layout_constructor_calls=planner.constructor_calls,admitted_group_geometries=len(unique),rejected_group_layouts=sum(not v['layout_status'].startswith('ACTUAL') for v in planner.cache.values()),full_exact_missing_cell_count=len(gaprows),legacy_geometry_only_matches=sum(x['legacy_geometry_match'] for x in gaprows),initial_proposal_cells=len(proposal),initial_trials_max=96*128,optional_topup_cells_max=24,optional_extra_trials_per_cell_max=384,LE2_total_packet_trial_cap=21504,LE2_not_executed=True,encoder_decoder_forward_calls=forward_calls[0],wireless_image_frames=0,train_updates=0,public_profile_design='Offline capacity domain has no paid IDs. Any future <=10 fixed strategies x(68 fallback edges+4 explicit empty)+1null =721IDs fits12bits; allocation/MCS/j/m/K must all be immutable catalogue entries before LE2. No catalogue generated by this planner.',data=[pin(data/'group_layouts.json'),pin(fullpath)]))
 assert forward_calls==[0] and not t.cuda.is_initialized();print('LE1_CAPACITY_PLAN_COMPLETE_ZERO_PACKET_CALLS',flush=True)

if __name__=='__main__':run()

"""Read existing formal events only; no PHY, model, metric, or bootstrap calls.

Run in the canonical checkout with frozen_ldpc_py311:
 python read_only_audit.py --output /new/output/directory
All source records are hash checked against closed delivery lineage. Energy
summaries retain the formal logical weights; a second view deduplicates physical
paths within each method/SNR. A physical event shared by two methods appears in
both method-specific summaries. std is population standard deviation (ddof=0).
"""
from pathlib import Path
import argparse, csv, hashlib, json, math, statistics, collections, time
R=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
B=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm')


def digest(b): return hashlib.sha256(b).hexdigest()
def csvrows(p):
 with Path(p).open(newline='') as f:return list(csv.DictReader(f))
def savecsv(p, rows):
 assert rows
 with p.open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def avg(v):return statistics.fmean(v)
def stat(v):return dict(mean=avg(v),std=statistics.pstdev(v),min=min(v),max=max(v))
def corr(x,y):
 a,b=avg(x),avg(y);dx=[v-a for v in x];dy=[v-b for v in y]
 d=math.sqrt(sum(v*v for v in dx)*sum(v*v for v in dy))
 return sum(a*b for a,b in zip(dx,dy))/d if d else None

def main(out):
 out.mkdir(parents=True,exist_ok=False);beg=time.time();pins={};cache={}
 def read(p,expected=None):
  p=Path(p);key=str(p)
  if key in cache:
   if expected:assert pins[key]['sha256']==expected,key
   return cache[key]
  b=p.read_bytes();s=digest(b)
  if expected:assert s==expected,(key,s,expected)
  pins[key]=dict(path=key,bytes=len(b),sha256=s)
  x=json.loads(b);cache[key]=x;return x
 def checked(d):return read(d['path'],d['sha256'])
 def meta(p):
  b=Path(p).read_bytes();pins[str(p)]=dict(path=str(p),bytes=len(b),sha256=digest(b));return p
 N512=R/'results/nestcom_n512_20261028_v1';L512=B/'nestcom_n512_20261028_v1'
 N1024=R/'results/n1024_consistency_20261019_v1/repair_v1';L1024=B/'n1024_consistency_20261019_v1/repair_v1'
 closure=read(N1024/'delivery/closure_audit.json');prior=checked(closure['metric_rows'])
 n512plan=read(L512/'formal_plan/completion.json');n512logic=checked(n512plan['logical'])
 n512rxplan=read(L512/'formal_rx_plan/completion.json');n512frames=checked(n512rxplan['frames'])
 n512final=csvrows(meta(N512/'delivery/frame_scores.csv'));n1024final=csvrows(meta(N1024/'delivery/frame_scores_9000.csv'))
 protocol=read(N512/'protocol.json');policies=read(N512/'frozen_policies.json');assert policies['protocol_sha256']==pins[str(N512/'protocol.json')]['sha256']
 common=checked(protocol['sources']['common500']);identity={int(r['source_index']):r['source_id']for r in common}
 assert len(identity)==500 and set(identity)==set(range(500))
 ref1024={(r['family'],int(r['snr_db']),int(r['source_index']),int(r['noise_seed'])):r for r in n1024final}
 ref512={(r['method'],int(r['snr_db']),int(r['source_index']),int(r['noise_seed'])):r for r in n512final}
 rows=[]
 def energy(tx,N):
  assert tx['N']==tx['total_symbols']==N and tx['header_symbols']==68 and tx['body_symbols']==N-68
  assert abs(tx['body_energy']+tx['header_energy']-tx['E_frame'])<1e-9
  assert tx['rho']==tx['E_frame']/(2*N) and tx['header_energy']==136
  return dict(E_frame=tx['E_frame'],E_per_symbol=tx['E_frame']/N,rho=tx['rho'],q=tx['q'],k=tx['k'],padding_bits=tx['known_information_padding_bits'],padding_fraction=tx['known_information_padding_bits']/tx['k'],frame_waveform_sha256=tx['transmitted_frame_sha256'],public_frame_counter=tx['public_frame_counter'],session=tx['session'])
 for row in prior:
  method=row['strategy_family'];snr=int(row['snr_db']);idx=int(row['source_index']);seed=int(row['noise_seed']);assert row['source_id']==identity[idx]
  f=ref1024[method,snr,idx,seed];d=row['physical_frame'];raw=checked(d);tx=raw['transmission'];rx=raw['actual_RX'];h=rx['header'];body=rx.get('body',{})
  assert (tx['m'],tx['K'],tx['arithmetic_bits'])==(row['actual_m'],row['actual_K'],row['arithmetic_bits'])
  assert (tx['m'],tx['K'],tx['arithmetic_bits'])==(int(f['sent_m']),int(f['sent_K']),int(f['arithmetic_bits']))
  assert rx['status']==f['RX_status']
  evidence=row['reconstruction']['evidence']
  evidence=checked(evidence)['evidence']if 'path'in evidence else evidence
  assert evidence['source_decode_called']and evidence['render_called']
  assert evidence['source_status']=='ARITHMETIC_SOURCE_DECODED'
  assert not any(evidence.get(k,False)for k in ['receiver_truth_used','comparison_truth_used','TX_CDF_used','transmitted_CDF_used'])
  # The closed delivery maps exact source identity and repeat to this hash-pinned raw event.
  for k in ['source_id','snr_db','noise_seed']:
   if k in raw:assert raw[k]==row[k],(k,raw[k],row[k])
   if 'event'in raw and k in raw['event']:assert raw['event'][k]==row[k]
  rows.append(dict(N=1024,method=method,snr_db=snr,source_index=idx,source_id=identity[idx],noise_seed=seed,m=tx['m'],K=tx['K'],rate=rx['rx_profile']['nominal_rate'],arithmetic_bits=tx['arithmetic_bits'],header_ok=bool(h['header_ok']),body_crc_ok=bool(body.get('crc_accepted')),body_parser_ok=bool(body.get('parser_accepted')),source_decoder_reject=evidence['source_status']!='ARITHMETIC_SOURCE_DECODED',TX_NO_FIT=False,NULL_received=False,receiver_status=rx['status'],source_status=evidence['source_status'],physical_path=d['path'],physical_sha256=d['sha256'],**energy(tx,1024)))
 for row in n512logic:
  method=row['method'];snr=int(row['snr_db']);idx=int(row['source_index']);seed=int(row['noise_seed']);assert row['source_id']==identity[idx]
  f=ref512[method,snr,idx,seed];key=row['physical_key'];assert f['physical_key']==key
  d=n512frames[key];raw=checked(d);assert raw['protocol_sha256']==pins[str(N512/'protocol.json')]['sha256']
  job=raw['job'];choice=raw['choice'];tx=raw['transmission'];rx=raw['actual_RX'];h=rx['header'];body=rx.get('body',{})
  for k in ['source_id','snr_db','noise_seed']:assert job[k]==row[k]
  assert (choice['m'],choice['K'])==(int(f['actual_m']),int(f['actual_K']))
  assert tx['arithmetic_bits']==int(f['arithmetic_bits'])+int(f['class_bits'])
  assert bool(h['header_ok'])==(f['header_accepted']=='True')
  assert bool(body.get('crc_accepted'))==(f['body_crc_accepted']=='True')
  assert bool(body.get('parser_accepted'))==(f['body_parser_accepted']=='True')
  assert (raw['projection'].get('reason')=='EXPLICIT_NULL_PROFILE')==(f['null_received']=='True')
  rows.append(dict(N=512,method=method,snr_db=snr,source_index=idx,source_id=identity[idx],noise_seed=seed,m=choice['m'],K=choice['K'],rate=choice['rate'],arithmetic_bits=int(f['arithmetic_bits']),header_ok=bool(h['header_ok']),body_crc_ok=bool(body.get('crc_accepted')),body_parser_ok=bool(body.get('parser_accepted')),source_decoder_reject=f['receiver_status']=='FIXED_GRAY_SOURCE_REJECT',TX_NO_FIT=choice['tx_no_fit'],NULL_received=f['null_received']=='True',receiver_status=rx['status'],source_status=f['receiver_status'],physical_path=d['path'],physical_sha256=d['sha256'],**energy(tx,512)))
 assert len(rows)==22500
 groups=collections.defaultdict(list)
 for r in rows:groups[r['N'],r['method'],r['snr_db']].append(r)
 summary=[];failures=[];endpoints=[]
 for (N,method,snr),rs in sorted(groups.items()):
  assert len(rs)==1500 and len({r['source_id']for r in rs})==500
  assert len({(r['source_id'],r['noise_seed'])for r in rs})==1500
  bysource=collections.defaultdict(list)
  for r in rs:bysource[r['source_id']].append(r)
  for ss in bysource.values():assert len(ss)==3 and len({(r['m'],r['K'])for r in ss})==1
  for (m,K),cnt in sorted(collections.Counter((ss[0]['m'],ss[0]['K'])for ss in bysource.values()).items()):endpoints.append(dict(N=N,method=method,snr_db=snr,m=m,K=K,sources=cnt,logical_frames=cnt*3,three_repeats_endpoint_identical=True))
  for weight,values in [('logical',rs),('physical_unique',list({r['physical_path']:r for r in rs}.values()))]:
   s=dict(N=N,method=method,snr_db=snr,weighting=weight,rows=len(values),sources=len({r['source_id']for r in values}),q_values=';'.join(map(str,sorted({r['q']for r in values}))))
   for k in ['E_frame','E_per_symbol','rho','padding_fraction']:
    s.update({k+'_'+a:b for a,b in stat([r[k]for r in values]).items()})
   s['rho_padding_Pearson_r']=corr([r['rho']for r in values],[r['padding_fraction']for r in values]);summary.append(s)
  failures.append(dict(N=N,method=method,snr_db=snr,logical_frames=1500,TX_NO_FIT=sum(r['TX_NO_FIT']for r in rs),NULL_received=sum(r['NULL_received']for r in rs),header_reject=sum(not r['header_ok']for r in rs),body_crc_reject=sum(r['header_ok']and not r['body_crc_ok']for r in rs),body_parser_reject=sum(r['body_crc_ok']and not r['body_parser_ok']for r in rs),source_decoder_reject=sum(r['source_decoder_reject']for r in rs),payload_parsed=sum(r['receiver_status']=='PAYLOAD_PARSED'for r in rs)))
 paired=[]
 for N in [512,1024]:
  for snr in [4,10,19]:
   p={ (r['source_id'],r['noise_seed']):r for r in groups[N,'NEST'if N==512 else'PARTIAL',snr]};w={ (r['source_id'],r['noise_seed']):r for r in groups[N,'WHOLE',snr]};assert set(p)==set(w)
   keys=sorted(w);de=[p[k]['E_frame']-w[k]['E_frame']for k in keys];dp=[p[k]['padding_fraction']-w[k]['padding_fraction']for k in keys]
   paired.append(dict(N=N,snr_db=snr,comparison=('NEST'if N==512 else'PARTIAL')+'_minus_WHOLE',paired_logical_frames=len(keys),sources=500,mean_energy_WHOLE=avg([w[k]['E_frame']for k in keys]),mean_energy_partial=avg([p[k]['E_frame']for k in keys]),ratio_of_means_dB=10*math.log10(avg([p[k]['E_frame']for k in keys])/avg([w[k]['E_frame']for k in keys])),delta_E_frame_mean=avg(de),delta_E_frame_std=statistics.pstdev(de),delta_E_frame_min=min(de),delta_E_frame_max=max(de),delta_rho_mean=avg(de)/(2*N),delta_padding_fraction_mean=avg(dp),delta_energy_delta_padding_Pearson_r=corr(de,dp),same_physical_event=sum(p[k]['physical_path']==w[k]['physical_path']for k in keys)))
 # This exact QPSK check includes the protected header and every formal frame.
 qpsk=[r for r in rows if r['N']==512 and r['snr_db']==4];assert len(qpsk)==4500 and all(r['q']==2 and r['E_frame']==1024 and r['rho']==1 for r in qpsk)
 sizes=[1,4,9,16,25,36,64,100,169,256]
 endpoint10=[]
 for method in ['WHOLE','NEST']:
  assert policies['policies']['10'][method]['target']==[7,0]and policies['policies']['10'][method]['q']==4 and policies['policies']['10'][method]['rate']=='2/3'
  rs=groups[512,method,10];source=[r for r in rs if r['noise_seed']==6201];assert len(source)==500
  endpoint10.append(dict(method=method,N=512,snr_db=10,sources=500,target_m=7,target_K=0,q=4,rate='2/3',fallback_sources=sum((r['m'],r['K'])!=(7,0)for r in source),within_scale_sources=sum(r['K']>0 for r in source),mean_retained_tokens=avg([sum(sizes[:r['m']])+r['K']for r in source]),mean_source_utilization=avg([r['arithmetic_bits']/(r['k']-29)for r in source]),three_repeats_endpoint_identical=True))
 for name,data in [('energy_summary',summary),('paired_energy',paired),('failure_summary',failures),('endpoint_distribution',endpoints),('n512_10_endpoint_explanation',endpoint10),('formal_event_mapping',rows)]:savecsv(out/(name+'.csv'),data)
 # Large arrays are not read. Exact event record hashes and all lineage inputs remain verifiable.
 manifest=dict(schema='READ_ONLY_FORMAL_EVENT_AUDIT_V2',base_commit='93205a2e007ac587e28246cd399a531347a4f3f7',logical_frames=22500,unique_physical_paths=len({r['physical_path']for r in rows}),population='current common500, exact source-ID/index/repeat joins',physical_weighting='deduplicate path within N/method/SNR; shared path appears in each method summary',logical_weighting='500 sources times 3 repetitions per method/SNR',standard_deviation='population (ddof=0)',source_decoder_status_N1024='all 9000 actual inline or hash-pinned decoder evidence read; no decoder rerun',QPSK_N512_4dB=dict(logical_frames=4500,E_frame=1024,rho=1,all_equal=True),new_PHY=0,new_model_calls=0,new_metric_calls=0,new_bootstrap_draws=0,elapsed_seconds=time.time()-beg,input_files=list(pins.values()))
 (out/'source_identity.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print(json.dumps(dict(output=str(out),logical=22500,unique_physical=manifest['unique_physical_paths'],paired=paired,endpoint10=endpoint10,failures=failures)))

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();main(args.output)

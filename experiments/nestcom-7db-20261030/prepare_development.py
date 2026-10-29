"""Identity-matched cached TX streams; no inference or access to new image quality."""
from g import *
import base64

def main():
 policy=read(D/'frozen_policies.json');assert policy['status']=='FROZEN_BEFORE_DEVELOPMENT'
 wait=read(L/'dev_nest_source_control/actual_wait.json');assert wait['returncode']==0 and gone(wait['process'])
 nest={s['source_id']:s for s in cr(read(L/'dev_nest_source/completion.json')['sources'])}
 adafile=B/'wcl_adatoken_20261027_v1/ada_development2_source/sources.json';assert sha(adafile)=='086cb6378b087de7c6b4e98800eae53b31cd1f53bde216eef9063717d4ffb5a0'
 ada={s['source_id']:s for s in read(adafile)};cfg=read(D/'timing_config.json')
 sys.path[:0]=[cfg['arpc_public']['upstream'],str(R/'experiments/arpc-external-baseline-20261019')]
 from wire import pack,unpack
 out=L/'dev_sources';out.mkdir();rows=[];audit=[]
 for s in read(D/'development_sources.json'):
  a=ada[s['source_id']];n=nest[s['source_id']]
  assert a['archive']==s['archive']and a['preprocessing_id']==s['preprocessing_id'];checked(a['streams']);checked(n['streams'])
  root=B/'five_method_timing_20261020_v1/ARPC/tx/cases';cached={};lengths=None;records=[]
  for snr in [4,10,19]:
   f=root/('s%d_i%04d_r3'%(snr,s['source_index']));v=read(f/'output.json');c=read(f/'complete.json')
   assert c['source_id']==s['source_id']and c['source_index']==s['source_index']
   now=c['evidence']['lengths_bytes'];assert lengths is None or now==lengths;lengths=now
   cached[v['endpoint']]=base64.b64decode(v['payload_b64']);records.extend([pin(f/'output.json'),pin(f/'complete.json')])
  emax=max(cached);caption,trans,modes=unpack(cached[emax],emax);streams=[]
  # Every possible 7 dB MCS is below endpoint7 for both fixed development sources.
  assert lengths['7']*8>int(956*4*2/3)-29
  for e in range(1,emax+1):
   blob,details=pack(caption,trans[:e],modes[:e],e);assert len(blob)==lengths[str(e)]
   if e in cached:assert blob==cached[e]
   p=out/('arpc_%04d_e%d.bin'%(s['source_index'],e));p.write_bytes(blob);streams.append(dict(endpoint=e,payload_bits=len(blob)*8,wire=pin(p)))
  rows.append(dict(s,nest_streams=n['streams'],ada_streams=a['streams'],predicted_class=a['predicted_class'],arpc_streams=streams))
  audit.append(dict(source_id=s['source_id'],original_tx_records=records,original_lengths=lengths,source_hash=s['archive']['sha256'],arpc_repack='Independently serialize canonical cached per-scale code blocks; no slicing terminated arithmetic stream',new_probability_passes=0,new_caption=0,new_tokenizer=0))
 save(D/'development_sources_ready.json',rows);save(D/'development_cache_reuse.json',dict(ada_source_manifest=pin(adafile),nest_new_source=pin(L/'dev_nest_source/completion.json'),ARPC=audit,not_selected_using_output_quality=True));print('DEVELOPMENT_SOURCES_READY',len(rows))
if __name__=='__main__':main()

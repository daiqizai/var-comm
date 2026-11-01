"""Verify exported real-model traces/arrays without rerunning model inference."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
def sha(a):
 a=np.ascontiguousarray(a);return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--vectors',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();checks=[]
 manifest=json.loads((a.vectors/'reader_vectors_manifest.json').read_text())
 for label in ('whole','partial'):
  m=json.loads((a.vectors/(label+'.json')).read_text());entry=manifest['vectors'][label];f=a.vectors/entry['json_path']
  assert hashlib.sha256(f.read_bytes()).hexdigest()==entry['json_sha256']
  assert entry['original_npz_sha256']==m['vector']['sha256']
  arrays=json.loads(f.read_text());v={}
  for name,item in arrays['arrays'].items():
   values=[float.fromhex(x) for x in item['values']]if item['encoding']=='float.hex'else item['values']
   v[name]=np.array(values,dtype=item['dtype']).reshape(item['shape']);assert sha(v[name])==item['array_sha256']
  assert len(v['bits'])==m['payload_bits'] and sha(v['bits'])==m['payload_sha256']
  assert sha(v['received_tokens'])==m['received_tokens_sha256']
  assert np.all(np.isin(v['bits'],[0,1]))
  assert m['TX_CDF']==m['RX_CDF'] and m['canonical'] and m['exact_tokens'] and m['zero_extension_reads']==30
  assert not m['receiver_TX_truth_used'] and not m['receiver_TX_CDF_used']
  checks.append(dict(vector=label,endpoint=m['endpoint'],payload_bits=m['payload_bits'],CDF_scales=len(m['TX_CDF']),status='PASS'))
 result=dict(status='PASS_EXPORTED_TRACE_INTEGRITY',model_replay=False,checks=checks)
 with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(result))
if __name__=='__main__':main()

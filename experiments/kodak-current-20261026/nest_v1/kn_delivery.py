"""CPU closure after output fixation: actual waits, exact hashes and truth-only audit."""
from kn_common import *
import numpy as np,collections,traceback
def run(expected):
 r=registration(expected);out=O/'delivery';out.mkdir(exist_ok=False)
 try:
  cs=closed('source',expected);cp=closed('phy',expected);crx=closed('receiver',expected)
  sources={s['source_index']:s for s in cr(cs['sources'])};frames=cr(cp['logical']);images=cr(crx['images']);rows=[];audit=[];seen={};accepted=collections.Counter()
  for v in images.values():
   p=checked(v['image_archive'])
   with np.load(p,allow_pickle=False)as z:a=z['image']
   assert a.dtype==np.float32 and a.shape==(3,256,256)and np.isfinite(a).all()and asha(a)==v['image_sha256'];seen[str(p)]=True
   checked(v['evidence'])
  for f in frames:
   im=images[f['source_input_key']];ev=cr(im['evidence'])['evidence'];p=cr(f['physical_frame']);s=sources[f['source_index']]
   with np.load(checked(s['source_archive']),allow_pickle=False)as z:tokens=z['tokens']
   if ev['received_tokens']is not None:
    received=np.asarray(ev['received_tokens'],dtype=np.int64);equal=bool(np.array_equal(received,tokens[:len(received)]))
   else:equal=None
   audit.append(dict(method=f['method'],source_index=f['source_index'],snr_db=f['snr_db'],noise_seed=f['noise_seed'],post_output_true_prefix_equal=equal,output_was_already_fixed=True,truth_corrected_RX=False))
   accepted[f['method'],f['snr_db'],im['rx_status']]+=1
   rows.append(dict(method=f['method'],source_index=f['source_index'],source_id=f['source_id'],snr_db=f['snr_db'],noise_seed=f['noise_seed'],rx_status=im['rx_status'],image_path=im['image_archive']['path'],image_key='image',image_file_sha256=im['image_archive']['sha256'],image_array_sha256=im['image_sha256'],array_hash_convention='sha256(str(dtype)+str(shape)+contiguous_bytes)',existing_psnr_db='',existing_lpips_alex='',family=f['family'],m=f['m'],K=f['K'],terminated_arithmetic_bits=f['terminated_arithmetic_bits'],source_capacity_bits=f['source_capacity_bits'],utilization=f['utilization'],source_input_key=f['source_input_key'],physical_frame_path=f['physical_frame']['path'],physical_frame_sha256=f['physical_frame']['sha256'],source_pixels_chw_uint8_sha256=s['pixels_chw_uint8_sha256']))
  assert len(rows)==432 and len({(x['method'],x['source_index'],x['snr_db'],x['noise_seed'])for x in rows})==432
  csvsave(out/'frame_manifest.csv',rows);csvsave(out/'post_output_truth_audit.csv',audit)
  csvsave(out/'acceptance_summary.csv',[dict(method=k[0],snr_db=k[1],rx_status=k[2],frames=v)for k,v in sorted(accepted.items())])
  ep=checked(cs['endpoint_utilization']);(out/'endpoint_utilization.csv').write_bytes(ep.read_bytes())
  save(out/'completion.json',dict(status='COMPLETE_NEST_WHOLE432_NATIVE_OUTPUTS',registration_sha256=expected,logical_frames=432,physical_events=cp['new_physical'],packet_ledger=cp['new_packets'],source_ledger=cs['ledger'],receiver_ledger=crx['ledger'],unique_native_outputs=len(images),logical_reconstruction_reuse=432-len(images),truth_audit_after_output=True,new_scores=0,frame_manifest=pin(out/'frame_manifest.csv'),endpoint_utilization=pin(out/'endpoint_utilization.csv'),acceptance=pin(out/'acceptance_summary.csv'),truth_audit=pin(out/'post_output_truth_audit.csv')))
 except BaseException:save(out/'failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False));raise
if __name__=='__main__':run(sys.argv[1])

"""Correct image-storage-only gray counting without changing any scientific score."""
from followup_common import *
from collections import defaultdict,Counter

def main():
 closed('E1_statistics');g=defaultdict(list);gray=asha(np.full((3,256,256),.5,np.float32))
 for v in read(L/'E1_join/frames.json'):g[v['method'],v['N'],v['snr_db']].append(v)
 result=[]
 for (method,N,snr),v in sorted(g.items()):
  status=Counter(x['receiver_status']for x in v);result.append(dict(method=method,N=N,snr_db=snr,logical=len(v),native_gray_pixel_count=sum(x['pixel_sha256']==gray for x in v),image_None_storage_count=sum(x['image']is None for x in v),receive_status_counts=json.dumps(dict(status),sort_keys=True)))
 csvout(D/'e1/failure_counts_native_v2.csv',result)
 save(D/'e1/failure_count_annotation.json',dict(original=pin(D/'e1/failure_counts.csv'),corrected=pin(D/'e1/failure_counts_native_v2.csv'),reason='Original fixed_gray column counts absent image pointers, whereas external BPG retains its gray image as a native array. The corrected column compares the authenticated native pixel hash, alongside actual receive status. No frame, image or metric was changed or rerun.',source=pin(L/'E1_join/frames.json'),new_scientific_calls=0))
 print('NATIVE_GRAY_AUDIT',[(x['method'],x['N'],x['snr_db'],x['native_gray_pixel_count'])for x in result if x['native_gray_pixel_count']])
if __name__=='__main__':main()

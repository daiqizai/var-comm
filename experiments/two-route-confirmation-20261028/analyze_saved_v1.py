"""Additional descriptive accounting of saved paired results; no model calls."""
import pathlib,json,csv,collections,hashlib
import numpy as np
R=pathlib.Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM');O=R/'results/two_route_confirmation_20261028_v1';L=pathlib.Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-tmp/liulu/liulu-var-comm/two_route_confirmation_20261028_v1')
def readcsv(p):return list(csv.DictReader(p.open()))
def main():
    a=readcsv(O/'a_source_accounting.csv');util={}
    for arm in ['QUARTER','TOKEN']:
        rr=[r for r in a if r['arm']==arm];assert len(rr)==32
        util[arm]={k:float(np.mean([float(r[k])for r in rr]))for k in ['source_bits','padding','utilization']}
    pair=[r for r in readcsv(O/'delivery/a_paired_source_differences.csv')if r['snr_db']=='2'];assert len(pair)==32
    wins={}
    for m,direction in [('psnr_db',1),('lpips_alex',-1),('dreamsim',-1),('dino_cosine',1),('clip_image_cosine',1)]:
        v=np.array([float(r[m])for r in pair]);wins[m]=dict(improved=int(np.sum(v*direction>1e-12)),worse=int(np.sum(v*direction< -1e-12)),equal=int(np.sum(np.abs(v)<=1e-12)),mean_difference=float(v.mean()),min_difference=float(v.min()),max_difference=float(v.max()))
    indices=json.loads((L/'b_dependence/indices.json').read_text());q=np.array([r['received']for r in indices],np.int64);p=np.array([r['predicted']for r in indices],np.int64)
    assert q.shape==(16,64)and np.array_equal(q,p)
    coarse=json.loads((R/'results/nested_residual_rd_20261028_v1/paused_coarse_identity.json').read_text());cb={r['source_id']:r['coarse_bits']for r in coarse['rows']}
    br=readcsv(O/'delivery/b_source_pairs.csv');inc=[float(r['actual_bits'])-cb[r['source_id']]for r in br]
    result=dict(A_utilization=util,A_source_direction_counts=wins,B_quantized_statistics=dict(distinct_received_vectors=len({tuple(row)for row in q}),distinct_vectors_are_descriptive_not_new_conditions=True,level_counts={str(k):int(v)for k,v in collections.Counter(q.ravel()).items()},source_invariant_scalar_positions=int(sum(len(set(q[:,i]))==1 for i in range(64))),mean_full_minus_coarse_finished_bits=float(np.mean(inc)),min_full_minus_coarse_finished_bits=min(inc),max_full_minus_coarse_finished_bits=max(inc)),no_new_inference=True,no_new_bootstrap=True)
    with(O/'delivery/saved_result_analysis.json').open('x')as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))
if __name__=='__main__':main()

"""Additional original-renderer versus selective-renderer alignment for novel endpoints."""
from fit_common import *
import traceback
def main(path,expected):
    assert sha(path)==expected;r=read(path);[checked(d)for d in r['code']];jobs=cr(r['inputs']);routes=cr(r['model_routes']);out=Path(r['output']);out.mkdir(exist_ok=False);os.sched_setaffinity(0,set(r['cpu_affinity']));ledger=Ledger(out/'calls.sqlite',r['caps']);begin=time.monotonic()
    try:
        sys.path.insert(0,str(R/'experiments/kodak-current-20261026/nest_v1'));from kn_native import load
        def boundary():assert time.monotonic()-begin<r['seconds']and not(L/'STOP').exists()
        n,render=ledger.call('model_load','once',lambda:load(routes,ledger,boundary,out,r['pci']));rows=[]
        for j in jobs:
            st=j['state'];K=len(st['positions']);assert st['positions']==list(range(K));tokens=np.asarray(st['coarse']+st['values'],np.int64)
            a=ledger.call('alignment_render',j['key'],lambda:render(n,tokens,st['m'],K))
            with np.load(checked(j['image']),allow_pickle=False)as z:b=z['image'].copy()
            row=dict(key=j['key'],exact=bool(np.array_equal(a,b)),max_abs_difference=float(np.abs(a-b).max()),original_renderer_pixel_sha256=asha(a),selective_renderer_pixel_sha256=j['pixel_sha256']);rows.append(row);save(out/(j['key']+'.json'),row)
            assert row['exact']
        save(out/'completion.json',dict(status='EXACT_ALIGNMENT_PASS',novel_endpoints=len(rows),rows=rows,ledger=ledger.summary(),seconds=time.monotonic()-begin))
    except BaseException:save(out/'failure.json',dict(traceback=traceback.format_exc(),ledger=ledger.summary(),automatic_retry=False));raise
if __name__=='__main__':main(*sys.argv[1:])

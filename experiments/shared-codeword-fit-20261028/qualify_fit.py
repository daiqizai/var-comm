"""Actual Sionna geometry/rate matching and noiseless roundtrip, no wireless trials."""
from fit_common import *
from fit_protocol import catalogue
from phy_runtime import runtime
import traceback
def main(path,expected):
    assert sha(path)==expected;r=read(path)
    for d in r['code']:checked(d)
    out=Path(r['output']);out.mkdir(exist_ok=False);os.sched_setaffinity(0,set(r['cpu_affinity']));ledger=Ledger(out/'calls.sqlite',r['caps']);begin=time.monotonic()
    try:
        b,phy,header,packet=runtime();rows=[];excluded=[]
        from profiles import UnsupportedConfiguration
        for k in(478,512,544,576,608,636,637):
            try:layout=ledger.call('geometry',str(k),lambda:b.plan(k,1912,2))
            except UnsupportedConfiguration as e:excluded.append(dict(k=k,reason=str(e)));continue
            rng=np.random.default_rng(2026102800+k);u=rng.integers(0,2,(4,k),dtype=np.uint8)
            def roundtrip():
                c=b.encode(u,1912,2);rx=b.decode((c*2-1)*20,k,1912,2).cpu().numpy().astype(np.uint8)
                assert np.array_equal(rx,u);return dict(layout=layout,rows=4,correct=4)
            rows.append(ledger.call('noiseless_roundtrip',str(k),roundtrip))
        supported=[x['layout']['k']for x in rows]
        assert all(x in supported for x in(636,637))
        save(out/'code_geometry.json',dict(qualified=rows,excluded=excluded))
        save(out/'profile_catalog.json',catalogue([x for x in supported if x in(512,544,576,608,637)]))
        save(out/'completion.json',dict(status='PASS',geometry=pin(out/'code_geometry.json'),catalogue=pin(out/'profile_catalog.json'),ledger=ledger.summary(),seconds=time.monotonic()-begin,wireless_frames=0))
    except BaseException:save(out/'failure.json',dict(traceback=traceback.format_exc(),ledger=ledger.summary(),automatic_retry=False));raise
if __name__=='__main__':main(*sys.argv[1:])

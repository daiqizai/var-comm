"""Actual independent source receiver with exact previously received content reuse."""
from fit_common import *
from fit_codec import FitCodec,WHOLE_FAMILY,OFFSETS,InvalidSourceStream
from protocol import state_identity
import tempfile,traceback
def main(path,expected):
    assert sha(path)==expected;r=read(path)
    for d in r['code']:checked(d)
    jobs=cr(r['inputs']);bundle=cr(r['old_receiver_cache']);routes=cr(r['model_routes']);out=Path(r['output']);out.mkdir(exist_ok=False);(out/'tmp').mkdir();tempfile.tempdir=str(out/'tmp')
    os.sched_setaffinity(0,set(r['cpu_affinity']));ledger=Ledger(out/'calls.sqlite',r['caps']);begin=time.monotonic();cache=bundle['cache'];rows=[];novel={};reused=[]
    def boundary():assert time.monotonic()-begin<r['seconds']and not(L/'STOP').exists()
    try:
        sys.path[:0]=[str(R/'experiments/t1b-m10-calibration-20261016'),str(R/'experiments/kodak-current-20261026/nest_v1')]
        from runtime_boundary import firewall
        allowed=[d['path']for d in routes['files']+routes['author_files']]+[routes['dist']['path']]
        firewall(out,allowed,[B,RT/'assets',RT/'qualification',R/'results',R/'outputs',Path('/mnt/pfs/pfs-yc2F4O/modelTeam/liberai-checkpoint/liulu')])
        from kn_native import load
        n,_=ledger.call('model_load','once',lambda:load(routes,ledger,boundary,out,r['pci']));codec=FitCodec(R,native=n)
        def take(projection,fn):
            key=digest(projection)
            if key in cache:assert cache[key]['projection']==projection;reused.append(key);return cache[key]['value']
            value=ledger.call('independent_source_decode',key,fn);record=dict(projection=projection,value=value,evidence=dict(study='shared_fit',key=key));cache[key]=record;novel[key]=record
            save(out/(key+'.json'),record);return value
        for j in jobs:
            boundary();p=j['projection']
            if p['type']=='SINGLE':
                wire=dict(status='SOURCE_INPUT',codec='NEST',m=p['m'],K=p['K'],family='EC_VAR_PARTIAL'if p['K']else'EC_VAR_WHOLE',bits=p['bits'])
                def single():
                    s=codec.receive_single(p['bits'],p['m'],p['K']);return dict(m=p['m'],coarse=s['C'],positions=list(range(p['K'])),values=s['A'])
                state=take(wire,single);states=dict(CAB=state)
            else:
                K=p['K'];wireC=dict(type='C',m=4,bits=p['C'])
                def coarse():
                    s=codec.receive_compact(0,p['C'],None,None);assert s['status_C']=='ACCEPTED';return s['C']
                C=take(wireC,coarse);values={};table=None
                for role,lo,hi in [('A',0,K//2),('B',K//2,K)]:
                    if lo==hi:values[role]=[];continue
                    projection=dict(type='BOUNDARY',m=4,C_bits=p['C'],positions=list(range(lo,hi)),bits=p[role])
                    def boundary_stream():
                        nonlocal table
                        if table is None:
                            with codec._inference(WHOLE_FAMILY),codec._context(WHOLE_FAMILY)as provider:
                                for scale in range(4):codec._table(provider,scale,[]);provider.advance(np.asarray(C[OFFSETS[scale]:OFFSETS[scale+1]],np.int64))
                                table=codec._table(provider,4,[])
                        return codec.decode_slice(p[role],table[lo:hi]).tolist()
                    values[role]=take(projection,boundary_stream)
                states={}
                for label in ['C','CA','CB','CAB']:
                    pos=[];vals=[]
                    for role,lo,hi in [('A',0,K//2),('B',K//2,K)]:
                        if role in label:pos+=list(range(lo,hi));vals+=values[role]
                    states[label]=dict(m=4,coarse=C,positions=pos,values=vals)
            rows.append(dict(key=j['key'],states=states,TX_truth_used=False))
            if len(rows)%16==0:print('FIT_RX_SAVED',len(rows),len(jobs),flush=True)
        save(out/'outputs.json',rows);save(out/'novel_decode_cache.json',novel)
        save(out/'completion.json',dict(status='INDEPENDENT_SOURCE_PASS',outputs=pin(out/'outputs.json'),novel_cache=pin(out/'novel_decode_cache.json'),new_decode_calls=len(novel),reuse_lookups=len(reused),unique_reuse_keys=len(set(reused)),ledger=ledger.summary(),seconds=time.monotonic()-begin))
    except BaseException:save(out/'failure.json',dict(traceback=traceback.format_exc(),ledger=ledger.summary(),automatic_retry=False));raise
if __name__=='__main__':main(*sys.argv[1:])

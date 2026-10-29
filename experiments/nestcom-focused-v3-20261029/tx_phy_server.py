"""Persistent CPU TX only; no noise generation, receiver, or packet decoding."""
import sys,os,json,time,base64,traceback
from pathlib import Path
R=Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
sys.path.insert(0,str(R/'experiments/adatoken-response-20261026'))
from common import checked,cr,save,ident,Ledger,asha
import numpy as np

def main(regpath,out):
    r=json.loads(Path(regpath).read_bytes());out=Path(out);out.mkdir(exist_ok=False)
    os.sched_setaffinity(0,set(r['phy_affinity']))
    save(out/'started.json',dict(process=ident(os.getpid()),parent_process=ident(os.getppid())))
    ledger=Ledger(out/'calls.sqlite',{'tx_waveform':610});begin=time.monotonic()
    try:
        from fixed_frame import runtime
        # Suppress constructor progress on the protocol pipe.
        with (out/'initialization.log').open('w') as log:
            old=sys.stdout;sys.stdout=log
            try:
                phy,rt=runtime(cr(r['original_link_registration']))
                rt.header.s.ROOT=Path(r['native_build']);rt.header.s.load_native()
                rt.backend.codecs(2549,3824,4)
            finally:sys.stdout=old
        assert not rt.torch.cuda.is_initialized()
        print(json.dumps(dict(ready=True,process=ident(os.getpid()),torch=rt.torch.__version__)),flush=True)
        for line in sys.stdin:
            request=json.loads(line)
            if request.get('quit'):break
            assert time.monotonic()-begin<r['max_seconds']
            def call():
                # Durable reservation is complete before the parent starts TX.
                print(json.dumps(dict(prepared=request['key'])),flush=True)
                payload=json.loads(sys.stdin.readline())
                bits=np.frombuffer(base64.b64decode(payload['bits']),np.uint8).copy()
                start=time.perf_counter();wave,meta=rt.transmit(request['profile_id'],bits,request['counter'])
                ms=(time.perf_counter()-start)*1000
                assert wave.shape==(1024,2) and np.isfinite(wave).all()
                result=dict(wave=base64.b64encode(wave.tobytes()).decode(),dtype=str(wave.dtype),packing_ldpc_qam_ms=ms,sha256=asha(wave),profile_id=request['profile_id'])
                # Audit completion is after the reply, outside measured TX.
                print(json.dumps(result),flush=True)
            ledger.call('tx_waveform',request['key'],call)
        save(out/'completion.json',dict(status='COMPLETE',ledger=ledger.summary(),seconds=time.monotonic()-begin,new_rx=0,new_noise=0))
    except BaseException:
        save(out/'failure.json',dict(traceback=traceback.format_exc(),ledger=ledger.summary(),automatic_retry=False));raise

if __name__=='__main__':main(*sys.argv[1:])

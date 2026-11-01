"""Bounded CPU-only transport/PHY qualification; no model or quality calls."""
from followup_common import *
from timing_arpc_phy_ipc_v1 import RuntimeProxy, MemoryCalls
import traceback


def main(request, expected):
    assert sha(request) == expected
    r = read(request); [checked(v) for v in r['code']]
    out=Path(r['output']);out.mkdir(exist_ok=False)
    os.sched_setaffinity(0,set(r['cpu_affinity']))
    rt=None; rows=[]
    try:
        rt=RuntimeProxy(r['sionna_package'],out); calls=MemoryCalls()
        # Fixed synthetic payload and paid NULL: transport engineering only.
        for endpoint,payload in [(1, bytes(range(32))), (0,b'')]:
            p=rt.profile(endpoint,4,'2/3');counter=75100000+endpoint
            wave,wire=rt.transmit(payload,p['profile_id'],counter)
            assert wave.shape==(1024,2) and np.isfinite(wave).all()
            actual=rt.receive(wave,30.,counter,calls,'qualification_'+str(endpoint))
            assert actual['body']['payload']==payload and actual['body']['parser_accepted']
            assert actual['status']==('PAYLOAD_PARSED' if endpoint else 'NULL_PROFILE')
            rows.append(dict(endpoint=endpoint,status=actual['status'],wire=wire,
                decoded_payload_sha256=hashlib.sha256(actual['body']['payload']).hexdigest()))
        rt.close(); w=read(out/'phy_service_actual_wait.json')
        assert w['actual_wait'] and w['returncode']==0 and w['final_identity'] is None
        assert w['report']['result']['TX']==w['report']['result']['RX']==2
        save(out/'completion.json',dict(status='COMPLETE',checks=rows,
            system_python=sys.version,PHY_identity=rt.identity,
            model_calls=0,quality_calls=0,engineering_TX=2,engineering_RX=2,
            packet_decodes=calls.counts,service_actual_wait=pin(out/'phy_service_actual_wait.json')))
    except BaseException:
        save(out/'failure.json',dict(traceback=traceback.format_exc(),completed_checks=rows,
             model_calls=0,quality_calls=0,automatic_retry=False));raise
    finally:
        if rt is not None:rt.close()


if __name__=='__main__':main(*sys.argv[1:])

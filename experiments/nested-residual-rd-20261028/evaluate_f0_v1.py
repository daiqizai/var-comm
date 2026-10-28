"""Frozen fine-token control on the same development sources and actual bit budgets."""
from residual_runtime import *
from train_residual_v1 import Data
from fit_common import digest,csvsave
from fit_protocol import endpoints
from selective_receiver import render
import torch,traceback

def main(path,expected):
    assert sha(path)==expected;r=read(path);[checked(x)for x in r['code']]
    out=pathlib.Path(r['output']);out.mkdir(exist_ok=False);os.sched_setaffinity(0,set(r['cpu_affinity']));ledger=Ledger(out/'calls.sqlite',r['caps']);begin=time.monotonic()
    try:
        n,dc,statehash=ledger.call('model_load','once',lambda:load_frozen(r,out));lp=ledger.call('lpips_load','once',lambda:load_lpips(r));codec=FitCodec(R,native=n)
        data=Data(r['cache_index']);dev=[s for s in data.rows if s['split']=='dev'];assert len(dev)==200
        fine=read(checked(r['fine_reuse']));native=read(checked(r['native_reuse']));scored={};rows=[];length_rows=[];reused_sources=0;reused_images=set();new_images=set()
        for s in dev:
            assert time.monotonic()-begin<r['seconds']and not(L/'STOP').exists();sid=s['source_id'];folder=out/str(s['source_index']);folder.mkdir()
            _,_,target,tokens=data.sample(s)
            if sid in fine:
                with np.load(checked(fine[sid]['streams']),allow_pickle=False)as z:streams={k:z[k].copy()for k in z.files if k.startswith('m')}
                reused_sources+=1
            else:
                def source():
                    streams,tables,trace=codec.encode_fit(tokens.numpy())
                    with(folder/'streams.npz').open('xb')as f:np.savez(f,**streams)
                    with(folder/'tx_cdfs.npz').open('xb')as f:np.savez(f,**tables)
                    save(folder/'source_commit.json',dict(streams=pin(folder/'streams.npz'),tables=pin(folder/'tx_cdfs.npz'),source_id=sid,trace=trace))
                    return streams
                streams=ledger.call('fine_source_extension',sid,source)
            chosen={b:max(((m,K)for m,K in endpoints()if len(streams[f'm{m}_K{K}'])<=b),key=lambda e:OFFSETS[e[0]]+e[1],default=None)for b in(449,547,608)}
            for m,K in endpoints():length_rows.append(dict(source_id=sid,source_index=s['source_index'],m=m,K=K,actual_bits=len(streams[f'm{m}_K{K}'])))
            for budget,endpoint in chosen.items():
                if endpoint is None:projection=dict(status='PAID_NULL');wirekey='GRAY';bits=0;m=4;K=0
                else:
                    m,K=endpoint;wire=streams[f'm{m}_K{K}'];bits=len(wire);projection=dict(status='SOURCE_INPUT',codec='NEST',m=m,K=K,family='EC_VAR_PARTIAL'if K else'EC_VAR_WHOLE',bits=wire.tolist());wirekey=digest(projection)
                if wirekey not in native:
                    def receive_render():
                        if endpoint is None:a=np.full((3,256,256),.5,np.float32);evidence=dict(status='PAID_NULL')
                        else:
                            rx=codec.receive_single(np.asarray(projection['bits'],np.uint8),m,K)
                            st=dict(m=m,coarse=rx['C'],positions=list(range(K)),values=rx['A']);a,trace=render(n,st);evidence=dict(received_state=st,trace=trace,independent_RX=True,TX_CDF_used=False)
                        ip=out/(wirekey+'.npz')
                        with ip.open('xb')as f:np.savez(f,image=a)
                        record=dict(image=pin(ip),pixel_sha256=asha(a),projection=projection,evidence=evidence);save(ip.with_suffix('.json'),record);return record
                    native[wirekey]=ledger.call('receive_render',wirekey,receive_render);new_images.add(wirekey)
                elif wirekey not in new_images:reused_images.add(wirekey)
                native_image=native[wirekey];pairkey=digest(dict(source_id=sid,pixel=native_image['pixel_sha256']))
                if pairkey not in scored:
                    def metric():
                        with np.load(checked(native_image['image']),allow_pickle=False)as z:a=z['image'].copy()
                        assert asha(a)==native_image['pixel_sha256'];x=torch.from_numpy(a)[None].cuda();ref=target[None].cuda()
                        with torch.no_grad():mse=float((x.double()-ref.double()).square().mean());pl=float(lp(x*2-1,ref*2-1).mean())
                        result=dict(mse=mse,lpips_alex=pl,psnr_db=float(-10*np.log10(mse)),D=mse+.1*pl);save(folder/(pairkey+'_score.json'),result);return result
                    scored[pairkey]=ledger.call('pair_metric',pairkey,metric)
                rows.append(dict(method='F0_FINE_TOKEN',source_id=sid,source_index=s['source_index'],budget=budget,m=m,K=K,actual_bits=bits,fields=29,padding=budget-bits,N=1024,energy=2048,k=budget+29,n=1912,status='PAID_NULL'if endpoint is None else'LOSSLESS_RX',image=native_image['image']['path'],pixel_sha256=native_image['pixel_sha256'],**scored[pairkey]))
            save(folder/'rows.json',rows[-3:]);print('F0_DEV',s['source_index'],len(rows)//3,flush=True)
        csvsave(out/'source_rd.csv',rows);csvsave(out/'source_lengths.csv',length_rows);save(out/'native_index.json',native)
        save(out/'completion.json',dict(status='F0_DEVELOPMENT_COMPLETE',sources=200,rows=pin(out/'source_rd.csv'),lengths=pin(out/'source_lengths.csv'),source_cache_reused=reused_sources,unique_native_reused=len(reused_images),new_native=len(new_images),ledger=ledger.summary(),seconds=time.monotonic()-begin,wireless_frames=0,scope='same fixed B dev200; exact terminated fine m4..m6 source endpoints; no strategy calibration'))
    except BaseException:save(out/'failure.json',dict(traceback=traceback.format_exc(),ledger=ledger.summary(),automatic_retry=False));raise
if __name__=='__main__':main(*sys.argv[1:])

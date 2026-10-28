"""One fixed checkpoint, 16 fixed sources, two j256 inputs; no training."""
import sys,pathlib,os,time,traceback,json
R=pathlib.Path('/mnt/pfs/pfs-yc2F4O/modelTeam/code/liulu/code/VAR_COMM')
sys.path.insert(0,str(R/'experiments/nested-residual-rd-20261028'))
from residual_runtime import *
from residual_model import ResidualCodec
from train_residual_v1 import Data
from mixed_arithmetic_v1 import classes
from nested_entropy import encode_candidates,decode_residual
from fit_common import csvsave
import torch

def main(path,expected):
    assert sha(path)==expected;r=read(path);[checked(d)for d in r['code']]
    out=pathlib.Path(r['output']);out.mkdir(exist_ok=False)
    os.sched_setaffinity(0,set(r['cpu_affinity']));ledger=Ledger(out/'calls.sqlite',r['caps']);begin=time.monotonic()
    try:
        n,dc,statehash=ledger.call('model_load','once',lambda:load_frozen(r,out))
        lp=ledger.call('lpips_load','once',lambda:load_lpips(r))
        cp=torch.load(checked(r['checkpoint']),map_location='cpu',weights_only=False)
        assert cp['step']==3000 and cp['order']=='progressive'
        model=ResidualCodec('progressive').cuda().eval().requires_grad_(False);model.load_state_dict(cp['model'])
        assert statehash(model)==cp['model_state_sha256']
        plan=read(checked(r['plan']));data=Data(r['cache_index']);sources=[data.by[s['source_id']]for s in plan['sources']]
        assert len(sources)==16;Encoder,Decoder=classes(R/'src/var_comm/whole_entropy.py')
        saved=read(checked(r['old_summary']))['rows'];old={(v['source_id'],v['budget']):v for v in saved}
        # Only earlier independent receiver contexts populate the RX CDF memo.
        memo={}
        for s in sources:
            tokens=torch.load(checked(s['rx_coarse']),map_location='cpu',weights_only=True)['tokens']
            with np.load(checked(s['rx_cdfs']),allow_pickle=False)as z:
                for scale in range(4):
                    key=(scale,tuple(tokens[:OFFSETS[scale]]));tab=z['cdf'+str(scale)]
                    if key in memo:assert np.array_equal(memo[key],tab)
                    else:memo[key]=tab
        def receive(bits):
            decoder=Decoder(bits);canonical=Encoder();tokens=[]
            for scale in range(4):
                key=(scale,tuple(tokens));assert key in memo,'Unknown received context'
                tab=memo[key];v=decoder.decode(tab);canonical.encode(v,tab);tokens+=v.tolist()
            fc=coarse_latent(n,tokens);pr=model.probabilities(fc)
            q,depth=decode_residual(decoder,canonical,256,pr[0].cpu().numpy(),'progressive')
            assert np.array_equal(canonical.finish(),bits)and decoder.position-len(bits)==30
            assert np.all(depth==4)
            return fc,torch.from_numpy(q)[None].cuda(),torch.from_numpy(depth)[None].cuda(),pr
        rows=[];indices=[];native=[]
        with torch.no_grad():
            for s in sources:
                assert time.monotonic()-begin<r['seconds'] and not(pathlib.Path(r['stop_file'])).exists()
                # The old training STOP must remain untouched throughout.
                assert sha(r['training_stop']['path'])==r['training_stop']['sha256']
                sid=s['source_id'];folder=out/str(s['source_index']);folder.mkdir()
                full,fc,target,tokens=data.sample(s);full=full[None].cuda();fc=fc[None].cuda();target=target[None].cuda()
                def encode():
                    u=model.encode(full,fc);pr=model.probabilities(fc)
                    values=torch.floor((u+1)*8).clamp(0,15).long()[0].cpu().numpy()
                    with np.load(checked(s['coarse_stream']),allow_pickle=False)as z:tables=[z['cdf'+str(k)]for k in range(4)]
                    streams,_=encode_candidates(Encoder,[tokens[OFFSETS[k]:OFFSETS[k+1]].numpy()for k in range(4)],tables,values,pr[0].cpu().numpy(),'progressive')
                    bits=streams[256]
                    assert all(old[sid,b]['j']==256 and old[sid,b]['actual_bits']==len(bits)for b in(449,547,608))
                    with(folder/'transmitted_stream.npz').open('xb')as f:np.savez(f,bits=bits)
                    return bits,values
                bits,txvalues=ledger.call('source_materialization',sid,encode)
                rxfc,q,depth,prob=ledger.call('independent_source_decode',sid,lambda:receive(bits))
                assert torch.equal(rxfc,fc),'Received coarse latent differs from bound evaluation'
                # TX values are used only for this post-decoding audit, never as RX input.
                received=torch.round((q+1)*8-.5).long()[0].cpu().numpy();assert np.array_equal(received,txvalues)
                predicted=prob.argmax(-1);pred_values=predicted[0].cpu().numpy()
                pred_q=-1+(2*predicted.to(q.dtype)+1)/16
                q_equal=bool(torch.equal(q,pred_q));same=int(np.sum(received==pred_values));bit_same=sum(int((int(a)>>b&1)==(int(c)>>b&1))for a,c in zip(received,pred_values)for b in range(4))
                images={};scores={};latents={}
                def render(values,label):
                    latent=model.decode(values,depth,rxfc);image=dc(latent);a=image[0].cpu().numpy()
                    assert a.dtype==np.float32 and np.isfinite(a).all()and a.min()>=0 and a.max()<=1
                    p=folder/(label+'.npz')
                    with p.open('xb')as f:np.savez(f,image=a)
                    mse=float((image.double()-target.double()).square().mean())
                    return latent,image,mse,dict(source_id=sid,source_index=s['source_index'],condition=label,image=pin(p),pixel_sha256=asha(a))
                latent,image,mse,entry=ledger.call('image_render',sid+'|received',lambda:render(q,'received'))
                assert mse==old[sid,449]['mse'],'Bound evaluation MSE does not match; stop without rescoring/retrying'
                native.append(entry);images['received']=image;latents['received']=latent
                scores['received']=dict(mse=mse,psnr_db=old[sid,449]['psnr_db'],lpips_alex=old[sid,449]['lpips_alex'],D=old[sid,449]['D'],lpips_reused=True,pixel_sha256=entry['pixel_sha256'])
                if q_equal:
                    scores['predicted']=dict(scores['received']);images['predicted']=image;latents['predicted']=latent
                    native.append(dict(entry,condition='predicted',reuse='EXACT_QUANTIZED_VECTOR_IDENTITY'))
                else:
                    latent,image,mse,entry=ledger.call('image_render',sid+'|predicted',lambda:render(pred_q,'predicted'))
                    native.append(entry);images['predicted']=image;latents['predicted']=latent
                    pl=ledger.call('predicted_lpips',sid,lambda:float(lp(image*2-1,target*2-1).mean()))
                    scores['predicted']=dict(mse=mse,psnr_db=float(-10*np.log10(mse)),lpips_alex=pl,D=mse+.1*pl,lpips_reused=False,pixel_sha256=entry['pixel_sha256'])
                pair=dict(source_id=sid,source_index=s['source_index'],step=3000,j=256,actual_bits=len(bits),scalar_matches=same,scalar_total=64,scalar_match_fraction=same/64,bit_matches=bit_same,bit_total=256,bit_match_fraction=bit_same/256,exact_quantized_vector=q_equal,latent_max_abs_difference=float((latents['received']-latents['predicted']).abs().max()),image_max_abs_difference=float((images['received']-images['predicted']).abs().max()),image_mse_between=float((images['received'].double()-images['predicted'].double()).square().mean()),received_fc_sha256=asha(rxfc[0].cpu().numpy()))
                pair.update({label+'_'+k:v for label in scores for k,v in scores[label].items()})
                pair.update({'received_minus_predicted_'+k:scores['received'][k]-scores['predicted'][k]for k in ['psnr_db','lpips_alex','mse','D']})
                rows.append(pair);indices.append(dict(source_id=sid,source_index=s['source_index'],received=received.tolist(),predicted=pred_values.tolist(),prediction='per-scalar argmax over16 levels',stream=pin(folder/'transmitted_stream.npz')))
                save(folder/'commit.json',dict(row=pair,indices=indices[-1]));print('DEPENDENCE_SOURCE',s['source_index'],same,flush=True)
        csvsave(out/'source_pairs.csv',rows);save(out/'indices.json',indices);save(out/'native_index.json',native)
        assert statehash(model)==cp['model_state_sha256'] and statehash(dc)==n.loaded['identity']['models']['decoder']
        save(out/'completion.json',dict(status='COMPLETE_FIXED16_DEPENDENCE',sources=16,checkpoint=r['checkpoint'],j=256,source_pairs=pin(out/'source_pairs.csv'),indices=pin(out/'indices.json'),native=pin(out/'native_index.json'),ledger=ledger.summary(),seconds=time.monotonic()-begin,wireless_frames=0,training_updates=0,old_stop_preserved=True))
    except BaseException:save(out/'failure.json',dict(traceback=traceback.format_exc(),ledger=ledger.summary(),automatic_retry=False));raise
if __name__=='__main__':main(*sys.argv[1:])

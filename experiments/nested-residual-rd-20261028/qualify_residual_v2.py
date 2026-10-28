"""Four training sources, full mixed-CDF qualification and frozen-Dc gradients."""
from residual_runtime import *
from residual_model import ResidualCodec
from nested_entropy import encode_candidates,decode_residual
from mixed_arithmetic_v1 import classes
import traceback,torch,csv
def main(path,expected):
    assert sha(path)==expected;r=read(path)
    for d in r['code']:checked(d)
    out=pathlib.Path(r['output']);out.mkdir(exist_ok=False);os.sched_setaffinity(0,set(r['cpu_affinity']));ledger=Ledger(out/'calls.sqlite',r['caps']);begin=time.monotonic()
    try:
        n,dc,statehash=ledger.call('model_load','once',lambda:load_frozen(r,out));codec=FitCodec(R,native=n);MixedEncoder,MixedDecoder=classes(R/'src/var_comm/whole_entropy.py')
        lp=ledger.call('lpips_load','once',lambda:load_lpips(r));models={}
        for order in('scalar','progressive'):
            torch.manual_seed(2026102801);models[order]=ResidualCodec(order).cuda().eval()
        assert statehash(models['scalar'])==statehash(models['progressive'])
        params=sum(p.numel()for p in models['scalar'].parameters());assert params<5_000_000
        save(out/'model_config.json',dict(parameters=params,latent=[32,16,16],bottleneck=[4,4,4],orders=list(models),initial_state_sha256=statehash(models['scalar'])))
        samples=read(checked(r['inputs']));rows=[];gradients=[];cache=[]
        for s in samples:
            assert time.monotonic()-begin<r['seconds'] and not(L/'STOP').exists()
            rgb=torch.load(checked(s['rgb_shard']),map_location='cpu',weights_only=True);old=torch.load(checked(s['old_latent_shard']),map_location='cpu',weights_only=True);i=s['within_shard'];assert rgb['image_ids'][i]==old['image_ids'][i]==s['source_id']
            target=rgb['targets_u8'][i].float().div(255).cuda()[None];tokens=old['full_tokens'][i].numpy().astype(np.int64)
            dest=out/('train_%05d'%s['source_index']);dest.mkdir()
            reuse=r.get('reuse_sources',{}).get(s['source_id'])
            if reuse:
                z=torch.load(checked(reuse['cache']),map_location='cpu',weights_only=True);assert z['source_id']==s['source_id'] and torch.equal(z['tokens'],torch.as_tensor(tokens))
                assert torch.equal(z['target'],target.cpu());full=z['full'].cuda();coarse=z['coarse'].cuda()
                with np.load(checked(reuse['cdf']),allow_pickle=False)as a:tables={k:a[k]for k in a.files}
                cache_desc=reuse['cache'];cdf_desc=reuse['cdf']
            else:
                full,coarse=ledger.call('direct_latent',s['source_id'],lambda:direct_latents(n,tokens))
                streams,tables,trace=ledger.call('source_probability',s['source_id'],lambda:codec.encode_fit(tokens))
                with(dest/'coarse_tx.npz').open('xb')as f:np.savez(f,**tables)
                torch.save(dict(full=full.cpu(),coarse=coarse.cpu(),target=target.cpu(),tokens=torch.as_tensor(tokens),source_id=s['source_id']),dest/'cache.pt')
                cache_desc=pin(dest/'cache.pt');cdf_desc=pin(dest/'coarse_tx.npz')
            delta=float((full.cpu()-old['Fq'][i][None]).abs().max());del rgb,old
            cache.append(dict(source=s,cache=cache_desc,cdf=cdf_desc,reused=bool(reuse),old_Fq_max_abs_difference=delta))
            save(dest/'input_commit.json',cache[-1])
            # Receiver CDF memo entries are built independently from decoded C.
            # Keys include the exact already-decoded prefix, never a source ID.
            receiver_tables={}
            for order,model in models.items():
                with torch.no_grad():
                    u=model.encode(full,coarse);probabilities=model.probabilities(coarse);prob=probabilities[0].cpu().numpy();v=torch.floor((u+1)*8).clamp(0,15).long()[0].cpu().numpy();rates=model.estimated_bits(u,probabilities,256,all_prefixes=True)[0].cpu().numpy()
                txparts=[tokens[OFFSETS[s]:OFFSETS[s+1]]for s in range(4)];txcdf=[tables[f'cdf{s}']for s in range(4)]
                streams_j,emitted=ledger.call('mixed_encode',s['source_id']+'|'+order,lambda:encode_candidates(MixedEncoder,txparts,txcdf,v,prob,order))
                selected={budget:max((j for j,b in streams_j.items()if len(b)<=budget),default=None)for budget in(449,547,608)}
                anchors=set(range(0,257,16))|{j for j in selected.values()if j is not None}
                def receive(bits,j):
                    d=MixedDecoder(bits);canonical=MixedEncoder();recovered=[]
                    for scale in range(4):
                        key=(scale,tuple(recovered))
                        if key not in receiver_tables:
                            with codec._inference('EC_VAR_WHOLE'),codec._context('EC_VAR_WHOLE')as p:
                                for prior in range(scale):
                                    codec._table(p,prior,[]);p.advance(np.asarray(recovered[OFFSETS[prior]:OFFSETS[prior+1]],np.int64))
                                receiver_tables[key]=codec._table(p,scale,[])
                        cdf=receiver_tables[key];values=d.decode(cdf);canonical.encode(values,cdf);recovered.extend(values.tolist())
                    fc=coarse_latent(n,recovered)
                    with torch.no_grad():rxprob=model.probabilities(fc)[0].cpu().numpy()
                    q,depth=decode_residual(d,canonical,j,rxprob,order)
                    assert np.array_equal(canonical.finish(),bits)and d.position-len(bits)==30
                    return recovered,fc,q,depth
                with(dest/(order+'_streams.npz')).open('xb')as f:np.savez(f,**{str(j):b for j,b in streams_j.items()})
                for j,bits in streams_j.items():
                    rec,fc,q,depth=ledger.call('mixed_roundtrip',s['source_id']+f'|{order}|{j}',lambda:receive(bits,j))
                    assert rec==tokens[:OFFSETS[4]].tolist()and torch.equal(fc,coarse)
                    expected_q,expected_d,_=model.quantize(u,j,gradient=False);assert np.array_equal(q,expected_q[0].cpu().numpy())
                    rate=float(rates[j])
                    item=dict(source_id=s['source_id'],source_index=s['source_index'],order=order,j=j,actual_bits=len(bits),coarse_bits=len(streams_j[0]),estimated_residual_bits=rate,emitted_bits=emitted[j],finish_bits=len(bits)-emitted[j],mse=None,psnr_db=None,lpips_alex=None)
                    if j in anchors:
                        with torch.no_grad():
                            latent=model.decode(torch.from_numpy(q)[None].cuda(),torch.from_numpy(depth)[None].cuda(),fc)
                            image=ledger.call('decoder_anchor',s['source_id']+f'|{order}|{j}',lambda:dc(latent))
                            mse=float((image.double()-target.double()).square().mean());pl=ledger.call('lpips_anchor',s['source_id']+f'|{order}|{j}',lambda:float(lp(image*2-1,target*2-1).mean()))
                            item.update(mse=mse,psnr_db=float(-10*np.log10(mse)),lpips_alex=pl)
                    rows.append(item)
                if s==samples[0]:
                    pred,rate=model(full,coarse,17);image=dc(pred);dist=(image-target).square().mean()+.1*lp(image*2-1,target*2-1).mean()
                    gd=torch.autograd.grad(dist,list(model.analysis.parameters()),retain_graph=True);gr=torch.autograd.grad(rate.mean(),list(model.analysis.parameters()))
                    for g in [gd,gr]:assert all(torch.isfinite(x).all()for x in g)and sum(float(x.abs().sum())for x in g)>0
                    assert all(p.grad is None and not p.requires_grad for p in dc.parameters())
                    gradients.append(dict(order=order,D_analysis_grad_l1=sum(float(x.abs().sum())for x in gd),R_analysis_grad_l1=sum(float(x.abs().sum())for x in gr)))
            print('RESIDUAL_B0_COMMITTED',s['source_id'],len(rows),flush=True)
        assert statehash(dc)==n.loaded['identity']['models']['decoder']
        for name in ['vae','var']:assert statehash(n.loaded[name])==n.loaded['identity']['models'][name]
        with(out/'mixed_stream_qualification.csv').open('x',newline='')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        save(out/'cache.json',cache);save(out/'gradients.json',gradients)
        save(out/'completion.json',dict(status='B0_PASS',sources=len(samples),mixed_roundtrips=len(rows),parameters=params,rows=pin(out/'mixed_stream_qualification.csv'),cache=pin(out/'cache.json'),gradients=pin(out/'gradients.json'),ledger=ledger.summary(),seconds=time.monotonic()-begin,training_updates=0,wireless_frames=0))
    except BaseException:save(out/'failure.json',dict(traceback=traceback.format_exc(),ledger=ledger.summary(),automatic_retry=False));raise
if __name__=='__main__':main(*sys.argv[1:])

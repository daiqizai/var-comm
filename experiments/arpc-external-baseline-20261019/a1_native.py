"""Bounded native ARPC TX/reference or independent RX process, no channel."""
import argparse,ast,hashlib,json,os,pathlib,sys,time,traceback

def digest(data):return hashlib.sha256(data).hexdigest()
def file_sha(p):
    h=hashlib.sha256()
    with pathlib.Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def put(p,d):
    with p.open('x') as f:json.dump(d,f,ensure_ascii=False,indent=2)
def pin(d):
    p=pathlib.Path(d['path']);assert p.is_file() and not p.is_symlink()
    assert p.stat().st_size==d['bytes'] and file_sha(p)==d['sha256'],p
    return p

class Ledger:
    def __init__(self,out,caps):self.out=out/'calls';self.out.mkdir();self.caps=caps;self.counts={}
    def call(self,kind,key,fn):
        import torch
        if torch.cuda.is_initialized() and torch.cuda.mem_get_info()[0]<8*1024**3:
            raise RuntimeError('Resource floor: less than 8 GiB device free before call')
        self.counts[kind]=self.counts.get(kind,0)+1
        assert self.counts[kind]<=self.caps[kind],(kind,self.counts)
        label=kind+'_'+key;p=self.out/(label+'.reserved.json');start=time.time()
        put(p,{'kind':kind,'key':key,'pid':os.getpid(),'started_unix':start})
        try:
            result=fn();put(self.out/(label+'.complete.json'),{'seconds':time.time()-start});return result
        except BaseException:
            put(self.out/(label+'.failed.json'),{'seconds':time.time()-start,'error':traceback.format_exc()});raise

def install_rx_boundary(allowed):
    blocked=[];root=pathlib.Path('/mnt/pfs/pfs-yc2F4O/modelTeam')
    allowed=[pathlib.Path(x).resolve() for x in allowed]
    def audit(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=pathlib.Path(os.fsdecode(args[0])).resolve()
        if p.is_relative_to(root):
            if not any(p==a or p.is_relative_to(a) for a in allowed):
                blocked.append(str(p));raise PermissionError('Independent RX disallows '+str(p))
    sys.addaudithook(audit)
    return blocked

def prefix_decoder(upstream):
    p=pathlib.Path(upstream)/'compression/reciever.py';tree=ast.parse(p.read_text())
    fn=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='decoding'][-1]
    assert not any(isinstance(n,ast.Name) and n.id=='gt_ls_Bl' and isinstance(n.ctx,ast.Load) for n in ast.walk(fn))
    loops=[n for n in ast.walk(fn) if isinstance(n,ast.For) and ast.unparse(n.iter)=='enumerate(vae_scale_schedule)']
    assert len(loops)==1
    loops[0].iter=ast.parse('enumerate(vae_scale_schedule[:len(trans_list)])',mode='eval').body
    fn.name='decode_prefix';code=ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[]))
    import compression.reciever as rx
    ns=dict(vars(rx));exec(compile(code,str(p)+'::bounded_prefix','exec'),ns)
    return ns['decode_prefix'],digest(ast.dump(code,include_attributes=False).encode())

def main():
    ap=argparse.ArgumentParser();ap.add_argument('role',choices=['tx','rx']);ap.add_argument('--public',required=True);ap.add_argument('--sources');a=ap.parse_args()
    cfg=json.loads(pathlib.Path(a.public).read_text());role=a.role
    if role=='rx':assert a.sources is None
    out=pathlib.Path(cfg['run'])/role;assert not out.exists();out.mkdir(parents=True)
    cache=out/'runtime_cache';cache.mkdir();(cache/'tmp').mkdir()
    os.environ.update(TMPDIR=str(cache/'tmp'),TORCHINDUCTOR_CACHE_DIR=str(cache/'inductor'),TRITON_CACHE_DIR=str(cache/'triton'),
        HF_HOME=str(cache/'hf'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    import tempfile;tempfile.tempdir=str(cache/'tmp')
    sys.path.insert(0,cfg['upstream'])
    blocked=[]
    if role=='rx':
        blocked=install_rx_boundary([cfg['upstream'],cfg['code'],cfg['environment'],cfg['weights_root'],out,pathlib.Path(cfg['run'])/'wire_inputs'])
    import numpy as np,torch
    from PIL import Image
    import compression.sender as sender
    import compression.util as util
    import compression.reciever as receiver
    from tools.run_infinity import load_tokenizer,load_visual_tokenizer,load_transformer
    from utils.arithmeticcoding import probabilities_to_cumul,compress_to_bit_list
    from wire import pack,unpack,SIDES
    torch.set_num_threads(4);torch.set_num_interop_threads(2)
    torch.manual_seed(0);np.random.seed(0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    torch.cuda.set_device(0)
    gpu=torch.cuda.get_device_properties(0)
    assert gpu.total_memory>=64*1024**3
    torch.cuda.set_per_process_memory_fraction(min(0.8,70*1024**3/gpu.total_memory),0)
    free,total=torch.cuda.mem_get_info();assert free>=cfg['resource']['start_free_bytes']
    caps={'model_load':2,'caption':8,'tokenizer':8,'tx_probability':8,'reference_render':24} if role=='tx' else {'model_load':1,'rx_prefix':24,'rx_render':24}
    ledger=Ledger(out,caps)
    args=argparse.Namespace(pn='0.06M',model_path=cfg['weights_root']+'/var_codec.pth',vae_type=16,
        vae_path=cfg['weights_root']+'/vae_d16_reg.ckpt',model_type='infinity_2b',cfg_insertion_layer=0,
        add_lvl_embeding_only_first_block=1,use_bit_label=1,rope2d_each_sa_layer=1,rope2d_normalized_by_hw=2,
        use_scale_schedule_embedding=0,sampling_per_bits=1,text_channels=2048,apply_spatial_patchify=0,
        use_flex_attn=0,cache_dir=str(cache),enable_model_cache=0,checkpoint_type='torch',seed=0,bf16=0)
    weights_audit={}
    def load():
        # Instrument the author's exact VAE loader to reject silently missing parameters.
        import infinity.models.bsq_vae.vae as vm
        native=vm.load_cnn
        def checked(model,state_dict,prefix,expand=False,use_linear=False):
            result=native(model,state_dict,prefix,expand,use_linear)
            got=set(result[2]);required=set(dict(model.named_parameters()))
            missing=sorted(required-got)
            weights_audit['vae']={'required_parameter_keys':len(required),'loaded_keys':len(got),'missing':missing,'unused_keys':sorted(result[1])}
            if missing:raise RuntimeError('VAE checkpoint missing parameters: '+str(missing))
            return result
        vm.load_cnn=checked
        tokenizer,textencoder=load_tokenizer(cfg['weights_root']+'/flan-t5-xl')
        vae=load_visual_tokenizer(args)
        prior=load_transformer(vae,args)  # Author uses strict load_state_dict.
        return tokenizer,textencoder,vae,prior
    text_tokenizer,text_encoder,vae,prior=ledger.call('model_load','shared',load)
    put(out/'model_load_audit.json',weights_audit)
    schedule=[(1,s,s) for s in SIDES]
    assert [(h,w) for _,h,w in util.dynamic_resolution_h_w[1.0]['0.06M']['scales']]==[(s,s) for s in SIDES]
    q_schedule=[(*p,int((i+1)//((len(schedule)//3)+1)+2)) for i,p in enumerate(schedule)]
    put(out/'runtime.json',{'pid':os.getpid(),'gpu_name':gpu.name,'gpu_uuid':str(getattr(gpu,'uuid','unavailable')),
        'torch':torch.__version__,'python':sys.version,'native_arguments':vars(args),'schedule':schedule,
        'mask_schedule':q_schedule,'cfg':3,'tau':0.5,'top_k':900,'top_p':0.97,'generation_seed':0,
        'matmul_tf32':False,'cudnn_tf32':False,'deterministic_algorithms':True,'started_unix':time.time()})
    def cdf_hash(prob):
        return digest(np.stack([probabilities_to_cumul(float(p)) for p in prob]).astype('<u8').tobytes())
    def render(tokens,e,key,kind):
        def go():
            prior.rng.manual_seed(0)
            with torch.inference_mode():
                image=receiver.decompress_cfg(prior,vae,schedule,caption,text_tokenizer,text_encoder,e-1,tokens,cfg_list=3,tau_list=0.5)
            torch.cuda.synchronize();assert tuple(image.shape)==(1,3,256,256) and bool(torch.isfinite(image).all())
            floats=image[0].float().cpu().numpy();pixels=image[0].mul(255).add(0.5).clamp(0,255).to(torch.uint8).cpu().numpy()
            with (out/(key+'.npz')).open('xb') as f:np.savez(f,pixels=pixels,float_output=floats)
            return {'pixel_sha256':digest(pixels.tobytes()),'float_sha256':digest(floats.tobytes()),'pixel_std':float(pixels.std()),
              'minimum':float(floats.min()),'maximum':float(floats.max()),'path':str(out/(key+'.npz'))}
        return ledger.call(kind,key,go)
    result=[]
    if role=='tx':
        from transformers import BlipProcessor,BlipForConditionalGeneration
        processor=BlipProcessor.from_pretrained(cfg['weights_root']+'/blip-image-captioning-base',local_files_only=True)
        blip=ledger.call('model_load','caption',lambda:BlipForConditionalGeneration.from_pretrained(cfg['weights_root']+'/blip-image-captioning-base',local_files_only=True).eval().cuda())
        blip.requires_grad_(False)
        sources=json.loads(pathlib.Path(a.sources).read_text())['a1_first8'];assert len(sources)==8
        wires=pathlib.Path(cfg['run'])/'wire_inputs';wires.mkdir()
        jobs=[]
        for s in sources:
            sid=f"{s['rank']:02d}";p=pin(s['archive'])
            with np.load(p,allow_pickle=False) as z:pixels=z['pixels'].copy()
            assert pixels.shape==(3,256,256) and pixels.dtype==np.uint8
            def caption_call():
                data=processor(images=Image.fromarray(pixels.transpose(1,2,0)),return_tensors='pt').to('cuda')
                with torch.inference_mode():generated=blip.generate(**data,do_sample=False,num_beams=3,max_length=20,min_length=5)
                return processor.decode(generated[0],skip_special_tokens=True)
            caption=ledger.call('caption',sid,caption_call)
            # Native encode_prompt truncates at 512. Reject before source coding if
            # that would silently change the transmitted caption's interpretation.
            assert len(text_tokenizer(caption,truncation=False).input_ids)<=512
            inp=torch.from_numpy(pixels.copy()).unsqueeze(0).cuda().float().div(255).mul(2).sub(1)
            def tokenize():
                with torch.inference_mode():
                    raw,_,_=vae.encode_for_raw_features(inp,scale_schedule=schedule)
                    _,tokens=util.mask_quant(vae,q_schedule,raw,'cuda')
                return tokens
            tokens=ledger.call('tokenizer',sid,tokenize)
            witnesses=[];original_prob=sender.get_prob
            def record_prob(*x,**kw):
                probs=original_prob(*x,**kw)
                witnesses.extend(cdf_hash(p[:,0].detach().cpu().tolist()) for p in probs)
                return probs
            sender.get_prob=record_prob
            def encode():
                with torch.inference_mode():return sender.encoding(args,prior,vae,schedule,caption,text_tokenizer,text_encoder,tokens)
            try:trans,modes,_=ledger.call('tx_probability',sid,encode)
            finally:sender.get_prob=original_prob
            with (out/(sid+'_truth.npz')).open('xb') as f:np.savez(f,**{f's{i+1}':t.cpu().numpy() for i,t in enumerate(tokens)})
            lengths=[]
            for e in range(1,8):
                payload,detail=pack(caption,trans[:e],modes[:e],e);key=f'{sid}_e{e}'
                (out/(key+'.arc')).open('xb').write(payload)
                lengths.append({'endpoint':e,**detail,'payload_sha256':digest(payload)})
                if e in (1,4,7):
                    (wires/(key+'.arc')).open('xb').write(payload)
                    jobs.append({'key':key,'path':str(wires/(key+'.arc')),'paid_profile':e,'generation_seed':0})
                    result.append({'key':key,'endpoint':e,**render(tokens[:e],e,key,'reference_render')})
            put(out/(sid+'_source.json'),{'source_id':s['source_id'],'source_index':s['source_index'],'pixels_sha256':digest(pixels.tobytes()),
              'caption':caption,'caption_bytes_hex':caption.encode().hex(),'cdf_hashes':witnesses,'lengths':lengths})
        put(wires/'jobs.json',jobs)
    else:
        decode,prefix_ast_sha=prefix_decoder(cfg['upstream']);put(out/'prefix_adapter.json',{'ast_sha256':prefix_ast_sha,'gt_reads':0,'loop_limit':'actual paid endpoint; full schedule retained for conditioning'})
        jobs=json.loads((pathlib.Path(cfg['run'])/'wire_inputs/jobs.json').read_text());assert len(jobs)==24
        for job in jobs:
            key=job['key'];e=job['paid_profile'];assert e in (1,4,7) and job['generation_seed']==0
            caption,trans,modes=unpack(pathlib.Path(job['path']).read_bytes(),e)
            cdfs=[];native_decompress=receiver.decompress_from_bit_list
            # The decoder's globals are a dedicated copy; interception records actual RX
            # CDFs and canonical recoding using only this receiver's probabilities.
            checked=decode.__globals__
            checks=[]
            def checked_block(bits,num,probabilities):
                result=native_decompress(bits,num,probabilities)
                if compress_to_bit_list(result,probabilities)!=bits:raise RuntimeError('Native source block canonical recoding failed')
                checks.append(True);return result
            checked['decompress_from_bit_list']=checked_block
            # Per-scale exact CDFs can be observed at get_logits without consulting TX.
            original_logits=prior.get_logits
            def record_logits(*x,**kw):
                logits=original_logits(*x,**kw)
                probs=(logits*2).reshape(-1,2).softmax(-1)[:,0].detach().cpu().tolist()
                cdfs.append(cdf_hash(probs));return logits
            prior.get_logits=record_logits
            def receive():
                with torch.inference_mode():return decode(args,prior,vae,schedule,caption,text_tokenizer,text_encoder,None,trans,modes)
            try:tokens=ledger.call('rx_prefix',key,receive)
            finally:prior.get_logits=original_logits
            assert len(tokens)==e and len(cdfs)==e
            with (out/(key+'_decoded.npz')).open('xb') as f:np.savez(f,**{f's{i+1}':t.cpu().numpy() for i,t in enumerate(tokens)})
            result.append({'key':key,'endpoint':e,'caption_bytes_hex':caption.encode().hex(),'cdf_hashes':cdfs,'canonical_blocks_checked':len(checks),
                **render(tokens,e,key,'rx_render')})
    torch.cuda.synchronize()
    put(out/'completion.json',{'status':'COMPLETE','counts':ledger.counts,'outputs':result,'blocked_reads':blocked,
        'max_allocated_bytes':torch.cuda.max_memory_allocated(),'max_reserved_bytes':torch.cuda.max_memory_reserved(),'ended_unix':time.time()})

if __name__=='__main__':main()

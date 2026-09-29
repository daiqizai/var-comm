"""Calibration-only verification of the failed clean-prior engineering gate."""
from probe import *
@torch.no_grad()
def main_verify():
 global SAFETY
 import probe
 cfg=read(CONFIG);configure_runtime();require_available();probe.SAFETY=Safety()
 vae,var=load_models(model_paths(),torch.device('cuda'))
 F,Fq,tokens,ids,bindings=load_calibration()
 stats=calibration(vae,F,Fq,tokens,ids,bindings,cfg)
 static=[frequency_log_probs(v,cfg['static_pseudocount']).float().cuda() for v in stats['counts']]
 variance=stats['s2'].clamp_min(cfg['variance_floor']);rows=[];literal_checks=[]
 for i in cfg['engineering_calibration_indices']:
  boundary();f=F[i:i+1].cuda();truth=split(tokens[i:i+1].cuda())
  for prior in ['A1','A2','V']:
   _,idx,metrics,logits=infer(vae,var,f,truth,prior,'TF',variance,static,details=True)
   rows.append(dict(calibration_index=i,prior=prior,mode='TF',eta=0,lambda_value=0 if prior=='A1' else 1,token_accuracy=float(torch.cat(idx,1).eq(tokens[i:i+1].cuda()).float().mean()),scale_accuracy=[r['acc'] for r in metrics]))
   if prior=='V':
    rest=f.clone()
    for k,pn in enumerate(PATCH_NUMS):
     d,z=distance(vae.quantize,rest,k)
     direct=((z[0,None,:].double()-vae.quantize.embedding.weight.double())**2).sum(-1)
     literal=fn.log_softmax(logits[k].double(),-1)[0,0]-direct/(2*float(variance[k]))
     actual=fn.log_softmax(logits[k],-1)[0,0]-d[0,0]/(2*float(variance[k]))
     err=float((literal-actual.double()).abs().max())
     assert err<0.001,(i,k,err)
     assert int(literal.argmax())==int(actual.argmax()),(i,k,'argmax mismatch')
     literal_checks.append(dict(calibration_index=i,scale=k+1,position=0,max_score_error_vs_direct_float64=err,argmax_exact=True))
     rest.sub_(contribution(vae.quantize,truth[k],k))
 result=dict(status='REAL_CALIBRATION_GATE2_IMPLEMENTATION_CROSSCHECK_COMPLETE',rows=rows,means={p:float(np.mean([r['token_accuracy'] for r in rows if r['prior']==p])) for p in ['A1','A2','V']},literal_likelihood_checks=literal_checks,source_ids=[ids[i] for i in cfg['engineering_calibration_indices']],inference='At eta=0 the specified denominator retains positive s_k^2. A prior can therefore override the exact nearest codeword; clean MAP is not mathematically required to reproduce the tokenizer.',new_development_accessed=False,holdout_accessed=False,formula_changed=False)
 write_json(OUT/'gate2_crosscheck.json',result);print(json.dumps(result['means']),flush=True)
if __name__=='__main__':
 try:main_verify()
 except ResourceBusy as e:print('SAFE_PROBE_PAUSE',e,flush=True);raise SystemExit(75)
 except Exception:
  write_json(OUT/'gate2_crosscheck_failure.json',dict(traceback=traceback.format_exc(),time=time.time()));raise

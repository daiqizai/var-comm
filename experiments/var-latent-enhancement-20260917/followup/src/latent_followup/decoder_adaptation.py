from __future__ import annotations
import argparse,copy,hashlib,json,signal,time
from pathlib import Path
import numpy as np,torch
from latent_enhancement.runtime import digest,image_losses,model_paths,perceptual_model,save_torch,settings,write_json
from latent_enhancement.latent import ContinuousDecoder
from latent_enhancement_b.common import load_gate,scale_statistics
from latent_enhancement_b.data import MatchedPopulation
from latent_enhancement_b.model import build_arms,latent_errors,render_received
from latent_enhancement.training import PairedOrder
from var_comm.next_scale_prior import load_models,state_sha256
from latent_followup.run_identity import checked_checkpoint, register_run, load_resume

PROJECT=Path(__file__).resolve().parents[5];VAR_COMM=PROJECT;EXP=VAR_COMM/'experiments/var-latent-enhancement-20260917';ROOT=VAR_COMM/'outputs/VAR-LATENT-ENHANCEMENT-20260917';CONFIG=EXP/'followup/decoder_adaptation.json';OUT=ROOT/'followup/decoder_adaptation_v1'

def read_json(p):return json.loads(Path(p).read_text())
def atomic_json(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def write_rows(path,rows):
 Path(path).parent.mkdir(parents=True,exist_ok=True)
 fields=list(dict.fromkeys(k for r in rows for k in r));
 with Path(path).open('w',newline='') as h:
  w=__import__('csv').DictWriter(h,fieldnames=fields);w.writeheader();w.writerows(rows)
def source_batch(pop,indices,device):
 return {k:v[indices].to(device) for k,v in pop.values.items()} | {'target':pop.images[indices].to(device).float().div(255)}
def calibration(models,decoders,pop,scale,perceptual,device,step):
 rows=[]
 for model in models.values(): model.eval()
 for start in range(0,len(pop),4):
  # Use all registered SNR/noise rows through MatchedPopulation's actual RX grid.
  for snr_i in range(len(pop.snrs)):
   for noise_i in range(len(pop.seeds)):
    ids=torch.arange(start,min(start+4,len(pop)))
    values=pop.batch(ids,torch.full((len(ids),),snr_i,dtype=torch.long),torch.full((len(ids),),noise_i,dtype=torch.long),[pop.seeds[noise_i]]*len(ids),device)
    for name,model in models.items():
     with torch.no_grad():
      latent,_=model.receive_training_sample(values['F'],values['Fb_TX'],values['Fb_RX'],values['snr_db'],values['rx_status'],values['standard_noise']);pred=render_received(decoders[name],latent,values['rx_status']);mse,lp=image_losses(pred,values['target'],perceptual);aux=latent_errors(latent,values['F'],scale,values['rx_status'])
     for j in range(len(ids)):rows.append({'branch':name,'source_index':int(ids[j]),'snr_db':float(pop.snrs[snr_i]),'seed':int(pop.seeds[noise_i]),'mse':float(mse[j]),'lpips':float(lp[j]),'latent':float(aux[j]),'utility':float(mse[j]+.1*lp[j]+.01*aux[j])})
 return rows

def decoder_diagnostics(decoder,pop,device,perceptual):
 out=[]
 for name,values in [('F',pop.source.values['F']),('Fq',pop.source.values['Fq']),('Fb_TX',pop.source.values['Fb_TX'])]:
  vals=[]
  with torch.no_grad():
   for start in range(0,len(pop),4):
    latent=values[start:start+4].to(device);target=pop.source.images[start:start+4].to(device).float().div(255);pred=decoder(latent);mse,lp=image_losses(pred,target,perceptual);vals.extend(zip(mse.cpu().tolist(),lp.cpu().tolist()))
  out.append({'input':name,'psnr_db':float(np.mean([-10*np.log10(max(x[0],1e-12)) for x in vals])),'lpips':float(np.mean([x[1] for x in vals])),'sources':len(vals)})
 return out

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',default=str(OUT));args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True);config=read_json(CONFIG)
 torch.set_num_threads(6);torch.set_num_interop_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 device=torch.device('cuda:0');vae,var=load_models(model_paths(),device);del var;torch.cuda.empty_cache();gate=load_gate();scale=scale_statistics(device);perceptual=perceptual_model(device);train=MatchedPopulation('train');cal=MatchedPopulation('calibration');recipe=settings()['stage_B'];
 selected=read_json(ROOT/'stage_B_v1/training/selected_enhancement1024.json');checkpoint=checked_checkpoint(selected,VAR_COMM);base_state=torch.load(checkpoint,map_location='cpu',weights_only=True);arms=build_arms(train.shape,scale,recipe).to(device);arms.load_state_dict(base_state['arms'],strict=True);control=arms['enhancement1024'].train();adapt=copy.deepcopy(control).train();
 frozen=__import__('latent_enhancement_b.common',fromlist=['load_decoder']).load_decoder(vae,device);decoder_control=frozen.eval().requires_grad_(False);decoder_adapt=copy.deepcopy(frozen).train().requires_grad_(True)
 # Optimizer state contains mutable tensor objects.  Deep-copy it for each
 # arm so restoring one branch cannot alias moments/steps in the other.
 opt_control=torch.optim.AdamW(control.parameters(),lr=recipe['learning_rate'],weight_decay=recipe['weight_decay']);opt_control.load_state_dict(copy.deepcopy(base_state['optimizers']['enhancement1024']));opt_adapt=torch.optim.AdamW(adapt.parameters(),lr=recipe['learning_rate'],weight_decay=recipe['weight_decay']);opt_adapt.load_state_dict(copy.deepcopy(base_state['optimizers']['enhancement1024']));opt_dc=torch.optim.AdamW(decoder_adapt.parameters(),lr=config['decoder_optimizer']['learning_rate'],weight_decay=config['decoder_optimizer']['weight_decay'])
 order=PairedOrder(len(train),recipe['data_seed']);channel_rng=torch.Generator().manual_seed(recipe['channel_seed']);state={'step':0,'last_full':-1,'selection':{'communication_continuation_Dc_frozen':{'step':0,'utility':None},'communication_plus_Dc_adaptation':{'step':0,'utility':None}},'plateau':{'communication_continuation_Dc_frozen':0,'communication_plus_Dc_adaptation':0},'update_seconds':0.0,'calibration_seconds':0.0,'calibration_calls':0}
 registration_sha=register_run(out,CONFIG,[ROOT/'stage_B_v1/training/selected_enhancement1024.json'])
 if (out/'latest.json').exists():
  latest=read_json(out/'latest.json');saved=load_resume(latest,VAR_COMM,registration_sha);control.load_state_dict(saved['control']);adapt.load_state_dict(saved['adapt']);decoder_adapt.load_state_dict(saved['decoder_adapt']);opt_control.load_state_dict(copy.deepcopy(saved['opt_control']));opt_adapt.load_state_dict(copy.deepcopy(saved['opt_adapt']));opt_dc.load_state_dict(copy.deepcopy(saved['opt_dc']));order.load_state_dict(saved['order']);channel_rng.set_state(saved['channel_rng']);torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng']);state=saved['state'];state.setdefault('calibration_calls',0);state.setdefault('calibration_seconds',0.0)
 def checkpoint(reason):
  path=out/'checkpoints'/f"step_{state['step']:05d}.pt";save_torch(path,{'registration_sha256':registration_sha,'control':control.state_dict(),'adapt':adapt.state_dict(),'decoder_adapt':decoder_adapt.state_dict(),'opt_control':opt_control.state_dict(),'opt_adapt':opt_adapt.state_dict(),'opt_dc':opt_dc.state_dict(),'order':order.state_dict(),'channel_rng':channel_rng.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'state':state,'selected_stageB_sha256':selected['checkpoint_sha256']});atomic_json(out/'latest.json',{'path':str(path),'sha256':digest(path),'step':state['step'],'reason':reason,'state':state})
 def full_calibration():
  nonlocal state
  calibration_start=time.perf_counter()
  models={'communication_continuation_Dc_frozen':control.eval(),'communication_plus_Dc_adaptation':adapt.eval()}; decs={'communication_continuation_Dc_frozen':decoder_control,'communication_plus_Dc_adaptation':decoder_adapt.eval()};rows=calibration(models,decs,cal,scale,perceptual,device,state['step']);write_rows(out/'calibration'/f"full_{state['step']:05d}.csv",rows)
  for name in models:
   utility=float(np.mean([r['utility'] for r in rows if r['branch']==name]));old=state['selection'][name]['utility'];improved=old is None or utility<old-abs(utility)*.001;state['plateau'][name]=0 if improved else state['plateau'][name]+1
   if old is None or utility<old:state['selection'][name]={'step':state['step'],'utility':utility}
  state['last_full']=state['step'];state['calibration_calls']=int(state.get('calibration_calls',0))+1;state['calibration_seconds']+=time.perf_counter()-calibration_start;checkpoint('full_calibration')
 if state['step']==0: atomic_json(out/'diagnostics_step0.json',{'frozen_Dc':decoder_diagnostics(decoder_control,cal,device,perceptual),'adapt_Dc_initial':decoder_diagnostics(decoder_adapt,cal,device,perceptual)})
 stop_requested=[False]
 def request_stop(*_):stop_requested[0]=True
 signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
 while state['step']<config['updates']['maximum'] and not all(state['step']>=config['updates']['minimum'] and v>=3 for v in state['plateau'].values()):
  control.train();adapt.train();decoder_adapt.train()
  indices=order.next(recipe['logical_batch_size']);sni=torch.randint(len(train.snrs),(len(indices),),generator=channel_rng);noi=torch.randint(len(train.seeds),(len(indices),),generator=channel_rng);eseed=torch.full_like(indices,recipe['channel_seed']+state['step']);batch=train.batch(indices,sni,noi,eseed,device);t=time.perf_counter();
  for model,decoder,opt in [(control,decoder_control,opt_control),(adapt,decoder_adapt,opt_adapt)]:
   if model is adapt: decoder_adapt.zero_grad(set_to_none=True)
   opt.zero_grad(set_to_none=True)
   for start in range(0,len(indices),recipe['microbatch_size']):
    sl=slice(start,min(start+recipe['microbatch_size'],len(indices)));part={k:v[sl] for k,v in batch.items()};latent,_=model.receive_training_sample(part['F'],part['Fb_TX'],part['Fb_RX'],part['snr_db'],part['rx_status'],part['standard_noise']);pred=render_received(decoder,latent,part['rx_status']);mse,lp=image_losses(pred,part['target'],perceptual);aux=latent_errors(latent,part['F'],scale,part['rx_status']);((mse+.1*lp+.01*aux).sum()/len(indices)).backward()
   torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
   if model is adapt: decoder_adapt.zero_grad(set_to_none=True)
  # Dc adaptation has a separate optimizer on the same adaptation graph; redo image path for the Dc-only gradient.
  opt_adapt.zero_grad(set_to_none=True)
  opt_dc.zero_grad(set_to_none=True)
  for start in range(0,len(indices),recipe['microbatch_size']):
   sl=slice(start,min(start+recipe['microbatch_size'],len(indices)));part={k:v[sl] for k,v in batch.items()};latent,_=adapt.receive_training_sample(part['F'],part['Fb_TX'],part['Fb_RX'],part['snr_db'],part['rx_status'],part['standard_noise']);pred=render_received(decoder_adapt,latent,part['rx_status']);mse,lp=image_losses(pred,part['target'],perceptual);aux=latent_errors(latent,part['F'],scale,part['rx_status']);((mse+.1*lp+.01*aux).sum()/len(indices)).backward()
  torch.nn.utils.clip_grad_norm_(decoder_adapt.parameters(),1);opt_dc.step();state['step']+=1;state['update_seconds']+=time.perf_counter()-t
  if stop_requested[0]:
   checkpoint('safe_boundary_pause');atomic_json(out/'status.json',{'status':'PAUSED_AT_PAIRED_UPDATE','step':state['step']});return
  if state['step']%config['updates']['full_calibration_interval']==0:full_calibration()
  elif state['step']%config['updates']['checkpoint_interval']==0:checkpoint('regular')
 if state['last_full']!=state['step']:full_calibration()
 atomic_json(out/'diagnostics_final.json',{'frozen_Dc':decoder_diagnostics(decoder_control,cal,device,perceptual),'adapted_Dc':decoder_diagnostics(decoder_adapt,cal,device,perceptual)})
 selected_records={}
 for name,record in state['selection'].items():
  selected_path=out/'checkpoints'/f"step_{int(record['step']):05d}.pt"
  if not selected_path.exists(): raise FileNotFoundError(f'selected checkpoint missing for {name}: {selected_path}')
  selected_records[name]={**record,'checkpoint':str(selected_path),'checkpoint_sha256':digest(selected_path)}
 atomic_json(out/'completion.json',{'status':'ALTERNATING_COMMUNICATION_AND_DECODER_ADAPTATION_COMPLETE','algorithm':'alternating_communication_update_then_decoder_update','updates':state['step'],'selection':state['selection'],'selected_checkpoints':selected_records,'plateau':state['plateau'],'calibration_calls':state.get('calibration_calls',0),'calibration_seconds':state.get('calibration_seconds',0.0),'new_holdout_used':False,'stop_reason':'safety_cap_or_three_plateau_checks'})
if __name__=='__main__':main()

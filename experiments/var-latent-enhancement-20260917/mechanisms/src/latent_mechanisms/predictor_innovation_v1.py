from __future__ import annotations
import argparse,json,time,copy,hashlib
from pathlib import Path
import torch
from latent_enhancement.runtime import image_losses,model_paths,perceptual_model,save_torch,settings
from latent_enhancement_b.common import load_decoder,scale_statistics
from latent_enhancement_b.data import MatchedPopulation
from latent_enhancement_b.model import build_arms,latent_errors,render_received
from var_comm.next_scale_prior import load_models

PROJECT=Path(__file__).resolve().parents[5];VAR=PROJECT;ROOT=VAR/'outputs/VAR-LATENT-ENHANCEMENT-20260917';EXP=VAR/'experiments/var-latent-enhancement-20260917';CONFIG=EXP/'mechanisms/predictor_innovation_config.json'
def read(p):return json.loads(Path(p).read_text())
def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',default=str(ROOT/'followup/predictor_innovation_v1'));a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True);cfg=read(CONFIG);device=torch.device('cuda:0');vae,var=load_models(model_paths(),device);del var;torch.cuda.empty_cache();scale=scale_statistics(device);recipe=settings()['stage_B'];train=MatchedPopulation('train');
 selected=read(ROOT/'stage_B_v1/training/selected_enhancement1024.json');
 if digest(selected['checkpoint'])!=selected['checkpoint_sha256']: raise RuntimeError('selected enhancement1024 checkpoint changed')
 base=torch.load(selected['checkpoint'],map_location='cpu',weights_only=True);arms=build_arms(train.shape,scale,recipe).to(device);arms.load_state_dict(base['arms'],strict=True);control=arms['enhancement1024'];candidate=build_arms(train.shape,scale,recipe).to(device)['enhancement1024'];candidate.load_state_dict({k.split('enhancement1024.',1)[1]:v for k,v in base['arms'].items() if k.startswith('enhancement1024.')},strict=True);predictor_arms=build_arms(train.shape,scale,recipe).to(device);predsel=read(ROOT/'stage_B_v1/training/selected_receiver_only_refiner.json');
 if digest(predsel['checkpoint'])!=predsel['checkpoint_sha256']: raise RuntimeError('selected predictor checkpoint changed')
 predck=torch.load(predsel['checkpoint'],map_location='cpu',weights_only=True);predictor_arms.load_state_dict(predck['arms'],strict=True);predictor=predictor_arms['receiver_only_refiner'];predictor.eval().requires_grad_(False);decoder=load_decoder(vae,device);perceptual=perceptual_model(device);optc=torch.optim.AdamW(control.parameters(),lr=recipe['learning_rate'],weight_decay=recipe['weight_decay']);optx=torch.optim.AdamW(candidate.parameters(),lr=recipe['learning_rate'],weight_decay=recipe['weight_decay']);
 # The two arms start from independent optimizer copies.  A shared state dict
 # aliases Adam moments and makes the control update contaminate the innovation arm.
 optc.load_state_dict(copy.deepcopy(base['optimizers']['enhancement1024']));optx.load_state_dict(copy.deepcopy(base['optimizers']['enhancement1024']));order=torch.randperm(len(train));
 status={'status':'PREDICTOR_INNOVATION_TRAINING','algorithm':'matched_tx_alternating_two_arm','step':0,'updates':cfg['updates'],'predictor_frozen':True,'control_checkpoint_sha256':selected['checkpoint_sha256'],'predictor_checkpoint_sha256':predsel['checkpoint_sha256']};(out/'status.json').write_text(json.dumps(status,indent=2)+'\n')
 for step in range(1,cfg['updates']['maximum']+1):
  idx=order[((step-1)*16)%len(train):((step-1)*16)%len(train)+16];
  if len(idx)<16:idx=order[:16]
  sni=torch.randint(len(train.snrs),(16,));noi=torch.randint(len(train.seeds),(16,));batch=train.batch(idx,sni,noi,[recipe['channel_seed']+step]*16,device);snr=batch['snr_db'];
  with torch.no_grad():
   txstatus=torch.tensor([[1.,1.,8.]],device=device).expand(16,3);ptx=predictor.receiver(None,batch['Fb_TX'],snr,txstatus);prx=predictor.receiver(None,batch['Fb_RX'],snr,batch['rx_status'])
  for model,opt,innovation in ((control,optc,False),(candidate,optx,True)):
   opt.zero_grad(set_to_none=True);target_base=ptx if innovation else batch['Fb_TX'];rx_base=prx if innovation else batch['Fb_RX'];continuous_arg=batch['F'];latent,_=model.receive_training_sample(continuous_arg,target_base,rx_base,snr,batch['rx_status'],batch['standard_noise']);pred=render_received(decoder,latent,batch['rx_status']);mse,lp=image_losses(pred,batch['target'],perceptual);aux=latent_errors(latent,batch['F'],scale,batch['rx_status']);loss=(mse+.1*lp+.01*aux).mean();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
  if step%500==0:
   status.update({'step':step,'last_loss':float(loss),'timestamp':time.time()});(out/'status.json').write_text(json.dumps(status,indent=2)+'\n');save_torch(out/f'step_{step:05d}.pt',{'control':control.state_dict(),'candidate':candidate.state_dict(),'opt_control':optc.state_dict(),'opt_candidate':optx.state_dict(),'predictor_state':predictor.state_dict(),'step':step})
 print('predictor innovation training complete')
if __name__=='__main__':main()
